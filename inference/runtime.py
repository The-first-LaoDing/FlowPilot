"""FlowPilotRuntime: FlowPilot model + VAE + basis + scheduler + stats with
inference and warmup entry points.
"""

import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class InferResult:
    wp_body:       np.ndarray   # [48, 12]  body-frame waypoints (p, v, a, j)
    ctrl_pts_body: np.ndarray   # [8, 3]    Bernstein control points (body, metres)


@dataclass(frozen=True)
class _Schedule:
    t_ids:    Tuple[torch.Tensor, ...]
    d_sigmas: Tuple[float, ...]


class FlowPilotRuntime:
    """FlowPilot model + VAE + Bernstein basis + flow-matching scheduler + stats.

    Construction:
      * Builds the Phase-2 model and loads the checkpoint.
      * Loads the Wan2.2 VAE condition-depth encoder.
      * Builds the Bernstein basis (degree=7, T=1.6, num_waypoints=48, hz=30)
        and the flow-matching scheduler (1000 flow timesteps, shift=5.0).
      * Loads action and coefficient normalization stats to the device.

    Warmup is NOT automatic. Call `.warmup()` to trigger a zero-input forward
    pass.

    `infer()` always returns InferResult(wp_body, ctrl_pts_body); the
    simulation only consumes `wp_body`.

    Required cfg fields (see configs/runtime/rviz.py):
        phase2_ckpt, vae_pth, action_stats_path, coeff_stats_path, device,
        num_layers, v_cmd, num_steps.
    """

    def __init__(self, cfg):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)
        self._schedule_cache: Dict[int, _Schedule] = {}
        self._v_cmd_cache: Dict[float, torch.Tensor] = {}
        self._video_latent_buf: Optional[torch.Tensor] = None
        self._free_latent_buf:  Optional[torch.Tensor] = None
        self.last_timings_ms: Dict[str, float] = {}
        self.profile_infer = bool(getattr(cfg, "profile_infer", False))
        self.use_model_amp = bool(getattr(cfg, "use_model_amp", False))
        self.model_amp_dtype = self._resolve_amp_dtype(
            getattr(cfg, "model_amp_dtype", "bfloat16")
        )
        self.allow_tf32 = bool(getattr(cfg, "allow_tf32", True))
        self.cudnn_benchmark = bool(getattr(cfg, "cudnn_benchmark", True))
        self.async_vae = bool(getattr(cfg, "async_vae", False))
        # Capture-age = (now - depth submit time). Rejects cached latents whose
        # underlying depth is older than this. Falls back to legacy
        # `async_vae_max_age_s` only if the new key isn't set (transitional).
        if hasattr(cfg, "async_vae_max_capture_age_s"):
            self.async_vae_max_capture_age_s = float(cfg.async_vae_max_capture_age_s)
        else:
            self.async_vae_max_capture_age_s = float(getattr(cfg, "async_vae_max_age_s", 0.06))
        self._vae_lock = threading.Lock()
        self._async_lock = threading.Condition()
        self._async_pending_depth: Optional[np.ndarray] = None
        self._async_pending_submit_time: float = 0.0
        self._async_pending_seq = 0
        self._async_latest_latent: Optional[torch.Tensor] = None
        self._async_latest_seq = 0
        # Time of `submit_depth_for_encoding` call for the depth frame whose
        # latent is currently in `_async_latest_latent`. NOT the time the worker
        # finished encoding. Used to compute capture-age.
        self._async_latest_capture_time: float = 0.0
        self._async_stop = False
        self._async_thread: Optional[threading.Thread] = None
        self._async_last_encode_ms = 0.0
        # Dedicated CUDA stream for VAE worker so that VAE encode kernels do
        # not serialize against model.forward on the default stream.
        self._vae_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        # Prev-trajectory conditioning state (see spec §7.1).
        self._prev_ctrl_pts_world: Optional[torch.Tensor] = None      # [8, 3]
        self._prev_free_norm_for_l2: Optional[torch.Tensor] = None    # [1, 5, 3]
        self.last_free_norm_l2: float = 0.0
        self._warned_no_frame_info: bool = False
        # When False, every infer() runs as cold-start (prev_mask=0). Useful for
        # ablation. Defaults to True for backward compatibility.
        self._use_prev_conditioning: bool = bool(
            getattr(cfg, "use_prev_conditioning", False)
        )
        self._configure_cuda_backends()
        self._build_model()
        self._load_stats()
        if self.async_vae:
            self.start_async_vae()

    def _configure_cuda_backends(self) -> None:
        if self.device.type != "cuda":
            return
        torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.benchmark = self.cudnn_benchmark

    @staticmethod
    def _resolve_amp_dtype(name: str) -> torch.dtype:
        name = str(name).lower()
        if name in ("bf16", "bfloat16"):
            return torch.bfloat16
        if name in ("fp16", "float16", "half"):
            return torch.float16
        if name in ("fp32", "float32"):
            return torch.float32
        raise ValueError(f"Unsupported model_amp_dtype: {name!r}")

    def _build_model(self) -> None:
        # Lazy imports keep flowpilot.* off the import path until a runtime is
        # actually constructed (the `flowpilot` package is a sibling of this one).
        from flowpilot.models.flowpilot   import FlowPilotPhase2
        from flowpilot.models.poly_action import BernsteinActionBasis
        from flowpilot.scheduler          import FlowMatchScheduler

        self.model = FlowPilotPhase2(
            num_layers=self.cfg.num_layers, d_video=512, d_action=256,
            num_heads=8, ffn_dim_video=2048, ffn_dim_action=1024,
            z_dim=48, num_actions=5, latent_t=3, grid_h=6, grid_w=10,
        ).to(self.device)
        ckpt = torch.load(self.cfg.phase2_ckpt, map_location=self.device,
                          weights_only=True)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

        self._build_condition_encoder()
        self.basis = BernsteinActionBasis(
            degree=7, T=1.6, num_waypoints=48, hz=30, device=self.device,
        )
        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=1000, shift=5.0,
        )

    def _build_condition_encoder(self) -> None:
        from flowpilot.models.vae_encoder import WanVAEEncoder

        self.vae = WanVAEEncoder(
            self.cfg.vae_pth, dtype=torch.bfloat16, device=self.device
        ).eval()

    def _load_stats(self) -> None:
        astats = torch.load(self.cfg.action_stats_path,
                            map_location=self.device, weights_only=True)
        cstats = torch.load(self.cfg.coeff_stats_path,
                            map_location=self.device, weights_only=True)
        self.sg_mean   = astats['sg_mean'].to(self.device)
        self.sg_std    = astats['sg_std'].to(self.device)
        self.free_mean = cstats['free_mean'].to(self.device)
        self.free_std  = cstats['free_std'].to(self.device)

    def _get_schedule(self, num_steps: int) -> _Schedule:
        num_steps = int(num_steps)
        cached = self._schedule_cache.get(num_steps)
        if cached is not None:
            return cached

        sigmas = self.scheduler.inference_sigmas(num_steps).to(self.device)
        t_linear = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        schedule = _Schedule(
            t_ids=tuple(
                (t_linear[i] * 999).long().clamp(0, 999).unsqueeze(0)
                for i in range(num_steps)
            ),
            d_sigmas=tuple(
                float((sigmas[i + 1] - sigmas[i]).item())
                for i in range(num_steps)
            ),
        )
        self._schedule_cache[num_steps] = schedule
        return schedule

    def _get_v_cmd_tensor(self, v_cmd: float) -> torch.Tensor:
        key = float(v_cmd)
        cached = self._v_cmd_cache.get(key)
        if cached is None:
            cached = torch.tensor([key], dtype=torch.float32, device=self.device)
            self._v_cmd_cache[key] = cached
        return cached

    def _ensure_latent_buffers(self) -> None:
        if self._video_latent_buf is None or self._video_latent_buf.device != self.device:
            self._video_latent_buf = torch.empty(
                1, 48, 3, 6, 10, dtype=torch.float32, device=self.device
            )
        if self._free_latent_buf is None or self._free_latent_buf.device != self.device:
            self._free_latent_buf = torch.empty(
                1, 5, 3, dtype=torch.float32, device=self.device
            )

    def _start_profile(self):
        if not (self.profile_infer and self.device.type == "cuda"):
            self.last_timings_ms = {}
            return None, None

        events = []

        def mark(name: str) -> None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            events.append((name, ev))

        mark("start")
        return mark, events

    def _finish_profile(self, events) -> None:
        if not events:
            return
        events[-1][1].synchronize()
        self.last_timings_ms = {
            name: prev_ev.elapsed_time(ev)
            for (prev_name, prev_ev), (name, ev) in zip(events, events[1:])
        }

    def start_async_vae(self) -> None:
        """Start a background worker that encodes the newest submitted depth."""
        if self._async_thread is not None:
            return
        self._async_stop = False
        self._async_thread = threading.Thread(
            target=self._async_vae_loop, name="flowpilot-vae-cache", daemon=True
        )
        self._async_thread.start()

    def shutdown(self) -> None:
        """Stop background workers. Safe to call from atexit."""
        with self._async_lock:
            self._async_stop = True
            self._async_lock.notify_all()
        if self._async_thread is not None:
            self._async_thread.join(timeout=1.0)
            self._async_thread = None

    def submit_depth_for_encoding(self, cond_depth: np.ndarray) -> None:
        """Queue the newest depth frame for async VAE encoding.

        The queue is intentionally size 1: depth frames are perishable, so the
        worker should always encode the newest available frame and drop stale
        pending work. Submit time (`time.perf_counter()` at this call) is
        captured alongside the frame so consumers can later compute the
        capture-age of the resulting latent — i.e. how long ago the underlying
        depth was actually observed, not how long ago encode finished.
        """
        if not getattr(self, "async_vae", False):
            return
        depth = np.asarray(cond_depth, dtype=np.float32).copy()
        submit_time = time.perf_counter()
        with self._async_lock:
            self._async_pending_depth = depth
            self._async_pending_submit_time = submit_time
            self._async_pending_seq += 1
            self._async_lock.notify()

    def _get_async_cond_latent(self) -> Optional[torch.Tensor]:
        with self._async_lock:
            latent = self._async_latest_latent
            if latent is None:
                return None
            age = time.perf_counter() - self._async_latest_capture_time
            if age > self.async_vae_max_capture_age_s:
                return None
            # Clone so the worker can safely overwrite `_async_latest_latent`
            # while the caller still holds the result.
            return latent.clone()

    def get_async_latent_capture_age(self) -> Optional[float]:
        """Seconds since the cached latent's depth was submitted, or None if
        no latent is cached. Diagnostic / instrumentation use only."""
        with self._async_lock:
            if self._async_latest_latent is None:
                return None
            return time.perf_counter() - self._async_latest_capture_time

    def _async_vae_loop(self) -> None:
        if self.device.type == "cuda":
            device_ctx = torch.cuda.device(self.device)
            stream_ctx = (
                torch.cuda.stream(self._vae_stream)
                if self._vae_stream is not None
                else nullcontext()
            )
        else:
            device_ctx = nullcontext()
            stream_ctx = nullcontext()

        with device_ctx:
            while True:
                with self._async_lock:
                    while self._async_pending_depth is None and not self._async_stop:
                        self._async_lock.wait()
                    if self._async_stop:
                        return
                    depth = self._async_pending_depth
                    seq = self._async_pending_seq
                    submit_time = self._async_pending_submit_time
                    self._async_pending_depth = None

                t0 = time.perf_counter()
                try:
                    with stream_ctx:
                        latent = self._encode_depth_to_latent(depth)
                        # Synchronize so encode_ms reflects real GPU completion
                        # and the latent is materialized before publication.
                        if self._vae_stream is not None:
                            self._vae_stream.synchronize()
                except Exception as exc:
                    print(f"[async_vae] encode failed: {exc}")
                    continue
                encode_ms = (time.perf_counter() - t0) * 1000.0

                with self._async_lock:
                    if seq >= self._async_latest_seq:
                        self._async_latest_latent = latent
                        self._async_latest_seq = seq
                        # Carry the original submit time through, so capture-age
                        # measures depth-arrival → now, not encode-completion → now.
                        self._async_latest_capture_time = submit_time
                        self._async_last_encode_ms = encode_ms

    def _encode_depth_to_latent(self, cond_depth: np.ndarray) -> torch.Tensor:
        cond_np = np.asarray(cond_depth, dtype=np.float32)
        cond_frame = torch.from_numpy(cond_np).unsqueeze(0).unsqueeze(0).to(
            self.device, non_blocking=True
        )
        with self._vae_lock, torch.inference_mode():
            return self.vae.encode(cond_frame.unsqueeze(1)).float()

    def reset_prev(self) -> None:
        """Clear cached previous trajectory and diagnostic state.

        Call on goal switch or any event that breaks trajectory continuity;
        the next infer() call falls back to null-prev (cold start)."""
        self._prev_ctrl_pts_world   = None
        self._prev_free_norm_for_l2 = None
        self.last_free_norm_l2      = 0.0
        self._warned_no_frame_info  = False

    def _invalidate_stale_prev_if_no_frame_info(self, r_wb, pos_world) -> None:
        """If caches exist but caller didn't supply (r_wb, pos_world), drop both
        caches so the next call cannot consume a stale prev. Also emits a single
        warning per runtime instance the first time this happens. Idempotent."""
        have_frame_info = (r_wb is not None and pos_world is not None)
        if not have_frame_info:
            if not self._warned_no_frame_info:
                print(
                    "[FlowPilotRuntime] WARNING: infer() called without "
                    "r_wb/pos_world; prev-trajectory conditioning disabled. "
                    "Caller should pass both kwargs to enable temporal coherence. "
                    "(Logged once per runtime instance.)"
                )
                self._warned_no_frame_info = True
            if self._prev_ctrl_pts_world is not None:
                self._prev_ctrl_pts_world   = None
                self._prev_free_norm_for_l2 = None

    def warmup(self) -> None:
        """Run one zero-input inference to warm up the CUDA/cuDNN caches."""
        self.infer(
            np.zeros((96, 160), dtype=np.float32),
            np.zeros(25,         dtype=np.float32),
        )

    @torch.inference_mode()
    def infer(
        self,
        cond_depth: np.ndarray,
        state_goal: np.ndarray,
        *,
        v_cmd:     Optional[float] = None,
        num_steps: Optional[int]   = None,
        r_wb:      Optional[np.ndarray] = None,
        pos_world: Optional[np.ndarray] = None,
    ) -> InferResult:
        """Run one FlowPilot inference step.

        Port of the `infer_one_step` loop from both scripts. Always returns
        both wp_body and ctrl_pts_body.

        Args:
            cond_depth: [96, 160] float32, normalized to [0, max_depth_m/20].
            state_goal: [25] float32 — body-frame state + goal (see state.py).
            v_cmd:     Commanded speed (m/s). None => cfg.v_cmd.
            num_steps: Flow-matching inference steps. None => cfg.num_steps.
            r_wb:      [3, 3] world<-body rotation at this call. Required to use a
                       cached prev anchor; if None, falls back to null prev.
            pos_world: [3] drone position in world. Same semantics as r_wb.

        Returns:
            InferResult(wp_body [48,12], ctrl_pts_body [8,3]).
        """
        v_cmd     = self.cfg.v_cmd     if v_cmd     is None else v_cmd
        num_steps = self.cfg.num_steps if num_steps is None else num_steps
        device    = self.device
        mark, events = self._start_profile()

        sg_np   = np.asarray(state_goal, dtype=np.float32)
        sg_t = torch.from_numpy(sg_np).unsqueeze(0).to(device, non_blocking=True)

        sg_norm    = (sg_t - self.sg_mean) / self.sg_std
        init_pva   = sg_t[:, 0:9]
        constrained = self.basis.constrained_coeffs(init_pva)
        v_cmd_t    = self._get_v_cmd_tensor(v_cmd)

        cond_latent = self._get_async_cond_latent() if self.async_vae else None
        if cond_latent is not None:
            if mark:
                mark("input_copy")
                mark("latent_fetch")
        else:
            if mark:
                mark("input_copy")
            cond_latent = self._encode_depth_to_latent(cond_depth)
            if mark:
                mark("vae_encode")

        self._ensure_latent_buffers()
        video_latent = self._video_latent_buf.normal_()
        video_latent[:, :, 0:1].copy_(cond_latent)
        free_latent = self._free_latent_buf.normal_()
        schedule = self._get_schedule(num_steps)
        if mark:
            mark("latent_init")

        have_frame_info = (r_wb is not None and pos_world is not None)
        if (
            self._use_prev_conditioning
            and self._prev_ctrl_pts_world is not None
            and have_frame_info
        ):
            r_wb_t = torch.as_tensor(np.asarray(r_wb, dtype=np.float32), device=device)
            p_t    = torch.as_tensor(np.asarray(pos_world, dtype=np.float32), device=device)
            prev_ctrl_pts_body = (
                (self._prev_ctrl_pts_world - p_t) @ r_wb_t
            ).unsqueeze(0)                                              # [1, 8, 3]
            prev_mask = torch.ones(1, dtype=torch.bool, device=device)
        else:
            # Stale-prev hazard guard
            self._invalidate_stale_prev_if_no_frame_info(r_wb, pos_world)
            prev_ctrl_pts_body = torch.zeros(1, 8, 3, device=device)
            prev_mask          = torch.zeros(1, dtype=torch.bool, device=device)

        amp_enabled = self.use_model_amp and device.type == "cuda"
        for i, (t_ids, d_sigma) in enumerate(zip(schedule.t_ids, schedule.d_sigmas)):
            with torch.autocast(
                device_type="cuda", dtype=self.model_amp_dtype, enabled=amp_enabled
            ):
                v_pred, c_pred = self.model(
                    video_latent, sg_norm, v_cmd_t, free_latent,
                    t_ids, t_ids,
                    prev_ctrl_pts_body, prev_mask,
                )
            video_latent[:, :, 1:].add_(v_pred, alpha=d_sigma)
            video_latent[:, :, 0:1].copy_(cond_latent)
            free_latent.add_(c_pred, alpha=d_sigma)
            if mark:
                mark(f"model_step_{i}")

        free_denorm = free_latent * self.free_std + self.free_mean
        wp_body     = self.basis.expand_to_12d(free_denorm, constrained)[0]

        # All 8 Bernstein control points in body frame [8, 3].
        all_ctrl_pts = torch.cat([constrained, free_denorm], dim=1)[0]
        if mark:
            mark("basis_expand")

        # Diagnostic L2 in normalized coeff space
        if self._prev_free_norm_for_l2 is not None:
            self.last_free_norm_l2 = float(
                (free_latent - self._prev_free_norm_for_l2).norm().item()
            )
        else:
            self.last_free_norm_l2 = 0.0
        self._prev_free_norm_for_l2 = free_latent.detach().clone()

        # Cache the prediction in world frame (only if finite and frame info supplied)
        if have_frame_info and torch.isfinite(all_ctrl_pts).all():
            r_wb_t = torch.as_tensor(np.asarray(r_wb, dtype=np.float32), device=device)
            p_t    = torch.as_tensor(np.asarray(pos_world, dtype=np.float32), device=device)
            self._prev_ctrl_pts_world = (
                all_ctrl_pts @ r_wb_t.T + p_t
            ).detach().clone()

        result = InferResult(
            wp_body=wp_body.cpu().numpy().astype(np.float32),
            ctrl_pts_body=all_ctrl_pts.cpu().numpy().astype(np.float64),
        )
        if mark:
            mark("cpu_output")
            self._finish_profile(events)
        return result
