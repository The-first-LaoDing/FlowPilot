"""Flightmare process lifecycle: launch, cleanup, ALSA stub."""

import atexit
import os
import signal
import subprocess
import time


_STUB_DLOPEN_SRC = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
void *dlopen(const char *f, int flags) {
    static void *(*real)(const char*,int)=NULL;
    if(!real)real=dlsym(RTLD_NEXT,"dlopen");
    if(f&&strstr(f,"libasound"))return NULL;
    return real(f,flags);
}
"""


def build_alsa_stub(so_path: str = "/tmp/stub_dlopen_flightmare.so") -> str:
    """Compile ALSA dlopen stub .so. Returns path or '' on failure."""
    if os.path.exists(so_path):
        return so_path
    src_path = so_path.replace(".so", ".c")
    with open(src_path, "w") as f:
        f.write(_STUB_DLOPEN_SRC)
    ret = subprocess.run(
        ["gcc", "-shared", "-fPIC", "-o", so_path, src_path, "-ldl"],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"[WARN] ALSA stub compile failed: {ret.stderr}", flush=True)
        return ""
    print("[INFO] compiled ALSA dlopen stub.", flush=True)
    return so_path


def cleanup_stale(binary_name: str = "flightmare.x86_64") -> None:
    """Kill stale Flightmare processes to avoid ZMQ/PLY conflicts."""
    result = subprocess.run(
        ["pgrep", "-f", binary_name],
        capture_output=True, text=True, check=False,
    )
    pids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)

    if not pids:
        return

    print(f"[INFO] terminating stale Flightmare processes: {pids}", flush=True)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.time() + 5.0
    while time.time() < deadline:
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                pass
        if not alive:
            return
        time.sleep(0.2)

    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _build_env_vars() -> dict:
    """Build environment with NVIDIA PRIME offload hints."""
    env = os.environ.copy()
    if env.get("DISPLAY"):
        env.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
        env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
        env.setdefault("__VK_LAYER_NV_optimus", "NVIDIA_only")
    return env


def launch(binary_path: str) -> subprocess.Popen:
    """Launch Flightmare binary with NVIDIA offload + ALSA stub.

    Registers atexit cleanup. Returns Popen handle.
    """
    cleanup_stale(os.path.basename(binary_path))
    env = _build_env_vars()
    stub = build_alsa_stub()
    if stub:
        preload = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = f"{stub}:{preload}" if preload else stub
    proc = subprocess.Popen(
        [binary_path],
        cwd=os.path.dirname(binary_path),
        env=env,
        start_new_session=True,
    )

    def _cleanup():
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    atexit.register(_cleanup)
    return proc


def terminate(proc: subprocess.Popen) -> None:
    """Gracefully terminate a Flightmare process."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
