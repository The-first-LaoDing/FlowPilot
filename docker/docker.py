#!/usr/bin/env python3
"""Manage the FlowPilot pure-simulation Docker container.

The image contains ROS Noetic, RViz, Flightmare flightgym and the Unity
renderer. Everything else (repository, checkpoints, VAE, stats) is mounted at
runtime. Model paths come from configs/runtime/rviz.py; GPU flags, X11
forwarding and the remaining mounts are defined here so the CLI stays short:

    python docker/docker.py build
    python docker/docker.py start
    python docker/docker.py enter
    python docker/docker.py stop

Extra commands for the three-terminal simulation workflow:

    python docker/docker.py roscore
    python docker/docker.py rviz
    python docker/docker.py run
"""

from __future__ import annotations

import argparse
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
IMAGE = os.environ.get("FLOWPILOT_SIM_IMAGE", "flowpilot-sim:latest")
CONTAINER = os.environ.get("FLOWPILOT_SIM_CONTAINER", "flowpilot-sim")
ROS_PORT = int(os.environ.get("FLOWPILOT_ROS_PORT", "11312"))
CONTAINER_WORKSPACE = "/root/workspace"


def die(message: str) -> None:
    print(f"[sim] ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def docker(*args: str) -> None:
    run(["docker", *args])


def _config_path(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        die(f"{label} not found: {path}. Set it in configs/runtime/rviz.py")
    return path


def resolve_model_paths() -> dict[str, Path]:
    """Read model paths from configs/runtime/rviz.py (no env overrides)."""
    config_path = REPO_ROOT / "configs" / "runtime" / "rviz.py"
    cfg = runpy.run_path(str(config_path))
    return {
        "phase2_ckpt": _config_path(cfg["phase2_ckpt"], "Phase 2 checkpoint"),
        "vae_pth": _config_path(cfg["vae_pth"], "VAE weights"),
        "action_stats": _config_path(cfg["action_stats_path"], "action_stats.pt"),
        "coeff_stats": _config_path(cfg["coeff_stats_path"], "coeff_stats.pt"),
    }


class SimContainer:
    def __init__(self) -> None:
        self.image = IMAGE
        self.name = CONTAINER
        self.display = os.environ.get("DISPLAY", ":1")
        self.ros_master_uri = os.environ.get("ROS_MASTER_URI", f"http://127.0.0.1:{ROS_PORT}")
        self.ros_ip = os.environ.get("ROS_IP", "127.0.0.1")

    # ------------------------------------------------------------------
    # Docker lifecycle
    # ------------------------------------------------------------------

    def build(self) -> None:
        args = [
            "build",
            "--network=host",
            "-f", str(DOCKERFILE),
            "-t", self.image,
        ]
        for env_name, build_arg in (
            ("FLOWPILOT_UBUNTU_MIRROR", "UBUNTU_MIRROR"),
            ("FLOWPILOT_ROS_MIRROR", "ROS_MIRROR"),
            ("FLOWPILOT_STANDALONE_URL", "STANDALONE_URL"),
            ("FLOWPILOT_PIP_INDEX_URL", "PIP_INDEX_URL"),
            ("HTTP_PROXY", "HTTP_PROXY"),
            ("HTTPS_PROXY", "HTTPS_PROXY"),
            ("NO_PROXY", "NO_PROXY"),
            ("http_proxy", "http_proxy"),
            ("https_proxy", "https_proxy"),
            ("no_proxy", "no_proxy"),
        ):
            value = os.environ.get(env_name)
            if value:
                args += ["--build-arg", f"{build_arg}={value}"]
        args.append(str(REPO_ROOT))
        docker(*args)

    def start(self) -> None:
        if not shutil.which("nvidia-smi"):
            die("nvidia-smi not found; an NVIDIA GPU is required.")
        if subprocess.run(["docker", "image", "inspect", self.image],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode != 0:
            die(f"image '{self.image}' not found. Run: python docker/docker.py build")
        if shutil.which("xhost"):
            subprocess.run(["xhost", "+local:root"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        model_paths = resolve_model_paths()

        volumes: list[str] = [
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix:ro",
            "-v", f"{REPO_ROOT}:{CONTAINER_WORKSPACE}",
        ]
        xauthority = Path.home() / ".Xauthority"
        if xauthority.is_file():
            volumes += ["-v", f"{xauthority}:/tmp/.Xauthority:ro"]

        # Files inside the repository are already covered by the workspace
        # mount. Files outside it are mounted read-only at the same absolute
        # path, so the paths in configs/runtime/rviz.py stay valid in-container.
        for path in sorted({p for p in model_paths.values()
                            if not p.is_relative_to(REPO_ROOT)}):
            volumes += ["-v", f"{path}:{path}:ro"]

        env = {
            "DISPLAY": self.display,
            "XAUTHORITY": "/tmp/.Xauthority",
            "ROS_MASTER_URI": self.ros_master_uri,
            "ROS_IP": self.ros_ip,
            "PYTHONDONTWRITEBYTECODE": "1",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "__NV_PRIME_RENDER_OFFLOAD": "1",
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            "__VK_LAYER_NV_optimus": "NVIDIA_only",
        }

        subprocess.run(["docker", "rm", "-f", self.name], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        args = [
            "run", "--rm", "-dit",
            "--name", self.name,
            "--gpus", "all",
            "--network=host",
            "--ipc=host",
        ]
        for key, value in env.items():
            args += ["-e", f"{key}={value}"]
        args += volumes
        args.append(self.image)
        docker(*args)

        docker("exec", self.name, "/usr/local/bin/flowpilot-entrypoint",
               "mkdir", "-p", "/opt/flightmare/run/flowpilot_sim")

        print("[sim] container started. Run the three terminals:")
        print("  python docker/docker.py roscore")
        print("  python docker/docker.py rviz")
        print("  python docker/docker.py run")

    def _running(self) -> bool:
        return subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.name],
            capture_output=True, text=True,
        ).stdout.strip() == "true"

    def enter(self) -> None:
        if not self._running():
            die("container is not running. Run: python docker/docker.py start")
        docker("exec", "-it", self.name,
               "/usr/local/bin/flowpilot-entrypoint", "bash")

    def stop(self) -> None:
        subprocess.run(["docker", "stop", self.name], check=False)

    def status(self) -> None:
        image_ok = subprocess.run(
            ["docker", "image", "inspect", self.image],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        running = self._running()
        print(f"  image:     {self.image} ({'present' if image_ok else 'missing'})")
        print(f"  container: {self.name} ({'running' if running else 'stopped'})")

    def logs(self) -> None:
        docker("logs", "--tail", "100", "-f", self.name)

    # ------------------------------------------------------------------
    # Simulation commands
    # ------------------------------------------------------------------

    def _exec(self, *args: str) -> None:
        if not self._running():
            die("container is not running. Run: python docker/docker.py start")
        docker("exec", "-it", self.name,
               "/usr/local/bin/flowpilot-entrypoint", *args)

    def roscore(self) -> None:
        self._exec("roscore", "-p", str(ROS_PORT))

    def rviz(self) -> None:
        self._exec("rviz", "-d", f"{CONTAINER_WORKSPACE}/sim.rviz")

    def run_sim(self) -> None:
        self._exec("python", f"{CONTAINER_WORKSPACE}/scripts/run_rviz_sim.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage the FlowPilot pure-simulation Docker container.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python docker/docker.py build\n"
            "  python docker/docker.py start\n"
            "  python docker/docker.py enter\n"
            "  python docker/docker.py stop\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("build", "build flowpilot-sim:latest"),
        ("start", "start container with GPU/X11/repo/weights"),
        ("enter", "open a shell inside the container"),
        ("stop", "stop the container"),
        ("status", "show image/container status"),
        ("logs", "follow container logs"),
        ("roscore", "run ROS master on port 11312"),
        ("rviz", "open RViz with sim.rviz"),
        ("run", "run scripts/run_rviz_sim.py"),
    ):
        sub.add_parser(name, help=help_text)

    args = parser.parse_args()
    sim = SimContainer()
    actions = {
        "build": sim.build,
        "start": sim.start,
        "enter": sim.enter,
        "stop": sim.stop,
        "status": sim.status,
        "logs": sim.logs,
        "roscore": sim.roscore,
        "rviz": sim.rviz,
        "run": sim.run_sim,
    }
    actions[args.command]()


if __name__ == "__main__":
    main()
