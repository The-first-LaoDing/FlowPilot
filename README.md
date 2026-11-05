<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/EN-English-0969da?style=flat-square" alt="English"></a>
  <a href="README_zh.md"><img src="https://img.shields.io/badge/CN-%E4%B8%AD%E6%96%87-d1242f?style=flat-square" alt="中文"></a>
</p>

# FlowPilot: Real-Time World-Action Modeling for Agile UAV Navigation

FlowPilot is a Flow Matching based framework for real-time onboard UAV navigation. A dual-stream MoT models the video latent and the action tokens jointly. The action is parameterized by a 7th-order Bernstein polynomial with 8 control points (3 fixed by the initial state and 5 predicted by the network), producing 48 waypoints at 30 Hz.

This repository contains the simulation and inference code. We currently provide a closed-loop test in Flightmare and RViz inside Docker. Training code and datasets are not included.

Paper: [arxiv]()

Video: [Bilibili](), [YouTube]()

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/flyingbitac/FlowPilot.git
cd FlowPilot
```

### 2. Download the checkpoints

The checkpoints are available on [Hugging Face](https://huggingface.co/YuDinging/FlowPilot), and the Wan2.2 VAE weights come from the official Wan2.2 repository. Download them into the repository root:

```bash
hf download YuDinging/FlowPilot --include "ckpt/*" --local-dir .
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir ckpt
```

If `hf` is not installed, run `pip install -U huggingface_hub` first.

The `--include "ckpt/*"` filter keeps the model card from overwriting this README. The downloaded files are:

| File | Description |
|---|---|
| `ckpt/phase2/ckpt_final.pt` | Phase 2 inference checkpoint |
| `ckpt/Wan2.2_VAE.pth` | Wan2.2 VAE weights from the official Wan2.2 repository, a symlink is fine |
| `ckpt/action_stats.pt` | Action and state normalization statistics |
| `ckpt/coeff_stats.pt` | Bernstein coefficient normalization statistics |

The paths are configured in `configs/runtime/rviz.py`.

### 3. Build the Docker image

```bash
python docker/docker.py build
```

The first build installs ROS Noetic, compiles `flightgym`, and downloads the Flightmare Unity renderer (about 1.7 GB) from Zenodo. The final image is about 30 GB. If these services are slow to reach, the Ubuntu, PyPI, and renderer mirrors can be overridden with `FLOWPILOT_UBUNTU_MIRROR`, `FLOWPILOT_ROS_MIRROR`, `FLOWPILOT_PIP_INDEX_URL`, and `FLOWPILOT_STANDALONE_URL`.

### 4. Verify the checkpoints

Check that the Phase-2 checkpoint and the statistics load inside the image:

```bash
docker run --rm -v "$PWD":/root/workspace flowpilot-sim:latest python -c "
import torch
ckpt = torch.load('ckpt/phase2/ckpt_final.pt', map_location='cpu', weights_only=False)
assert 'model' in ckpt
print('phase2:', len(ckpt['model']), 'tensors')
torch.load('ckpt/action_stats.pt', map_location='cpu')
torch.load('ckpt/coeff_stats.pt', map_location='cpu')
print('checkpoints OK')
"
```

A successful run prints `phase2: 377 tensors` and `checkpoints OK`. The Wan2.2 VAE weights are checked when the container starts; if `ckpt/Wan2.2_VAE.pth` is a symlink, `docker/docker.py start` mounts its target into the container as well.

## Simulation

Start the container:

```bash
export DISPLAY=:1
python docker/docker.py start
```

Then run the three processes in separate terminals:

```bash
python docker/docker.py roscore
python docker/docker.py rviz
python docker/docker.py run
```

Set the Fixed Frame to `world` in RViz and use `2D Nav Goal` to send a target. The node publishes the planned trajectory, the current depth image, the UAV marker, and a local point cloud. To stop the container:

```bash
python docker/docker.py stop
```

## Configuration

Runtime parameters, including the model paths, scene id, tree spacing, and depth noise, are in `configs/runtime/rviz.py`. Model files outside the repository are mounted read-only at the same path by `docker/docker.py start`.

## TODO

- Release the dataset.
- Release the training code.
- Detail the control module of FlowPilot.

## Acknowledgements

We thank the authors of the following projects for making their work available:

- [YOPO](https://github.com/TJU-Aerial-Robotics/YOPO) — Our simulation uses `flightgym` from a YOPO fork.
- [Flightmare](https://github.com/uzh-rpg/flightmare) — The simulator and Unity renderer used for our closed-loop evaluation.
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) — The source of our VAE implementation and weights.

## License

FlowPilot code is released under the Apache License 2.0. See [LICENSE](LICENSE). Third-party components have their own copyright notices and licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights and the renderer are distributed separately and may have their own terms.
