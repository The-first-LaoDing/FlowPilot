<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/EN-English-0969da?style=flat-square" alt="English"></a>
  <a href="README_zh.md"><img src="https://img.shields.io/badge/CN-%E4%B8%AD%E6%96%87-d1242f?style=flat-square" alt="中文"></a>
</p>

# FlowPilot：面向敏捷无人机导航的实时世界-动作建模

FlowPilot 是一个基于 Flow Matching 的无人机机载自主导航框架。模型采用双流 MoT，将视频 latent 与动作 token 放在同一序列中建模；动作由 7 阶 Bernstein 多项式参数化，包含 8 个控制点，其中 3 个由初始状态约束、5 个由网络预测，输出 30 Hz、48 个路点的轨迹。

本仓库开源仿真与推理部分的代码，目前提供 Docker 下的 Flightmare + RViz 闭环测试。训练代码与数据集不在本次发布范围内。

论文：[arxiv]()

视频：[Bilibili](), [Youtobe]()


## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/flyingbitac/FlowPilot.git
cd FlowPilot
```

### 2. 下载模型权重

Phase 2 checkpoint 和统计量已上传至 [Hugging Face](https://huggingface.co/YuDinging/FlowPilot)，Wan2.2 VAE 权重来自官方 Wan2.2 仓库。在仓库根目录执行：

```bash
hf download YuDinging/FlowPilot --include "ckpt/*" --local-dir .
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir ckpt
```

如果还没有 `hf` 命令，先执行 `pip install -U huggingface_hub`。

`--include "ckpt/*"` 用于避免 Hugging Face 的模型卡片覆盖本仓库的 README。下载的文件如下：

| 文件 | 说明 |
|---|---|
| `ckpt/phase2/ckpt_final.pt` | Phase 2 推理 checkpoint |
| `ckpt/Wan2.2_VAE.pth` | 官方 Wan2.2 仓库的 VAE 权重，可以是软链接 |
| `ckpt/action_stats.pt` | 动作与状态归一化统计量 |
| `ckpt/coeff_stats.pt` | Bernstein 系数归一化统计量 |

模型路径在 `configs/runtime/rviz.py` 中配置。

### 3. 构建镜像

```bash
python docker/docker.py build
```

首次构建会安装 ROS Noetic、编译 `flightgym`，并从 Zenodo 下载约 1.7 GB 的 Flightmare Unity renderer，最终镜像约 30 GB。如果这些服务访问较慢，可以通过 `FLOWPILOT_UBUNTU_MIRROR`、`FLOWPILOT_ROS_MIRROR`、`FLOWPILOT_PIP_INDEX_URL` 和 `FLOWPILOT_STANDALONE_URL` 替换镜像源。

### 4. 检查权重

在镜像中加载 Phase 2 checkpoint 和统计量：

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

正常输出为 `phase2: 377 tensors` 和 `checkpoints OK`。Wan2.2 VAE 权重会在启动容器时检查；如果 `ckpt/Wan2.2_VAE.pth` 是软链接，`docker/docker.py start` 会把它的目标路径也挂载进容器。

## 仿真测试

启动容器：

```bash
export DISPLAY=:1
python docker/docker.py start
```

然后在三个终端中分别运行：

```bash
python docker/docker.py roscore
python docker/docker.py rviz
python docker/docker.py run
```

在 RViz 中将 Fixed Frame 设为 `world`，用 `2D Nav Goal` 给出目标点。节点会发布规划轨迹、当前深度图、无人机 marker 和局部点云。停止容器：

```bash
python docker/docker.py stop
```

## 配置说明

模型路径、场景编号、树木间距和深度噪声等运行参数都在 `configs/runtime/rviz.py` 中。仓库外的模型文件会由 `docker/docker.py start` 以只读方式挂载到容器内的同一路径。


## TODO

- 发布数据集。
- 发布训练代码。
- 详细介绍 FlowPilot 的控制模块。

## 致谢

感谢以下项目的作者开源他们的工作：

- [YOPO](https://github.com/TJU-Aerial-Robotics/YOPO)：仿真中使用的 `flightgym` 来自 YOPO 的一个 fork。
- [Flightmare](https://github.com/uzh-rpg/flightmare)：闭环测试使用的仿真器和 Unity 渲染器。
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)：VAE 实现和权重的来源。

## 许可证

FlowPilot 代码采用 Apache License 2.0，详见 [LICENSE](LICENSE)。第三方组件保留各自的版权声明和许可证，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。模型权重和 renderer 单独分发，可能适用各自的条款。
