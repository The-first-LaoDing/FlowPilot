# FlowPilot pure-simulation image: Flightmare + ROS Noetic + RViz.
#
# This image contains only what is needed to run scripts/run_rviz_sim.py:
#   - ROS Noetic (roscore, rviz, tf2, sensor/visualization messages)
#   - Flightmare flightlib / flightgym (public Flightmare fork, pinned commit)
#   - Flightmare Unity renderer (standalone.tar from the upstream release)
#   - a minimal Python 3.10 environment for FlowPilot inference
#
# The repository itself is intentionally NOT copied into the image; mount it at
# /root/workspace at runtime.
#
# Build (from the repository root):
#   docker build -f docker/Dockerfile -t flowpilot-sim:latest .
#
# Optional build args for restricted networks:
#   --build-arg UBUNTU_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu
#   --build-arg ROS_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu
#   --build-arg STANDALONE_URL=http://your-mirror/standalone.tar

FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ARG UBUNTU_MIRROR=
ARG ROS_MIRROR=http://packages.ros.org/ros/ubuntu
ARG FLIGHTMARE_REPO=https://github.com/ZombieRUNs/YOPO.git
ARG FLIGHTMARE_REF=13de2e417d625a3eebcfbe0a8bf97d4316cb7a91
ARG STANDALONE_URL=https://zenodo.org/records/5517791/files/standalone.tar
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=noetic \
    TZ=Asia/Shanghai

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
RUN if [ -n "${UBUNTU_MIRROR}" ]; then \
      sed -i "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g; \
              s|http://security.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g" \
        /etc/apt/sources.list; \
    fi \
 && apt-get -o Acquire::Retries=5 -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false update \
 && apt-get -o Acquire::Retries=5 -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false install -y --no-install-recommends \
      build-essential cmake git wget curl ca-certificates gnupg2 lsb-release \
      ccache pkg-config \
      libzmqpp-dev libopencv-dev libpcl-dev libeigen3-dev \
      libboost-all-dev libflann-dev libgomp1 \
      libgl1-mesa-glx libgl1-mesa-dri libglu1-mesa libglfw3 libglfw3-dev \
      libvulkan1 libgbm1 \
      libx11-6 libxext6 libxrender1 libsm6 libice6 libfontconfig1 \
      libxrandr2 libxi6 libxcursor1 libxinerama1 libxcomposite1 \
      libxdamage1 libxfixes3 libxss1 libxtst6 \
      libasound2 libgconf-2-4 libgstreamer1.0-0 \
      libgstreamer-plugins-base1.0-0 libfuse2 libnss3 libgtk-3-0 \
      python3-pip python3-dev python3-empy \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. ROS Noetic
# ---------------------------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
      | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] ${ROS_MIRROR} focal main" \
      > /etc/apt/sources.list.d/ros1-latest.list \
 && apt-get -o Acquire::Retries=5 -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false update \
 && apt-get -o Acquire::Retries=5 -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false install -y --no-install-recommends \
      ros-noetic-ros-base \
      ros-noetic-roscpp \
      ros-noetic-catkin \
      ros-noetic-cmake-modules \
      ros-noetic-message-generation \
      ros-noetic-message-runtime \
      ros-noetic-tf \
      ros-noetic-tf2-ros \
      ros-noetic-nav-msgs \
      ros-noetic-sensor-msgs \
      ros-noetic-std-msgs \
      ros-noetic-geometry-msgs \
      ros-noetic-visualization-msgs \
      ros-noetic-cv-bridge \
      ros-noetic-image-transport \
      ros-noetic-pcl-ros \
      ros-noetic-pcl-conversions \
      ros-noetic-rviz \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 3. Minimal Python environment for FlowPilot inference
# ---------------------------------------------------------------------------
RUN wget -q --tries=5 --timeout=30 "${MINICONDA_URL}" -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p /opt/conda \
 && rm /tmp/miniconda.sh \
 && /opt/conda/bin/conda tos accept \
      --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && /opt/conda/bin/conda tos accept \
      --override-channels --channel https://repo.anaconda.com/pkgs/r \
 && /opt/conda/bin/conda create -y -n flowpilot python=3.10 \
 && /opt/conda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=flowpilot \
    PATH=/opt/conda/envs/flowpilot/bin:/opt/conda/bin:${PATH}

ARG PIP_INDEX_URL=https://pypi.org/simple

RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" \
      --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" \
      torch==2.7.1 \
      numpy==1.26.4 \
      opencv-python-headless==4.10.0.84 \
      "einops>=0.7" \
      ruamel.yaml==0.17.21 \
      rospkg \
      catkin_pkg \
      empy==3.3.4 \
      pyyaml

# ---------------------------------------------------------------------------
# 4. Flightmare flightlib / flightgym
# ---------------------------------------------------------------------------
RUN git init /opt/flightmare \
 && cd /opt/flightmare \
 && git remote add origin "${FLIGHTMARE_REPO}" \
 && git fetch --depth 1 origin "${FLIGHTMARE_REF}" \
 && git checkout FETCH_HEAD \
 && rm -rf .git

# Pin the CMake-downloaded dependencies so the build is reproducible.
RUN sed -i 's#GIT_TAG[[:space:]]*master#GIT_TAG v2.13.6#' \
      /opt/flightmare/flightlib/cmake/pybind11_download.cmake \
 && sed -i 's#GIT_TAG[[:space:]]*master#GIT_TAG yaml-cpp-0.7.0#' \
      /opt/flightmare/flightlib/cmake/yaml_download.cmake

RUN bash -lc 'cd /opt/flightmare/flightlib && mkdir -p build && cd build && source /opt/ros/noetic/setup.bash && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_UNITY_BRIDGE_TESTS=OFF -DCMAKE_PREFIX_PATH=/opt/ros/noetic -DPYTHON_EXECUTABLE=/opt/conda/envs/flowpilot/bin/python -DPython_EXECUTABLE=/opt/conda/envs/flowpilot/bin/python && make -j"$(nproc)"'

# ---------------------------------------------------------------------------
# 5. Flightmare Unity renderer
# ---------------------------------------------------------------------------
RUN wget -q --tries=5 --timeout=30 "${STANDALONE_URL}" -O /tmp/standalone.tar \
 && tar -xf /tmp/standalone.tar -C /opt/flightmare/flightrender \
 && rm /tmp/standalone.tar \
 && test -x /opt/flightmare/flightrender/RPG_Flightmare/flightmare.x86_64

# ---------------------------------------------------------------------------
# 6. Runtime environment
# ---------------------------------------------------------------------------
ENV FLIGHTMARE_PATH=/opt/flightmare \
    PYTHONPATH=/opt/flightmare/flightlib/build:/opt/flightmare \
    ROS_MASTER_URI=http://127.0.0.1:11312 \
    ROS_IP=127.0.0.1 \
    PYTHONDONTWRITEBYTECODE=1 \
    __NV_PRIME_RENDER_OFFLOAD=1 \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    __VK_LAYER_NV_optimus=NVIDIA_only \
    QT_X11_NO_MITSHM=1

RUN mkdir -p /opt/flightmare/run/flowpilot_sim /root/workspace \
 && bash -lc 'source /opt/ros/noetic/setup.bash && PYTHONPATH=/opt/flightmare/flightlib/build:/opt/flightmare:${PYTHONPATH} python -c "import cv2, einops, numpy, rospy, torch, flightgym; from ruamel.yaml import YAML; print(\"Flightmare/ROS/runtime imports OK\"); print(\"torch\", torch.__version__, \"numpy\", numpy.__version__)"'

COPY docker/entrypoint.sh /usr/local/bin/flowpilot-entrypoint
RUN chmod +x /usr/local/bin/flowpilot-entrypoint

WORKDIR /root/workspace
ENTRYPOINT ["/usr/local/bin/flowpilot-entrypoint"]
CMD ["bash"]
