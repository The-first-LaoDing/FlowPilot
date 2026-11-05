#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash
source /opt/conda/etc/profile.d/conda.sh
conda activate flowpilot

export FLIGHTMARE_PATH="${FLIGHTMARE_PATH:-/opt/flightmare}"
export PYTHONPATH="/root/workspace:${FLIGHTMARE_PATH}/flightlib/build:${FLIGHTMARE_PATH}:${PYTHONPATH:-}"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11312}"
export ROS_IP="${ROS_IP:-127.0.0.1}"

cd /root/workspace 2>/dev/null || cd "${FLIGHTMARE_PATH}"
exec "$@"
