#!/usr/bin/env bash
# Plays the ped_vehicle bag and opens RViz2. Extra args go to the launch file, e.g. ./run.sh rate:=0.5
set -e
source /opt/ros/humble/setup.bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec ros2 launch "$DIR/view_bag.launch.py" "$@"
