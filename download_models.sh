#!/usr/bin/env bash
# Downloads the pretrained Autoware CenterPoint (v2) ONNX models into ./models
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models"
URL=https://awf.ml.dev.web.auto/perception/models/centerpoint/v2
mkdir -p "$DIR"
for f in pts_voxel_encoder_centerpoint.onnx pts_backbone_neck_head_centerpoint.onnx centerpoint_ml_package.param.yaml; do
  [ -s "$DIR/$f" ] && { echo "have $f"; continue; }
  echo "downloading $f"
  curl -fSL --retry 3 -o "$DIR/$f" "$URL/$f"
done
