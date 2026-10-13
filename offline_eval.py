#!/usr/bin/env python3
"""Run the detector over a rosbag2 without RViz: prints detections per frame and saves bird's-eye-view images.

  python3 offline_eval.py --bag ../rosbag2_2022_04_14-ped_vehicle --every 20 --out results/
"""
import argparse
import os
import sys
import time

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from detector import LidarDetector  # noqa: E402
from lidar_detector_node import cloud_to_xyz  # noqa: E402

COLORS = {'CAR': 'red', 'TRUCK': 'orange', 'BUS': 'orange', 'BICYCLE': 'cyan',
          'PEDESTRIAN': 'yellow', 'TREE': 'lime'}


def save_bev(path, pts, boxes, ground_z, extent=40.0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    ax.scatter(pts[:, 0], pts[:, 1], s=0.15, c=pts[:, 2], cmap='Greys_r', vmin=ground_z - 1, vmax=ground_z + 4)
    for b in boxes:
        c, s = np.cos(b.yaw), np.sin(b.yaw)
        cr = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]]) * [b.l / 2, b.w / 2]
        cr = cr @ np.array([[c, s], [-s, c]]) + [b.x, b.y]
        ax.plot(cr[:, 0], cr[:, 1], color=COLORS[b.label], lw=1.5)
        if abs(b.x) < extent and abs(b.y) < extent:
            ax.text(b.x, b.y + b.w / 2 + 0.4, b.label.lower(), color=COLORS[b.label], fontsize=7, ha='center')
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.set_xlabel('x [m]', color='white')
    ax.set_ylabel('y [m]', color='white')
    fig.savefig(path, dpi=100, bbox_inches='tight', facecolor='black')
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.realpath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', default=os.path.join(os.path.dirname(here), 'rosbag2_2022_04_14-ped_vehicle'))
    ap.add_argument('--topic', default='/sensing/lidar/top/rectified/pointcloud')
    ap.add_argument('--models', default=os.path.join(here, 'models'))
    ap.add_argument('--every', type=int, default=20, help='process every N-th frame')
    ap.add_argument('--out', default='', help='folder for BEV images (empty = no images)')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    det = LidarDetector(args.models, use_gpu=not args.cpu)
    print('execution provider:', det.cp.providers[0])
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    i = 0
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != args.topic:
            continue
        i += 1
        if (i - 1) % args.every:
            continue
        pts = cloud_to_xyz(deserialize_message(data, PointCloud2))
        t0 = time.perf_counter()
        boxes = det(pts)
        ms = (time.perf_counter() - t0) * 1000
        print(f'frame {i}: {len(pts)} points, {len(boxes)} objects, {ms:.0f} ms')
        for b in sorted(boxes, key=lambda b: (b.label, -b.score)):
            print('   ', b)
        if args.out:
            save_bev(os.path.join(args.out, f'bev_{i:04d}.png'), pts, boxes, det.ground_z)


if __name__ == '__main__':
    main()
