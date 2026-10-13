#!/usr/bin/env python3
"""ROS 2 node: lidar PointCloud2 -> 3D boxes (people, cars, trucks, bikes, trees) as RViz markers."""
import math
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from detector import LidarDetector  # noqa: E402

COLORS = {  # r, g, b
    'CAR': (1.0, 0.2, 0.2),
    'TRUCK': (1.0, 0.5, 0.0),
    'BUS': (1.0, 0.5, 0.0),
    'BICYCLE': (0.0, 0.8, 1.0),
    'PEDESTRIAN': (1.0, 1.0, 0.0),
    'TREE': (0.1, 0.9, 0.2),
}
# 12 edges of a box, as pairs of corner indices (corners 0-3 bottom, 4-7 top)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


class LidarDetectorNode(Node):
    def __init__(self):
        super().__init__('lidar_detector')
        here = os.path.dirname(os.path.realpath(__file__))
        topic = self.declare_parameter('input_topic', '/sensing/lidar/top/rectified/pointcloud').value
        model_dir = self.declare_parameter('model_dir', os.path.join(here, 'models')).value
        score = self.declare_parameter('score_threshold', 0.35).value
        use_gpu = self.declare_parameter('use_gpu', True).value
        trees = self.declare_parameter('detect_trees', True).value

        self.det = LidarDetector(model_dir, use_gpu=use_gpu, score_thresh=score, detect_trees=trees)
        self.get_logger().info(f'CenterPoint running on {self.det.cp.providers[0]}')

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub = self.create_subscription(PointCloud2, topic, self.on_cloud, qos)
        self.pub = self.create_publisher(MarkerArray, '/perception/object_markers', 10)

    def on_cloud(self, msg):
        pts = cloud_to_xyz(msg)
        t0 = time.perf_counter()
        boxes = self.det(pts)
        dt = (time.perf_counter() - t0) * 1000
        counts = {}
        for b in boxes:
            counts[b.label] = counts.get(b.label, 0) + 1
        self.get_logger().info(f'{dt:.0f} ms  ' + '  '.join(f'{k}:{v}' for k, v in sorted(counts.items())),
                               throttle_duration_sec=1.0)
        self.pub.publish(boxes_to_markers(boxes, msg.header))


def cloud_to_xyz(msg):
    off = {f.name: f.offset for f in msg.fields}
    dt = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4'] * 3,
                   'offsets': [off['x'], off['y'], off['z']], 'itemsize': msg.point_step})
    a = np.frombuffer(msg.data, dt, msg.width * msg.height)
    p = np.stack([a['x'], a['y'], a['z']], axis=1)
    return p[np.isfinite(p).all(axis=1) & (np.abs(p).sum(axis=1) > 0.1)].astype(np.float32)


def boxes_to_markers(boxes, header):
    arr = MarkerArray()
    clear = Marker()
    clear.header = header
    clear.action = Marker.DELETEALL
    arr.markers.append(clear)

    for i, b in enumerate(boxes):
        r, g, bl = COLORS.get(b.label, (1.0, 1.0, 1.0))
        qz, qw = math.sin(b.yaw / 2), math.cos(b.yaw / 2)

        edges = Marker()
        edges.header, edges.ns, edges.id = header, 'boxes', i
        edges.type, edges.action = Marker.LINE_LIST, Marker.ADD
        edges.pose.position.x, edges.pose.position.y, edges.pose.position.z = b.x, b.y, b.z
        edges.pose.orientation.z, edges.pose.orientation.w = qz, qw
        edges.scale.x = 0.08
        edges.color.r, edges.color.g, edges.color.b, edges.color.a = r, g, bl, 1.0
        corners = [Point(x=sx * b.l / 2, y=sy * b.w / 2, z=sz * b.h / 2)
                   for sz in (-1, 1) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        for a_, c_ in EDGES:
            edges.points += [corners[a_], corners[c_]]
        arr.markers.append(edges)

        fill = Marker()
        fill.header, fill.ns, fill.id = header, 'fill', i
        fill.type, fill.action = Marker.CUBE, Marker.ADD
        fill.pose = edges.pose
        fill.scale.x, fill.scale.y, fill.scale.z = b.l, b.w, b.h
        fill.color.r, fill.color.g, fill.color.b, fill.color.a = r, g, bl, 0.15
        arr.markers.append(fill)

        text = Marker()
        text.header, text.ns, text.id = header, 'labels', i
        text.type, text.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        text.pose.position.x, text.pose.position.y = b.x, b.y
        text.pose.position.z = b.z + b.h / 2 + 0.5
        text.pose.orientation.w = 1.0
        text.scale.z = 0.8
        text.color.r, text.color.g, text.color.b, text.color.a = r, g, bl, 1.0
        text.text = b.label.lower() if b.label == 'TREE' else f'{b.label.lower()} {b.score:.2f}'
        arr.markers.append(text)
    return arr


def main():
    rclpy.init()
    node = LidarDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
