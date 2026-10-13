"""Lidar 3D object detection (no ROS dependency, numpy + onnxruntime).

* CenterPoint (Autoware pretrained ONNX, classes CAR/TRUCK/BUS/BICYCLE/PEDESTRIAN)
* Tree detection by geometry: ground removal + clustering + trunk/canopy shape test
  (none of the public pretrained lidar detectors has a "tree" class).
"""
import glob
import os
import site

import numpy as np
from scipy import ndimage

CLASS_NAMES = ['CAR', 'TRUCK', 'BUS', 'BICYCLE', 'PEDESTRIAN']


def _cuda_lib_dirs():
    """Library folders of the pip-installed nvidia-* CUDA/cuDNN wheels."""
    roots = site.getsitepackages() + [site.getusersitepackages()]
    return sorted({d for r in roots for d in glob.glob(os.path.join(r, 'nvidia', '*', 'lib'))})


def _make_session(path, use_gpu, disable_opt=False):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if disable_opt:
        # The Autoware encoder breaks ORT's MatMul+BN fusion pass; it is tiny anyway.
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    providers = ['CPUExecutionProvider']
    if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
    return ort.InferenceSession(path, so, providers=providers)


class Box3D:
    __slots__ = ('label', 'score', 'x', 'y', 'z', 'l', 'w', 'h', 'yaw')

    def __init__(self, label, score, x, y, z, l, w, h, yaw):
        self.label, self.score = label, float(score)
        self.x, self.y, self.z = float(x), float(y), float(z)  # box centre
        self.l, self.w, self.h, self.yaw = float(l), float(w), float(h), float(yaw)

    def __repr__(self):
        return (f'{self.label}({self.score:.2f}) c=({self.x:.1f},{self.y:.1f},{self.z:.1f}) '
                f'lwh=({self.l:.1f},{self.w:.1f},{self.h:.1f}) yaw={self.yaw:.2f}')


class CenterPoint:
    """Autoware lidar_centerpoint (v2) re-implemented in numpy around its two ONNX graphs."""

    def __init__(self, model_dir, use_gpu=True, score_thresh=0.35,
                 pc_range=(-76.8, -76.8, -4.0, 76.8, 76.8, 6.0), voxel=(0.32, 0.32, 10.0),
                 max_voxels=40000, max_pts=32):
        self.enc = _make_session(os.path.join(model_dir, 'pts_voxel_encoder_centerpoint.onnx'),
                                 use_gpu, disable_opt=True)
        self.head = _make_session(os.path.join(model_dir, 'pts_backbone_neck_head_centerpoint.onnx'),
                                  use_gpu)
        self.providers = self.head.get_providers()
        self.range = np.array(pc_range, np.float32)
        self.vs = np.array(voxel, np.float32)
        self.grid = np.round((self.range[3:] - self.range[:3]) / self.vs).astype(int)  # nx, ny, nz
        self.max_voxels, self.max_pts = max_voxels, max_pts
        self.score_thresh = score_thresh

    def _voxelize(self, pts):
        r, vs = self.range, self.vs
        keep = np.all((pts >= r[:3]) & (pts < r[3:]), axis=1)
        pts = pts[keep]
        ix = ((pts[:, 0] - r[0]) / vs[0]).astype(np.int64)
        iy = ((pts[:, 1] - r[1]) / vs[1]).astype(np.int64)
        vid = iy * self.grid[0] + ix

        order = np.argsort(vid, kind='stable')
        pts, vid = pts[order], vid[order]
        uniq, start, counts = np.unique(vid, return_index=True, return_counts=True)
        uniq, start, counts = uniq[:self.max_voxels], start[:self.max_voxels], counts[:self.max_voxels]
        n_vox = len(uniq)

        group = np.minimum(np.searchsorted(uniq, vid), n_vox - 1)  # voxel index of each point
        slot = np.arange(len(vid)) - start[group]                   # point index inside its voxel
        valid = (uniq[group] == vid) & (slot < self.max_pts)
        pts, group, slot = pts[valid], group[valid], slot[valid]

        n_in = np.minimum(counts, self.max_pts).astype(np.float32)
        mean = np.zeros((n_vox, 3), np.float32)
        np.add.at(mean, group, pts)
        mean /= n_in[:, None]

        vx, vy = uniq % self.grid[0], uniq // self.grid[0]
        cx = vx * vs[0] + vs[0] / 2 + r[0]
        cy = vy * vs[1] + vs[1] / 2 + r[1]

        feat = np.zeros((n_vox, self.max_pts, 9), np.float32)
        f = np.empty((len(pts), 9), np.float32)
        f[:, 0:3] = pts
        f[:, 3] = 0.0                                           # time lag (single sweep)
        f[:, 4:7] = pts - mean[group]
        f[:, 7] = pts[:, 0] - cx[group]
        f[:, 8] = pts[:, 1] - cy[group]
        feat[group, slot] = f
        return feat, vx, vy

    def __call__(self, pts):
        """pts: (N,3) float32 in a frame whose z=0 is the ground. Returns list[Box3D]."""
        feat, vx, vy = self._voxelize(pts)
        if len(feat) == 0:
            return []
        pillar = self.enc.run(None, {self.enc.get_inputs()[0].name: feat})[0].reshape(len(feat), -1)
        nx, ny = self.grid[0], self.grid[1]
        bev = np.zeros((1, pillar.shape[1], ny, nx), np.float32)
        bev[0, :, vy, vx] = pillar
        heat, reg, height, dim, rot, _ = self.head.run(None, {'spatial_features': bev})
        return self._decode(heat[0], reg[0], height[0], dim[0], rot[0])

    def _decode(self, heat, reg, height, dim, rot):
        score = 1.0 / (1.0 + np.exp(-heat))                     # (C,H,W)
        peak = score == ndimage.maximum_filter(score, size=(1, 3, 3))
        cls, yi, xi = np.nonzero(peak & (score > self.score_thresh))
        out = []
        for c, j, i in zip(cls, yi, xi):
            x = self.vs[0] * (i + reg[0, j, i]) + self.range[0]
            y = self.vs[1] * (j + reg[1, j, i]) + self.range[1]
            w, l, h = np.exp(dim[:, j, i])
            yaw_mm = np.arctan2(rot[0, j, i], rot[1, j, i])
            yaw = -yaw_mm - np.pi / 2                           # mmdet3d -> ROS convention (as Autoware)
            out.append(Box3D(CLASS_NAMES[c], score[c, j, i], x, y, height[0, j, i], l, w, h, yaw))
        return _nms_bev(out, dist=1.0)


def _nms_bev(boxes, dist):
    """Greedy centre-distance NMS across classes (keeps the highest score)."""
    boxes = sorted(boxes, key=lambda b: -b.score)
    kept = []
    for b in boxes:
        if all((b.x - k.x) ** 2 + (b.y - k.y) ** 2 > (dist + 0.25 * max(k.w, b.w)) ** 2 for k in kept):
            kept.append(b)
    return kept


def estimate_ground_z(pts, r_min=3.0, r_max=30.0):
    """Ground height (z) in the sensor frame: densest low z-bin of nearby points."""
    r = np.hypot(pts[:, 0], pts[:, 1])
    z = pts[(r > r_min) & (r < r_max), 2]
    if len(z) < 100:
        return None
    hist, edges = np.histogram(z, bins=np.arange(z.min(), np.percentile(z, 60) + 0.1, 0.1))
    if len(hist) == 0:
        return None
    return float(edges[np.argmax(hist)] + 0.05)


class TreeDetector:
    """Trees = clusters with a narrow trunk below ~2 m and a wider canopy above it."""

    def __init__(self, cell=0.25, max_range=40.0, min_height=3.0, trunk_band=(0.4, 1.6),
                 canopy_from=2.2, max_trunk_width=1.2, min_canopy_width=1.5, max_footprint=14.0):
        self.cell, self.max_range = cell, max_range
        self.min_height, self.trunk_band, self.canopy_from = min_height, trunk_band, canopy_from
        self.max_trunk_width, self.min_canopy_width = max_trunk_width, min_canopy_width
        self.max_footprint = max_footprint

    def __call__(self, pts, avoid=()):
        """pts: (N,3) with z=0 at ground. avoid: boxes (cars/people) to ignore."""
        r = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[(r < self.max_range) & (r > 2.0)]
        pts = pts[_above_local_ground(pts, 0.3)]
        if len(pts) < 50:
            return []
        c = self.cell
        n = int(np.ceil(2 * self.max_range / c))
        ix = ((pts[:, 0] + self.max_range) / c).astype(int).clip(0, n - 1)
        iy = ((pts[:, 1] + self.max_range) / c).astype(int).clip(0, n - 1)
        occ = np.zeros((n, n), bool)
        occ[iy, ix] = True
        occ = ndimage.binary_closing(occ, iterations=1)
        lab, nlab = ndimage.label(occ, structure=np.ones((3, 3)))
        plab = lab[iy, ix]

        trees = []
        order = np.argsort(plab, kind='stable')
        sl, st, cnt = np.unique(plab[order], return_index=True, return_counts=True)
        for l, s, k in zip(sl, st, cnt):
            if l == 0 or k < 40:
                continue
            p = pts[order[s:s + k]]
            zmax = np.percentile(p[:, 2], 98)
            ext = p[:, :2].max(0) - p[:, :2].min(0)
            if zmax < self.min_height or ext.max() > self.max_footprint:
                continue
            trunk = p[(p[:, 2] > self.trunk_band[0]) & (p[:, 2] < self.trunk_band[1])]
            canopy = p[p[:, 2] > self.canopy_from]
            if len(trunk) < 3 or len(canopy) < 20:
                continue
            tw = (trunk[:, :2].max(0) - trunk[:, :2].min(0)).max()
            cw = (canopy[:, :2].max(0) - canopy[:, :2].min(0)).max()
            if tw > self.max_trunk_width or cw < self.min_canopy_width or cw < 1.8 * tw:
                continue
            cx, cy = (p[:, :2].max(0) + p[:, :2].min(0)) / 2
            if any(abs(cx - b.x) < b.l / 2 + 0.5 and abs(cy - b.y) < b.l / 2 + 0.5 for b in avoid):
                continue
            score = min(1.0, 0.5 + 0.1 * (cw / max(tw, 0.1)))
            trees.append(Box3D('TREE', score, cx, cy, zmax / 2, max(ext[0], 0.5), max(ext[1], 0.5), zmax, 0.0))
        return trees


def _above_local_ground(pts, min_h, tile=2.0):
    """Mask of points more than min_h above the lowest point in their 2 m tile (handles slopes)."""
    tx = np.floor(pts[:, 0] / tile).astype(np.int64)
    ty = np.floor(pts[:, 1] / tile).astype(np.int64)
    key = (tx - tx.min()) * (ty.max() - ty.min() + 1) + (ty - ty.min())
    _, inv = np.unique(key, return_inverse=True)
    zmin = np.full(inv.max() + 1, np.inf)
    np.minimum.at(zmin, inv, pts[:, 2])
    zmin = np.minimum(zmin, 0.3)  # a tile with no ground return (e.g. under canopy) uses the global ground
    return pts[:, 2] > zmin[inv] + min_h


class LidarDetector:
    def __init__(self, model_dir, use_gpu=True, score_thresh=0.35, detect_trees=True):
        self.cp = CenterPoint(model_dir, use_gpu, score_thresh)
        self.trees = TreeDetector() if detect_trees else None
        self.ground_z = None

    def __call__(self, pts_sensor):
        """pts_sensor: (N,3) in the lidar frame. Returns boxes in the lidar frame."""
        g = estimate_ground_z(pts_sensor)
        if g is not None:
            self.ground_z = g if self.ground_z is None else 0.8 * self.ground_z + 0.2 * g
        gz = self.ground_z or 0.0
        pts = pts_sensor - np.array([0.0, 0.0, gz], np.float32)
        boxes = self.cp(pts)
        if self.trees is not None:
            boxes += self.trees(pts, avoid=boxes)
        for b in boxes:
            b.z += gz
        return boxes
