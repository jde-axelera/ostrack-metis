"""Shared helpers for the OSTrack -> Metis scripts.

Preprocessing, target cropping, the Hann window, box utilities and the OSTrack
tracking loop, factored out so run_pth.py / run_ptgraph.py / compare_pth_ptgraph.py
all use identical logic and only differ in how they produce the score/size/offset
maps.
"""
import math

import numpy as np
import cv2 as cv
import torch

# ImageNet normalisation (OSTrack training-time preprocessing).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sample_target(im, bb, factor, out_sz):
    """Square crop centred on bb=[x,y,w,h] of side ceil(sqrt(w*h)*factor), resized to
    out_sz. Returns (crop_rgb, resize_factor). Mirrors
    lib/train/data/processing_utils.sample_target (resize_factor = out_sz / crop_sz)."""
    x, y, w, h = bb
    crop = math.ceil(math.sqrt(w * h) * factor)
    x1 = round(x + 0.5 * w - crop * 0.5); x2 = x1 + crop
    y1 = round(y + 0.5 * h - crop * 0.5); y2 = y1 + crop
    xp1, xp2 = max(0, -x1), max(x2 - im.shape[1] + 1, 0)
    yp1, yp2 = max(0, -y1), max(y2 - im.shape[0] + 1, 0)
    c = im[y1 + yp1:y2 - yp2, x1 + xp1:x2 - xp2, :]
    c = cv.copyMakeBorder(c, yp1, yp2, xp1, xp2, cv.BORDER_CONSTANT)
    return cv.resize(c, (out_sz, out_sz)), out_sz / crop


def preprocess(rgb_crop):
    """HWC uint8 RGB crop -> (1,3,H,W) float32, ImageNet-normalised."""
    t = torch.from_numpy(rgb_crop).float().permute(2, 0, 1).unsqueeze(0)
    return ((t / 255.0) - IMAGENET_MEAN) / IMAGENET_STD


def _hann1d(sz):
    return 0.5 * (1 - torch.cos((2 * math.pi / (sz + 1)) * torch.arange(1, sz + 1).float()))


def hann2d(n):
    """Centred 2D cosine window (n x n), as in lib/test/utils/hann.hann2d."""
    return _hann1d(n).reshape(1, 1, -1, 1) * _hann1d(n).reshape(1, 1, 1, -1)


def clip_box(box, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W - margin); x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H - margin); y2 = min(max(margin, y2), H)
    return [x1, y1, max(margin, x2 - x1), max(margin, y2 - y1)]


def iou(a, b):
    """IoU of two [x,y,w,h] boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def maps_fp32(net):
    """Score/size/offset provider that runs the full fp32 OSTrack network."""
    def fn(z, x):
        with torch.no_grad():
            o = net.forward(template=z, search=x, ce_template_mask=None)
        return o["score_map"], o["size_map"], o["offset_map"]
    return fn


def maps_ptgraph(gm, box_head):
    """Score/size/offset provider for the quantised backbone (the .ptgraph, i.e. what
    the .axm computes). The backbone runs quantised; the conv head runs on the host in
    fp32 -- exactly the head_on_host deployment split."""
    def fn(z, x):
        with torch.no_grad():
            feat = gm(z, x)
            feat = feat if isinstance(feat, torch.Tensor) else torch.as_tensor(feat)
            score, size, off = box_head.get_score_map(feat)   # sigmoid(score), sigmoid(size), raw off
        return score, size, off
    return fn


class Tracker:
    """OSTrack single-object tracking loop (search window follows the prediction, Hann
    window on the score map, cal_bbox decode, map_box_back). `maps_fn(z, x)` returns the
    (score_map, size_map, offset_map) triple; `box_head` supplies cal_bbox/get_score_map."""

    def __init__(self, maps_fn, box_head, cfg):
        self.maps_fn = maps_fn
        self.box_head = box_head
        self.ssz = cfg.TEST.SEARCH_SIZE
        self.tsz = cfg.TEST.TEMPLATE_SIZE
        self.sf = cfg.TEST.SEARCH_FACTOR
        self.tf = cfg.TEST.TEMPLATE_FACTOR
        self.hann = hann2d(self.ssz // cfg.MODEL.BACKBONE.STRIDE)
        self.z = None
        self.state = None

    def init(self, rgb0, init_bbox):
        zc, _ = sample_target(rgb0, list(init_bbox), self.tf, self.tsz)
        self.z = preprocess(zc)
        self.state = list(init_bbox)
        return self.state

    def track(self, rgb):
        H, W, _ = rgb.shape
        xc, rf = sample_target(rgb, self.state, self.sf, self.ssz)
        x = preprocess(xc)
        score, size, off = self.maps_fn(self.z, x)
        pb = self.box_head.cal_bbox(self.hann * score, size, off).view(-1, 4)[0]
        cx, cy, w, h = (pb * self.ssz / rf).tolist()
        cxp, cyp = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        half = 0.5 * self.ssz / rf
        self.state = clip_box([cx + (cxp - half) - 0.5 * w,
                               cy + (cyp - half) - 0.5 * h, w, h], H, W, margin=10)
        return self.state
