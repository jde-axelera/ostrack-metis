"""Calibration dataset for post-training quantization of OSTrack.

Yields (template, search) pairs the same way the model is driven at inference:
a single fixed template cropped from frame 0, plus N search crops sampled across
the clip at the init-bbox location. This is the exact activation distribution the
AxMO calibrator observes in convert_pth_to_ptgraph.py.

NOTE: this crops every search region at the *fixed* init bbox (not a real tracking
loop). It is deliberately simple -- enough activation diversity for PTQ range
collection. For better int8 accuracy, calibrate on more/representative data
(multiple clips, real tracking crops).
"""
import numpy as np
import cv2 as cv
import torch

from ostrack_common import sample_target, preprocess


class OSTrackCalibSet(torch.utils.data.Dataset):
    def __init__(self, video, init_bbox, n, tsz=128, ssz=256, tfac=2.0, sfac=4.0):
        cap = cv.VideoCapture(video)
        total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        ok, f0 = cap.read()
        assert ok, f"cannot read {video}"
        rgb0 = cv.cvtColor(f0, cv.COLOR_BGR2RGB)
        crop, _ = sample_target(rgb0, init_bbox, tfac, tsz)
        self.template = preprocess(crop)[0]                 # (3,tsz,tsz), no batch dim
        idxs = np.linspace(0, max(total - 1, 1), n).astype(int)
        self.searches = []
        for i in idxs:
            cap.set(cv.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                continue
            rgb = cv.cvtColor(fr, cv.COLOR_BGR2RGB)
            crop, _ = sample_target(rgb, init_bbox, sfac, ssz)
            self.searches.append(preprocess(crop)[0])       # (3,ssz,ssz)
        cap.release()
        print(f"calibration: 1 template + {len(self.searches)} search crops from {video}")

    def __len__(self):
        return len(self.searches)

    def __getitem__(self, i):
        return self.template, self.searches[i]              # DataLoader -> model(template, search)
