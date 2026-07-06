"""Run the quantized OSTrack .ptgraph as a single-object tracker on a video.

The .ptgraph is the AxMO-quantized backbone (the exact graph compiled into the Metis
.axm); running it on CPU reproduces the device's int8 arithmetic (fake-quant). It is
head_on_host: the quantized backbone outputs a (1,768,16,16) feature map and the conv
head + sigmoid run on the host in fp32 (loaded from the checkpoint). Same tracking loop
as run_pth.py, so trajectories are directly comparable.
"""
import os
import sys
import argparse

import numpy as np
import cv2 as cv
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("OSTRACK_ROOT", HERE))

from lib.config.ostrack.config import cfg, update_config_from_file
from lib.models.ostrack import build_ostrack
from axelera.model_optimizer.api import load_model
from ostrack_common import Tracker, maps_ptgraph


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="vitb_256_mae_32x4_ep300")
    ap.add_argument("--ptgraph", required=True, help="quantized model.ptgraph")
    ap.add_argument("--ckpt", required=True, help="checkpoint for the fp32 host conv head")
    ap.add_argument("--video", required=True)
    ap.add_argument("--init_bbox", required=True, help="frame-0 target box x,y,w,h")
    ap.add_argument("--n_frames", type=int, default=500)
    ap.add_argument("--out", default="ptgraph_boxes.txt")
    ap.add_argument("--out_video", default=None, help="optional overlay .mp4")
    args = ap.parse_args()

    update_config_from_file(os.path.join(os.environ.get("OSTRACK_ROOT", HERE),
                                          "experiments/ostrack/%s.yaml" % args.config))
    net = build_ostrack(cfg, training=False)                 # only box_head is used (head on host)
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu")["net"], strict=True)
    net.eval()
    gm = load_model(args.ptgraph)

    tracker = Tracker(maps_ptgraph(gm, net.box_head), net.box_head, cfg)
    init = [float(v) for v in args.init_bbox.split(",")]

    cap = cv.VideoCapture(args.video)
    ok, f0 = cap.read()
    assert ok, f"cannot read {args.video}"
    tracker.init(cv.cvtColor(f0, cv.COLOR_BGR2RGB), init)

    vw = None
    if args.out_video:
        vw = cv.VideoWriter(args.out_video, cv.VideoWriter_fourcc(*"mp4v"), 20,
                            (int(cap.get(3)), int(cap.get(4))))
    boxes = [init]
    for i in range(1, args.n_frames):
        ok, fr = cap.read()
        if not ok:
            break
        state = tracker.track(cv.cvtColor(fr, cv.COLOR_BGR2RGB))
        boxes.append(list(state))
        if vw is not None:
            x, y, w, h = [int(v) for v in state]
            cv.rectangle(fr, (x, y), (x + w, y + h), (0, 0, 255), 2)
            vw.write(fr)
        if i % 50 == 0:
            print(f"frame {i}: {[round(v, 1) for v in state]}", flush=True)
    if vw is not None:
        vw.release()
    cap.release()

    np.savetxt(args.out, np.array(boxes), fmt="%.2f", delimiter=",")
    print(f"wrote {len(boxes)} boxes -> {args.out}"
          + (f" and overlay -> {args.out_video}" if args.out_video else ""))


if __name__ == "__main__":
    main()
