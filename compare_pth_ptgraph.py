"""Compare the fp32 .pth model against the quantized .ptgraph as two INDEPENDENT
trackers on the same video, and report how closely they agree.

Both run the full OSTrack loop from the same init box, each updating its own state.
Prints per-frame and summary IoU / center-error, and writes an overlay video
(fp32 = green, quantized = red) plus the trajectories.
"""
import os
import sys
import argparse
import time

import numpy as np
import cv2 as cv
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("OSTRACK_ROOT", HERE))

from lib.config.ostrack.config import cfg, update_config_from_file
from lib.models.ostrack import build_ostrack
from axelera.model_optimizer.api import load_model
from ostrack_common import Tracker, maps_fp32, maps_ptgraph, iou


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="vitb_256_mae_32x4_ep300")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ptgraph", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--init_bbox", required=True, help="frame-0 target box x,y,w,h")
    ap.add_argument("--n_frames", type=int, default=500)
    ap.add_argument("--out_video", default="compare.mp4")
    ap.add_argument("--out_traj", default="traj.npz")
    args = ap.parse_args()

    update_config_from_file(os.path.join(os.environ.get("OSTRACK_ROOT", HERE),
                                          "experiments/ostrack/%s.yaml" % args.config))
    net = build_ostrack(cfg, training=False)
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu")["net"], strict=True)
    net.eval()
    gm = load_model(args.ptgraph)

    fp32 = Tracker(maps_fp32(net), net.box_head, cfg)
    quant = Tracker(maps_ptgraph(gm, net.box_head), net.box_head, cfg)
    init = [float(v) for v in args.init_bbox.split(",")]

    cap = cv.VideoCapture(args.video)
    ok, f0 = cap.read()
    assert ok, f"cannot read {args.video}"
    rgb0 = cv.cvtColor(f0, cv.COLOR_BGR2RGB)
    fp32.init(rgb0, init); quant.init(rgb0, init)

    vw = cv.VideoWriter(args.out_video, cv.VideoWriter_fourcc(*"mp4v"), 20,
                        (int(cap.get(3)), int(cap.get(4))))
    fB, qB, ious, cerr = [list(init)], [list(init)], [], []
    t0 = time.time()
    for i in range(1, args.n_frames):
        ok, fr = cap.read()
        if not ok:
            break
        rgb = cv.cvtColor(fr, cv.COLOR_BGR2RGB)
        sf = fp32.track(rgb); sq = quant.track(rgb)
        fB.append(list(sf)); qB.append(list(sq)); ious.append(iou(sf, sq))
        cerr.append(((sf[0] + sf[2] / 2 - sq[0] - sq[2] / 2) ** 2 +
                     (sf[1] + sf[3] / 2 - sq[1] - sq[3] / 2) ** 2) ** 0.5)
        x, y, w, h = [int(v) for v in sf]; cv.rectangle(fr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x, y, w, h = [int(v) for v in sq]; cv.rectangle(fr, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(fr, "fp32=green quant=red f%d IoU%.2f" % (i, ious[-1]),
                   (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        vw.write(fr)
        if i % 50 == 0:
            print("frame %d IoU=%.3f cerr=%.1fpx %.1fs" % (i, ious[-1], cerr[-1], time.time() - t0),
                  flush=True)
    vw.release(); cap.release()

    ious, cerr = np.array(ious), np.array(cerr)
    print("=== SUMMARY over %d frames ===" % len(ious))
    print("mean IoU        : %.4f" % ious.mean())
    print("median IoU      : %.4f" % np.median(ious))
    print("min IoU         : %.4f" % ious.min())
    print("%% frames IoU>0.5 : %.1f" % (100 * (ious > 0.5).mean()))
    print("%% frames IoU>0.7 : %.1f" % (100 * (ious > 0.7).mean()))
    print("mean center err : %.2f px" % cerr.mean())
    print("max center err  : %.2f px" % cerr.max())
    np.savez(args.out_traj, fp32=np.array(fB), quant=np.array(qB), iou=ious, cerr=cerr)
    print(f"saved {args.out_video} and {args.out_traj}")


if __name__ == "__main__":
    main()
