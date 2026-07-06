# ostrack-metis

Scripts to quantize and deploy **OSTrack** (the non-CE ViT-B tracker) to the
**Axelera Metis** accelerator, and to validate the quantized model against the
original fp32 checkpoint.

```
OSTrack_ep0300.pth.tar
      │  quantize_and_compile.py  (model surgery + torch.export + AxMO calibrate)
      ▼
model.ptgraph                        ← int8-quantized backbone (what the .axm computes)
      │  quantize_and_compile.py --compile  (axelera-graph-compiler + axkernelcc)
      ▼
model.axm                            ← Metis device artifact (backbone; head runs on host)
```

## Scripts

| Script | What it does |
| --- | --- |
| `quantize_and_compile.py` | `.pth` → quantized `.ptgraph` (+ optional `--compile` → `.axm`). Applies the compile-enabling model surgery, `torch.export`, and AxMO post-training quantization. |
| `calibration.py` | `OSTrackCalibSet` — the calibration data fed to the quantizer (template from frame 0 + N search crops). |
| `run_pth.py` | Run the fp32 `.pth` checkpoint as a tracker on a video. |
| `run_ptgraph.py` | Run the quantized `.ptgraph` (the exact graph in the `.axm`) as a tracker on CPU. |
| `compare_pth_ptgraph.py` | Run both as independent trackers and report IoU / center-error + overlay video. |
| `ostrack_common.py` | Shared preprocessing, cropping, Hann window, and the OSTrack tracking loop. |

## Requirements

These scripts depend on the **OSTrack project code** (`lib/`, `experiments/`) and the
**Axelera SDK** Python packages (`axelera-model-optimizer`, `axelera-graph-compiler`,
`axelera-compiler`, `axelera-runtime`, plus `torch`, `opencv-python`, `numpy`).

Run them from inside your OSTrack checkout (so `lib/` and `experiments/` are importable),
or point `OSTRACK_ROOT` at it:

```bash
export OSTRACK_ROOT=/path/to/ostrack_metis   # dir containing lib/ and experiments/
```

## Usage

Convert a checkpoint to a quantized `.ptgraph` (and optionally a device `.axm`):

```bash
python quantize_and_compile.py \
    --config vitb_256_mae_32x4_ep300 \
    --ckpt   output/checkpoints/train/ostrack/vitb_256_mae_32x4_ep300/OSTrack_ep0300.pth.tar \
    --video  /path/to/calibration.mp4 \
    --init_bbox 348,147,38,84 \
    --n_calib 100 \
    --out_dir build/ostrack_ptgraph
    # add --compile [--target device|sim] to also produce model.axm
```

Run either model as a tracker:

```bash
python run_pth.py      --ckpt CKPT --video coyote.mp4 --init_bbox 690,348,16,27 --out_video pth.mp4
python run_ptgraph.py  --ckpt CKPT --ptgraph build/ostrack_ptgraph/model.ptgraph \
                       --video coyote.mp4 --init_bbox 690,348,16,27 --out_video ptgraph.mp4
```

Compare them:

```bash
python compare_pth_ptgraph.py --ckpt CKPT --ptgraph build/ostrack_ptgraph/model.ptgraph \
    --video coyote.mp4 --init_bbox 690,348,16,27 --n_frames 500
```

## Notes

- **head_on_host:** the compiled backbone outputs a `(1,768,16,16)` feature map; the
  center-head convolutions + sigmoid + argmax decode run on the host in fp32. All the
  `run_*`/`compare` scripts follow this split.
- **Model surgery** in `quantize_and_compile.py` (SDPA rewrite, all-zeros attention
  mask, unshared patch-embed, ReLU→clamp, zero-bias) is required only to let the model
  quantize/legalize/compile; each step is numerically a no-op (fp32 output matches the
  original checkpoint / an ONNX export to ~1e-6).
- **`enable_buffer_promotion=False`** is set at compile time: the ALTO `PromoteBuffers`
  pass otherwise aborts with `Cannot substitute buffers: alias '' cannot rebind onto ''`
  on this graph. It is an optimization pass, not required for a correct artifact.
- **On-device execution** requires the Metis firmware/driver/runtime versions to match
  the toolchain that rendered the `.axm` (i.e. `axkernelcc` version == device firmware).
  Running the `.ptgraph` on CPU (via `run_ptgraph.py`) reproduces the device's int8
  arithmetic and is the reference used for the comparison.
