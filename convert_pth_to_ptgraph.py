"""Convert a trained OSTrack .pth checkpoint into a quantized .ptgraph (and,
optionally, a Metis .axm), following the AxMO + axelera-graph-compiler path.

Pipeline:
    build fp32 OSTrack  ->  model surgery (compile-enabling, numerically no-op)
    torch.export.export(wrapper, (template, search))          -> ExportedProgram
    amo.import_exported_program -> prepare -> calibrate -> finalize
    save_model('model.ptgraph')                               -> quantized reference
    [--compile] compile_single_graph(...)                     -> model.axm  (Metis)

The surgery is required only so the graph legalizes/quantizes/compiles; each step
is a numerical no-op verified against the fp32 model:
  * hand-rolled attention  -> F.scaled_dot_product_attention (so AxMO can quantize it)
  * an all-zeros additive attention mask (SDPA lowering segfaults on a null mask)
  * unshared patch_embed   (AxMO's bias pass can't handle one conv called twice)
  * head ReLU              -> clamp(min=0)  (PReLU custom op won't legalize)
  * zero bias added to bias-free conv/linear layers
Compiled backbone-only (head_on_host): the conv head + sigmoid + decode run on host.
"""
import os
import sys
import argparse
import copy
import types

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("OSTRACK_ROOT", HERE))   # OSTrack project root (lib/, experiments/)

from lib.config.ostrack.config import cfg, update_config_from_file
from lib.models.ostrack import build_ostrack
from calibration import OSTrackCalibSet


# ---------------------------------------------------------------- model surgery
def patch_vit_attention_sdpa():
    """Swap vit.py's raw bmm+softmax attention for F.scaled_dot_product_attention so
    AxMO's SDPA passes recognise/quantize it. Uses a registered all-zeros additive mask
    (see register_sdpa_zero_masks) so the SDPA op gets a materialized mask operand --
    a null mask segfaults the compiler's SDPA lowering. Math is identical."""
    import torch.nn.functional as F
    from lib.models.ostrack import vit as _vit

    def sdpa_forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask_zero)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return (out, None) if return_attention else out

    _vit.Attention.forward = sdpa_forward


def register_sdpa_zero_masks(backbone):
    """Register a (1,1,N,N) all-zeros additive mask (a graph constant, no-op) on every
    attention block. N = template tokens + search tokens (concatenated backbone)."""
    lens = backbone.pos_embed_z.shape[1] + backbone.pos_embed_x.shape[1]
    for blk in backbone.blocks:
        blk.attn.register_buffer("attn_mask_zero", torch.zeros(1, 1, lens, lens), persistent=True)
    print(f"registered zero SDPA mask (1,1,{lens},{lens}) on {len(backbone.blocks)} blocks")


def unshare_patch_embed(backbone):
    """Give the template its own copy of patch_embed so each conv module is called once
    (AxMO's RewriteModuleWithBias crashes on one module called twice). Non-CE forward."""
    from lib.models.ostrack.utils import combine_tokens, recover_tokens
    backbone.patch_embed_z = copy.deepcopy(backbone.patch_embed)

    def forward_features(self, z, x):
        x = self.patch_embed(x)
        z = self.patch_embed_z(z)
        z = z + self.pos_embed_z
        x = x + self.pos_embed_x
        x = combine_tokens(z, x, mode=self.cat_mode)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = recover_tokens(x, self.pos_embed_z.shape[1], self.pos_embed_x.shape[1], mode=self.cat_mode)
        return self.norm(x), {"attn": None}

    backbone.forward_features = types.MethodType(forward_features, backbone)


class ClampReLU(nn.Module):
    """relu(x) == clamp(x, min=0). AxMO maps nn.ReLU to a PReLU custom op the backend
    can't legalize; aten.clamp is supported. No accuracy change."""
    def forward(self, x):
        return torch.clamp(x, min=0.0)


def replace_relu_with_clamp(module):
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, ClampReLU()); n += 1
        else:
            n += replace_relu_with_clamp(child)
    return n


class OSTrackMapsOnly(nn.Module):
    """Backbone -> center-head score/size/offset maps (RAW logits; host applies sigmoid),
    bypassing cal_bbox (argmax/gather won't quantize). With head_on_host=True, returns
    the (B,768,16,16) backbone feature map and the conv head runs on the host."""
    def __init__(self, net, head_on_host=False):
        super().__init__()
        self.net = net
        self.head_on_host = head_on_host

    def _score_maps(self, x):
        h = self.net.box_head
        ctr = h.conv5_ctr(h.conv4_ctr(h.conv3_ctr(h.conv2_ctr(h.conv1_ctr(x)))))
        off = h.conv5_offset(h.conv4_offset(h.conv3_offset(h.conv2_offset(h.conv1_offset(x)))))
        siz = h.conv5_size(h.conv4_size(h.conv3_size(h.conv2_size(h.conv1_size(x)))))
        return ctr, siz, off

    def forward(self, template, search):
        x, _ = self.net.backbone(z=template, x=search, ce_template_mask=None,
                                 ce_keep_rate=None, return_last_attn=False)
        feat = x[-1] if isinstance(x, list) else x
        enc_opt = feat[:, -self.net.feat_len_s:]
        C = enc_opt.shape[-1]
        f = self.net.feat_sz_s
        opt_feat = enc_opt.transpose(1, 2).reshape(-1, C, f, f)
        if self.head_on_host:
            return opt_feat
        return self._score_maps(opt_feat)


def build_wrapper(config, ckpt, head_on_host=True):
    update_config_from_file(os.path.join(os.environ.get("OSTRACK_ROOT", HERE),
                                          "experiments/ostrack/%s.yaml" % config))
    net = build_ostrack(cfg, training=False)
    net.load_state_dict(torch.load(ckpt, map_location="cpu")["net"], strict=True)
    net.eval()

    patch_vit_attention_sdpa()
    unshare_patch_embed(net.backbone)
    register_sdpa_zero_masks(net.backbone)
    if not head_on_host:
        print(f"replaced {replace_relu_with_clamp(net.box_head)} head ReLU with clamp")

    wrapper = OSTrackMapsOnly(net, head_on_host=head_on_host).eval()
    nb = 0
    for mod in wrapper.modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and mod.bias is None:
            oc = getattr(mod, "out_channels", None) or mod.out_features
            mod.bias = nn.Parameter(torch.zeros(oc)); nb += 1
    print(f"added zero-bias to {nb} bias-free conv/linear modules")
    return wrapper, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="vitb_256_mae_32x4_ep300")
    ap.add_argument("--ckpt", required=True, help="OSTrack .pth.tar checkpoint")
    ap.add_argument("--out_dir", default="build/ostrack_ptgraph")
    ap.add_argument("--video", default="/home/ubuntu/Arquimea/ir_crop.mp4",
                    help="calibration video")
    ap.add_argument("--init_bbox", default="348,147,38,84", help="calib crop location x,y,w,h")
    ap.add_argument("--n_calib", type=int, default=100)
    ap.add_argument("--smooth_qk_alpha", type=float, default=None)
    ap.add_argument("--compile", action="store_true", help="also compile to a Metis .axm")
    ap.add_argument("--target", default="device", choices=["device", "sim"])
    args = ap.parse_args()

    from pathlib import Path
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    bbox = [float(v) for v in args.init_bbox.split(",")]

    wrapper, cfg_ = build_wrapper(args.config, args.ckpt, head_on_host=True)
    tsz, ssz = cfg_.TEST.TEMPLATE_SIZE, cfg_.TEST.SEARCH_SIZE
    t, s = torch.ones(1, 3, tsz, tsz), torch.ones(1, 3, ssz, ssz)

    print("torch.export.export ...")
    ep = torch.export.export(wrapper, args=(t, s))

    import axelera.model_optimizer as amo
    from axelera.model_optimizer.api import (prepare_model_for_optimization,
                                             finalize_optimized_model, save_model)
    from axelera.model_optimizer.trainer.calibration import calibrate_model
    from torch.utils.data import DataLoader

    axmo_config = amo.get_default_config(generation=amo.HardwareGeneration.METIS,
                                         smooth_quant_alpha=0.5,
                                         smooth_qk_alpha=args.smooth_qk_alpha)
    gm = amo.import_exported_program(ep); gm.to("cpu")
    prepare_model_for_optimization(gm, axmo_config)

    loader = DataLoader(OSTrackCalibSet(args.video, bbox, args.n_calib, tsz, ssz),
                        batch_size=1, shuffle=False)
    print("calibrating ...")
    calibrate_model(gm, loader, progress_bar=True, device=torch.device("cpu"))
    finalize_optimized_model(gm, axmo_config)

    ptgraph = out_dir / "model.ptgraph"
    save_model(gm, str(ptgraph))
    print(f"saved quantized reference -> {ptgraph}")

    if not args.compile:
        return

    # ---- compile to .axm ----
    from axelera.compiler.alto.compiler.config import HardwareGeneration, Target
    from axelera.graph_compiler.api import compile_single_graph
    from axelera.graph_compiler.config import CompilerConfig
    from axelera.graph_compiler.types import (DeviceQuantBoundarySetting, DType,
                                              Pipeline, TensorType)
    compiler_config = CompilerConfig(
        generation=HardwareGeneration.OMEGA,
        target=Target.AICORE_SIM if args.target == "sim" else Target.DEVICE,
        output_dir=str(out_dir),
        pipeline=Pipeline.GENERIC,
        enable_fold_constant_ops=False,
        device_quant_boundary_setting=DeviceQuantBoundarySetting.HOST_ONLY,
        # PromoteBuffers aborts ("alias '' cannot rebind onto ''") on this graph; it is
        # an optimization pass, not required for a correct artifact.
        enable_buffer_promotion=False,
    )
    input_types = (
        TensorType(shape=(1, 3, tsz, tsz), dtype=DType.FLOAT32, name="template"),
        TensorType(shape=(1, 3, ssz, ssz), dtype=DType.FLOAT32, name="search"),
    )
    print("compile_single_graph ...")
    axm = compile_single_graph(gm, compiler_config, input_types)
    print("COMPILED:", axm)


if __name__ == "__main__":
    main()
