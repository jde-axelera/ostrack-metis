# Graph surgery — operator-level walkthrough

`quantize_and_compile.py::build_wrapper` builds the fp32 OSTrack model and then applies
a set of transformations that are each **mathematically a no-op** but are required so the
exported ATen graph contains only operators that (a) AxMO has a quantization factory for
and (b) the OMEGA / TVM backend can legalize. The compiler's failures were never "wrong
math" — they were "unknown/unsupported operator" or "null operand". Each fix rewrites the
*operator*, not the *computation* (verified: fp32-with-surgery ≡ checkpoint ≡ ONNX export
to ~1e-6, identical decoded boxes).

This document ties each surgery to the exact lines in OSTrack's `lib/models/ostrack/vit.py`,
`lib/models/ostrack/base_backbone.py` and `lib/models/layers/head.py`.

## The concrete model (vitb_256_mae_32x4_ep300)

- embed dim **C = 768**, **num_heads = 12**, **head_dim = 64**, so
  `self.scale = head_dim**-0.5 = 64**-0.5 = 0.125`
- template `z`: 128x128, patch 16 -> 8x8 = **64 tokens**
- search `x`: 256x256, patch 16 -> 16x16 = **256 tokens**
- concatenated sequence **N = 64 + 256 = 320 tokens**
- activations flowing through a block are `(B=1, N=320, C=768)`

---

## 1. `patch_vit_attention_sdpa` — replacing the attention math

**Original `vit.py::Attention.forward`:**
```python
qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
q, k, v = qkv[0], qkv[1], qkv[2]                 # each (1, 12, 320, 64)
attn = (q @ k.transpose(-2, -1)) * self.scale    # (1, 12, 320, 320)   <- matmul #1
attn = attn.softmax(dim=-1)                      # (1, 12, 320, 320)
x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (1, 320, 768)       <- matmul #2
x = self.proj(x)
```

Each `@` on 4-D tensors lowers to **`aten.bmm` / `aten.matmul`** (batched over the `(1,12)`
leading dims). AxMO's `MetisAnnotator` looks up an annotation factory *keyed by aten op*,
and there is **no factory for `aten.bmm.default`**. Confirmed via the `--no_sdpa` experiment:
```
KeyError: Missing annotation factory function for 'call_function' node bmm with target aten.bmm.default
```

**Replacement** collapses those five lines into one fused op:
```python
out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask_zero)
```
The exporter emits a single `scaled_dot_product_attention` node, which AxMO's
`ModularizeSDPAFunction -> InjectFakeQuantizedSDPA -> InjectCustomSDPA` passes recognize
and quantize as one attention primitive.

**Numerical equivalence** — SDPA is `softmax(Q.Kᵀ · scale + mask) · V`, and its *default*
`scale` is `head_dim**-0.5 = 0.125`, exactly `self.scale`. For one query row `q_i` vs key `k_j`:
- hand-rolled: `attn_ij = (q_i · k_j) * 0.125`
- SDPA:        `attn_ij = (q_i · k_j) / sqrt(64) = (q_i · k_j) / 8 = (q_i · k_j) * 0.125`

Same scores -> same softmax -> same `Σ_j softmax_ij · v_j`.

**The one real difference:** the hand-rolled path can *return* the `attn` weights
(`return_attention=True`) — candidate elimination (CE) needs them. SDPA fuses and discards
them. Fine here: this is the **non-CE** model (`return_attention=False`, `ce_template_mask=None`).

---

## 2. `register_sdpa_zero_masks` — giving SDPA a mask operand

The ALTO SDPA op (`axol.scaled_dot_product_attention`) has an **`attention_mask` operand**
at index 13. Calling `F.scaled_dot_product_attention(q, k, v)` with **no** `attn_mask`
records nothing there, so at codegen that operand resolves to `None` and ALTO passes `None`
into TVM's TIR builder `A.axop(...)` -> TVM builds a `Call` with a null operand ->
**null-pointer deref inside `libtvm.so`** (faulting frame `scaled_dot_product_attention.py:268`,
instrumentation printed `attention_mask None=True type=NoneType`).

**Why OSTrack specifically:** a stock ViT-Small classifier's 197 tokens get padded to 256,
and that padding forces a mask constant to be materialized (to `-inf` the pad columns),
shape `(1,1,197,256)`. OSTrack's **N=320 = 5x64 is already tile-aligned**, so no padding, no
mask -> the `None`.

**Fix:**
```python
lens = pos_embed_z.shape[1] + pos_embed_x.shape[1]        # 64 + 256 = 320
blk.attn.register_buffer("attn_mask_zero", torch.zeros(1, 1, lens, lens), persistent=True)
```
SDPA uses an **additive float mask**: `scores = Q.Kᵀ·scale + mask`. All-zeros adds 0 ->
`softmax(s + 0) = softmax(s)`, an identity. But operand 13 is now a real `(1,1,320,320)`
constant buffer instead of null.

**Why a registered buffer, not `torch.zeros(...)` inline:** an inline `torch.zeros` becomes
an **`aten.full`** node, and AxMO has no factory for that either:
```
KeyError: Missing annotation factory function for 'call_function' node full with target aten.full.default
```
A `register_buffer` is *lifted* by `torch.export` as a graph **constant** (`get_attr`),
not a runtime op, so there is nothing for the annotator to choke on. `persistent=True`
keeps it a normal named constant.

---

## 3. `unshare_patch_embed` — one conv module called twice

**Original `base_backbone.py::forward_features`:**
```python
x = self.patch_embed(x)   # search:   Conv2d(3,768,k=16,s=16) -> (1,768,16,16) -> (1,256,768)
z = self.patch_embed(z)   # template: SAME module            -> (1,64,768)
```
`patch_embed.proj` is a single `nn.Conv2d(3, 768, kernel=16, stride=16)` invoked **twice**,
so the exported graph has two conv nodes pointing at the *same* module/parameters.

**What breaks:** AxMO's `RewriteModuleWithBias` rewrites each conv's bias by mutating the
module. It processes the first node (rewrites/nulls the bias), then reaches the second node
whose module was already mutated -> crash on the inconsistent state.

**Fix** — an independent, identical copy for the template:
```python
backbone.patch_embed_z = copy.deepcopy(backbone.patch_embed)
# patched forward_features:
x = self.patch_embed(x)      # module A, called once
z = self.patch_embed_z(z)    # module B (deepcopy), called once
```
Each module now produces exactly one graph node.

**Numerical equivalence:** `deepcopy` copies weight `W (768,3,16,16)` and bias `b (768)`
bit-for-bit, so `patch_embed_z(z)` computes the identical convolution. The patched
`forward_features` also drops the CE args and returns `self.norm(x), {"attn": None}` — valid
because this is the non-CE model. (Running the *CE* config by mistake gives
`forward_features() got an unexpected keyword argument 'ce_template_mask'`.)

---

## 4. `replace_relu_with_clamp` — and an honest caveat

**Original head, `head.py::conv`:**
```python
nn.Sequential(nn.Conv2d(..., bias=True), nn.BatchNorm2d(out), nn.ReLU(inplace=True))
```
Every `CenterPredictor` tower ends `... -> BN -> ReLU`.

**What breaks:** AxMO maps `nn.ReLU` to a **PReLU custom op** (slope 0 ≡ ReLU), and the OMEGA
MLIR backend can't legalize the `x >= 0` select PReLU lowering emits. `ClampReLU` swaps it:
```python
torch.clamp(x, min=0.0)   # clamp(-2,min=0)=0 ; clamp(3,min=0)=3  -> exactly ReLU
```
`aten.clamp` is a legalizable op.

**Honest caveat** (why this only runs when `head_on_host=False`): in the full-model,
head-on-device path this very clamp hit a *different* legalization failure —
```
error: failed to legalize operation 'torch.aten.clamp'
```
because `clamp(min=0, max=None)` with an unbounded upper end wasn't accepted. That is a major
reason **head_on_host is the working path**: the head is not compiled at all, so this is moot.
The shipped artifacts run the conv head on the host in fp32.

---

## 5. The zero-bias loop

```python
for mod in wrapper.modules():
    if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) and mod.bias is None:
        mod.bias = nn.Parameter(torch.zeros(out_channels))
```
`RewriteModuleWithBias` assumes every conv/linear has a `.bias` **Parameter**; a `bias=None`
layer crashes it. A zero bias is a no-op (`conv(x) + 0`). For this ViT-B in `head_on_host`
mode it reported **"added zero-bias to 0"** — the backbone is fully biased (`qkv_bias=True`,
`proj` and patch-embed `proj` all have bias). It is a safeguard that fires for models/heads
using `bias=False` convs; here it is a no-op both numerically and count-wise.

---

## 6. `OSTrackMapsOnly` — cutting the graph before the un-quantizable decode

**Original `OSTrack.forward`:** `backbone -> (1,320,768) -> take search tokens ->
reshape (1,768,16,16) -> box_head.get_score_map -> cal_bbox -> decoded box`.

**The wrapper stops early:**
```python
x, _ = self.net.backbone(z=template, x=search, ce_template_mask=None, ...)
feat = x[-1] if isinstance(x, list) else x          # (1, 320, 768)
enc_opt = feat[:, -self.net.feat_len_s:]            # (1, 256, 768)  -> search tokens only
opt_feat = enc_opt.transpose(1, 2).reshape(-1, 768, 16, 16)   # (1, 768, 16, 16)
if self.head_on_host:
    return opt_feat                                 # STOP: head runs on host
return self._score_maps(opt_feat)                   # else raw conv logits (no sigmoid)
```

Two operator-level decisions:

- **Bypass `cal_bbox`.** It does `torch.max(...)` (argmax) + `.gather(...)` +
  `idx // feat_sz` (floor-divide) -> `aten.max`/`aten.gather`/`aten.floor_divide`/`aten.lt`
  nodes the Metis annotator can't quantize, for a trivial argmax we do on the host anyway.
  So the graph is cut at the feature map (`head_on_host`) or at raw score maps
  (`_score_maps`, with **sigmoid removed** so the host applies it).
- **`transpose(1,2).reshape(...)` instead of the original `unsqueeze(-1).permute(...).view(...)`.**
  Numerically identical — both realize `opt_feat[b, c, i, j] = enc_opt[b, i*16 + j, c]` — but
  the original's trailing-dim `unsqueeze` produced a reshape the MLIR backend couldn't
  legalize, whereas `transpose + reshape` maps cleanly to supported ops.

---

## Through-line

Every surgery is chosen so the exported ATen graph contains only ops that AxMO can quantize
and the backend can legalize, while the values are provably unchanged. The compile failures
we resolved were all operator/representation problems (unknown op, null operand, un-legalizable
reshape/clamp), never numerical ones.
