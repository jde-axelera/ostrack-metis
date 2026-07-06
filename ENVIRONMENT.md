# Environment — Axelera SDK libraries & tools

The setup used to quantize/compile/run OSTrack on Metis. There are **two layers**:

1. a **Python virtualenv** with the Axelera pip packages (compiler, quantizer, runtime bindings), and
2. **system packages** (`apt`/`.deb`) providing the on-device runtime + firmware + PCIe driver.

All Axelera packages come from Axelera's private Artifactory (needs account/network):

- pip index: `https://artifactory.production.aws.axelera.ai/artifactory/api/pypi/axelera-pypi/simple/`
- apt repo:  `https://artifactory.production.aws.axelera.ai/artifactory/axelera-apt-source/ ubuntu22 main`

SDK: **Voyager SDK 1.8.0-rc1**, host **Ubuntu 22.04 / x86_64**, **Python 3.10.12**.

---

## 1. Python virtualenv (compile + quantize + host runtime bindings)

### Create & install

```bash
cd /path/to/internal-voyager-sdk           # the Voyager SDK checkout
python3 -m venv axelera-env
source axelera-env/bin/activate
pip install \
  --index-url https://artifactory.production.aws.axelera.ai/artifactory/api/pypi/axelera-pypi/simple/ \
  --extra-index-url https://pypi.org/simple \
  --trusted-host artifactory.production.aws.axelera.ai \
  "axelera-rt==1.8.0rc1" "axelera-devkit[hf-transformers]==1.8.0rc1"
```

Two meta-packages pull everything else in:
- **`axelera-devkit`** → the compile/quantize stack (`axelera-compiler`, `axelera-graph-compiler`, `axelera-model-optimizer`, `axelera_tvm`, `axelera-codegen`, …).
- **`axelera-rt`** → the host-side runtime bindings + CLI (`axdevice`, etc.).

### Exact versions resolved (`pip list`)

| Package | Version | Role |
| --- | --- | --- |
| `axelera-compiler` | `1.8.0rc1` | ALTO compiler front/back-end |
| `axelera-graph-compiler` | `1.1.0` | `compile_single_graph` (graph → `.axm`) |
| `axelera-model-optimizer` | `0.2.0.dev29+gecf1522fa9` | AxMO — PTQ (`import_exported_program`, calibrate, `save_model`) |
| `axelera_tvm` | `1.8.0rc1` | TVM backend (`libtvm.so`) used by ALTO |
| `axelera-codegen` | `1.8.0rc1` | kernel/axmodel codegen (invokes `axkernelcc`) |
| `axelera-qtoolsv2` | `1.8.0rc1` | quantization tooling |
| `axelera-qtools-tvm-interface` | `1.8.0rc1` | |
| `axelera-onnx2torch` | `1.8.0rc1` | |
| `axelera-onnx-graph-cleaner` | `1.8.0rc1` | |
| `axelera-functions-registry` | `1.8.0rc1` | |
| `axelera-config` | `1.8.0rc1` | |
| `axelera-types` | `1.8.0rc1` | |
| `axelera-llm` | `1.8.0rc1` | |
| `axelera-runtime` | `1.8.0rc1` | Python host runtime (`axelera.runtime`) |
| `axelera-runtime2` | `0.1.9` | |
| `axelera-rt` | `1.7.0` | runtime meta / `axdevice` CLI |
| `axelera-devkit` | `1.7.0` | devkit meta |
| `axelera-firmware` | `1.7.0` | |
| `axelera-aipu-api` | `3.3.1.post2+git.da85b00` | |
| `axelera-aipu-aingine-api` | `0.5.0.post6+git.890ed85` | |
| `axelera-package-lib` | `0.4.6` | |
| `axelera-miraculix` | `0.3.0` | |
| `axelera-transformers` | `0.2.1.dev18+gecf1522fa9` | |
| `axelera-zoo` | `0.1.0` | model zoo |
| `axelera-riscv-gnu-newlib-toolchain` | `1.0.0` | RISC-V toolchain |
| `axelera-riscv-llvm-toolchain-minimal` | `0.5.5` | |

Key third-party pins in the same venv: `torch 2.9.1`, `torchvision 0.24.1`,
`onnx 1.17.0`, `onnxruntime 1.23.2`, `onnxscript 0.7.1`.

> Note: Axelera wheels are **pyarmor-obfuscated**; source isn't readable.

---

## 2. System runtime + firmware + driver (on-device)

Needed only to run a `.axm` **on the card**. Two ways to install:

### A. Via the SDK installer (needs network to the Artifactory apt repo)

```bash
cd /path/to/internal-voyager-sdk
./install.sh --runtime      # adds the apt repo + installs axelera-runtime-<ver>
./install.sh --driver       # builds/installs the metis-dkms PCIe driver
./install.sh --status       # show what's installed
```

### B. Offline, from local `.deb`s (what we used — no network)

```bash
sudo dpkg -i axelera-device_1.8.0-rc1-1_amd64.deb \
             axelera-runtime_1.8.0-rc1-1_amd64.deb
# then source the runtime env before using the device (see below)
```

### Installed system packages / versions

| Package | Version | Provides |
| --- | --- | --- |
| `axelera-runtime-1.8.0-rc1` | `1.8.0-rc1` | `/opt/axelera/runtime-1.8.0-rc1-1` — host libs (`libaxldev`), CLI tools |
| `axelera-device-1.8.0-rc1` | `1.8.0-rc1` | `/opt/axelera/device-1.8.0-rc1-1/omega` — AIPU firmware ELF (`start_axelera_runtime.elf`) |
| `metis-dkms` | `1.6.1` | PCIe kernel driver `metis.ko` (module version `1.6.1`) |

### CLI tools (`/opt/axelera/runtime-1.8.0-rc1-1/bin`)

| Tool | Version | Use |
| --- | --- | --- |
| `axkernelcc` | `v1.8.0-rc1` | AICore kernel compiler (final `.axm` render step) |
| `axcmd`, `axmonitor`, `axdma`, `axmem`, `axtrace` | (1.8.0-rc1) | device cmd / monitor / DMA / mem / trace utilities |
| `axdevice` | (from `axelera-rt` pip pkg) | list/query devices; **provided by the venv**, not the runtime bin |

### Runtime environment (`env-1.8.0-rc1.sh`)

Source before using the device so the runtime finds the matching firmware/libs:

```bash
export AXELERA_RUNTIME_DIR=/opt/axelera/runtime-1.8.0-rc1-1
export AXELERA_DEVICE_DIR=/opt/axelera/device-1.8.0-rc1-1/omega
export AIPU_FIRMWARE_OMEGA=${AXELERA_DEVICE_DIR}/bin/start_axelera_runtime.elf
export AIPU_RUNTIME_STAGE0_OMEGA=${AXELERA_DEVICE_DIR}/bin/start_axelera_runtime_stage0.bin
export LD_LIBRARY_PATH=${AXELERA_RUNTIME_DIR}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=${AXELERA_RUNTIME_DIR}/bin${PATH:+:$PATH}
export PKG_CONFIG_PATH=${AXELERA_RUNTIME_DIR}/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}
export GST_PLUGIN_PATH=${AXELERA_RUNTIME_DIR}/lib/gstreamer-1.0${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}
```

---

## Version-consistency (important for on-device runs)

The `.axm` kernel is rendered by whatever **`axkernelcc`** is on `PATH`, and it must
match the **device firmware** version loaded on the card:

- graph compiled by `axelera-graph-compiler 1.1.0` / `axelera_tvm 1.8.0rc1` (venv)
- kernel rendered by `axkernelcc 1.8.0-rc1` (system runtime bin, on PATH)
- device firmware `start_axelera_runtime.elf` 1.8.0-rc1 (system device pkg)
- PCIe driver `metis-dkms 1.6.1`

If the card runs older firmware (e.g. `flver=1.6.0`), a 1.8.0-rendered kernel fails
to load (`Invalid kernel_info section: bad magic` / `Failed to initialise AICoreKernel`).
The firmware is (re)loaded by the runtime over PCIe DMA at connect; a matching
`metis-dkms` driver is required for that DMA path to complete.
