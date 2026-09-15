# HSWQ DisTorch2 UNet Loader — Technical Guide (Complete)

Target: `HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader (DisTorch2)`
Base commit: `299c8c951ff920d1caa33b9eca5a10810bfe0665`
Scope: `299c8c9..HEAD` (the DisTorch2 node and every change that comes with it)

---

## ① Why an in-repo implementation was necessary

### ①-1. The upstream (ComfyUI-MultiGPU) loader cannot read HSWQ checkpoints

Upstream `ComfyUI-MultiGPU` already ships DisTorch2 loaders (`UNETLoaderDisTorch2MultiGPU`, ...). Their base class, however, is ComfyUI's stock `UNETLoader`. A stock loader does not understand:

- **comfy_quant metadata** (per-layer `int8_tensorwise` / `nvfp4` formats, the `convrot` flag, `convrot_groupsize`)
- **ConvRot layers** (pre-rotated weights: inverse rotation at load, online rotation at run time)
- **HSWQ's LoRA handling** (8-bit / 4-bit requant rounds away small LoRA deltas, so the low-rank residual must be baked)
- **Krea2's `txtfusion.projector`** (the Krea2 detection marker) and its packed-K correction
- **Z Image parity residue cleanup** (a wrapper left over from a previous load contaminates Krea2)

All of this lives in this repository's loader (`HSWQFP8E4M3UNetLoader` → `load_unet_hswq_weight_dtype` / `load_unet_nvfp4_weight_dtype`). Feeding an HSWQ checkpoint to the upstream DisTorch2 loader leaves the quantization metadata uninterpreted, so it cannot load correctly.

### ①-2. Planned large ConvRot NVFP4 packs must live outside VRAM

HSWQ shrinks models with NVFP4 (4-bit), but checkpoints are still coming that do not fit on a single GPU (e.g. a 16 GB-class card). Those need part (or all) of the UNet outside VRAM, with only the required blocks resident on the GPU. That is exactly DisTorch's job.

So a node that satisfies **"HSWQ loader" + "DisTorch2 offload placement"** at the same time is required. Upstream has no such combination.

### ①-3. Conclusion

- **Port the upstream DisTorch2 backend (`distorch_2.py`)** (GPL-3.0, modified copy; the attribution is recorded in this repo's README / LICENSE)
- **Use the HSWQ loader (`HSWQFP8E4M3UNetLoader`) as the base class**, so HSWQ's quantization, LoRA and architecture detection are used unchanged
- Register it as a standalone node, `HSWQUNetLoaderDisTorch2`

That is why an in-repo implementation was necessary.

---

## ② Overview and usage

### ②-1. The node

- **Display name**: `HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader (DisTorch2)`
- **Class name**: `HSWQUNetLoaderDisTorch2`
- **Registry key**: `NODE_CLASS_MAPPINGS["HSWQUNetLoaderDisTorch2"]`
- **Base**: `HSWQFP8E4M3UNetLoader` (this repo's UNet loader) wrapped with `override_class_with_distorch_safetensor_v2()`

### ②-2. What it does

The whole UNet can live outside VRAM (virtual VRAM + a donor device), so packs larger than the GPU can still run.

### ②-3. Inputs

| Input | Meaning |
|---|---|
| `unet_name` | checkpoint under `diffusion_models` |
| `weight_dtype` | `default` / FP8 options / `int8_tensorwise` / `Z Image ConvRot NVFP4` / `Krea2 ConvRot NVFP4` |
| `hswq_bake` | **ON** = HSWQ path (HSWQ LoRA bake + legacy patcher) / **OFF** = stock ComfyUI path (DynamicVRAM) |
| `compute_device` | device used for compute |
| `virtual_vram_gb` | virtual VRAM budget (GB) |
| `donor_device` | offload target (e.g. `cpu`) |
| `expert_mode_allocations` | manual allocation string |
| `eject_models` | evict other models before loading |

### ②-4. Usage

1. Add **HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader (DisTorch2)** to the graph.
2. Pick the HSWQ checkpoint in `unet_name`.
3. Choose `weight_dtype` (`default` also auto-detects; Krea2 ConvRot INT8 is `int8_tensorwise`).
4. Set `virtual_vram_gb` to **at least the UNet size** (**14 GB or more** for Krea2 ConvRot INT8).
5. Set `donor_device` to `cpu` (offload outside VRAM).
6. Set `hswq_bake` to **OFF** (the faster side for Krea2 ConvRot INT8; offloaded weights land in the DynamicVRAM host buffer, i.e. shared VRAM).
7. Connect the `MODEL` output to the sampler / LoRA nodes.

> Note: `hswq_bake = ON` is the HSWQ path (HSWQ LoRA bake + legacy patcher), intended for the **planned large Hybrid ConvRot NVFP4** packs. On Krea2 ConvRot INT8 it is slower than `OFF`.

### ②-5. Requirements and performance

- **Verified**: **Krea2 ConvRot INT8 only** (other `weight_dtype` choices are not verified with DisTorch2)
- **16 GB-class GPU** (e.g. RTX 5060 Ti 16GB): faster than the plain UNet loader when `virtual_vram_gb` is at least the UNet size
- **System RAM**: **64 GB minimum** (offloaded weights are held on the host side)

### ②-6. Behaviour split (`hswq_bake`)

| `hswq_bake` | Path | Patcher | Patches | Result |
|---|---|---|---|---|
| **OFF** | stock ComfyUI (DynamicVRAM) | stays dynamic | no HSWQ patches at all | fast (measured 5.34-5.51 s/it on Krea2 ConvRot INT8); offload lands in shared VRAM |
| **ON** | HSWQ path | legacy (`disable_dynamic`) | HSWQ LoRA bake / parity / mp stack installed | for the planned Hybrid ConvRot NVFP4; slower than OFF on Krea2 ConvRot INT8 |

---

## ③ Files created / modified

Diff from `299c8c9` to `HEAD` (`git diff --stat 299c8c9..HEAD`):

| File | Kind | Lines |
|---|---|---|
| `distorch_2.py` | new | +791 |
| `nodes/hswq_unet_distorch2.py` | new | +160 |
| `__init__.py` | modified | +16 |
| `device_utils.py` | modified | +1 |
| `patches/comfy_quant_int8.py` | modified | +45 |
| `hswq/zimage_fp8_e4m3_unet.py` | modified | +2/-2 |
| `nodes/krea2_convrot_nvfp4/nvfp4_lora_bake.py` | modified | +2 |
| `nodes/zimage_nvfp4/nvfp4_lora_bake.py` | modified | +2 |
| `wrappers/__init__.py` | deleted | -0 |
| `png/distorch.png` | new (binary, README figure) | - |
| `README.md` / `zhmd/README.md` | modified (DisTorch2 section, Overview, License) | +29 each |
| `changelog.md` / `zhmd/CHANGELOG.md` | modified (v3.5.4 entry) | +5 each |

> The code proper is **`distorch_2.py`** and **`nodes/hswq_unet_distorch2.py`**; the rest is registration, supporting fixes and documentation.

### ③-1. Change summary

- **`distorch_2.py` (new)**: port of the upstream ComfyUI-MultiGPU DisTorch2 backend. Patches `mm.load_models_gpu` and `ModelPatcher.partially_load`, and places safetensor blocks according to `virtual_vram_gb` / `donor_device`.
- **`nodes/hswq_unet_distorch2.py` (new)**: the DisTorch2 node built on the HSWQ UNet loader. Contains the `hswq_bake` INPUT_TYPES (DisTorch2 only), the `disable_dynamic` switch driven by it, and the host-pin fix (handling `cudaHostRegister` rc 712 = already-registered correctly).
- **`__init__.py` (modified)**: registers `HSWQUNetLoaderDisTorch2` (`build_distorch2_unet_loader(HSWQFP8E4M3UNetLoader)`).
- **`device_utils.py` (modified)**: restores `torch.cuda.synchronize()` in `soft_empty_cache_multigpu()` (matches upstream).
- **`patches/comfy_quant_int8.py` (modified)**: accepts `hswq_bake`; under aimdo (DynamicVRAM) it skips the HSWQ Krea2 patches (parity peel / mp stack / LoRA bake) and stamps the model with `_hswq_bake_enabled`.
- **`hswq/zimage_fp8_e4m3_unet.py` (modified)**: keeps `hswq_bake` **off the plain loader** (DisTorch2 only); restores `attention_accel`; adds `**kwargs` to `load_unet`.
- **`nodes/krea2_convrot_nvfp4/nvfp4_lora_bake.py` / `nodes/zimage_nvfp4/nvfp4_lora_bake.py` (modified)**: add an `_hswq_bake_enabled` check to the `Dynamic.load` bake hooks (no bake on the stock path).
- **`wrappers/__init__.py` (deleted)**: empty package removed (the functionality lives in the single module `wrappers.py`).

---

## ④ Full source of the created / modified code

### ④-1 `distorch_2.py` (full, new)

DisTorch2 backend (new; ported from upstream ComfyUI-MultiGPU)

```python
"""
DisTorch Safetensor Memory Management Module
Contains all safetensor related code for distributed memory management.

Ported from ComfyUI-MultiGPU (pollockjj)
  https://github.com/pollockjj/ComfyUI-MultiGPU
Upstream license: GNU GPL-3.0 (see LICENSE in this repository).
This file is a MODIFIED COPY (GPL-3.0 section 5a notice):
  2026-09-14  ported into ComfyUI-HSWQ-Loader-and-Tools;
              mgpu_mm_log(...) log calls replaced with logger.info(...);
              logger name "MultiGPU" -> "SDXL" (matches the other ported
              MultiGPU modules in this repository).
"""

import torch
import logging
import re
from collections import defaultdict

logger = logging.getLogger("SDXL")
import comfy.model_management as mm
import comfy.model_patcher
from .device_utils import get_device_list
from .model_management_mgpu import multigpu_memory_log



def unpack_load_item(item):
    """Handle ComfyUI 0.6.0+ 5-tuple vs legacy 4-tuple"""
    if len(item) == 5:
        # (module_offload_mem, module_mem, module_name, module_object, params)
        return item[1], item[2], item[3], item[4]
    # (module_mem, module_name, module_object, params)
    return item[0], item[1], item[2], item[3]






def register_patched_safetensor_modelpatcher():
    """Register and patch the ModelPatcher for distributed safetensor loading"""
    # Patch ComfyUI's ModelPatcher
    if not hasattr(comfy.model_patcher.ModelPatcher, '_distorch_patched'):


        # PATCH load_models_gpu with correct memory calculations per model flags
        def patched_load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
            from comfy.model_management import cleanup_models_gc, get_free_memory, free_memory, current_loaded_models
            from comfy.model_management import VRAMState, vram_state, lowvram_available, MIN_WEIGHT_MEMORY_RATIO
            from comfy.model_management import minimum_inference_memory, extra_reserved_memory, is_device_cpu

            multigpu_memory_log("load_models_gpu_top_level", "start")

            cleanup_models_gc()

            inference_memory = minimum_inference_memory()
            extra_reserved_mem = extra_reserved_memory()
            memory_required_total = memory_required + extra_reserved_mem
            extra_mem = max(inference_memory, memory_required_total)
            if minimum_memory_required is None:
                minimum_memory_required = extra_mem
            else:
                minimum_memory_required = max(inference_memory, minimum_memory_required + extra_reserved_mem)

            models_temp = set()
            for m in models:
                models_temp.add(m)
                model_type = type(m).__name__

                if ("GGUF" in model_type or "ModelPatcher" in model_type) and hasattr(m, "model_patches_to") and not hasattr(m, "model_patches_models"):
                    logger.info(f"[MultiGPU DisTorch V2] {type(m).__name__} missing 'model_patches_models' attribute, using 'model_patches_to' fallback.")
                    target_device = m.load_device
                    logger.debug(f"[MultiGPU DisTorch V2] Target device: {target_device}")
                    patches = m.model_patches_to(target_device)
                    if patches:
                        logger.debug(f"[MultiGPU DisTorch V2] Found {len(patches)} mm_patch(es) for {type(m).__name__} on device {target_device}")
                        for mm_patch in patches:
                            logger.debug(f"[MultiGPU DisTorch V2] Registering mm_patch: {type(mm_patch).__name__}")
                            models_temp.add(mm_patch)
                    continue

                for mm_patch in m.model_patches_models():
                    models_temp.add(mm_patch)
                patches = m.model_patches_to(m.load_device)
                if patches:
                    for mm_patch in patches:
                        models_temp.add(mm_patch)

            models = models_temp

            models_to_load = []

            for x in models:
                loaded_model = mm.LoadedModel(x)
                try:
                    loaded_model_index = current_loaded_models.index(loaded_model)
                except ValueError:
                    loaded_model_index = None

                if loaded_model_index is not None:
                    loaded = current_loaded_models[loaded_model_index]
                    loaded.currently_used = True
                    models_to_load.append(loaded)
                else:
                    if hasattr(x, "model"):
                        logging.info(f"Requested to load {x.model.__class__.__name__}")
                    models_to_load.append(loaded_model)

            for loaded_model in models_to_load:
                to_unload = []
                for i, current_loaded_model in enumerate(current_loaded_models):
                    if loaded_model.model.is_clone(current_loaded_model.model):
                        to_unload = [i] + to_unload
                for i in to_unload:
                    model_to_unload = current_loaded_models.pop(i)
                    model_to_unload.model.detach(unpatch_all=False)
                    model_to_unload.model_finalizer.detach()

            # DisTorch Processing
            total_memory_required = {}
            eject_device = None

            for loaded_model in models_to_load:
                device = loaded_model.device
                base_memory = loaded_model.model_memory_required(device)

                inner_model = loaded_model.model.model

                if hasattr(inner_model, '_distorch_v2_meta'):
                    meta = inner_model._distorch_v2_meta
                    allocation_str = meta['full_allocation']

                    # Parse allocation string: "expert#compute_device;virtual_vram_gb;donors"
                    parts = allocation_str.split('#')
                    virtual_vram_gb = 0.0
                    has_eject = False

                    if len(parts) > 1:
                        virtual_vram_str = parts[1]
                        virtual_info = virtual_vram_str.split(';')
                        if len(virtual_info) > 1:
                            virtual_vram_gb = float(virtual_info[1])
                        if len(virtual_info) > 2 and virtual_info[2]:
                            has_eject = True

                    if has_eject:
                        eject_device = device
                        logger.info("DisTorch eject_models detected - MAX memory eviction")

                    virtual_vram_bytes = virtual_vram_gb * (1024**3)
                    adjusted_memory = max(0, base_memory - virtual_vram_bytes)
                    total_memory_required[device] = total_memory_required.get(device, 0) + adjusted_memory
                    logger.info(f"DisTorch model adjusted {(base_memory - virtual_vram_bytes)/(1024**3):.2f}GB for device {device}")
                else:
                    # Standard model: use full model size
                    total_memory_required[device] = total_memory_required.get(device, 0) + base_memory
                    logger.info(f"[LOAD_MODELS_GPU] Standard model {(base_memory)/(1024**3):.2f}GB for device {device}")

            for device, device_memory in total_memory_required.items():
                if device != torch.device("cpu"):
                    requested_mem = device_memory * 1.1 + extra_mem
                    logger.info(f"[FREE_MEMORY_CALL] Device {device}: requesting {requested_mem/(1024**3):.2f}GB = {device_memory/(1024**3):.2f}GB * 1.1 + {extra_mem/(1024**3):.2f}GB inference")


            multigpu_memory_log("free_memory", "pre")

            for device, device_memory in total_memory_required.items():
                if device != torch.device("cpu"):
                    if device == eject_device:
                        total_device_memory = mm.get_total_memory(device)
                        logger.info(f"[LOAD_MODELS_GPU] eject_models=1, is_distorch=1 → using MAX memory ({total_device_memory/(1024**3):.2f}GB) for eviction")
                        free_memory(total_device_memory,device)
                    else:
                        logger.info(f"[LOAD_MODELS_GPU] eject_models=0, using Comfy Core Computed memory ({(device_memory * 1.1 + extra_mem)/(1024**3):.2f}GB) for eviction")
                        free_memory(device_memory * 1.1 + extra_mem, device)

            multigpu_memory_log("free_memory/minimum_memory_required", "post/pre")

            for device in total_memory_required:
                if device != torch.device("cpu"):
                    free_mem = get_free_memory(device)
                    free_mem_gb = free_mem / (1024**3)
                    min_required_gb = minimum_memory_required / (1024**3)
                    logger.info(f"[MIN_MEMORY_CHECK] Device {device}: free={free_mem_gb:.2f}GB, required={min_required_gb:.2f}GB, will_evict={free_mem < minimum_memory_required}")

                    if free_mem < minimum_memory_required:
                        models_l = free_memory(minimum_memory_required, device)
                        logger.info(f"[EVICTION] Device {device}: unloaded {len(models_l)} models due to insufficient memory")
                        logging.info(f"{len(models_l)} models unloaded.")

            multigpu_memory_log("minimum_memory_required", "post")

            for loaded_model in models_to_load:
                model = loaded_model.model
                torch_dev = model.load_device
                if is_device_cpu(torch_dev):
                    vram_set_state = VRAMState.DISABLED
                else:
                    vram_set_state = vram_state
                lowvram_model_memory = 0
                if lowvram_available and vram_set_state in (VRAMState.LOW_VRAM, VRAMState.NORMAL_VRAM) and not force_full_load:
                    loaded_memory = loaded_model.model_loaded_memory()
                    current_free_mem = get_free_memory(torch_dev) + loaded_memory

                    lowvram_model_memory = max(128 * 1024 * 1024, (current_free_mem - minimum_memory_required), min(current_free_mem * MIN_WEIGHT_MEMORY_RATIO, current_free_mem - minimum_inference_memory()))
                    lowvram_model_memory = max(0.1, lowvram_model_memory - loaded_memory)

                if vram_set_state == VRAMState.NO_VRAM:
                    lowvram_model_memory = 0.1

                loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
                current_loaded_models.insert(0, loaded_model)

        # Replace the module function
        mm.load_models_gpu = patched_load_models_gpu

        original_partially_load = comfy.model_patcher.ModelPatcher.partially_load

        def new_partially_load(self, device_to, extra_memory=0, full_load=False, force_patch_weights=False, **kwargs):
            """Override to use direct model annotation for allocation"""

            mp_id = id(self)
            inner_model = self.model
            inner_model_id = id(inner_model)

            if not hasattr(inner_model, "_distorch_v2_meta"):
                logger.debug(f"[DISTORCH_SKIP] ModelPatcher=0x{mp_id:x} inner_model=0x{inner_model_id:x} type={type(inner_model).__name__} - no metadata, using standard loading")
                result = original_partially_load(self, device_to, extra_memory, force_patch_weights)
                if hasattr(self, '_distorch_block_assignments'):
                    del self._distorch_block_assignments
                return result

            allocations = inner_model._distorch_v2_meta['full_allocation']

            if not hasattr(self.model, '_distorch_high_precision_loras'):
                self.model._distorch_high_precision_loras = True

            if not hasattr(self.model, 'current_weight_patches_uuid'):
                self.model.current_weight_patches_uuid = None

            unpatch_weights = self.model.current_weight_patches_uuid is not None and (self.model.current_weight_patches_uuid != self.patches_uuid or force_patch_weights)

            if unpatch_weights:
                logger.debug("[MultiGPU DisTorch V2] Patches changed or forced. Unpatching model.")
                self.unpatch_model(self.offload_device, unpatch_weights=True)

            self.patch_model(load_weights=False)

            mem_counter = 0

            is_clip_model = getattr(self, 'is_clip', False)
            ## TODO - I do not believe this code is needed and needs to be flagged for proof it is needed
            # Check for valid cache
            allocations_match = hasattr(self, '_distorch_last_allocations') and self._distorch_last_allocations == allocations
            cache_exists = hasattr(self, '_distorch_cached_assignments')

            if cache_exists and allocations_match and not unpatch_weights and not force_patch_weights:
                device_assignments = self._distorch_cached_assignments
                logger.debug(f"[MultiGPU DisTorch V2] Reusing cached analysis for {type(inner_model).__name__}")
            else:
                device_assignments = analyze_safetensor_loading(self, allocations, is_clip=is_clip_model)  ## This should be the only required line - that is how it worked previous release so if it doesn't it is Comfy changes
                self._distorch_cached_assignments = device_assignments
                self._distorch_last_allocations = allocations

            model_original_dtype = comfy.utils.weight_dtype(self.model.state_dict())
            high_precision_loras = getattr(self.model, "_distorch_high_precision_loras", True)
            # Use standard ComfyUI load list - the device comparison fix ensures we don't crash
            loading = self._load_list()
            loading.sort(reverse=True)
            for item in loading:
                module_size, module_name, module_object, params = unpack_load_item(item)
                if not unpatch_weights and hasattr(module_object, "comfy_patched_weights") and module_object.comfy_patched_weights is True:
                    block_target_device = device_assignments['block_assignments'].get(module_name, device_to)
                    current_module_device = None
                    try:
                        if any(p.numel() > 0 for p in module_object.parameters(recurse=False)):
                            current_module_device = next(module_object.parameters(recurse=False)).device
                    except StopIteration:
                        pass

                    if current_module_device is not None and str(current_module_device) != str(block_target_device):
                        logger.debug(f"[MultiGPU DisTorch V2] Moving already patched {module_name} to {block_target_device}")
                        module_object.to(block_target_device)

                    mem_counter += module_size
                    continue

                # Step 1: Write block/tensor to compute device first
                module_object.to(device_to)

                # Step 2: Apply LoRa patches while on compute device
                weight_key = f"{module_name}.weight"
                bias_key = f"{module_name}.bias"

                if weight_key in self.patches:
                    self.patch_weight_to_device(weight_key, device_to=device_to)
                if weight_key in self.weight_wrapper_patches:
                    module_object.weight_function.extend(self.weight_wrapper_patches[weight_key])

                if bias_key in self.patches:
                    self.patch_weight_to_device(bias_key, device_to=device_to)
                if bias_key in self.weight_wrapper_patches:
                    module_object.bias_function.extend(self.weight_wrapper_patches[bias_key])

                # Step 3: FP8 casting for CPU storage (if enabled)
                block_target_device = device_assignments['block_assignments'].get(module_name, device_to)
                has_patches = weight_key in self.patches or bias_key in self.patches

                if not high_precision_loras and block_target_device == "cpu" and has_patches and model_original_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    for param_name, param in module_object.named_parameters():
                        if param.dtype.is_floating_point:
                            cast_data = comfy.float.stochastic_rounding(param.data, torch.float8_e4m3fn)
                            new_param = torch.nn.Parameter(cast_data.to(torch.float8_e4m3fn))
                            new_param.requires_grad = param.requires_grad
                            setattr(module_object, param_name, new_param)
                            logger.debug(f"[MultiGPU DisTorch V2] Cast {module_name}.{param_name} to FP8 for CPU storage")

                # Step 4: Move to ultimate destination based on DisTorch assignment
                if str(block_target_device) != str(device_to):
                    logger.debug(f"[MultiGPU DisTorch V2] Moving {module_name} from {device_to} to {block_target_device}")
                    module_object.to(block_target_device)
                    module_object.comfy_cast_weights = True

                # Mark as patched and update memory counter
                module_object.comfy_patched_weights = True
                mem_counter += module_size

            self.model.current_weight_patches_uuid = self.patches_uuid

            self.model.device = device_to

            logger.info("[MultiGPU DisTorch V2] DisTorch loading completed.")
            logger.info(f"[MultiGPU DisTorch V2] Total memory: {mem_counter / (1024 * 1024):.2f}MB")

            return 0


        comfy.model_patcher.ModelPatcher.partially_load = new_partially_load
        comfy.model_patcher.ModelPatcher._distorch_patched = True
        logger.info("[MultiGPU Core Patching] Successfully patched ModelPatcher.partially_load")

def _extract_clip_head_blocks(raw_block_list, compute_device):
    """Identify and pre-assign CLIP head blocks to compute device returning head_blocks, distributable_blocks, block_assignments, and head_memory."""
    head_keywords = ['embed', 'wte', 'wpe', 'token_embedding', 'position_embedding']
    head_blocks = []
    distributable_blocks = []
    head_memory = 0
    block_assignments = {}

    block_assignments = {}

    for item in raw_block_list:
        module_size, module_name, module_object, params = unpack_load_item(item)
        if any(kw in module_name.lower() for kw in head_keywords):
            head_blocks.append((module_size, module_name, module_object, params))
            block_assignments[module_name] = compute_device
            head_memory += module_size
        else:
            distributable_blocks.append((module_size, module_name, module_object, params))

    return head_blocks, distributable_blocks, block_assignments, head_memory

def analyze_safetensor_loading(model_patcher, allocations_string, is_clip=False):
    """
    Analyze and distribute safetensor model blocks across devices.
    Supports CLIP head preservation when is_clip=True.
    """
    DEVICE_RATIOS_DISTORCH = {}
    device_table = {}
    distorch_alloc = ""
    virtual_vram_str = ""
    if '#' in allocations_string:
        distorch_alloc, virtual_vram_str = allocations_string.split('#', 1)
    else:
        distorch_alloc = allocations_string

    compute_device = virtual_vram_str.split(';')[0] if virtual_vram_str else "cuda:0"
    logger.debug(f"[MultiGPU DisTorch V2] Compute Device: {compute_device}")

    if not distorch_alloc:
        distorch_alloc = calculate_safetensor_vvram_allocation(model_patcher, virtual_vram_str)

    elif any(c in distorch_alloc.lower() for c in ['g', 'm', 'k', 'b']):
        distorch_alloc = calculate_fraction_from_byte_expert_string(model_patcher, distorch_alloc)
    elif "%" in distorch_alloc:
        distorch_alloc = calculate_fraction_from_ratio_expert_string(model_patcher, distorch_alloc)

    all_devices = get_device_list()
    present_devices = {item.split(',')[0] for item in distorch_alloc.split(';') if ',' in item}
    for device in all_devices:
        if device not in present_devices:
            distorch_alloc += f";{device},0.0"

    eq_line = "=" * 50
    dash_line = "-" * 50
    fmt_assign = "{:<18}{:>7}{:>14}{:>10}"

    logger.info(eq_line)
    logger.info(f"[MultiGPU DisTorch V2] Final Allocation String:\n{distorch_alloc}")

    for allocation in distorch_alloc.split(';'):
        if ',' not in allocation:
            continue
        dev_name, fraction = allocation.split(',')
        fraction = float(fraction)
        total_mem_bytes = mm.get_total_memory(torch.device(dev_name))
        alloc_gb = (total_mem_bytes * fraction) / (1024**3)
        DEVICE_RATIOS_DISTORCH[dev_name] = alloc_gb
        device_table[dev_name] = {
            "fraction": fraction,
            "total_gb": total_mem_bytes / (1024**3),
            "alloc_gb": alloc_gb
        }

    logger.info(eq_line)
    logger.info("    DisTorch2 Model Device Allocations")
    logger.info(eq_line)

    fmt_rosetta = "{:<8}{:>9}{:>9}{:>11}{:>10}"
    logger.info(fmt_rosetta.format("Device", "VRAM GB", "Dev %", "Model GB", "Dist %"))
    logger.info(dash_line)

    sorted_devices = sorted(device_table.keys(), key=lambda d: (d == "cpu", d))

    total_allocated_model_bytes = sum(d["alloc_gb"] * (1024**3) for d in device_table.values())

    for dev in sorted_devices:
        total_dev_gb = device_table[dev]["total_gb"]
        alloc_fraction = device_table[dev]["fraction"]
        alloc_gb = device_table[dev]["alloc_gb"]

        dist_ratio_percent = (alloc_gb * (1024**3) / total_allocated_model_bytes) * 100 if total_allocated_model_bytes > 0 else 0

        logger.info(fmt_rosetta.format(
            dev,
            f"{total_dev_gb:.2f}",
            f"{alloc_fraction*100:.1f}%",
            f"{alloc_gb:.2f}",
            f"{dist_ratio_percent:.1f}%"
        ))

    logger.info(dash_line)

    block_summary = {}
    block_list = []
    memory_by_type = defaultdict(int)
    total_memory = 0

    raw_block_list = model_patcher._load_list()
    total_memory = sum(unpack_load_item(x)[0] for x in raw_block_list)

    MIN_BLOCK_THRESHOLD = total_memory * 0.0001
    logger.debug(f"[MultiGPU DisTorch V2] Total model memory: {total_memory} bytes")
    logger.debug(f"[MultiGPU DisTorch V2] Tiny block threshold (0.01%): {MIN_BLOCK_THRESHOLD} bytes")

    # CLIP-specific: Extract head blocks and get pre-assignments
    head_memory = 0
    block_assignments = {}
    if is_clip:
        head_blocks, distributable_raw, block_assignments, head_memory = \
            _extract_clip_head_blocks(raw_block_list, compute_device)
        logger.info(f"[MultiGPU DisTorch V2 CLIP] Preserving {len(head_blocks)} head layer(s) ({head_memory/(1024**2):.2f} MB) on compute device: {compute_device}")
    else:
        distributable_raw = raw_block_list

    # Build all_blocks list for summary (using full raw_block_list)
    all_blocks = []
    for item in raw_block_list:
        module_size, module_name, module_object, params = unpack_load_item(item)
        block_type = type(module_object).__name__
        # Populate summary dictionaries
        block_summary[block_type] = block_summary.get(block_type, 0) + 1
        memory_by_type[block_type] += module_size
        all_blocks.append((module_name, module_object, block_type, module_size))

    # Use distributable blocks for actual allocation (for CLIP, this excludes heads)
    distributable_all_blocks = []
    for item in distributable_raw:
        module_size, module_name, module_object, params = unpack_load_item(item)
        distributable_all_blocks.append((module_name, module_object, type(module_object).__name__, module_size))

    block_list = [b for b in distributable_all_blocks if (b[3] >= MIN_BLOCK_THRESHOLD and hasattr(b[1], "bias"))]
    tiny_block_list = [b for b in distributable_all_blocks if b not in block_list]

    logger.debug(f"[MultiGPU DisTorch V2] Total blocks: {len(all_blocks)}")
    logger.debug(f"[MultiGPU DisTorch V2] Distributable blocks: {len(block_list)}")
    logger.debug(f"[MultiGPU DisTorch V2] Tiny blocks (<0.01%): {len(tiny_block_list)}")

    logger.info("    DisTorch2 Model Layer Distribution")
    logger.info(dash_line)
    fmt_layer = "{:<18}{:>7}{:>14}{:>10}"
    logger.info(fmt_layer.format("Layer Type", "Layers", "Memory (MB)", "% Total"))
    logger.info(dash_line)

    for layer_type, count in block_summary.items():
        mem_mb = memory_by_type[layer_type] / (1024 * 1024)
        mem_percent = (memory_by_type[layer_type] / total_memory) * 100 if total_memory > 0 else 0
        logger.info(fmt_layer.format(layer_type[:18], str(count), f"{mem_mb:.2f}", f"{mem_percent:.1f}%"))

    logger.info(dash_line)

    # Distribute blocks sequentially from the tail of the model

    device_assignments = {device: [] for device in DEVICE_RATIOS_DISTORCH}
    # Create a memory quota for each donor device based on its calculated allocation.
    donor_devices = list(sorted_devices)
    donor_quotas = {
        dev: device_table[dev]["alloc_gb"] * (1024**3)
        for dev in donor_devices
    }

    # CLIP-specific: Adjust compute_device quota to account for locked head blocks
    if is_clip and compute_device in donor_quotas and head_memory > 0:
        donor_quotas[compute_device] = max(0, donor_quotas[compute_device] - head_memory)
        logger.debug(f"[MultiGPU DisTorch V2 CLIP] Adjusted {compute_device} quota by -{head_memory/(1024**2):.2f} MB for head preservation")

    # Iterate from the TAIL of the model, assigning blocks to donors until their quotas are filled.
    for block_name, module, block_type, block_memory in reversed(block_list):
        assigned_to_donor = False
        for donor in donor_devices:
            if donor_quotas[donor] >= block_memory:
                block_assignments[block_name] = donor
                donor_quotas[donor] -= block_memory
                assigned_to_donor = True
                break # Move to the next block

        if not assigned_to_donor:  #Note - small rounding errors and tensor-fitting on devices make a block occasionally an orphan. We treat orphans the same as tiny_block_list as they are generally small rounding errors
            block_assignments[block_name] = compute_device

    if tiny_block_list:
        for block_name, module, block_type, block_memory in tiny_block_list:
            block_assignments[block_name] = compute_device

    # Populate device_assignments from the final block_assignments
    for block_name, device in block_assignments.items():
        # Find the block in the original list to get all its info
        for b_name, b_module, b_type, b_mem in all_blocks:
            if b_name == block_name:
                device_assignments[device].append((b_name, b_module, b_type, b_mem))
                break

    logger.info("DisTorch2 Model Final Device/Layer Assignments")
    logger.info(dash_line)
    logger.info(fmt_assign.format("Device", "Layers", "Memory (MB)", "% Total"))
    logger.info(dash_line)

    if tiny_block_list:
        tiny_block_memory = sum(b[3] for b in tiny_block_list)
        tiny_mem_mb = tiny_block_memory / (1024 * 1024)
        tiny_mem_percent = (tiny_block_memory / total_memory) * 100 if total_memory > 0 else 0
        device_label = f"{compute_device} (<0.01%)"
        logger.info(fmt_assign.format(device_label, str(len(tiny_block_list)), f"{tiny_mem_mb:.2f}", f"{tiny_mem_percent:.1f}%"))
        logger.debug(f"[MultiGPU DisTorch V2] Tiny block memory breakdown: {tiny_block_memory} bytes ({tiny_mem_mb:.2f} MB), which is {tiny_mem_percent:.4f}% of total model memory.")

    total_assigned_memory = 0
    device_memories = {}

    for device, blocks in device_assignments.items():
        dist_blocks = [b for b in blocks if b[3] >= MIN_BLOCK_THRESHOLD]
        if not dist_blocks:
            continue

        device_memory = sum(b[3] for b in dist_blocks)
        device_memories[device] = device_memory
        total_assigned_memory += device_memory

    sorted_assignments = sorted(device_memories.keys(), key=lambda d: (d == "cpu", d))

    for dev in sorted_assignments:
        # Get only the distributed blocks for the count
        dist_blocks = [b for b in device_assignments[dev] if b[3] >= MIN_BLOCK_THRESHOLD]
        if not dist_blocks:
            continue

        mem_mb = device_memories[dev] / (1024 * 1024)
        mem_percent = (device_memories[dev] / total_memory) * 100 if total_memory > 0 else 0
        logger.info(fmt_assign.format(dev, str(len(dist_blocks)), f"{mem_mb:.2f}", f"{mem_percent:.1f}%"))

    logger.info(dash_line)

    return {
        "device_assignments": device_assignments,
        "block_assignments": block_assignments
    }

def parse_memory_string(mem_str):
    """Parses a memory string (e.g., '4.0g', '512M') and returns bytes."""
    mem_str = mem_str.strip().lower()
    match = re.match(r'(\d+\.?\d*)\s*([gmkb]?)', mem_str)
    if not match:
        raise ValueError(f"Invalid memory string format: {mem_str}")

    val, unit = match.groups()
    val = float(val)

    if unit == 'g':
        return val * (1024**3)
    elif unit == 'm':
        return val * (1024**2)
    elif unit == 'k':
        return val * 1024
    else: # b or no unit
        return val

def calculate_fraction_from_byte_expert_string(model_patcher, byte_str):
    """Convert byte allocation string (e.g. 'cuda:1,4gb;cpu,*') to fractional VRAM allocation string respecting device order and byte quotas."""
    raw_block_list = model_patcher._load_list()
    total_model_memory = sum(unpack_load_item(x)[0] for x in raw_block_list)
    remaining_model_bytes = total_model_memory

    # Use a list of tuples to preserve the user-defined order
    parsed_allocations = []
    wildcard_device = "cpu"  # Default wildcard device

    for allocation in byte_str.split(';'):
        if ',' not in allocation:
            continue
        dev_name, val_str = allocation.split(',', 1)
        is_wildcard = '*' in val_str

        if is_wildcard:
            wildcard_device = dev_name
            # Don't add wildcard to the priority list yet
        else:
            byte_val = parse_memory_string(val_str)
            parsed_allocations.append({'device': dev_name, 'bytes': byte_val})

    final_byte_allocations = defaultdict(int)

    # Process devices with specific byte allocations first, in order
    for alloc in parsed_allocations:
        dev = alloc['device']
        requested_bytes = alloc['bytes']

        # Determine the actual bytes to allocate to this device
        bytes_to_assign = min(requested_bytes, remaining_model_bytes)

        if bytes_to_assign > 0:
            final_byte_allocations[dev] = bytes_to_assign
            remaining_model_bytes -= bytes_to_assign
            logger.info(f"[MultiGPU DisTorch V2] Assigning {bytes_to_assign / (1024**2):.2f}MB of model to {dev} (requested {requested_bytes / (1024**2):.2f}MB).")

        if remaining_model_bytes <= 0:
            logger.info("[MultiGPU DisTorch V2] All model blocks have been allocated. Subsequent devices in the string will receive no assignment.")
            break

    # Assign any leftover model bytes to the wildcard device
    if remaining_model_bytes > 0:
        final_byte_allocations[wildcard_device] += remaining_model_bytes
        logger.info(f"[MultiGPU DisTorch V2] Assigning remaining {remaining_model_bytes / (1024**2):.2f}MB of model to wildcard device '{wildcard_device}'.")

    # Convert the final byte allocations to VRAM fractions
    allocation_parts = []
    for dev, bytes_alloc in final_byte_allocations.items():
        total_device_vram = mm.get_total_memory(torch.device(dev))
        if total_device_vram > 0:
            fraction = bytes_alloc / total_device_vram
            allocation_parts.append(f"{dev},{fraction:.4f}")

    allocations_string = ";".join(allocation_parts)

    return allocations_string

def calculate_fraction_from_ratio_expert_string(model_patcher, ratio_str):
    """Convert ratio allocation string (e.g. 'cuda:0,25%;cpu,75%') describing model split to fractional VRAM allocation string."""
    raw_block_list = model_patcher._load_list()
    total_model_memory = sum(unpack_load_item(x)[0] for x in raw_block_list)

    raw_ratios = {}
    for allocation in ratio_str.split(';'):
        if ',' not in allocation:
            continue
        dev_name, val_str = allocation.split(',', 1)
        # Assumes the value is a unitless ratio number, ignores '%' for simplicity.
        value = float(val_str.replace('%','').strip())
        raw_ratios[dev_name] = value

    total_ratio_parts = sum(raw_ratios.values())
    allocation_parts = []

    for dev, ratio_val in raw_ratios.items():
        bytes_of_model_for_device = (ratio_val / total_ratio_parts) * total_model_memory

        total_vram_of_device = mm.get_total_memory(torch.device(dev))

        if total_vram_of_device > 0:
            required_fraction = bytes_of_model_for_device / total_vram_of_device
            allocation_parts.append(f"{dev},{required_fraction:.4f}")

    ratio_values = [str(v) for v in raw_ratios.values()]
    ratio_string = ":".join(ratio_values)

    normalized_pcts = [(v / total_ratio_parts) * 100 for v in raw_ratios.values()]

    put_parts = []
    for i, dev_name in enumerate(raw_ratios.keys()):
        put_parts.append(f"{int(normalized_pcts[i])}% on {dev_name}")

    if len(put_parts) == 1:
        put_part = put_parts[0]
    elif len(put_parts) == 2:
        put_part = f"{put_parts[0]} and {put_parts[1]}"
    else:
        put_part = ", ".join(put_parts[:-1]) + f", and {put_parts[-1]}"

    logger.info(f"[MultiGPU DisTorch V2] Ratio(%) Mode - {ratio_str} -> {ratio_string} ratio, put {put_part}")

    allocations_string = ";".join(allocation_parts)

    return allocations_string

def calculate_safetensor_vvram_allocation(model_patcher, virtual_vram_str):
    """Calculate virtual VRAM allocation string for distributed safetensor loading"""
    recipient_device, vram_amount, donors = virtual_vram_str.split(';')
    virtual_vram_gb = float(vram_amount)

    eq_line = "=" * 47
    dash_line = "-" * 47
    fmt_assign = "{:<8} {:<6} {:>11} {:>9} {:>9}"

    logger.info(eq_line)
    logger.info("    DisTorch2 Model Virtual VRAM Analysis")
    logger.info(eq_line)
    logger.info(fmt_assign.format("Object", "Role", "Original(GB)", "Total(GB)", "Virt(GB)"))
    logger.info(dash_line)

    # Calculate recipient VRAM
    recipient_vram = mm.get_total_memory(torch.device(recipient_device)) / (1024**3)
    recipient_virtual = recipient_vram + virtual_vram_gb

    logger.info(fmt_assign.format(recipient_device, 'recip', f"{recipient_vram:.2f}GB",f"{recipient_virtual:.2f}GB", f"+{virtual_vram_gb:.2f}GB"))

    # Handle donor devices
    ram_donors = list(donors.split(','))
    remaining_vram_needed = virtual_vram_gb

    donor_device_info = {}
    donor_allocations = {}

    for donor in ram_donors:
        donor_vram = mm.get_total_memory(torch.device(donor)) / (1024**3)
        max_donor_capacity = donor_vram

        donation = min(remaining_vram_needed, max_donor_capacity)
        donor_virtual = donor_vram - donation
        remaining_vram_needed -= donation
        donor_allocations[donor] = donation

        donor_device_info[donor] = (donor_vram, donor_virtual)
        logger.info(fmt_assign.format(donor, 'donor', f"{donor_vram:.2f}GB",  f"{donor_virtual:.2f}GB", f"-{donation:.2f}GB"))


    logger.info(dash_line)

    # Calculate model size
    model = model_patcher.model if hasattr(model_patcher, 'model') else model_patcher
    total_memory = 0

    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            if module.weight is not None:
                total_memory += module.weight.numel() * module.weight.element_size()
            if hasattr(module, "bias") and module.bias is not None:
                total_memory += module.bias.numel() * module.bias.element_size()

    model_size_gb = total_memory / (1024**3)
    new_model_size_gb = max(0, model_size_gb - virtual_vram_gb)

    logger.info(fmt_assign.format('model', 'model', f"{model_size_gb:.2f}GB",f"{new_model_size_gb:.2f}GB", f"-{virtual_vram_gb:.2f}GB"))

    # Warning if model too large
    if model_size_gb > (recipient_vram * 0.9):
        required_offload_gb = model_size_gb - (recipient_vram * 0.9)
        logger.warning(f"\n\n[MultiGPU DisTorch V2] Model size ({model_size_gb:.2f}GB) is larger than 90% of available VRAM on: {recipient_device} ({recipient_vram * 0.9:.2f}GB).")
        logger.warning(f"[MultiGPU DisTorch V2] To prevent an OOM error, set 'virtual_vram_gb' to at least {required_offload_gb:.2f}.\n\n")

    new_on_recipient = max(0, model_size_gb - virtual_vram_gb)

    # Build allocation string
    allocation_parts = []
    recipient_percent = new_on_recipient / recipient_vram
    allocation_parts.append(f"{recipient_device},{recipient_percent:.4f}")

    for donor in ram_donors:
        donor_vram = donor_device_info[donor][0]
        donor_percent = donor_allocations[donor] / donor_vram
        allocation_parts.append(f"{donor},{donor_percent:.4f}")

    allocations_string = ";".join(allocation_parts)
    return allocations_string
```

### ④-2 `nodes/hswq_unet_distorch2.py` (full, new)

DisTorch2 node (new)

```python
"""HSWQ UNet Loader (DisTorch2).

Upstream DisTorch2 wrapper (same widgets/behaviour as ComfyUI-MultiGPU's
``UNETLoaderDisTorch2MultiGPU``) applied to the HSWQ UNet loader registered in
``NODE_CLASS_MAPPINGS``, plus two fixes on the load/placement path:

1. Placement: DisTorch assigns blocks inside the patched
   ``ModelPatcher.partially_load``. Under DynamicVRAM the loaded model would be a
   ``ModelPatcherDynamic``, whose own ``partially_load``/``load`` win, so that
   placement never runs. ``clone(disable_dynamic=True)`` (same inner model) puts
   the load back on the standard path where the placement runs.

2. Host pinning: on that path ComfyUI pins offloaded weights via
   ``comfy.model_management.pin_memory()`` -> ``cudaHostRegister``. ComfyUI's own
   guard (``tensor.is_pinned()``) cannot see registrations made outside the torch
   allocator (e.g. comfy-aimdo's host buffers), so the call returns
   ``cudaErrorHostMemoryAlreadyRegistered`` (712) for every such weight and
   ComfyUI logs ``Pin error.`` for each one. That is the already-pinned case its
   own comment describes; this patch handles the error code properly instead of
   spamming warnings. Genuine failures are still reported (with the real CUDA
   error string and the size), no message is hidden.
"""

import logging

import torch

from ..wrappers import override_class_with_distorch_safetensor_v2

logger = logging.getLogger("SDXL")

_CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
_installed = False
_reported = set()


def _real_error_string(rc):
    try:
        res = torch.cuda.cudart().cudaGetErrorString(rc)
        return res[1] if isinstance(res, tuple) else str(res)
    except Exception:
        return "rc=%s" % rc


def _install_pin_memory_fix():
    """Make pin_memory handle the 'already registered' case instead of warning per weight."""
    global _installed
    if _installed:
        return True
    try:
        import comfy.memory_management
        import comfy.model_management as mm

        original = mm.pin_memory

        def pin_memory_fixed(tensor):
            if mm.MAX_PINNED_MEMORY <= 0:
                return False
            if type(tensor).__name__ not in mm.PINNING_ALLOWED_TYPES:
                return False
            if not mm.is_device_cpu(tensor.device):
                return False
            if tensor.is_pinned():
                return False
            if not tensor.is_contiguous():
                return False
            size = tensor.nbytes
            comfy.memory_management.extra_ram_release(comfy.memory_management.RAM_CACHE_HEADROOM)
            mm.ensure_pin_registerable(size)
            ptr = tensor.data_ptr()
            if ptr == 0:
                return False
            rc = int(torch.cuda.cudart().cudaHostRegister(ptr, size, 1))
            if rc == 0:
                mm.PINNED_MEMORY[ptr] = size
                mm.TOTAL_PINNED_MEMORY += size
                return True
            if rc == _CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
                # Already host-registered by another subsystem (comfy-aimdo host buffers).
                # The memory is pinnable; do not track it (so it is never unregistered
                # here) and do not warn - ComfyUI's is_pinned() guard cannot see this.
                mm.discard_cuda_async_error()
                key = "already"
                if key not in _reported:
                    _reported.add(key)
                    logger.info(
                        "[HSWQ] host-pin: weights are already host-registered (cudaErrorHostMemoryAlreadyRegistered); "
                        "skipping re-registration (normal with comfy-aimdo host buffers)"
                    )
                return False
            mm.discard_cuda_async_error()
            key = _real_error_string(rc)
            if key not in _reported:
                _reported.add(key)
                logger.warning(
                    "[HSWQ] host-pin failed: %s (size=%d bytes, rc=%s) - pinning is optional, continuing",
                    key, size, rc,
                )
            return False

        pin_memory_fixed._hswq_orig = original
        mm.pin_memory = pin_memory_fixed
        _installed = True
        return True
    except Exception:
        logger.exception("[HSWQ] could not install the host-pin fix")
        return False


_install_pin_memory_fix()


def build_distorch2_unet_loader(base_cls):
    """Return the DisTorch2 override class for ``base_cls`` (the HSWQ UNet loader)."""
    wrapped = override_class_with_distorch_safetensor_v2(base_cls)

    class HSWQUNetLoaderDisTorch2(wrapped):
        """DisTorch2 wrapper; forces the placement-capable patcher for the load."""

        @classmethod
        def INPUT_TYPES(s):
            inputs = super().INPUT_TYPES()
            _req = inputs.setdefault("required", {})
            # The DisTorch2 node does not expose the SA2 option (attention_accel)
            _req.pop("attention_accel", None)
            _req["hswq_bake"] = (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": (
                        "ON = HSWQ path (HSWQ LoRA bake + legacy patcher; required for "
                        "HSWQ-only formats such as Hybrid ConvRot NVFP4). OFF = stock "
                        "ComfyUI path under DynamicVRAM (faster; offloaded weights land "
                        "in shared VRAM)."
                    ),
                },
            )
            return inputs

        def override(self, *args, **kwargs):
            out = super().override(*args, **kwargs)
            try:
                # UI toggle: hswq_bake ON = HSWQ path -> legacy patcher (DisTorch
                # placement + HSWQ bake). OFF = stock path -> keep the model dynamic
                # so offloaded weights live in the HostBuffer (shared VRAM).
                _hswq_path = bool(kwargs.get("hswq_bake", True))
                if _hswq_path:
                    patcher = out[0]
                    if patcher is not None and getattr(patcher, "is_dynamic", None) and patcher.is_dynamic():
                        patcher = patcher.clone(disable_dynamic=True)
                        logger.info(
                            "[HSWQ DisTorch2] legacy ModelPatcher for this load: DisTorch block "
                            "placement / GPU-addressable offload is otherwise skipped by DynamicVRAM"
                        )
                        return (patcher,) + tuple(out[1:])
            except Exception:
                logger.exception("[HSWQ DisTorch2] legacy-patcher switch failed; keeping the dynamic patcher")
            return out

    return HSWQUNetLoaderDisTorch2
```

### ④-3 Diffs of the modified files (`299c8c9`..`HEAD`)

#### `__init__.py`

```diff
diff --git a/__init__.py b/__init__.py
index 9207351..ffa2205 100644
--- a/__init__.py
+++ b/__init__.py
@@ -695,6 +695,22 @@ try:
 except Exception as e:
     logger.exception("[HSWQ NVFP4] install_nvfp4_option_dispatch: %s", e)
 
+# DisTorch2 variant of the HSWQ UNet Loader: the whole UNet can be held on CPU
+# (virtual VRAM, donor_device=cpu) when a run heavily exceeds VRAM.
+# Backend ported from ComfyUI-MultiGPU (pollockjj, GPL-3.0) - see distorch_2.py.
+try:
+    from .nodes.hswq_unet_distorch2 import build_distorch2_unet_loader
+    # DisTorch2 wrapper on the dispatched HSWQ UNet loader (Krea2 / Z Image
+    # ConvRot INT8 + NVFP4 routing and the INT8 LoRA bake stay consistent).
+    _hswq_distorch2_base = NODE_CLASS_MAPPINGS.get("HSWQFP8E4M3UNetLoader") or HSWQFP8E4M3UNetLoader
+    NODE_CLASS_MAPPINGS["HSWQUNetLoaderDisTorch2"] = build_distorch2_unet_loader(_hswq_distorch2_base)
+    logger.info(
+        "Registered HSWQ UNet Loader (DisTorch2) build=hswq-base+distorch2 base=%s",
+        getattr(_hswq_distorch2_base, "__name__", _hswq_distorch2_base),
+    )
+except Exception as e:
+    logger.exception("HSWQ UNet Loader (DisTorch2) not registered: %s", e)
+
 # SDXL LoRA Stack V3 (INT8 / NVFP4 / standard via load_lora_for_models bake path)
 try:
     from .nodes.lora.sdxl_v3 import GENERATED_NODES as _SDXL_LORA_V3_NODES
```

#### `device_utils.py`

```diff
diff --git a/device_utils.py b/device_utils.py
index bbe16d7..d05c0c0 100644
--- a/device_utils.py
+++ b/device_utils.py
@@ -175,6 +175,7 @@ def soft_empty_cache_multigpu():
                 logger.info(f"Clearing CUDA cache on {device_str} (idx={device_idx})")
                 multigpu_memory_log("general", f"pre-empty:{device_str}")
                 with torch.cuda.device(device_idx):
+                    torch.cuda.synchronize()
                     torch.cuda.empty_cache()
                     if hasattr(torch.cuda, "ipc_collect"):
                         torch.cuda.ipc_collect()
```

#### `patches/comfy_quant_int8.py`

```diff
diff --git a/patches/comfy_quant_int8.py b/patches/comfy_quant_int8.py
index 6b31182..1913e14 100644
--- a/patches/comfy_quant_int8.py
+++ b/patches/comfy_quant_int8.py
@@ -1554,6 +1554,8 @@ def _patch_model_patcher_dynamic_int8_lora_bake() -> bool:
             full_load=full_load,
             dirty=dirty,
         )
+        if not bool(getattr(self.model, "_hswq_bake_enabled", True)):
+            return result
         # INT8 LoRA bake only — never touch Nunchaku SVDQ (class is often Lumina2).
         if _model_is_nunchaku_svdq(self.model):
             return result
@@ -3181,7 +3183,7 @@ def tag_krea2_model(model) -> bool:
     return True
 
 
-def load_unet_hswq_weight_dtype(unet_name, weight_dtype, attention_accel="default"):
+def load_unet_hswq_weight_dtype(unet_name, weight_dtype, attention_accel="default", hswq_bake=True):
     import logging
     import torch
     import folder_paths
@@ -3202,16 +3204,28 @@ def load_unet_hswq_weight_dtype(unet_name, weight_dtype, attention_accel="defaul
         # Krea2 does not need or want Z Image parity - it uses stock MixedPrecision -
         # but the leftover wrapper makes forward_parity fire Hadamard rotations every
         # step, causing progressive slowdown (4s/step -> 16s -> 22s -> 26s across runs).
-        try:
-            from ..nodes.nvfp4.comfy_quant_nvfp4 import (
-                _clear_zimage_parity_contamination_for_sdxl,
+        # UI toggle: hswq_bake ON = HSWQ path (HSWQ patches).
+        # OFF = stock ComfyUI path (DynamicVRAM / shared VRAM, no HSWQ patches).
+        _aimdo_dynamic = not bool(hswq_bake)
+        if _aimdo_dynamic:
+            # DynamicVRAM (comfy-aimdo): run the stock ComfyUI path exactly like
+            # upstream ComfyUI-MultiGPU. HSWQ parity/mp-stack/LoRA-bake are skipped
+            # (they fight the mmap/vbar representation; measured 2x slowdown).
+            logging.info(
+                "[HSWQ INT8] Krea2 ConvRot INT8: DynamicVRAM (comfy-aimdo) active -> "
+                "stock path (HSWQ parity/mp-stack/LoRA-bake skipped, upstream parity)"
             )
+        else:
+            try:
+                from ..nodes.nvfp4.comfy_quant_nvfp4 import (
+                    _clear_zimage_parity_contamination_for_sdxl,
+                )
 
-            _clear_zimage_parity_contamination_for_sdxl()
-        except Exception as e:
-            logging.warning(
-                "[HSWQ INT8] clear Z Image NVFP4 contamination for Krea2 failed: %s", e
-            )
+                _clear_zimage_parity_contamination_for_sdxl()
+            except Exception as e:
+                logging.warning(
+                    "[HSWQ INT8] clear Z Image NVFP4 contamination for Krea2 failed: %s", e
+                )
         model_options = {}
         # Krea2 ConvRot INT8 needs the SAME low-rank residual LoRA bake +
         # forward as Krea2 ConvRot NVFP4 (INT8 8-bit requant rounds away small
@@ -3219,7 +3233,7 @@ def load_unet_hswq_weight_dtype(unet_name, weight_dtype, attention_accel="defaul
         # projector only - never other DiT (FLUX), SDXL, or Z Image ConvRot INT8.
         is_krea2 = checkpoint_is_krea2(unet_path)
         krea2_bake_ok = False
-        if is_krea2:
+        if is_krea2 and not _aimdo_dynamic:
             try:
                 from ..nodes.krea2_convrot_nvfp4.comfy_quant_nvfp4 import (
                     apply_comfy_quant_nvfp4_patches,
@@ -3265,6 +3279,15 @@ def load_unet_hswq_weight_dtype(unet_name, weight_dtype, attention_accel="defaul
             flush=True,
         )
         model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
+        # Stamp the HSWQ-path choice so the Dynamic.load bake hooks (built at
+        # import time) can be skipped on the stock path.
+        try:
+            _bk = bool(hswq_bake)
+            setattr(model, "_hswq_bake_enabled", _bk)
+            if getattr(model, "model", None) is not None:
+                setattr(model.model, "_hswq_bake_enabled", _bk)
+        except Exception:
+            pass
         if krea2_bake_ok:
             # Stamp for the Krea2 bake hook (mirrors the NVFP4 loader stamp; the
             # inner-model stamp survives ModelPatcher clones made by LoRA nodes).
@@ -3459,6 +3482,7 @@ def install_int8_option_dispatch(node_class_mappings) -> bool:
                 return load_unet_hswq_weight_dtype(
                     unet_name, weight_dtype,
                     attention_accel=kwargs.get("attention_accel", "default"),
+                    hswq_bake=kwargs.get("hswq_bake", True),
                 )
             # default: auto-detect INT8 checkpoints only; otherwise original FP path.
             import folder_paths
@@ -3468,6 +3492,7 @@ def install_int8_option_dispatch(node_class_mappings) -> bool:
                 return load_unet_hswq_weight_dtype(
                     unet_name, weight_dtype,
                     attention_accel=kwargs.get("attention_accel", "default"),
+                    hswq_bake=kwargs.get("hswq_bake", True),
                 )
             return _orig_load_unet(self, unet_name, weight_dtype, **kwargs)
 
```

#### `hswq/zimage_fp8_e4m3_unet.py`

```diff
diff --git a/hswq/zimage_fp8_e4m3_unet.py b/hswq/zimage_fp8_e4m3_unet.py
index 8b85f5c..8029250 100644
--- a/hswq/zimage_fp8_e4m3_unet.py
+++ b/hswq/zimage_fp8_e4m3_unet.py
@@ -974,7 +974,7 @@ class HSWQFP8E4M3UNetLoader:
     CATEGORY = "advanced/loaders"
     TITLE = "HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader"
 
-    def load_unet(self, unet_name, weight_dtype, attention_accel="default"):
+    def load_unet(self, unet_name, weight_dtype, attention_accel="default", **kwargs):
         model_options = {}
         if weight_dtype == "fp8_e4m3fn":
             model_options["dtype"] = torch.float8_e4m3fn
```

#### `nodes/krea2_convrot_nvfp4/nvfp4_lora_bake.py`

```diff
diff --git a/nodes/krea2_convrot_nvfp4/nvfp4_lora_bake.py b/nodes/krea2_convrot_nvfp4/nvfp4_lora_bake.py
index 26a57d2..81efba9 100644
--- a/nodes/krea2_convrot_nvfp4/nvfp4_lora_bake.py
+++ b/nodes/krea2_convrot_nvfp4/nvfp4_lora_bake.py
@@ -827,6 +827,8 @@ def install_krea2_nvfp4_lora_bake(force: bool = False) -> bool:
             full_load=full_load,
             dirty=dirty,
         )
+        if not bool(getattr(self.model, "_hswq_bake_enabled", True)):
+            return result
         run_krea2_nvfp4_lora_bake_on_patcher(
             self, device_to=device_to, reason="Dynamic.load"
         )
```

#### `nodes/zimage_nvfp4/nvfp4_lora_bake.py`

```diff
diff --git a/nodes/zimage_nvfp4/nvfp4_lora_bake.py b/nodes/zimage_nvfp4/nvfp4_lora_bake.py
index 28f1353..305148f 100644
--- a/nodes/zimage_nvfp4/nvfp4_lora_bake.py
+++ b/nodes/zimage_nvfp4/nvfp4_lora_bake.py
@@ -688,6 +688,8 @@ def install_zimage_nvfp4_lora_bake(force: bool = False) -> bool:
             full_load=full_load,
             dirty=dirty,
         )
+        if not bool(getattr(self.model, "_hswq_bake_enabled", True)):
+            return result
         run_zimage_nvfp4_lora_bake_on_patcher(
             self, device_to=device_to, reason="Dynamic.load"
         )
```

#### `wrappers/__init__.py` (deleted)

An empty package (`__init__.py` only) was removed. The functionality lives in the single module `wrappers.py` at the repository root, so `from ..wrappers import ...` resolves there. The deleted content was 0 lines (an empty file).


---

## ⑤ What it all means

### ⑤-1. `distorch_2.py` (new; port)

A straight port of the upstream `ComfyUI-MultiGPU` DisTorch2 backend. It is a GPL-3.0 modified copy; the only changes made during the port are:

- `logger` name `"MultiGPU"` -> `"SDXL"` (matches the other ported modules in this repo)
- `logger.mgpu_mm_log(...)` -> `logger.info(...)` (the upstream logger extension becomes normal logging)
- a port header at the top (origin, license, changes, date)

**The placement logic itself (`patched_load_models_gpu` / `new_partially_load` / `calculate_safetensor_vvram_allocation`) is identical to upstream.** What it means:

- `mm.load_models_gpu` is replaced by `patched_load_models_gpu`, which computes a DisTorch model's memory requirement as `max(0, base_memory - virtual_vram_bytes)`. When `virtual_vram_gb` is at least the UNet size the requirement becomes 0, no eviction happens and the model is left to DynamicVRAM.
- `ModelPatcher.partially_load` is replaced by `new_partially_load`, which performs block placement on the legacy path (`module.to(device)` according to `analyze_safetensor_loading`). Crucially this only runs on the legacy path (see ⑤-2).

### ⑤-2. `nodes/hswq_unet_distorch2.py` (new; the node)

This file is the core of the change. Four points.

**(a) `build_distorch2_unet_loader(base_cls)`**: wraps this repo's UNet loader with `override_class_with_distorch_safetensor_v2()` to produce `HSWQUNetLoaderDisTorch2`. This keeps HSWQ's quantization / LoRA / architecture detection intact while adding DisTorch's allocation inputs (`compute_device`, `virtual_vram_gb`, `donor_device`, `expert_mode_allocations`, `eject_models`). The upstream loader is not used as the base, for the reasons in 1-1.

**(b) `INPUT_TYPES`**: adds `hswq_bake` to the DisTorch2 node only, and explicitly removes `attention_accel`. SA2 (`attention_accel`) is removed because the DisTorch2 operation (DynamicVRAM / stock path) does not need that entry point exposed here. Because the base `INPUT_TYPES` is inherited, the removal is done with `_req.pop("attention_accel", None)`.

**(c) `disable_dynamic` switching in `override()`**: only when `hswq_bake` is ON does it apply `clone(disable_dynamic=True)` to switch to the legacy patcher; when OFF the model stays dynamic. This is based on measurements:

- legacy + HSWQ patches ... 5.64 s/it
- dynamic + HSWQ patches ... 11.28 s/it (progressive slowdown: HSWQ's LoRA bake wraps `ModelPatcherDynamic.load` and fights the mmap/vbar weight representation, and `forward_parity` rebuilds its Hadamard every step)
- dynamic without HSWQ patches (stock path) ... 5.34-5.51 s/it (equal to upstream, and the offload lands in the HostBuffer, i.e. shared VRAM)

So the correct pairing is "use the HSWQ patches -> legacy" and "do not use them -> dynamic", and `hswq_bake` is the UI switch that selects it.

**(d) host-pin fix**: replaces `comfy.model_management.pin_memory()`. When `cudaHostRegister` returns **712 (`cudaErrorHostMemoryAlreadyRegistered`)** the memory is already registered (comfy-aimdo host buffers), so it is skipped silently with a single INFO. Any other failure is **not hidden**: it emits one WARNING with the real CUDA error string, the size and the rc (the log line `host-pin failed: rc=2 (size=13107200 bytes, rc=2)` is that path). ComfyUI's own `tensor.is_pinned()` guard cannot see registrations made outside the torch allocator, which is why 712 must be handled here.

### ⑤-3. `__init__.py` (modified)

Adds `NODE_CLASS_MAPPINGS["HSWQUNetLoaderDisTorch2"] = build_distorch2_unet_loader(HSWQFP8E4M3UNetLoader)` and logs the registration (`Registered HSWQ UNet Loader (DisTorch2) build=hswq-base+distorch2 base=...`). It registers the DisTorch2 node whose base is the HSWQ loader.

### ⑤-4. `device_utils.py` (modified)

Restores `torch.cuda.synchronize()` at the top of each device's handling in `soft_empty_cache_multigpu()`, matching upstream so that the cache is emptied only after prior work has completed (calling `empty_cache()` while kernels are still running can free memory they reference).

### ⑤-5. `patches/comfy_quant_int8.py` (modified)

- `load_unet_hswq_weight_dtype(...)` gains `hswq_bake=True`, and the dispatch (`install_int8_option_dispatch`'s `load_unet`) passes `kwargs.get("hswq_bake", True)`.
- With `hswq_bake` OFF (`_aimdo_dynamic = not bool(hswq_bake)` is True) the Krea2 HSWQ patches (parity peel / mp stack / LoRA bake) are **all skipped**; with ON they are applied as before.
- After loading, `model._hswq_bake_enabled` is stamped (on the inner model too), which the bake hooks use to decide "no bake on the stock path".

The meaning: whether to install the HSWQ patches is selectable through the UI (`hswq_bake`), and the stock path installs none. The measurements in 5-2(c) are the basis.

### ⑤-6. `hswq/zimage_fp8_e4m3_unet.py` (modified)

- `attention_accel` (`default` / `sa2`) is **restored** (SA2 selectable on the plain loader again).
- `load_unet` gains `**kwargs` (the dispatch passes `hswq_bake` etc. through; without it the fp8 path raises `TypeError`).
- `hswq_bake` is **not** added to the plain loader's `INPUT_TYPES` (DisTorch2 only). The switch only has meaning for the DisTorch2 path split.

### ⑤-7. `nodes/krea2_convrot_nvfp4/nvfp4_lora_bake.py` / `nodes/zimage_nvfp4/nvfp4_lora_bake.py` (modified)

Both add, right after `true_orig`/`prev_load` inside the `Dynamic.load` wrapper:
`if not bool(getattr(self.model, "_hswq_bake_enabled", True)): return result`
So a model with `hswq_bake` OFF (stock path) does not run the post-load LoRA bake. The default is `True`, so models without the stamp (e.g. loaded through the plain loader) bake as before.

### ⑤-8. `wrappers/__init__.py` (deleted)

The empty package was removed so that `wrappers` is a single module (`wrappers.py`). The `from ..wrappers import override_class_with_distorch_safetensor_v2` in `nodes/hswq_unet_distorch2.py` resolves to `wrappers.py`.

### ⑤-9. Overall meaning

The essence is one sentence:

> **Keep the HSWQ loader as the base, make DisTorch2 offload usable on top of it, and expose the path choice (HSWQ patches on/off) explicitly in the UI.**

To achieve it we ported the upstream placement logic (`distorch_2.py`), identified by measurement the conflict between HSWQ's patches (parity / mp stack / LoRA bake) and DynamicVRAM (the 11.28 s/it progressive slowdown), implemented a mechanism that installs those patches only on the selected path (`hswq_bake` + the `_hswq_bake_enabled` stamp + the `pop` / `disable_dynamic` switch), and thereby obtained **5.34-5.51 s/it (equal to upstream) with the offload in shared VRAM** on the stock path.

`hswq_bake = ON` is kept as the path for the planned large Hybrid ConvRot NVFP4 (a format upstream cannot read at all). In that case the HSWQ patches are mandatory, and functionality takes priority over speed.
