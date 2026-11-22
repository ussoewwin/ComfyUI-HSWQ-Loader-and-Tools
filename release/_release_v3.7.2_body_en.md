<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.7.2.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# ComfyUI-HSWQ-Loader-and-Tools v3.7.2

## Overview

**A new loader node for IP-Adapter models, from any folder, with ConvRot INT8 support.** ComfyUI IPAdapter Plus's `IPAdapter Model Loader` loads only from the registered `models/ipadapter` folder and has no ConvRot INT8 path. `HSWQ Load IPAdapter Model` is a drop-in replacement that loads from the folder the user points at, and keeps the INT8 codes INT8 instead of dequantising them.

Float packs (no `comfy_quant`) load byte-identically to the stock node - same key-layout auto-detection (`image_proj` / `ip_adapter`), same `plusv2` / `unnorm` filename flags, same output. ConvRot INT8 packs keep their INT8 codes: each declared layer is installed as a `ConvRotInt8Linear` holding the codes in an `int8_weight` buffer and the per-output-channel scale in a float32 `weight_scale` buffer, and its forward runs `comfy_kitchen.int8_linear(..., convrot=True)` so the Hadamard activation rotation happens inside the INT8 GEMM.

The folder given in `ipadapter_dir` is the only place the model is loaded from. An empty folder is rejected and there is no fallback to another folder.

---

## Features

- **Drop-in replacement**: inputs `ipadapter_file` + `ipadapter_dir`, output `IPADAPTER` (standard, consumable by IPAdapter Plus' `IPAdapter` / `IPAdapter Advanced` / `IPAdapter Unified Loader`).
- **Float packs**: byte-identical to stock - same key layout auto-detection, same filename flags, same output.
- **ConvRot INT8 packs**: INT8 codes stay INT8 (not dequantised). Only packs that declare `comfy_quant` with `format == int8_tensorwise` and `convrot == true` are accepted; any other declared format is rejected explicitly rather than silently running on codes-as-float.
- **Arbitrary folder**: `ipadapter_dir` is required. The model is loaded from that folder only; an empty folder is rejected, with no fallback to another folder.
- **Runtime patch**: the IP-Adapter extension's `To_KV` and `IPAdapter` classes are replaced at runtime (no file of ComfyUI or of the extension is modified) so the graph evaluates the INT8 payload. Float packs take the stock path unchanged - the patched classes behave bit-identically to the originals when no ConvRot INT8 sidecar is present.

---

## Usage

1. Add the `HSWQ Load IPAdapter Model` node (category `loaders`).
2. Set `ipadapter_dir` to the folder containing the IP-Adapter model (e.g. `D:\...\models\ControlNet`).
3. Pick the file from the `ipadapter_file` dropdown (it follows `ipadapter_dir`).
4. Connect the `IPADAPTER` output to IPAdapter Plus' `IPAdapter` / `IPAdapter Advanced` / `IPAdapter Unified Loader`.

---

## Modified Files

| File | Type | Responsibility |
|---|---|---|
| `nodes/hswq_load_ipadapter_model.py` | Added | Node class `HSWQLoadIPAdapterModel` (category `loaders`, TITLE "HSWQ Load IPAdapter Model") |
| `hswq/ipadapter_paths.py` | Added | Directory listing + in-memory registration for the `ipadapter` category |
| `hswq/ipadapter_convrot_int8.py` | Added | `ConvRotInt8Linear`, `split_convrot_state_dict`, `install_convrot_linears`, `patch_ipadapter_plus` |
| `patches/ipadapter_convrot_int8.py` | Added | Runtime patch installer for the IP-Adapter extension (`To_KV` / `IPAdapter`) |
| `js/hswq_ipadapter_model_dir.js` | Added | Frontend combo follow-up for `ipadapter_dir` |
| `__init__.py` | Modified | Register `HSWQLoadIPAdapterModel` and the listing route |
| `README.md` | Modified | New "HSWQ Load IPAdapter Model" section (with screenshot) |
| `png/ipadapter.png` | Added | Node screenshot |
| `changelog.md` | Modified | New Version 3.7.2 entry |
| `zhmd/CHANGELOG.md` | Modified | New Version 3.7.2 entry (Chinese) |

---

## Notes

- ComfyUI core (`comfy/`, `execution.py`) is not modified. The runtime patch touches the IP-Adapter extension's classes in memory only.
- Float packs take the stock path unchanged.
- ConvRot INT8 packs must declare `comfy_quant` with `format == int8_tensorwise` and `convrot == true`; other formats are rejected explicitly rather than silently running on codes-as-float.
- The folder given in `ipadapter_dir` is the only place the model is loaded from.
