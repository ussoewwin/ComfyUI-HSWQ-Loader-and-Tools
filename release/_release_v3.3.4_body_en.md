<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><font color="#4b5563"><b>EN</b></font></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.3.4.md"><font color="#ffffff"><b>中文</b></font></a></td>
  </tr>
</table>

## Summary

**v3.3.4** fixes **2nd+ generation quality decay** on Z Image / ZIT **ConvRot NVFP4** (and INT8 protect online act rotate) after a **Distorch** VRAM purge, when module-local Hadamard `H` was reused under a weaker check than the global Hadamard cache.

Global `build_hadamard` already rejects poisoned tensors with `_tensor_storage_ok`. Parity forward kept `_hswq_nvfp4_parity_H` alive using mainly device/dtype plus `nbytes==0`. Empty/garbage shells that still looked valid on device/dtype could keep rotating activations across later gens.

Parity forward now gates module-local `H` with the same `_tensor_storage_ok` before reuse.

## Fixed

### Module-local Hadamard reuse after Distorch purge

- **Symptom:** Gen 1 OK → Distorch / General Purge VRAM → gen 2+ quality decay / noise on ConvRot NVFP4 (and INT8 protect layers that use the same parity rotate path).
- **Cause:** Module attr `_hswq_nvfp4_parity_H` reused without `_tensor_storage_ok`; global cache was already hardened.
- **Fix:** `nodes/zimage_nvfp4/nvfp4_comfy_parity.py` — rebuild when `not _tensor_storage_ok(h)` (same helper as `nodes/nvfp4/nvfp4_hadamard.py`).

## Docs

- Changelog: `changelog.md` / `zhmd/CHANGELOG.md` → Version **3.3.4**

## Operator notes

1. Update this custom node to tag **v3.3.4**.
2. Restart ComfyUI completely.
3. Keep **General Purge VRAM V2** (`HSWQ` on) at workflow end when using HSWQ NVFP4 / INT8, as before.
4. Confirm 2nd (and later) gens after purge no longer decay from poisoned module-local Hadamard reuse.

## Compatibility

| Item | Policy |
|------|--------|
| Scope | Z Image / ZIT **ConvRot NVFP4** / INT8 protect online rotate on **HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader** |
| Quantizer | **Only** [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization) |
| SDXL ConvRot NVFP4 | Unchanged (Checkpoint Loader + `nodes/nvfp4` TC product path) |
| ComfyUI-master | Not modified by this extension |
