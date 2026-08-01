<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><font color="#4b5563"><b>EN</b></font></td>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.3.2.md"><font color="#ffffff"><b>中文</b></font></a></td>
  </tr>
</table>

## Summary

**v3.3.2** fixes **salt-and-pepper noise on the 2nd generation** of Z Image / ZIT **ConvRot NVFP4** after a **DistOrch VRAM purge**.

After purge, DistOrch rebuilds the stack. When INT8 decode wrap had stripped NVFP4 stack markers, a later “upgrade” re-wrapped the **Tensor Core product** path over the **Comfy parity** stack. Refresh then peeled only the TC layer and left **double online act rotate** on reload — visible as 2nd-gen noise while the 1st gen looked fine.

Markers are now preserved through the INT8 wrap so parity refresh no longer re-arms a second rotate.

## Fixed

### Double online act rotate after DistOrch purge

- **Symptom:** Gen 1 OK → DistOrch / General Purge VRAM → Gen 2 salt-and-pepper / broken texture on Z Image / ZIT ConvRot NVFP4.
- **Cause:** INT8 decode wrap dropped NVFP4 stack markers → TC product path nested over Comfy parity → DistOrch refresh left **two** online act rotates.
- **Fix:** Preserve NVFP4 / parity stack markers through INT8 wrap (`nodes/nvfp4/comfy_quant_nvfp4.py`, `nodes/zimage_nvfp4/nvfp4_comfy_parity.py`, `patches/comfy_quant_int8.py`) so refresh re-arms a single rotate.

## Docs

- Changelog: `changelog.md` / `zhmd/CHANGELOG.md` → Version **3.3.2**

## Operator notes

1. Update this custom node to tag **v3.3.2** (commit `c04811e` or later on `main`).
2. Restart ComfyUI completely.
3. Keep **General Purge VRAM V2** (`HSWQ` on) at workflow end when using HSWQ NVFP4 / INT8, as before.
4. Confirm 2nd (and later) gens after purge no longer show salt-and-pepper from double act rotate.

## Compatibility

| Item | Policy |
|------|--------|
| Scope | Z Image / ZIT **ConvRot NVFP4** on **HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader** |
| Quantizer | **Only** [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization) |
| SDXL ConvRot NVFP4 | Unchanged (Checkpoint Loader + `nodes/nvfp4` TC product path) |
| ComfyUI-master | Not modified by this extension |
