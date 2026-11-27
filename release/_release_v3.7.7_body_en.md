<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.7.7.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

## Overview

**DisTorch2 loads again on ComfyUI v0.46.0.** ComfyUI v0.46.0 removed the `lowvram_available` module global from `comfy.model_management` (the directml removal, upstream commit `c5751c81ab`), and the DisTorch2 load path imported that name directly - so every DisTorch2 load raised `ImportError: cannot import name 'lowvram_available' from 'comfy.model_management'` before any model was loaded.

The fix is a small compatibility helper in `distorch/distorch_2.py`: the patched `load_models_gpu` reads the flag through `_hswq_lowvram_available()` instead of importing it. When the global is present (ComfyUI <= 0.45.x) the helper returns its value, so the old behaviour is kept unchanged; when it is absent (v0.46.0+) the helper returns `True`, which is exactly the semantics of v0.46.0's own rewritten condition `(vram_set_state == LOW_VRAM or NORMAL_VRAM) and not force_full_load` - upstream removed the variable together with the only case that ever set it to `False` (directml), so the low-vram path is now always eligible.

---

## Features

- **Works on both ComfyUI lines**: the same file loads on ComfyUI <= 0.45.x (the global is read as before) and on v0.46.0+ (the new always-eligible semantics), with no version sniffing.
- **Minimal change**: only `distorch/distorch_2.py` is touched - one module-level helper, the import line, and the single `if` that referenced the global (+13 / -2 lines). No other DisTorch2 memory logic, arming order, or placement behaviour is modified.
- **No new dependencies**: the helper reads `comfy.model_management` lazily inside the call, so import order and startup behaviour are unchanged.

---

## Compatibility

- **Scope**: this fixes the DisTorch2 loader path (`HSWQ UNet Loader (DisTorch2)` and the DisTorch2 memory management module). All other HSWQ paths - INT8 / NVFP4 / W4A8 / FP8 / FP16 loaders, the prompt enhancer, and every patch - are untouched.
- **Measured**: verified against the running ComfyUI v0.46.0 (`comfy.model_management` has no `lowvram_available` attribute, helper returns `True`), and the module compiles and imports cleanly on both lines.

---

## Links

- **Changelog**: see [`changelog.md`](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/changelog.md) (v3.7.7 entry).
- **Chinese release notes**: [`zhmd/v3.7.7.md`](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.7.7.md).
- **Previous release**: [v3.7.6](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.7.6).
