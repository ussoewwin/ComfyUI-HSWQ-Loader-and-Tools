<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.7.8.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

## Overview

**The pack name now names all three ConvRot tiers.** The Registry display name, the package description, and the README titles (English and 中文) now read `ComfyUI-HSWQ-ConvRot-INT8/ConvRot-W4A8/ConvRot-NVFP4-Loader-and-Tools` and mention the **Flux1 Hybrid ConvRot W4A8** loader alongside the existing ConvRot INT8 and ConvRot NVFP4 lines, matching the formats the loader has supported since v3.7.6 and v3.7.7.

This release is a **rename only**. Four documentation/metadata commits are included (`2cade57`, `faeb5f8`, `64a1aab`, `074fa30`): the `pyproject.toml` `DisplayName` and `description`, the two README H1 titles, and the bilingual changelog entry. Node classes, `weight_dtype` options, file names, loading behaviour, and quantization logic are unchanged, and no code was touched.

---

## Features

- **Registry display name**: `ComfyUI-HSWQ-ConvRot-INT8/ConvRot-W4A8/ConvRot-NVFP4-Loader-and-Tools` — now names the INT8, W4A8, and NVFP4 ConvRot tiers together.
- **Package description**: now mentions the **Flux1 Hybrid ConvRot W4A8 Loader** in addition to the SDXL/Z Image ConvRot INT8/Hybrid ConvRot NVFP4 loader, the KREA2 ConvRot INT8 loader, and the various tools.
- **README titles (EN + 中文)**: the same new name, applied to both language sides in the same commit.
- **Bilingual changelog**: v3.7.8 entries added to `changelog.md` and `zhmd/CHANGELOG.md` with the same structure as previous versions.

---

## Compatibility

- **Naming only**: upgrading from v3.7.7 changes no behaviour. Workflows, node names, `weight_dtype` dropdown values, and model files all stay as they are.
- **Registry note**: the display name and description take effect on the Registry when this version is published there; publishing is a separate step from this release.

---

## Links

- **Changelog**: see [`changelog.md`](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/changelog.md) (v3.7.8 entry).
- **Chinese release notes**: [`zhmd/v3.7.8.md`](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/blob/main/zhmd/v3.7.8.md).
- **Previous release**: [v3.7.7](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.7.7).
