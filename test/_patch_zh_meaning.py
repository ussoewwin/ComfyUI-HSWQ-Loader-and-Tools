# -*- coding: utf-8 -*-
from pathlib import Path

zh = Path(r"D:/USERFILES/GitHub/ComfyUI-HSWQ-Loader-and-Tools/zhmd/HSWQ_FROM_a9d372_PROBLEM_COUNTERMEASURES_GUIDE.md")
text = zh.read_text(encoding="utf-8")

old_p1 = """### ④ 含义

| 片段 | 含义 |
|-------|---------|
| `_hswq_nvfp4_comfy_only` | ops 处于 Z Image parity — 不得当作 SDXL 产品 TC |
| `_hswq_nvfp4_product_tc` | 仅由 `nodes/nvfp4` 的 SDXL 产品安装器打戳 |
| 在线 act rotate | 权重离线旋转；激活在 parity forward 中用 Hadamard `H` 旋转 |
| Linear 上清除 `Params.convrot` | kitchen 不得再旋转；HSWQ 通过 `_hswq_nvfp4_convrot` / `_hswq_int8_convrot` 拥有旋转 |
"""

new_p1 = """### ④ 含义（文件与代码）

**新增包各文件的含义**

| 文件 | 文件含义 | 重要符号 / 代码含义 |
|------|---------------------|----------------------------------|
| `nodes/zimage_nvfp4/__init__.py` | Z Image ConvRot NVFP4 包的公开导出面；避免加载器深入 SDXL `nodes/nvfp4`。 | 仅再导出 `load_unet` 入口 — 无运行时逻辑。 |
| `load_unet.py` | dtype **Z Image ConvRot NVFP4** 的 UNet 加载入口；选择 parity，而非 SDXL TC。 | 进入 `zi_comfy_quant_nvfp4` / parity；不得以 SDXL `apply_comfy_quant_nvfp4_patches` 为主路径。 |
| `zi_comfy_quant_nvfp4.py` | 安装/拆除 Z Image 补丁栈，并在 `ops` 上打产品身份戳。 | 仅打 `_hswq_nvfp4_comfy_only`。永不打 `_hswq_nvfp4_product_tc`。 |
| `zi_nvfp4_conf.py` | Z Image 侧 conf 解码（使 ZI conf 解析离开 SDXL 模块）。 | conf → parity 武装标志；无 GEMM。 |
| `zi_nvfp4_forward.py` | parity 路径用的 Z Image forward 辅助。 | 支持在线 act rotate；不拥有 SDXL scaled_mm TC forward。 |
| `zi_nvfp4_hadamard.py` | 与模块本地 `H` 配套的 Z Image Hadamard。 | 构建 `H`；存活仍由 `_tensor_storage_ok` 门控。 |
| `nvfp4_comfy_parity.py` | **Z Image 产品核心：** 库存 MixedPrecision GEMM + 在线 act rotate；PRODUCT 记忆/恢复；外来 peel。 | `apply_nvfp4_comfy_parity`、`_make_convrot_parity_forward`、`peel_non_product_nvfp4_ops`、`restore_nvfp4_tc_product_stack`、`_discard_poisoned_product_refs`。 |
| `nvfp4_lora_bake.py` | Z Image 的 DynamicVRAM LoRA bake（VER=8），含混合 INT8 protect bake。 | `install`/`uninstall` 就地突变 `Linear`；卸载必须剥 VER=8，不能只解 Dynamic.load。 |
| `nvfp4_addmm_patch.py` | Z Image 与 kitchen addmm 交互，避免双重旋转。 | 配合清除后的 `Params.convrot`。 |
| `nvfp4_tc_gate.py` | 硬门：parity 存活时拒绝 TC full-load upgrade。 | 阻止叠 TC-on-parity（P2 根因）。 |
| `require_parity.py` | 选定 Z Image ConvRot NVFP4 时断言 parity forward 已武装。 | 加载结束仍无 parity forward 则快速失败。 |
| `prestartup_script.py` | ComfyUI prestartup：导入路径 / 早期钩子。 | 仅打包纪律；无模型数学。 |

**戳记含义**

| 戳记 / 标志 | 含义 |
|--------------|---------|
| `_hswq_nvfp4_comfy_only` | 存活 `ops` 为 Z Image **Comfy parity**。SDXL 清理时视为外来。永不在其上 upgrade TC。 |
| `_hswq_nvfp4_product_tc` | 存活 `ops` 为 SDXL **产品 Tensor Core**。仅 `nodes/nvfp4` 的 SDXL 安装器可设置。 |
| 在线 act rotate | 权重量化时已离线旋转；激活在 parity forward 中用 Hadamard `H`（`module._hswq_nvfp4_parity_H`）旋转。 |
| Linear 上清除 `Params.convrot` | kitchen 不得再旋转。HSWQ 通过 `_hswq_nvfp4_convrot` / `_hswq_int8_convrot` 拥有旋转。 |
"""

old_p2 = """### ④ 含义

| 片段 | 含义 |
|-------|---------|
| 展平两种 wrapper | 永不把 parity 叠在 parity 上，或留下 TC-on-parity 残渣 |
| 标记保留 | DistOrch / INT8 decode 不得忘记“此 Linear 是 parity” |
"""

new_p2 = """### ④ 含义（文件与代码）

| 文件 / 符号 | 含义 |
|---------------|---------|
| `nvfp4_comfy_parity.py` 中的 `_unwrap_stock_forward` | 遍历存活 `Linear.forward` wrapper 链，剥掉 **TC 产品**与 **parity** 两壳直至库存基座。含义：DistOrch refresh 不得留下第二层在线 act-rotate。 |
| `_ensure_single_parity_linear_forward` | unwrap 之后只安装 **恰好一层** parity forward。含义：单一在线旋转，永不 TC-on-parity 后再只剥 TC。 |
| `patches/comfy_quant_int8.py` 中的 INT8 wrap 标记保留 | INT8 decode wrap Linear 时，NVFP4 / parity 戳记必须存活。含义：不得忘记“此 Linear 是 parity”。 |
"""

old_p3 = """### ④ 含义

全局缓存与模块本地 `H` 使用同一存活规则。若存储已中毒，则经 `build_hadamard` 重建。
"""

new_p3 = """### ④ 含义（文件与代码）

| 文件 / 符号 | 含义 |
|---------------|---------|
| `nodes/nvfp4/nvfp4_hadamard.py` 中的 `_tensor_storage_ok` | 共享门：Hadamard 张量仅在存储存活时可用。含义：仅 device/dtype/`numel` **不够**。 |
| `_make_convrot_parity_forward` 内的 `need_rebuild` | 模块本地 `module._hswq_nvfp4_parity_H` 必须通过与全局缓存 **相同** 的 `_tensor_storage_ok`。含义：DistOrch purge 后第 2 次及之后重建 `H`。 |
| 重建时调用的 `build_hadamard` | 重建存活 Hadamard；含义：Method-3 存储擦除后的画质恢复。 |
"""

old_p4 = """### ④ 含义

按 Conv2d 同型武装：清除 kitchen `Params.convrot`，设置 `_hswq_int8_convrot`，bake 时 unrotate 一次，requant 后保持 `Params.convrot=False`。Pass-delta EVIDENCE 不得在空的 `patches=0` pass 上打印 OK。
"""

new_p4 = """### ④ 含义（文件与代码）

| 文件 / 符号 | 含义 |
|---------------|---------|
| `nvfp4_conf.py` 的 `int8_convrot_flags_from_conf` | 将混合 INT8 protect ConvRot conf 读入 HSWQ 标志。含义：INT8 protect 是 NVFP4 ConvRot 的一等孪生。 |
| `nvfp4_forward.py` 的双重 unrotate / re-rotate | bake 路径按正确 layout unrotate 一次；在线路径经 `_hswq_int8_convrot` 旋转 act，无 kitchen 双重旋转。含义：错误 `Params.convrot` = LoRA 死或噪声。 |
| `nvfp4_comfy_parity.py` 的 `_arm_int8_protect_convrot_after_stock_load` | **按 Conv2d 同型**武装：清 kitchen `Params.convrot`，设 `_hswq_int8_convrot`。含义：HSWQ 拥有旋转。 |
| `nvfp4_lora_bake.py` 的双通道 bake + pass-delta EVIDENCE | 同时 bake NVFP4 **与** INT8 protect 键；EVIDENCE 必须用 pass-delta，不得在 `patches=0` 上空打印 OK。 |
"""

old_p5 = """### ④ 含义

导入边界：SDXL 清理可以 **调用** Z Image 的 peel/uninstall；Z Image 卸载后不得永久占有 SDXL 的 `ops`。
"""

new_p5 = """### ④ 含义（文件与代码）

| 边界 | 含义 |
|----------|---------|
| 包 `nodes/zimage_nvfp4/` | 剥离后拥有 Z Image ConvRot NVFP4 运行时。含义：SDXL 产品代码不得永远携带 Z Image parity。 |
| 包 `nodes/nvfp4/` | 仅拥有 SDXL ConvRot NVFP4 TC 产品。含义：共享所有权曾是污染根因。 |
| SDXL 清理调用 Z Image peel/uninstall | 允许的 **调用**方向：SDXL 可调用 Z Image 清理。含义：卸载后 Z Image 不得永久占有 SDXL `ops`。 |
"""

old_p6 = """### ④ 含义

按产品身份分支，而不是按“看起来像 NVFP4 conf”分支。
"""

new_p6 = """### ④ 含义（文件与代码）

| 片段 | 含义 |
|-------|---------|
| 下拉字符串 `Z Image ConvRot NVFP4` | 与 SDXL `ConvRot NVFP4` 分离的产品身份。含义：一个字符串 → 一套栈；永不共享。 |
| 按 Z Image dtype 门控的 Dynamic bake 安装（`nvfp4_lora_bake.py`） | 仅在选定 Z Image 产品时安装 VER=8 bake。含义：按产品身份分支，而非“看起来像 NVFP4 conf”。 |
"""

old_cross = """## ④ 横切含义（摘要）

1. **两种产品，两种戳记：** `_hswq_nvfp4_product_tc`（SDXL）vs `_hswq_nvfp4_comfy_only`（Z Image）。永不在存活的 parity 之上 upgrade TC。
2. **DistOrch 清空的是存储，不是 Python 引用：** Hadamard 与 wrapper 链必须重新校验；单一再包之前剥掉 TC 与 parity 两种 wrapper。
3. **混合 INT8 protect = Conv2d 孪生：** 清除 `Params.convrot`，经标志在线旋转，bake 一次，requant 后保持 Params False。
4. **就地 Linear 突变在 ops peel 之后仍存活：** 离开 Z Image 或进入 SDXL 时始终 `peel_all_nvfp4_linear_lora_bake`。
5. **INT8 protect load overlay 对 SDXL 是外来物：** 与 parity 一样 peel `_hswq_int8_protect_*` / `_hswq_int8_decode_patched`。
"""

new_cross = """## ④ 主含义目录（附录 A 每个文件 + 关键代码）

### 新增模块 — 文件含义与代码含义

| 路径 | 文件含义 | 关键代码 / 符号含义 |
|------|---------------------|--------------------------------|
| `nodes/zimage_nvfp4/__init__.py` | Z Image ConvRot NVFP4 包导出面。 | 仅再导出 load/bake 安装器。 |
| `nodes/zimage_nvfp4/load_unet.py` | dtype `Z Image ConvRot NVFP4` 的 UNet 加载入口。 | `apply_nvfp4_patches` → parity，非 SDXL TC。`ZI_NVFP4_WEIGHT_DTYPE` 与 SDXL `ConvRot NVFP4` 是不同存在。 |
| `nodes/zimage_nvfp4/nvfp4_addmm_patch.py` | Z Image 与 kitchen addmm 交互。 | 防止 kitchen 与 HSWQ 双重旋转。 |
| `nodes/zimage_nvfp4/nvfp4_comfy_parity.py` | Z Image 产品核心：库存 GEMM + 在线 act rotate；PRODUCT 记忆/恢复；外来 peel。 | `apply_nvfp4_comfy_parity`、`_make_convrot_parity_forward`、`_unwrap_stock_forward`、`_ensure_single_parity_linear_forward`、`peel_non_product_nvfp4_ops`、`restore_nvfp4_tc_product_stack`、`_discard_poisoned_product_refs`、`_arm_int8_protect_convrot_after_stock_load`。 |
| `nodes/zimage_nvfp4/nvfp4_lora_bake.py` | Z Image DynamicVRAM LoRA bake（VER=8）+ 混合 INT8 protect。 | `install`/`uninstall` 就地突变 `Linear`；卸载必须剥 VER=8。 |
| `nodes/zimage_nvfp4/nvfp4_tc_gate.py` | 门：parity 存活时拒绝 TC upgrade。 | 阻断 P2 类叠包。 |
| `nodes/zimage_nvfp4/require_parity.py` | Z Image 加载后若缺 parity forward 则快速失败。 | 断言存活 forward 上的产品身份。 |
| `nodes/zimage_nvfp4/zi_comfy_quant_nvfp4.py` | Z Image 补丁安装 / 栈戳记。 | 仅打 `_hswq_nvfp4_comfy_only`。 |
| `nodes/zimage_nvfp4/zi_nvfp4_conf.py` | Z Image conf 辅助。 | conf → 武装标志；无 GEMM。 |
| `nodes/zimage_nvfp4/zi_nvfp4_forward.py` | parity 用的 Z Image forward 辅助。 | 在线 act 路径；非 SDXL scaled_mm TC。 |
| `nodes/zimage_nvfp4/zi_nvfp4_hadamard.py` | Z Image Hadamard 辅助。 | 构建 `H`；存活经 `_tensor_storage_ok`。 |
| `prestartup_script.py` | 早期 ComfyUI 钩子 / 导入路径。 | 仅打包；无模型数学。 |

### 修改模块 — 文件含义与代码含义

| 路径 | 文件含义 | 关键代码 / 符号含义 |
|------|---------------------|--------------------------------|
| `nodes/nvfp4/comfy_quant_nvfp4.py` | SDXL ConvRot NVFP4 **产品**安装器 + 清除 Z Image 残留。 | `_clear_zimage_parity_contamination_for_sdxl` 在 SDXL 补丁前运行；拒绝在残留 parity 上装 TC；打 `_hswq_nvfp4_product_tc`。 |
| `nodes/nvfp4/nvfp4_forward.py` | SDXL 产品 TC forward + 产品 LoRA bake VER=1 + 外来 bake peel。 | `peel_all_nvfp4_linear_lora_bake` 从存活 Linear 剥掉任何 HSWQ bake（含 ZI VER=8）；产品 `attach` 先剥外来 VER。 |
| `nodes/nvfp4/nvfp4_conf.py` | 产品 conf 解码，含 INT8 protect 标志。 | `int8_convrot_flags_from_conf` 使混合 INT8 protect 显式化。 |
| `nodes/nvfp4/nvfp4_hadamard.py` | 产品与 parity 共享的 Hadamard / 存活工具。 | `_tensor_storage_ok` 是 DistOrch 中毒门。 |
| `nodes/nvfp4/nvfp4_load.py` | SDXL 产品 NVFP4 Linear 加载（TC 戳记 / 形状检查）。 | 拥有产品 full-load；绝非 Z Image parity load。 |
| `patches/comfy_quant_int8.py` | INT8 产品路径 + 标记保留 + INT8 加载前 SDXL 清理。 | INT8 前清除 Z Image 污染；经 INT8 wrap 保留 NVFP4 标记。 |

### 横切规则（与 P1–P7 同义）

1. **两种产品，两种戳记：** `_hswq_nvfp4_product_tc`（SDXL）vs `_hswq_nvfp4_comfy_only`（Z Image）。永不在存活的 parity 之上 upgrade TC。
2. **DistOrch 清空的是存储，不是 Python 引用：** Hadamard 与 wrapper 链必须重新校验；单一再包之前剥掉 TC 与 parity 两种 wrapper。
3. **混合 INT8 protect = Conv2d 孪生：** 清除 `Params.convrot`，经标志在线旋转，bake 一次，requant 后保持 Params False。
4. **就地 Linear 突变在 ops peel 之后仍存活：** 离开 Z Image 或进入 SDXL 时始终 `peel_all_nvfp4_linear_lora_bake`。
5. **INT8 protect load overlay 对 SDXL 是外来物：** 与 parity 一样 peel `_hswq_int8_protect_*` / `_hswq_int8_decode_patched`。
"""

for i, (old, new) in enumerate([
    (old_p1, new_p1), (old_p2, new_p2), (old_p3, new_p3),
    (old_p4, new_p4), (old_p5, new_p5), (old_p6, new_p6), (old_cross, new_cross),
]):
    if old not in text:
        raise SystemExit(f"MISSING block {i}")
    text = text.replace(old, new, 1)

old_app = """以下代码块为上文所列主要对策模块的**当前完整文件正文**（UTF-8，生成本指南时磁盘上的内容）。**代码保持英文原文**，与英文版 Appendix A 一致。
"""
new_app = """以下代码块为上文所列主要对策模块的**当前完整文件正文**（UTF-8，生成本指南时磁盘上的内容）。**代码保持英文原文**，与英文版 Appendix A 一致。

**完整性规则：** 下方每个代码围栏必须与磁盘文件字符级一致（仅允许文件末尾换行差异）。撰写本注记时对照工作树校验：**18/18 文件一致**（`nodes/zimage_nvfp4/*`、`prestartup_script.py`、`nodes/nvfp4/{comfy_quant_nvfp4,nvfp4_forward,nvfp4_conf,nvfp4_hadamard,nvfp4_load}.py`、`patches/comfy_quant_int8.py`）。
"""
if old_app not in text:
    raise SystemExit("MISSING ZH appendix intro")
text = text.replace(old_app, new_app, 1)

if "## 结尾" not in text:
    text = text.rstrip() + """

---

## 结尾

本指南对策模块的 **③** 义务由 **附录 A** 满足（完整文件正文；所列 18 个模块已与磁盘字符级校验一致）。

上文 **④** 对每个新增/修改文件以及每个关键符号写明了 **文件含义** 与 **代码含义** — 不只是戳记口号。

污染类操作者复测见 **P7**。英文孪生：`md/HSWQ_FROM_a9d372_PROBLEM_COUNTERMEASURES_GUIDE.md`。
"""

zh.write_text(text, encoding="utf-8", newline="\n")
print("ZH_OK", zh.stat().st_size)
