# HSWQ 问题对策完全解说（自 `a9d372` / v3.3.0 之后）

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../md/HSWQ_FROM_a9d372_PROBLEM_COUNTERMEASURES_GUIDE.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

**基线（本文档所记录的问题对策工作的排他起点）：**  
`a9d372089c2314bcfa9a1d314a3bf81f0dfde9fb` — `docs:point-zhmd-changelog-v3.3.0-link-to-zh`（v3.3.0 已打标签；此提交为开发者指定的文档起点）。

**记载时点 HEAD：** `f030d71afb116ff0b53c2186ebc133a6a6d4ed3a` — peel Z Image NVFP4 contamination so SDXL INT8 LoRA survives after Z Image。

**范围：** Z Image ConvRot NVFP4 出现之后，针对该路径与 SDXL 交互的运行时 / 加载 / LoRA / DistOrch / 包拆分 **问题对策**。纯文档 / CI / 版本号 bump 的提交仅在其承载修复的发布面时列出。

**量化器：** 仅 [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)。

**绝对分支（禁止混用）：**

| 表面 | 加载器 / 路径 |
|---------|----------------|
| SDXL ConvRot NVFP4 | Checkpoint Loader → `nodes/nvfp4` Tensor Core 产品路径（`_hswq_nvfp4_product_tc`） |
| Z Image / ZIT ConvRot NVFP4 | **HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader** → `nodes/zimage_nvfp4` Comfy parity（`_hswq_nvfp4_comfy_only`） |
| 下拉标签 | SDXL：`ConvRot NVFP4` · Z Image：`Z Image ConvRot NVFP4` |

**同期相关专题文档（单主题更深）：**

- `md/HSWQ_ZIMAGE_CONVROT_NVFP4_TECHNICAL_GUIDE.md`
- `md/HSWQ_ZI_HYBRID_NVFP4_INT8_LORA_BAKE_FROM_v3.3.2.md`
- `release/_release_v3.3.2_body_en.md` / `_release_v3.3.3_body_en.md` / `_release_v3.3.4_body_en.md`

本文档是问题 + 文件 + HEAD 对策代码树 **完整当前源码** 的 **单一地图**。

---

## 问题索引（① 一览）

| # | 时代 / tip | 症状 | 根因 |
|---|-----------|---------|------|
| **P1** | Feat → v3.3.1 | Z Image 需要 ConvRot NVFP4 且不能破坏 SDXL TC | 共享 `nodes/nvfp4` TC 栈 ≠ Z Image parity（stock GEMM + 在线 act rotate） |
| **P2** | v3.3.1→v3.3.2 | DistOrch purge 后第 2 次生成 = 椒盐 / 噪声 | INT8 decode wrap 丢掉 NVFP4 标记 → TC 叠在 parity 上 → purge 只剥 TC → **双重在线 act rotate** |
| **P3** | v3.3.1→v3.3.4 | DistOrch 后多代画质衰减 | 模块本地 `_hswq_nvfp4_parity_H` 的复用门弱于全局 `_tensor_storage_ok` |
| **P4** | v3.3.2→v3.3.3 | 混合包 LoRA 失效 / 噪声（约 60 层 INT8 protect 残留） | INT8 protect ConvRot 未按 Conv2d 同型武装/bake；kitchen `Params.convrot` 双重 unrotate / 双重 act-rotate 陷阱 |
| **P5** | `6b52de2` | Z Image 仍与 SDXL 产品模块纠缠 | 共享 `nodes/nvfp4` 所有权；需要专用 `nodes/zimage_nvfp4` 包 |
| **P6** | `916bb89` | SDXL 与 Z Image 共用同一 dtype 字符串 / 同一 bake 路径 | 下拉 + Dynamic bake 必须按 **Z Image ConvRot NVFP4** vs **ConvRot NVFP4** 分支 |
| **P7** | → `f030d71` | SDXL → Z Image → SDXL：椒盐、第 3 次提示词 LoRA 脱落、随后全噪声 | Z Image 留下 `comfy_parity` load overlay + **VER=8 就地 Linear bake** + INT8 protect 武装标志；SDXL INT8 / TC 加载未 peel |

---

## P1 — Z Image ConvRot NVFP4 产品路径（引入）

### ① 问题是什么

SDXL ConvRot NVFP4 使用 **Tensor Core** 全栈（`load_nvfp4_linear_module` + scaled_mm + TC forward 内的 act ConvRot）。Z Image / ZIT ConvRot NVFP4 在 **Comfy parity** 路径上验证：stock MixedPrecision GEMM + **在线** act rotate（`_hswq_nvfp4_parity_H`）、NVFP4 + INT8 protect 混合包、Dynamic VRAM LoRA bake。把 Z Image 强行走 SDXL TC 栈（或留下 kitchen `Params.convrot=True` 导致 kitchen 与 HSWQ 双重旋转）会产生错误激活 / 噪声。AIMDO DynamicVRAM 也与 parity bake wrap 冲突。

### ② 新增 / 修改的文件（问题相关代码）

**新增（包 `nodes/zimage_nvfp4/`）：**

| 路径 | 作用 |
|------|------|
| `__init__.py` | 包导出 |
| `load_unet.py` | Z Image ConvRot NVFP4 的 UNet 加载入口 |
| `zi_comfy_quant_nvfp4.py` | Z Image 补丁安装 / 栈戳记 |
| `zi_nvfp4_conf.py` | Conf 辅助 |
| `zi_nvfp4_forward.py` | Z Image forward 辅助 |
| `zi_nvfp4_hadamard.py` | Z Image Hadamard 辅助 |
| `nvfp4_comfy_parity.py` | Comfy parity 安装、Hadamard 清理、peel/restore PRODUCT |
| `nvfp4_lora_bake.py` | Dynamic.load bake wrap（VER=8）、卸载 |
| `nvfp4_addmm_patch.py` | addmm / kitchen 交互 |
| `nvfp4_tc_gate.py` | TC vs parity 门控 |
| `require_parity.py` | 要求 parity 处于活动状态 |
| `prestartup_script.py` | 早期钩子（sys.path 纪律） |

**修改：** `nodes/nvfp4/*`（SDXL 产品路径保留）、`patches/comfy_quant_int8.py`、`nodes/models/zimage_fp8_e4m3_unet.py`、`hswq/zimage_fp8_e4m3_unet.py`、加载器 UI / README。

### ③ 新增 / 修改代码全文

见 **附录 A**（HEAD `f030d71` 上的完整文件）。

### ④ 含义（文件与代码）

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

---

## P2 — DistOrch purge → 双重在线 act rotate（v3.3.2）

### ① 问题是什么

DistOrch VRAM purge 之后，INT8 decode wrap 可能丢掉 NVFP4 栈标记。随后的“upgrade”会把 **TC 产品**路径叠到已经是 parity 的 Linear 上。DistOrch refresh 只剥掉 **TC** 层，留下 **两层** 在线 act-rotate wrap → **第 2 次生成**出现椒盐 / 噪声。

### ② 文件

| 路径 | 变更 |
|------|--------|
| `nodes/zimage_nvfp4/nvfp4_comfy_parity.py` | `_unwrap_stock_forward` 同时剥 TC **与** parity；单一 parity 再包 |
| `patches/comfy_quant_int8.py` | 经 INT8 wrap 保留 NVFP4 标记 |
| Release / changelog | v3.3.2 发布面 |

### ③ 全文

见附录 A — 尤其 `nvfp4_comfy_parity.py` 中的 `_unwrap_stock_forward` / `_ensure_single_parity_linear_forward`，以及 `patches/comfy_quant_int8.py` 中的 INT8 wrap 标记保留。

### ④ 含义（文件与代码）

| 文件 / 符号 | 含义 |
|---------------|---------|
| `nvfp4_comfy_parity.py` 中的 `_unwrap_stock_forward` | 遍历存活 `Linear.forward` wrapper 链，剥掉 **TC 产品**与 **parity** 两壳直至库存基座。含义：DistOrch refresh 不得留下第二层在线 act-rotate。 |
| `_ensure_single_parity_linear_forward` | unwrap 之后只安装 **恰好一层** parity forward。含义：单一在线旋转，永不 TC-on-parity 后再只剥 TC。 |
| `patches/comfy_quant_int8.py` 中的 INT8 wrap 标记保留 | INT8 decode wrap Linear 时，NVFP4 / parity 戳记必须存活。含义：不得忘记“此 Linear 是 parity”。 |

---

## P3 — DistOrch purge → 中毒的模块本地 Hadamard（v3.3.4）

### ① 问题是什么

Parity 保存 `module._hswq_nvfp4_parity_H`。DistOrch Method 3 可清空/毒化张量存储，而 Python 仍持有该属性。全局缓存已使用 `_tensor_storage_ok`；模块本地复用仅检查 device/dtype/`numel`/`nbytes==0`，可能复用死亡壳 → **第 2 次及之后**画质衰减。

### ② 文件

| 路径 | 变更 |
|------|--------|
| `nodes/zimage_nvfp4/nvfp4_comfy_parity.py` | `need_rebuild` 使用 `_tensor_storage_ok` |
| `nodes/nvfp4/nvfp4_hadamard.py` | `_tensor_storage_ok`（共享门） |

### ③ 全文

见附录 A — `_tensor_storage_ok` 与 `_make_convrot_parity_forward`。

### ④ 含义（文件与代码）

| 文件 / 符号 | 含义 |
|---------------|---------|
| `nodes/nvfp4/nvfp4_hadamard.py` 中的 `_tensor_storage_ok` | 共享门：Hadamard 张量仅在存储存活时可用。含义：仅 device/dtype/`numel` **不够**。 |
| `_make_convrot_parity_forward` 内的 `need_rebuild` | 模块本地 `module._hswq_nvfp4_parity_H` 必须通过与全局缓存 **相同** 的 `_tensor_storage_ok`。含义：DistOrch purge 后第 2 次及之后重建 `H`。 |
| 重建时调用的 `build_hadamard` | 重建存活 Hadamard；含义：Method-3 存储擦除后的画质恢复。 |

---

## P4 — 混合 NVFP4 + INT8 protect LoRA bake（v3.3.3）

### ① 问题是什么

混合包（约 120 层 NVFP4 + 约 60 层 INT8 protect ConvRot）bake 了 NVFP4 键，却留下 **INT8 protect LowVramPatch** 残留。错误的 `Params.convrot` 处理导致 **双重 unrotate**（LoRA 死）或 **双重 act rotate**（噪声）。证据日志把 INT8 成功埋在 NVFP4 采样配额之下。

### ② 文件

| 路径 | 作用 |
|------|------|
| `nodes/nvfp4/nvfp4_conf.py` | `int8_convrot_flags_from_conf` |
| `nodes/nvfp4/nvfp4_forward.py` | 双重 unrotate / re-rotate、计数器、EVIDENCE |
| `nodes/zimage_nvfp4/nvfp4_comfy_parity.py` | 按 Conv2d 同型武装 INT8 protect |
| `nodes/zimage_nvfp4/nvfp4_lora_bake.py` | 双通道 bake + pass-delta EVIDENCE |

### ③ 全文

见附录 A。叙事细节另见 `md/HSWQ_ZI_HYBRID_NVFP4_INT8_LORA_BAKE_FROM_v3.3.2.md`。

### ④ 含义（文件与代码）

| 文件 / 符号 | 含义 |
|---------------|---------|
| `nvfp4_conf.py` 的 `int8_convrot_flags_from_conf` | 将混合 INT8 protect ConvRot conf 读入 HSWQ 标志。含义：INT8 protect 是 NVFP4 ConvRot 的一等孪生。 |
| `nvfp4_forward.py` 的双重 unrotate / re-rotate | bake 路径按正确 layout unrotate 一次；在线路径经 `_hswq_int8_convrot` 旋转 act，无 kitchen 双重旋转。含义：错误 `Params.convrot` = LoRA 死或噪声。 |
| `nvfp4_comfy_parity.py` 的 `_arm_int8_protect_convrot_after_stock_load` | **按 Conv2d 同型**武装：清 kitchen `Params.convrot`，设 `_hswq_int8_convrot`。含义：HSWQ 拥有旋转。 |
| `nvfp4_lora_bake.py` 的双通道 bake + pass-delta EVIDENCE | 同时 bake NVFP4 **与** INT8 protect 键；EVIDENCE 必须用 pass-delta，不得在 `patches=0` 上空打印 OK。 |

---

## P5 — 将 Z Image 从共享 `nodes/nvfp4` 剥离（`6b52de2`）

### ① 问题是什么

Z Image 的 parity / bake 与 SDXL 产品模块纠缠。共享所有权使污染与“upgrade”竞态不可避免。

### ② 文件

重构：Z Image 运行时位于 `nodes/zimage_nvfp4/`；SDXL 产品仍在 `nodes/nvfp4/`。

### ③ 全文

附录 A 即为 peel 之后的 HEAD 树。

### ④ 含义（文件与代码）

| 边界 | 含义 |
|----------|---------|
| 包 `nodes/zimage_nvfp4/` | 剥离后拥有 Z Image ConvRot NVFP4 运行时。含义：SDXL 产品代码不得永远携带 Z Image parity。 |
| 包 `nodes/nvfp4/` | 仅拥有 SDXL ConvRot NVFP4 TC 产品。含义：共享所有权曾是污染根因。 |
| SDXL 清理调用 Z Image peel/uninstall | 允许的 **调用**方向：SDXL 可调用 Z Image 清理。含义：卸载后 Z Image 不得永久占有 SDXL `ops`。 |

---

## P6 — 分离下拉项 + Dynamic bake 分支（`916bb89`）

### ① 问题是什么

单一 dtype / 单一 bake 路径诱使 SDXL 与 Z Image 共享 wrapper。开发者绝对规则：分离 **ConvRot NVFP4** 与 **Z Image ConvRot NVFP4**。

### ② 文件

加载器 UI / weight_dtype 字符串；Dynamic bake 安装以 Z Image dtype 为门控。

### ③ 全文

见附录 A 中的加载器 / `nvfp4_lora_bake.py` 安装路径。

### ④ 含义（文件与代码）

| 片段 | 含义 |
|-------|---------|
| 下拉字符串 `Z Image ConvRot NVFP4` | 与 SDXL `ConvRot NVFP4` 分离的产品身份。含义：一个字符串 → 一套栈；永不共享。 |
| 按 Z Image dtype 门控的 Dynamic bake 安装（`nvfp4_lora_bake.py`） | 仅在选定 Z Image 产品时安装 VER=8 bake。含义：按产品身份分支，而非“看起来像 NVFP4 conf”。 |

---

## P7 — Z Image 之后的 SDXL 污染（`f030d71` 及前驱）

### ① 问题是什么

实测序列 **SDXL → Z Image → SDXL**：

1. Z Image 在 `ops._load_quantized_module` / `mixed_precision_ops` 上安装 `comfy_parity`。
2. Z Image 用 LoRA bake **VER=8**（`[HSWQ ConvRot LoRA] … int8_protect`）**就地突变 `mp0.Linear`**。
3. INT8 protect **武装 overlay** 在加载链上打上 `_hswq_int8_protect_arm_v2` / `_hswq_int8_protect_in_load`。
4. 早期 peel 只把 `_hswq_nvfp4_comfy_only` / 非产品 `_hswq_nvfp4_full_load` 当外来物 — **漏掉** INT8 protect overlay 标志。
5. 仅 peel `ops.mixed_precision_ops` **不会**撤销就地 Linear 类突变 → SDXL INT8 仍命中 ZI VER=8 bake → **第 3 次提示词** LoRA 脱落。
6. Overlay 武装了 SDXL INT8 ConvRot（`int8_tensorwise`+`convrot`）→ `_hswq_int8_convrot` + 错误 `Params` → **噪声**（第 3 次 SDXL 上大量 `arm INT8 protect ConvRot #80…#760`）。

### ② 新增 / 修改的文件（本对策）

| 路径 | 变更 |
|------|--------|
| `nodes/nvfp4/comfy_quant_nvfp4.py` | `_clear_zimage_parity_contamination_for_sdxl`；在 SDXL NVFP4 补丁前调用；残留 parity 时拒绝 TC |
| `nodes/nvfp4/nvfp4_forward.py` | `peel_all_nvfp4_linear_lora_bake`；产品 `attach_nvfp4_linear_lora_bake` 先剥外来 VER |
| `nodes/zimage_nvfp4/nvfp4_comfy_parity.py` | `peel_non_product_nvfp4_ops` 外来 load 标志；`_discard_poisoned_product_refs`；PRODUCT remember 要求 `_hswq_nvfp4_product_tc` |
| `nodes/zimage_nvfp4/nvfp4_lora_bake.py` | `uninstall_zimage_nvfp4_lora_bake` + Linear peel |
| `patches/comfy_quant_int8.py` | SDXL INT8 加载前清除污染 |

### ③ 全文

附录 A — HEAD 完整文件。关键符号：`_clear_zimage_parity_contamination_for_sdxl`、`peel_all_nvfp4_linear_lora_bake`、`peel_non_product_nvfp4_ops`、`_discard_poisoned_product_refs`。

### ④ 含义

| 片段 | 含义 |
|-------|---------|
| `restore_nvfp4_tc_product_stack()` | 当打有 `_hswq_nvfp4_product_tc` 时，把记住的 SDXL PRODUCT load/mp 放回 |
| `peel_non_product_nvfp4_ops` | 遍历 wrapper 链；把 INT8 protect overlay / decode / 非产品 full_load 当外来 |
| `peel_all_nvfp4_linear_lora_bake(Lin)` | 从 **活着的** `Linear.convert_weight` / `set_weight` 上剥掉 VER=8（任意 HSWQ bake） |
| 仅当 product_tc 时再 `attach` VER=1 | SDXL NVFP4 得到产品 bake；SDXL INT8 在 peel 后保持 stock |
| INT8 加载时也清除 | 进入 INT8 UNet 路径前同样清污染 |

### 操作者复测

1. 完全重启 ComfyUI。
2. SDXL INT8（+LoRA）→ Z Image ConvRot NVFP4 → 再次 SDXL INT8（+LoRA）。
3. SDXL 上 **不得** 出现 ZI `Dynamic.load ENTER`、`[HSWQ ConvRot LoRA] int8_protect`，或第 3 次 SDXL 上的大量 `arm INT8 protect ConvRot`。
4. 清理时优先看到 peel / restore 控制台行。

---

## ② 主文件列表（代码对策，`a9d372`..HEAD）

### 新增（新模块）

- `nodes/zimage_nvfp4/__init__.py`
- `nodes/zimage_nvfp4/load_unet.py`
- `nodes/zimage_nvfp4/nvfp4_addmm_patch.py`
- `nodes/zimage_nvfp4/nvfp4_comfy_parity.py`
- `nodes/zimage_nvfp4/nvfp4_lora_bake.py`
- `nodes/zimage_nvfp4/nvfp4_tc_gate.py`
- `nodes/zimage_nvfp4/require_parity.py`
- `nodes/zimage_nvfp4/zi_comfy_quant_nvfp4.py`
- `nodes/zimage_nvfp4/zi_nvfp4_conf.py`
- `nodes/zimage_nvfp4/zi_nvfp4_forward.py`
- `nodes/zimage_nvfp4/zi_nvfp4_hadamard.py`
- `prestartup_script.py`

### 修改（既有模块；全文见附录 A）

| 路径 | 说明 |
|------|-------|
| `nodes/nvfp4/comfy_quant_nvfp4.py` | SDXL 产品 + 污染清除 |
| `nodes/nvfp4/nvfp4_forward.py` | 产品 bake VER=1 + peel_all |
| `nodes/nvfp4/nvfp4_conf.py` | INT8 protect conf 标志 |
| `nodes/nvfp4/nvfp4_hadamard.py` | `_tensor_storage_ok` |
| `nodes/nvfp4/nvfp4_load.py` | 产品加载 |
| `patches/comfy_quant_int8.py` | 标记 + SDXL 清除 |
| 加载器 / README / changelog / zhmd / release 正文 | 版本发布面（附录 A 不重复） |

---

## ④ 主含义目录（附录 A 每个文件 + 关键代码）

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

---

## 附录 A — HEAD `f030d71` 完整源码

以下代码块为上文所列主要对策模块的**当前完整文件正文**（UTF-8，生成本指南时磁盘上的内容）。**代码保持英文原文**，与英文版 Appendix A 一致。

**完整性规则：** 下方每个代码围栏必须与磁盘文件字符级一致（仅允许文件末尾换行差异）。撰写本注记时对照工作树校验：**18/18 文件一致**（`nodes/zimage_nvfp4/*`、`prestartup_script.py`、`nodes/nvfp4/{comfy_quant_nvfp4,nvfp4_forward,nvfp4_conf,nvfp4_hadamard,nvfp4_load}.py`、`patches/comfy_quant_int8.py`）。

### `nodes/zimage_nvfp4/__init__.py`

```python
"""Z Image ConvRot NVFP4 — comfy_parity (stock GEMM + act rotate); INT8 = core."""

from .load_unet import (
    apply_nvfp4_patches,
    install_zimage_nvfp4_unet_dispatch,
    load_unet_nvfp4_weight_dtype,
)
from .nvfp4_lora_bake import install_zimage_nvfp4_lora_bake

__all__ = [
    "apply_nvfp4_patches",
    "install_zimage_nvfp4_unet_dispatch",
    "install_zimage_nvfp4_lora_bake",
    "load_unet_nvfp4_weight_dtype",
]
```

### `nodes/zimage_nvfp4/load_unet.py`

```python
"""Z Image / ZIT UNet load — ConvRot NVFP4 (parity) + INT8 ConvRot (ComfyUI core).

Z Image ConvRot NVFP4 is **not** the SDXL TC Linear.forward path.
``hswq/benchmark/zi_convrot_nvfp4_bench.py`` ``require_convrot_parity_forward``:
TC wrap (``_hswq_nvfp4_full_forward``) destroys SSIM; need stock GEMM + online
act rotate (``_hswq_nvfp4_convrot_parity``) via ``apply_nvfp4_comfy_parity``.

  - Arm detect/load/LoRA bake with ``zi_comfy_quant_nvfp4.apply_comfy_quant_nvfp4_patches``,
    then **replace** Linear.forward with comfy_parity (not stacked double-rotate).
  - INT8 ConvRot: ComfyUI core / kitchen as-is. ``apply_comfy_quant_int8_patches``
    only for int8_tensorwise load.

All logic under ``nodes/zimage_nvfp4``. Does not edit ``nodes/nvfp4`` (SDXL TC).
"""
from __future__ import annotations

import logging
import sys

# ZI/Krea UNet dropdown ONLY — never share the SDXL Checkpoint Loader string.
# SDXL uses nodes/nvfp4 NVFP4_WEIGHT_DTYPE == "ConvRot NVFP4" (separate being).
ZI_NVFP4_WEIGHT_DTYPE = "Z Image ConvRot NVFP4"

_DISPATCH_INSTALLED = False
_INSTALL_HOOKED = False

logger = logging.getLogger(__name__)


def apply_nvfp4_patches() -> None:
    """Arm Z Image ConvRot NVFP4 (parity) + INT8 load (core ConvRot)."""
    from .zi_comfy_quant_nvfp4 import apply_comfy_quant_nvfp4_patches
    from ...patches.comfy_quant_int8 import apply_comfy_quant_int8_patches
    from .nvfp4_comfy_parity import (
        apply_nvfp4_comfy_parity,
        require_convrot_parity_forward,
    )
    from .nvfp4_lora_bake import install_zimage_nvfp4_lora_bake

    if not apply_comfy_quant_nvfp4_patches():
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image: apply_comfy_quant_nvfp4_patches failed "
            "(detect/load/LoRA bake required; see nodes/zimage_nvfp4/zi_comfy_quant_nvfp4)"
        )
    # Replace TC Linear.forward with stock GEMM + act rotate (not double-rotate).
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image: apply_nvfp4_comfy_parity failed "
            "(stock GEMM + act rotate required; TC destroys SSIM)"
        )
    require_convrot_parity_forward()
    # INT8 tensorwise load only — ConvRot INT8 remains ComfyUI core / kitchen.
    apply_comfy_quant_int8_patches()
    # After INT8 Dynamic bake wrap: force ConvRot NVFP4 LoRA bake outermost.
    if not install_zimage_nvfp4_lora_bake(force=True):
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image: install_zimage_nvfp4_lora_bake failed "
            "(Dynamic ConvRot NVFP4 LoRA bake required)"
        )
    print(
        "  [HSWQ NVFP4] Z Image: ConvRot NVFP4 (comfy_parity) + INT8 ConvRot "
        "+ Dynamic NVFP4 LoRA bake",
        flush=True,
    )


def _ensure_dynamic_load_bake_wrap() -> None:
    """Re-arm ZI NVFP4 bake wrap if MultiGPU/INT8 overwrote Dynamic.load or load_models_gpu."""
    from .nvfp4_lora_bake import (
        _BAKE_HOOK_VER,
        install_load_models_gpu_bake_hook,
        install_zimage_nvfp4_lora_bake,
    )

    try:
        import comfy.model_management as mm
        import comfy.model_patcher as mp
    except ImportError:
        return
    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    need_force = True
    if Dynamic is not None:
        cur = getattr(Dynamic, "load", None)
        if (
            cur is not None
            and getattr(cur, "_hswq_zi_nvfp4_lora_bake", False)
            and getattr(cur, "_hswq_zi_nvfp4_lora_bake_ver", 0) >= _BAKE_HOOK_VER
        ):
            need_force = False
    if need_force:
        install_zimage_nvfp4_lora_bake(force=True)
    gpu = getattr(mm, "load_models_gpu", None)
    if (
        gpu is None
        or not getattr(gpu, "_hswq_zi_nvfp4_gpu_bake", False)
        or getattr(gpu, "_hswq_zi_nvfp4_gpu_bake_ver", 0) < _BAKE_HOOK_VER
    ):
        install_load_models_gpu_bake_hook(force=True)
    else:
        install_load_models_gpu_bake_hook(force=False)


def load_unet_nvfp4_weight_dtype(unet_name, weight_dtype):
    """Load Z Image / ZIT UNet with ConvRot NVFP4 parity (not SDXL TC forward)."""
    import folder_paths
    import comfy.sd

    from .zi_comfy_quant_nvfp4 import apply_comfy_quant_nvfp4_patches
    from .zi_nvfp4_forward import reset_nvfp4_lora_log_counters
    from ...patches.comfy_quant_int8 import (
        _int8_quant_conv_scope,
        apply_comfy_quant_int8_patches,
        reset_int8_lora_log_counters,
        summarize_int8_lora_capability,
    )
    from .nvfp4_comfy_parity import (
        apply_nvfp4_comfy_parity,
        require_convrot_parity_forward,
    )
    from .nvfp4_lora_bake import (
        install_zimage_nvfp4_lora_bake,
        reset_zimage_nvfp4_lora_bake_log_counters,
    )

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    if not apply_comfy_quant_nvfp4_patches():
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image UNet requires NVFP4 detect/load/LoRA bake "
            "(zi_comfy_quant_nvfp4.apply_comfy_quant_nvfp4_patches)"
        )
    if not apply_nvfp4_comfy_parity():
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image UNet requires comfy_parity "
            "(stock GEMM + act rotate; not HSWQ TC Linear.forward)"
        )
    require_convrot_parity_forward()
    # Mixed pack: Linear=nvfp4 parity, INT8 = ComfyUI core ConvRot path.
    apply_comfy_quant_int8_patches()
    if not install_zimage_nvfp4_lora_bake(force=True):
        raise RuntimeError(
            "[HSWQ NVFP4] Z Image UNet requires Dynamic ConvRot NVFP4 LoRA bake"
        )
    _ensure_dynamic_load_bake_wrap()
    reset_int8_lora_log_counters()
    reset_nvfp4_lora_log_counters()
    reset_zimage_nvfp4_lora_bake_log_counters()
    logging.info(
        "[HSWQ NVFP4] Loading UNet (ConvRot NVFP4 comfy_parity + INT8 ConvRot ComfyUI core): "
        "%s (weight_dtype=%s)",
        unet_name,
        weight_dtype,
    )
    print(
        f"[HSWQ NVFP4] Loading UNet (ConvRot NVFP4 / comfy_parity): {unet_name}",
        flush=True,
    )
    with _int8_quant_conv_scope():
        model = comfy.sd.load_diffusion_model(unet_path, model_options={})
    summarize_int8_lora_capability(model)
    return (model,)


def _attach_to_comfy_quant_module() -> None:
    """Expose this loader on comfy_quant_nvfp4 so prestartup can bind it."""
    for name, mod in list(sys.modules.items()):
        if not (
            name.endswith("nodes.nvfp4.comfy_quant_nvfp4")
            or name.endswith(".comfy_quant_nvfp4")
            or name == "comfy_quant_nvfp4"
        ):
            continue
        cur = getattr(mod, "load_unet_nvfp4_weight_dtype", None)
        if cur is None or cur is load_unet_nvfp4_weight_dtype:
            mod.load_unet_nvfp4_weight_dtype = load_unet_nvfp4_weight_dtype


def install_zimage_nvfp4_unet_dispatch(node_class_mappings=None) -> bool:
    """Wrap HSWQFP8E4M3UNetLoader for weight_dtype ConvRot NVFP4.

    Must run *after* ``install_int8_option_dispatch``: mixed NVFP4 packs also
    contain ``int8_tensorwise`` layers, so INT8-only auto-detect would otherwise
    steal the load without NVFP4 Linear patches. INT8 ConvRot stays core.
    """
    global _DISPATCH_INSTALLED
    if node_class_mappings is None:
        wrapped_any = False
        for _n, mod in list(sys.modules.items()):
            mappings = getattr(mod, "NODE_CLASS_MAPPINGS", None)
            if isinstance(mappings, dict) and install_zimage_nvfp4_unet_dispatch(mappings):
                wrapped_any = True
        return wrapped_any

    if not isinstance(node_class_mappings, dict):
        return False

    from ..nvfp4.nvfp4_conf import checkpoint_looks_like_comfy_quant_nvfp4

    unet_cls = node_class_mappings.get("HSWQFP8E4M3UNetLoader")
    if unet_cls is None:
        return False
    if getattr(unet_cls, "_hswq_zi_nvfp4_dispatch", False):
        _DISPATCH_INSTALLED = True
        return True

    _fp8 = frozenset({"fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"})
    _prev = unet_cls.load_unet

    def load_unet(self, unet_name, weight_dtype):
        _ensure_dynamic_load_bake_wrap()
        if weight_dtype in _fp8:
            return _prev(self, unet_name, weight_dtype)
        if weight_dtype == ZI_NVFP4_WEIGHT_DTYPE:
            return load_unet_nvfp4_weight_dtype(unet_name, weight_dtype)
        import folder_paths

        if weight_dtype == "default":
            unet_path = folder_paths.get_full_path_or_raise(
                "diffusion_models", unet_name
            )
            if checkpoint_looks_like_comfy_quant_nvfp4(unet_path):
                return load_unet_nvfp4_weight_dtype(unet_name, weight_dtype)
        # Never treat SDXL's "ConvRot NVFP4" string as ZI — different being.
        # int8_tensorwise / other: leave to INT8 dispatch / original (core ConvRot).
        return _prev(self, unet_name, weight_dtype)

    unet_cls.load_unet = load_unet
    unet_cls._hswq_zi_nvfp4_dispatch = True  # type: ignore[attr-defined]
    _DISPATCH_INSTALLED = True
    print(
        f"[HSWQ NVFP4] Z Image UNet dispatch: {ZI_NVFP4_WEIGHT_DTYPE!r} "
        "-> nodes.zimage_nvfp4 (comfy_parity; not SDXL ConvRot NVFP4)",
        flush=True,
    )
    return True


def _hook_nvfp4_install_for_unet_dispatch() -> None:
    """When package ``__init__`` runs SDXL NVFP4 install, also wrap Z Image UNet."""
    global _INSTALL_HOOKED
    if _INSTALL_HOOKED:
        return
    for name, mod in list(sys.modules.items()):
        if not (
            name.endswith("nodes.nvfp4.comfy_quant_nvfp4")
            or name.endswith(".comfy_quant_nvfp4")
            or name == "comfy_quant_nvfp4"
        ):
            continue
        prev = getattr(mod, "install_nvfp4_option_dispatch", None)
        if prev is None or getattr(prev, "_hswq_zi_unet_hook", False):
            continue

        def install_nvfp4_option_dispatch(node_class_mappings, _prev=prev):
            ok = _prev(node_class_mappings)
            install_zimage_nvfp4_unet_dispatch(node_class_mappings)
            return ok

        install_nvfp4_option_dispatch._hswq_zi_unet_hook = True  # type: ignore[attr-defined]
        mod.install_nvfp4_option_dispatch = install_nvfp4_option_dispatch
        _INSTALL_HOOKED = True
        return


# Import-time: register on comfy_quant; hook SDXL install so UNet wrap runs after INT8.
_attach_to_comfy_quant_module()
_hook_nvfp4_install_for_unet_dispatch()
install_zimage_nvfp4_unet_dispatch()
```

### `nodes/zimage_nvfp4/nvfp4_addmm_patch.py`

```python
"""Fill kitchen NVFP4 gap: register aten.addmm for TensorCoreNVFP4Layout.

comfy_kitchen registers addmm for INT8 / MXFP8 / FP8 / SVDQuant / ConvRotW4A4,
but NOT for TensorCoreNVFP4Layout. PyTorch F.linear(bias=...) often decomposes
to aten.addmm.default → unhandled → full dequantize of both operands.

That is why stock MixedPrecision Linear (Comfy ops.py) can look "NVFP4 loaded"
(uint8 packed weights in state_dict) while peak VRAM exceeds FP16: packed
storage stays resident AND dequant materializes FP16 weights every forward.

Runtime-only registration — does not edit ComfyUI-master or site-packages files.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_REGISTERED = False


def register_nvfp4_addmm_handler() -> bool:
    """Register aten.addmm.default → scaled_mm_nvfp4 (same contract as MXFP8 addmm)."""
    global _REGISTERED
    if _REGISTERED:
        return True

    try:
        import torch
        import comfy_kitchen as ck
        from comfy_kitchen.tensor.base import (
            QuantizedTensor,
            dequantize_args,
            register_layout_op,
            _LAYOUT_DISPATCH_TABLE,
        )
        from comfy_kitchen.tensor.nvfp4 import (
            TensorCoreNVFP4Layout,
            _slice_to_original_shape,
        )
        from .nvfp4_tc_gate import (
            announce_tc_status_at_register,
            note_scaled_mm_failure,
            nvfp4_tc_enabled,
        )
    except Exception as e:
        logger.warning("[HSWQ NVFP4] addmm register skipped (import): %s", e)
        return False

    announce_tc_status_at_register()

    # Already present in a newer kitchen — do not double-register.
    op = torch.ops.aten.addmm.default
    table = _LAYOUT_DISPATCH_TABLE.get(op, {})
    if TensorCoreNVFP4Layout in table:
        _REGISTERED = True
        logger.info("[HSWQ NVFP4] aten.addmm already registered for NVFP4")
        return True

    @register_layout_op(op, TensorCoreNVFP4Layout)
    def _handle_nvfp4_addmm(qt, args, kwargs):
        """NVFP4 addmm: bias + input @ weight.T (F.linear with bias decomposition)."""
        bias, mat1, mat2 = args[0], args[1], args[2]

        if not (isinstance(mat1, QuantizedTensor) and isinstance(mat2, QuantizedTensor)):
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))
        if mat1._qdata.dim() != 2:
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

        input_transposed = getattr(mat1._params, "transposed", False)
        weight_transposed = getattr(mat2._params, "transposed", False)
        # F.linear → addmm(bias, x, w.t()): weight must be logically transposed.
        if input_transposed or not weight_transposed:
            logger.debug(
                "NVFP4 addmm: unsupported transpose configuration, falling back to dequantize"
            )
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

        # Cloud Ada/Hopper etc.: skip scaled_mm after first CUBLAS NOT_SUPPORTED
        # (otherwise WARNING floods every Linear every step).
        if not nvfp4_tc_enabled():
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

        input_qdata, scale_a, block_scale_a = TensorCoreNVFP4Layout.get_plain_tensors(mat1)
        weight_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(mat2)
        out_dtype = kwargs.get("out_dtype", mat1._params.orig_dtype)

        try:
            result = ck.scaled_mm_nvfp4(
                input_qdata,
                weight_qdata,
                tensor_scale_a=scale_a,
                tensor_scale_b=scale_b,
                block_scale_a=block_scale_a,
                block_scale_b=block_scale_b,
                bias=bias,
                out_dtype=out_dtype,
            )
            orig_m = mat1._params.orig_shape[0]
            orig_n = mat2._params.orig_shape[1]
            return _slice_to_original_shape(result, orig_m, orig_n)
        except (RuntimeError, TypeError) as e:
            note_scaled_mm_failure(e)
            return torch.addmm(*dequantize_args((bias, mat1, mat2)))

    _REGISTERED = True
    print(
        "[HSWQ NVFP4] registered aten.addmm.default for TensorCoreNVFP4Layout "
        "(stock F.linear+bias -> scaled_mm_nvfp4; was dequant-only)",
        flush=True,
    )
    return True
```

### `nodes/zimage_nvfp4/nvfp4_comfy_parity.py`

```python
"""Z Image / ZIT ConvRot NVFP4 — ComfyUI stock GEMM + online act rotate.

Ported from ``hswq/benchmark/nvfp4_comfy_parity.py`` (same math as
``zi_convrot_nvfp4_bench.py``). Product HSWQ Tensor Core Linear.forward
breaks Pixel SSIM on Z Image ConvRot packs; the bench path does not.

Call ``apply_nvfp4_comfy_parity()`` **after** ``apply_comfy_quant_nvfp4_patches()``
for UNet / Z Image loads. SDXL product path keeps TC via
``restore_nvfp4_tc_product_stack()`` before SDXL checkpoint load.

Does not edit ComfyUI-master.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_PARITY_APPLIED = False
_PRODUCT_LOAD: Optional[Callable] = None
_PRODUCT_MP: Optional[Callable] = None

# Runtime / load diagnostics (console — owner-ordered visibility).
_LOAD_NVFP4_SEEN = 0
_LOAD_CONVROT_ARMED = 0
_LOAD_NVFP4_NO_CONVROT = 0
_LOAD_INT8_CONVROT_ARMED = 0


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def reset_nvfp4_parity_load_counters() -> None:
    global _LOAD_NVFP4_SEEN, _LOAD_CONVROT_ARMED, _LOAD_NVFP4_NO_CONVROT
    global _LOAD_INT8_CONVROT_ARMED
    _LOAD_NVFP4_SEEN = 0
    _LOAD_CONVROT_ARMED = 0
    _LOAD_NVFP4_NO_CONVROT = 0
    _LOAD_INT8_CONVROT_ARMED = 0


def clear_nvfp4_parity_hadamard_caches(root=None) -> int:
    """Drop parity ``H`` attrs + global Hadamard dicts after Distorch purge.

    Method 3 may ``t.data = empty`` on module ``_hswq_nvfp4_parity_H`` while the
    same tensor remains in ``zi_nvfp4_hadamard._HADAMARD_CACHE``. The next gen then
    gets a dead/garbage ``H`` from the global cache (nbytes==0 rebuild still
    returns the poisoned entry) and quality decays as CUDA reuses the region
    (2nd→3rd→4th gen gradually worse). Distorch Method 2c calls this via
    ``sys.modules``.
    """
    import gc

    import torch

    from .zi_nvfp4_hadamard import clear_hadamard_global_caches

    cleared = 0
    cleared += int(clear_hadamard_global_caches() or 0)

    def _drop_attr(mod, name: str) -> None:
        nonlocal cleared
        if not hasattr(mod, name):
            return
        try:
            delattr(mod, name)
            cleared += 1
        except Exception:
            try:
                setattr(mod, name, None)
                cleared += 1
            except Exception:
                pass

    def _clear_one(mod) -> None:
        if not isinstance(mod, torch.nn.Module):
            return
        _drop_attr(mod, "_hswq_nvfp4_parity_H")
        _drop_attr(mod, "_hswq_nvfp4_H")
        # Z Image Dynamic LoRA bake bookkeeping — Distorch INT8 clear missed these.
        _drop_attr(mod, "_hswq_zi_nvfp4_baked_keys")
        _drop_attr(mod, "_hswq_zi_nvfp4_baked_uuid")

    if root is not None:
        if isinstance(root, torch.nn.Module):
            for m in root.modules():
                _clear_one(m)
            _clear_one(root)
        return cleared

    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.nn.Module):
                _clear_one(obj)
        except Exception:
            continue
    return cleared


def log_nvfp4_parity_load_summary(label: str = "") -> None:
    """Print how many nvfp4 / int8protect ConvRot layers were armed during load."""
    tag = f" ({label})" if label else ""
    _console(
        f"[HSWQ NVFP4][diag] load summary{tag}: "
        f"nvfp4_seen={_LOAD_NVFP4_SEEN} "
        f"convrot_armed={_LOAD_CONVROT_ARMED} "
        f"nvfp4_no_convrot={_LOAD_NVFP4_NO_CONVROT} "
        f"int8_convrot_armed={_LOAD_INT8_CONVROT_ARMED}"
    )
    if _LOAD_NVFP4_SEEN == 0:
        _console(
            "[HSWQ NVFP4][diag] WARNING: zero nvfp4 layers seen during load — "
            "comfy_quant markers may be missing / wrong prefix "
            "(kitchen bare→prefixed remap should have run)"
        )
    elif _LOAD_CONVROT_ARMED == 0:
        _console(
            "[HSWQ NVFP4][diag] WARNING: nvfp4 layers loaded but "
            "convrot_armed=0 — act rotate will never run"
        )
    if _LOAD_INT8_CONVROT_ARMED == 0:
        _console(
            "[HSWQ NVFP4][diag] WARNING: int8protect ConvRot Linear armed=0 — "
            "mixed packs need online act rotate on protect Linears "
            "(offline W@H^T without x@H → bit-crush)"
        )


def summarize_nvfp4_parity_modules(model, max_names: int = 8) -> None:
    """Post-load walk: Linear counts + forward type + sample ConvRot names."""
    import torch.nn as nn

    try:
        import comfy.ops as ops
    except Exception as e:
        _console(f"[HSWQ NVFP4][diag] post-load skipped (ops): {e}")
        return

    # ModelPatcher -> BaseModel -> diffusion_model (same as INT8 summary).
    diffusion = model
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        diffusion = model.model.diffusion_model
    elif hasattr(model, "diffusion_model"):
        diffusion = model.diffusion_model

    n_linear = 0
    n_convrot = 0
    n_int8_convrot = 0
    n_tc_arm = 0
    names: list[str] = []
    names_i8: list[str] = []
    for name, mod in diffusion.named_modules():
        if not isinstance(mod, nn.Linear) and "Linear" not in type(mod).__name__:
            continue
        n_linear += 1
        if getattr(mod, "_hswq_nvfp4_convrot", False):
            n_convrot += 1
            if len(names) < max_names:
                gs = getattr(mod, "_hswq_nvfp4_convrot_groupsize", "?")
                names.append(f"{name}(gs={gs})")
        if getattr(mod, "_hswq_int8_convrot", False):
            n_int8_convrot += 1
            if len(names_i8) < max_names:
                gs = getattr(mod, "_hswq_int8_convrot_groupsize", "?")
                names_i8.append(f"{name}(gs={gs})")
        if getattr(mod, "_hswq_nvfp4", False):
            n_tc_arm += 1

    fwd = ops.mixed_precision_ops().Linear.forward
    fwd_parity = bool(getattr(fwd, "_hswq_nvfp4_convrot_parity", False))
    fwd_tc = bool(getattr(fwd, "_hswq_nvfp4_full_forward", False))
    load_fn = ops._load_quantized_module
    load_parity = bool(getattr(load_fn, "_hswq_nvfp4_comfy_only", False))
    # INT8 may wrap load outside; peel once for display.
    if not load_parity and getattr(load_fn, "_hswq_int8_decode_patched", False):
        inner = _closure_named(load_fn, "original_load")
        if inner is not None:
            load_parity = bool(getattr(inner, "_hswq_nvfp4_comfy_only", False))
            load_fn = inner
    load_tc = bool(
        getattr(load_fn, "_hswq_nvfp4_full_load", False)
        and not getattr(load_fn, "_hswq_nvfp4_comfy_only", False)
    )

    _console(
        "[HSWQ NVFP4][diag] ===== post-load ====="
    )
    _console(
        f"[HSWQ NVFP4][diag] Linear={n_linear} "
        f"_hswq_nvfp4_convrot={n_convrot} "
        f"_hswq_int8_convrot={n_int8_convrot} "
        f"_hswq_nvfp4(TC arm)={n_tc_arm}"
    )
    _console(
        f"[HSWQ NVFP4][diag] Linear.forward: "
        f"parity={fwd_parity} tc_full={fwd_tc} "
        f"load: parity={load_parity} tc_full={load_tc} "
        f"_PARITY_APPLIED={_PARITY_APPLIED}"
    )
    if names:
        _console(
            "[HSWQ NVFP4][diag] sample NVFP4 ConvRot: "
            + ", ".join(names)
        )
    if names_i8:
        _console(
            "[HSWQ NVFP4][diag] sample INT8 protect ConvRot: "
            + ", ".join(names_i8)
        )
    _console("[HSWQ NVFP4][diag] =====================")


def remember_nvfp4_tc_product_stack(load_fn, mp_fn) -> None:
    """Store SDXL product TC refs (call from apply_comfy_quant_nvfp4_patches only).

    Never overwrite with parity wrappers — SDXL must always be able to restore.

    Z Image ``zi_comfy_quant_nvfp4`` also stamps ``_hswq_nvfp4_stack_ver`` /
    ``_hswq_nvfp4_full_forward`` without ``_hswq_nvfp4_comfy_only``. Treating that
    as PRODUCT poisoned SDXL INT8 after Z Image (ZI VER=8 ``int8_protect`` bake
    on ConvRot INT8 → LoRA falls off on the 3rd prompt). Require
    ``_hswq_nvfp4_product_tc`` stamped only by ``nodes/nvfp4`` SDXL product.
    """
    global _PRODUCT_LOAD, _PRODUCT_MP
    if load_fn is not None and getattr(load_fn, "_hswq_nvfp4_product_tc", False):
        if not getattr(load_fn, "_hswq_nvfp4_comfy_only", False):
            _PRODUCT_LOAD = load_fn
    if mp_fn is not None and getattr(mp_fn, "_hswq_nvfp4_product_tc", False):
        if not getattr(mp_fn, "_hswq_nvfp4_comfy_only", False):
            _PRODUCT_MP = mp_fn


def _discard_poisoned_product_refs() -> None:
    """Drop PRODUCT refs that are Z Image stack / parity mistaken for SDXL TC."""
    global _PRODUCT_LOAD, _PRODUCT_MP
    if _PRODUCT_MP is not None and not getattr(
        _PRODUCT_MP, "_hswq_nvfp4_product_tc", False
    ):
        logger.warning(
            "[HSWQ NVFP4] discarding poisoned PRODUCT_MP "
            "(not SDXL product_tc — likely Z Image stack)"
        )
        _PRODUCT_MP = None
    if _PRODUCT_LOAD is not None and not getattr(
        _PRODUCT_LOAD, "_hswq_nvfp4_product_tc", False
    ):
        logger.warning(
            "[HSWQ NVFP4] discarding poisoned PRODUCT_LOAD "
            "(not SDXL product_tc — likely Z Image stack)"
        )
        _PRODUCT_LOAD = None


def peel_non_product_nvfp4_ops(ops) -> bool:
    """Peel Z Image / comfy_parity mp+load wrappers down to stock or SDXL product_tc.

    Used when PRODUCT was never saved (INT8-only → Z Image → SDXL) so restore
    cannot reinstate TC, but ZI mp must not keep attaching VER=8 Linear bake.
    """
    changed = False
    cur = getattr(ops, "mixed_precision_ops", None)
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if getattr(cur, "_hswq_nvfp4_product_tc", False):
            if ops.mixed_precision_ops is not cur:
                ops.mixed_precision_ops = cur
                changed = True
            break
        is_foreign = bool(
            getattr(cur, "_hswq_nvfp4_comfy_only", False)
            or (
                getattr(cur, "_hswq_nvfp4_stack_ver", 0)
                and not getattr(cur, "_hswq_nvfp4_product_tc", False)
            )
        )
        if not is_foreign:
            if ops.mixed_precision_ops is not cur:
                ops.mixed_precision_ops = cur
                changed = True
            break
        nxt = getattr(cur, "_hswq_nvfp4_orig_mp", None) or getattr(
            cur, "_hswq_orig_mixed_precision_ops", None
        )
        if nxt is None:
            break
        ops.mixed_precision_ops = nxt
        changed = True
        cur = nxt

    cur_l = getattr(ops, "_load_quantized_module", None)
    seen_l: set[int] = set()
    while cur_l is not None and id(cur_l) not in seen_l:
        seen_l.add(id(cur_l))
        if getattr(cur_l, "_hswq_nvfp4_product_tc", False):
            if ops._load_quantized_module is not cur_l:
                ops._load_quantized_module = cur_l
                changed = True
            break
        is_foreign_l = bool(
            getattr(cur_l, "_hswq_nvfp4_comfy_only", False)
            or getattr(cur_l, "_hswq_int8_protect_in_load", False)
            or getattr(cur_l, "_hswq_int8_protect_arm_v2", False)
            or getattr(cur_l, "_hswq_int8_decode_patched", False)
            or (
                getattr(cur_l, "_hswq_nvfp4_full_load", False)
                and not getattr(cur_l, "_hswq_nvfp4_product_tc", False)
            )
        )
        if not is_foreign_l:
            if ops._load_quantized_module is not cur_l:
                ops._load_quantized_module = cur_l
                changed = True
            break
        nxt_l = _closure_named(cur_l, "orig_load") or _closure_named(
            cur_l, "original_load"
        )
        if nxt_l is None:
            nxt_l = getattr(cur_l, "_hswq_nvfp4_orig_load", None)
        if nxt_l is None:
            break
        ops._load_quantized_module = nxt_l
        changed = True
        cur_l = nxt_l
    return changed


def is_nvfp4_comfy_parity_active() -> bool:
    return bool(_PARITY_APPLIED)


def _closure_named(fn, name: str):
    try:
        cells = fn.__closure__ or ()
        for n, c in zip(fn.__code__.co_freevars, cells):
            if n == name:
                return c.cell_contents
    except Exception:
        return None
    return None


def _is_tc_full_load(fn) -> bool:
    """True for SDXL product TC load only (not Z Image / parity).

    Z Image also stamps ``_hswq_nvfp4_full_load`` without ``_hswq_nvfp4_product_tc``.
    Treating that as product TC left ZI VER=8 Linear bake on SDXL INT8 (3rd prompt).
    """
    return bool(
        getattr(fn, "_hswq_nvfp4_full_load", False)
        and getattr(fn, "_hswq_nvfp4_product_tc", False)
        and not getattr(fn, "_hswq_nvfp4_comfy_only", False)
    )


def _parity_load_in_chain(fn) -> bool:
    """True if comfy_parity load wrapper is already somewhere under ``fn``."""
    cur = fn
    seen = set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            return False
        seen.add(id(cur))
        if getattr(cur, "_hswq_nvfp4_comfy_only", False):
            return True
        if getattr(cur, "_hswq_int8_decode_patched", False):
            cur = _closure_named(cur, "original_load")
            continue
        if _is_tc_full_load(cur):
            cur = _closure_named(cur, "_orig_load")
            continue
        return False
    return False


def _resolve_load_under_tc(patched_load):
    """Callable under TC for parity to close over (stock Comfy or INT8 normalize).

    Peel **only** TC ``load_nvfp4_linear_module``. Keep INT8 decode wrap so
    int8protect layers still normalize ``comfy_quant`` tensors.
    Never return TC itself (ones(1) / ``_hswq_nvfp4`` arm).
    """
    if _is_tc_full_load(patched_load):
        inner = _closure_named(patched_load, "_orig_load")
        if inner is None:
            raise RuntimeError(
                "[HSWQ NVFP4] comfy_parity: TC load has no _orig_load "
                "(cannot recover Comfy / INT8 load under TC)"
            )
        if _is_tc_full_load(inner):
            raise RuntimeError(
                "[HSWQ NVFP4] comfy_parity: nested TC load; refusing"
            )
        return inner
    return patched_load


def _chain_has_int8_protect_in_load(fn) -> bool:
    """True if load chain already arms INT8 protect ConvRot after stock load."""
    cur = fn
    seen = set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            return False
        seen.add(id(cur))
        if getattr(cur, "_hswq_int8_protect_in_load", False):
            return True
        if getattr(cur, "_hswq_int8_protect_arm_v2", False):
            return True
        if getattr(cur, "_hswq_int8_decode_patched", False):
            cur = _closure_named(cur, "original_load")
            continue
        if _is_tc_full_load(cur):
            cur = _closure_named(cur, "_orig_load")
            continue
        if getattr(cur, "_hswq_nvfp4_comfy_only", False):
            return False
        return False
    return False


def _ensure_int8_protect_arm_overlay() -> None:
    """Hot-refresh: wrap current load so INT8 protect Linears get act-rotate arm.

    No-op when ``_load_quantized_module_comfy_only`` already has
    ``_hswq_int8_protect_in_load`` (fresh install path).
    """
    try:
        import comfy.ops as ops
    except Exception:
        return
    cur = ops._load_quantized_module
    if _chain_has_int8_protect_in_load(cur):
        return
    from ..nvfp4.nvfp4_conf import decode_comfy_quant_conf

    def _load_int8_protect_arm_overlay(
        module,
        super_load,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        load_extra_params=False,
    ):
        conf = decode_comfy_quant_conf(state_dict.get(f"{prefix}comfy_quant"))
        out = cur(
            module,
            super_load,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            load_extra_params=load_extra_params,
        )
        _arm_int8_protect_convrot_after_stock_load(module, conf)
        return out

    _load_int8_protect_arm_overlay._hswq_int8_protect_arm_v2 = True  # type: ignore[attr-defined]
    ops._load_quantized_module = _load_int8_protect_arm_overlay
    _console(
        "[HSWQ NVFP4] comfy_parity: INT8 protect ConvRot arm overlay installed "
        "(hot refresh; online act rotate for protect Linears)"
    )


def _unwrap_stock_forward(forward_fn):
    """Peel HSWQ TC *and* ConvRot parity wrappers until Comfy stock forward.

    After Distorch purge, Z Image reload may hit NVFP4 ``upgraded stack`` which
    wraps TC over an already-parity ``Linear.forward``. Refresh then used to
    peel only TC and re-wrap parity on top of parity → double online rotate →
    noise. Always flatten both wrapper kinds before a single parity wrap.
    """
    f = forward_fn
    for _ in range(8):
        if getattr(f, "_hswq_nvfp4_full_forward", False) or getattr(
            f, "_hswq_nvfp4_convrot_parity", False
        ):
            stock = _closure_named(f, "stock_forward")
            if stock is None:
                return None
            f = stock
            continue
        return f
    return None


def _ensure_single_parity_linear_forward(Lin) -> None:
    """Idempotent: one ConvRot parity wrap over true stock MixedPrecision forward."""
    fwd = Lin.forward
    stock = _unwrap_stock_forward(fwd)
    if stock is None:
        raise RuntimeError(
            "[HSWQ NVFP4] comfy_parity: could not unwrap Linear.forward "
            "to Comfy stock (TC/parity chain broken)"
        )
    # Already exactly parity(stock) with no nested wrappers under stock.
    if (
        getattr(fwd, "_hswq_nvfp4_convrot_parity", False)
        and not getattr(fwd, "_hswq_nvfp4_full_forward", False)
        and _closure_named(fwd, "stock_forward") is stock
        and not getattr(stock, "_hswq_nvfp4_full_forward", False)
        and not getattr(stock, "_hswq_nvfp4_convrot_parity", False)
    ):
        return
    Lin.forward = _make_convrot_parity_forward(stock)


def _is_int8_tensorwise_convrot_conf(conf) -> bool:
    """True for INT8 protect Linear layers stamped with ConvRot offline rotate."""
    from .zi_nvfp4_conf import int8_convrot_flags_from_conf

    enabled, _gs = int8_convrot_flags_from_conf(conf)
    return bool(enabled)


def _make_convrot_parity_forward(stock_forward):
    """Stock MixedPrecision forward + online act rotate for ConvRot Linears.

    NVFP4: ``_hswq_nvfp4_convrot`` (Params.convrot cleared at load).
    INT8 protect: ``_hswq_int8_convrot`` (Params.convrot cleared at load —
    same as Conv2d). Kitchen must **not** see Params.convrot=True or
    int8_linear double-rotates with this path.
    """
    from .zi_nvfp4_hadamard import build_hadamard, rotate_last_dim

    def forward_parity(self, input, *args, **kwargs):
        nv = bool(getattr(self, "_hswq_nvfp4_convrot", False))
        i8 = bool(getattr(self, "_hswq_int8_convrot", False))
        if nv or i8:
            if nv:
                gs = int(getattr(self, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
            else:
                gs = int(getattr(self, "_hswq_int8_convrot_groupsize", 256) or 256)
            h = getattr(self, "_hswq_nvfp4_parity_H", None)
            need_rebuild = True
            if h is not None:
                try:
                    from .zi_nvfp4_hadamard import _tensor_storage_ok

                    # Global cache already rejects poisoned H via
                    # _tensor_storage_ok; module-local H must use the same
                    # check. nbytes==0 alone misses emptied shells that still
                    # report device/dtype (Distorch Method 3), so 2nd+ gen
                    # rotates with garbage and quality decays.
                    need_rebuild = (
                        h.device != input.device
                        or h.dtype != input.dtype
                        or (bool(input.is_cuda) and not bool(h.is_cuda))
                        or not _tensor_storage_ok(h)
                    )
                except Exception:
                    need_rebuild = True
            if need_rebuild:
                h = build_hadamard(gs, device=input.device, dtype=input.dtype)
                self._hswq_nvfp4_parity_H = h
            input = rotate_last_dim(input, h, gs)
        return stock_forward(self, input, *args, **kwargs)

    forward_parity._hswq_nvfp4_convrot_parity = True  # type: ignore[attr-defined]
    return forward_parity


def _arm_convrot_after_stock_load(module, conf) -> None:
    global _LOAD_NVFP4_SEEN, _LOAD_CONVROT_ARMED, _LOAD_NVFP4_NO_CONVROT
    from ..nvfp4.nvfp4_conf import convrot_flags_from_conf, is_nvfp4_conf

    if not is_nvfp4_conf(conf):
        return
    _LOAD_NVFP4_SEEN += 1
    enabled, gs = convrot_flags_from_conf(conf)
    module._hswq_nvfp4_convrot = bool(enabled)
    module._hswq_nvfp4_convrot_groupsize = int(gs)
    try:
        import comfy.quant_ops as quant_ops

        p = getattr(module, "weight", None)
        layout = getattr(p, "layout_params", None) if p is not None else None
        if isinstance(layout, quant_ops.Params) and getattr(layout, "convrot", False):
            layout.convrot = False
    except Exception:
        pass
    if enabled:
        _LOAD_CONVROT_ARMED += 1
        if _LOAD_CONVROT_ARMED <= 4 or _LOAD_CONVROT_ARMED % 40 == 0:
            fmt = conf.get("format")
            top = conf.get("convrot")
            params = conf.get("params") if isinstance(conf.get("params"), dict) else {}
            _console(
                f"[HSWQ NVFP4][diag] arm ConvRot #{_LOAD_CONVROT_ARMED} "
                f"gs={gs} format={fmt} convrot={top!r} "
                f"params.convrot={params.get('convrot')!r}"
            )
    else:
        _LOAD_NVFP4_NO_CONVROT += 1
        if _LOAD_NVFP4_NO_CONVROT <= 4:
            _console(
                f"[HSWQ NVFP4][diag] nvfp4 without convrot "
                f"(#{_LOAD_NVFP4_NO_CONVROT}) keys={list(conf.keys())[:12]}"
            )
    # Do not set _hswq_nvfp4 (TC full-forward arm).


def _clear_int8_qt_params_convrot(module) -> bool:
    """Force ``Params.convrot=False`` on INT8 QT weight (Conv2d-style).

    ``layout_params`` on Parameter is unreliable for QuantizedTensor; clear
    ``qt._params`` via dataclasses.replace. Leaving Params.convrot=True while
    ``_hswq_int8_convrot`` is set double-rotates acts (parity + kitchen).
    """
    import dataclasses

    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    w = getattr(module, "weight", None)
    qt = w if isinstance(w, QuantizedTensor) else getattr(w, "data", None)
    if qt is None or not isinstance(qt, QuantizedTensor):
        return False
    params = getattr(qt, "_params", None)
    if params is None or not bool(getattr(params, "convrot", False)):
        return False
    new_params = dataclasses.replace(params, convrot=False)
    try:
        object.__setattr__(qt, "_params", new_params)
        return True
    except Exception:
        pass
    try:
        qt._params = new_params
        return True
    except Exception:
        return False


def _arm_int8_protect_convrot_after_stock_load(module, conf) -> None:
    """Arm INT8 protect ConvRot Linear like Conv2d: flag + clear Params.convrot.

    Kitchen ``dequantize_int8_convrot_weight`` already unrotates when
    Params.convrot=True — LoRA bake must see Params=False so convert gets
    rotated-basis float and unrotates once. Online act rotate is parity
    (``_hswq_int8_convrot``). Requant must keep Params.convrot=False
    (see ``zi_nvfp4_forward`` set_weight).
    """
    global _LOAD_INT8_CONVROT_ARMED
    from .zi_nvfp4_conf import int8_convrot_flags_from_conf

    enabled, gs = int8_convrot_flags_from_conf(conf)
    if not enabled:
        return
    module._hswq_int8_convrot = True
    module._hswq_int8_convrot_groupsize = int(gs)
    cleared = _clear_int8_qt_params_convrot(module)
    _LOAD_INT8_CONVROT_ARMED += 1
    if _LOAD_INT8_CONVROT_ARMED <= 4 or _LOAD_INT8_CONVROT_ARMED % 20 == 0:
        _console(
            f"[HSWQ NVFP4][diag] arm INT8 protect ConvRot "
            f"#{_LOAD_INT8_CONVROT_ARMED} gs={gs} params_cleared={cleared}"
        )



def require_convrot_parity_forward() -> None:
    """Fail fast if TC full-forward is still installed (bench guard)."""
    import comfy.ops as ops

    mp = ops.mixed_precision_ops()
    fwd = mp.Linear.forward
    if getattr(fwd, "_hswq_nvfp4_full_forward", False):
        raise RuntimeError(
            "ConvRot NVFP4 parity requires stock Comfy Linear.forward + act rotate; "
            "HSWQ TC full-forward is still installed (_hswq_nvfp4_full_forward)."
        )
    if not getattr(fwd, "_hswq_nvfp4_convrot_parity", False):
        raise RuntimeError(
            "ConvRot NVFP4 parity forward missing "
            "(_hswq_nvfp4_convrot_parity). Call apply_nvfp4_comfy_parity()."
        )


def restore_nvfp4_tc_product_stack() -> bool:
    """Put SDXL product TC load + forward back; peel Z Image if PRODUCT missing.

    Z Image stamps ``_hswq_nvfp4_stack_ver`` / ``_hswq_nvfp4_full_forward`` without
    ``_hswq_nvfp4_product_tc``. Treating that as ``already_tc`` left ZI VER=8
    ``int8_protect`` Linear bake on SDXL INT8 after Z Image (LoRA falls off on
    the 3rd prompt). Only ``product_tc`` counts as product TC.
    """
    global _PARITY_APPLIED
    try:
        import comfy.ops as ops
    except Exception as e:
        logger.warning("[HSWQ NVFP4] restore TC stack skipped: %s", e)
        return False

    _discard_poisoned_product_refs()

    mp = ops.mixed_precision_ops
    load_fn = ops._load_quantized_module
    already_product = bool(
        getattr(mp, "_hswq_nvfp4_product_tc", False)
        and getattr(load_fn, "_hswq_nvfp4_product_tc", False)
        and not getattr(mp, "_hswq_nvfp4_comfy_only", False)
        and not getattr(load_fn, "_hswq_nvfp4_comfy_only", False)
    )
    if already_product and not _PARITY_APPLIED:
        return True

    if _PRODUCT_LOAD is not None and _PRODUCT_MP is not None:
        ops._load_quantized_module = _PRODUCT_LOAD
        ops.mixed_precision_ops = _PRODUCT_MP
        _PARITY_APPLIED = False
        _console("[HSWQ NVFP4] restored product TC stack (SDXL path; parity off)")
        return True

    # INT8-only → Z Image → SDXL: PRODUCT was never saved. Peel ZI / parity so
    # SDXL INT8 does not keep attaching VER=8 ``[HSWQ ConvRot LoRA] int8_protect``.
    peeled = peel_non_product_nvfp4_ops(ops)
    _PARITY_APPLIED = False
    if peeled:
        if (
            getattr(ops.mixed_precision_ops, "_hswq_nvfp4_product_tc", False)
            and getattr(ops._load_quantized_module, "_hswq_nvfp4_product_tc", False)
        ):
            remember_nvfp4_tc_product_stack(
                ops._load_quantized_module, ops.mixed_precision_ops
            )
            _console(
                "[HSWQ NVFP4] restored product TC stack via peel "
                "(SDXL path; parity off)"
            )
        else:
            _console(
                "[HSWQ NVFP4] peeled non-product NVFP4 ops "
                "(stock/INT8 base; no product_tc PRODUCT — SDXL INT8 LoRA safe)"
            )
        return True

    logger.warning(
        "[HSWQ NVFP4] restore TC stack: no saved product refs "
        "(SDXL needs apply_comfy_quant_nvfp4_patches first)"
    )
    return False


def apply_nvfp4_comfy_parity() -> bool:
    """Switch NVFP4 Linear path to stock Comfy GEMM + online act rotate.

    Also registers aten.addmm for TensorCoreNVFP4Layout (kitchen gap).
    Saves product TC refs so SDXL can restore later.
    """
    global _PARITY_APPLIED, _PRODUCT_LOAD, _PRODUCT_MP
    try:
        import comfy.ops as ops
        from comfy.quant_ops import QUANT_ALGOS
    except Exception as e:
        logger.warning("[HSWQ NVFP4] comfy_parity import failed: %s", e)
        return False

    from .nvfp4_addmm_patch import register_nvfp4_addmm_handler
    from ..nvfp4.nvfp4_conf import decode_comfy_quant_conf, is_nvfp4_conf
    from .zi_nvfp4_forward import attach_nvfp4_linear_lora_bake
    # Product Z Image: keep ConvRot Linear LoRA bake (same as SDXL). Do not peel.

    register_nvfp4_addmm_handler()

    if "nvfp4" not in QUANT_ALGOS:
        logger.warning("[HSWQ NVFP4] comfy_parity: nvfp4 not in QUANT_ALGOS")
        return False

    patched_load = ops._load_quantized_module
    # Prefer refs already saved by apply_comfy_quant_nvfp4_patches (TC only).
    remember_nvfp4_tc_product_stack(patched_load, ops.mixed_precision_ops)

    def _parity_mp_base(mp_fn):
        """Innermost non-parity ``mixed_precision_ops`` (TC / product stack).

        Refresh used to wrap the previous refresh wrapper every reload, stacking
        ``attach_nvfp4_linear_lora_bake`` / forward ensure. Peel to ``_orig_mp``.
        """
        cur = mp_fn
        seen: set[int] = set()
        while id(cur) not in seen:
            seen.add(id(cur))
            if not getattr(cur, "_hswq_nvfp4_comfy_only", False):
                return cur
            nxt = getattr(cur, "_hswq_nvfp4_orig_mp", None)
            if nxt is None or nxt is cur:
                return cur
            cur = nxt
        return cur

    def _refresh_parity_mp() -> None:
        _cur_mp = ops.mixed_precision_ops
        _base_mp = _parity_mp_base(_cur_mp)

        def mixed_precision_ops_parity_refresh(*args, **kwargs):
            mp = _base_mp(*args, **kwargs)
            Lin = mp.Linear
            attach_nvfp4_linear_lora_bake(Lin)
            _ensure_single_parity_linear_forward(Lin)
            return mp

        mixed_precision_ops_parity_refresh._hswq_nvfp4_comfy_only = True  # type: ignore[attr-defined]
        mixed_precision_ops_parity_refresh._hswq_nvfp4_stack_ver = getattr(
            _base_mp, "_hswq_nvfp4_stack_ver", getattr(_cur_mp, "_hswq_nvfp4_stack_ver", 0)
        )  # type: ignore[attr-defined]
        mixed_precision_ops_parity_refresh._hswq_nvfp4_orig_mp = _base_mp  # type: ignore[attr-defined]
        ops.mixed_precision_ops = mixed_precision_ops_parity_refresh

    # Already on parity load (possibly under INT8 decode wrap): keep load chain.
    if _parity_load_in_chain(patched_load):
        _ensure_int8_protect_arm_overlay()
        _refresh_parity_mp()
        _PARITY_APPLIED = True
        _console(
            "[HSWQ NVFP4] comfy_parity refresh: stock GEMM + act rotate "
            "(NVFP4 + INT8 protect) + ConvRot Linear LoRA bake (Z Image)"
        )
        return True

    orig_load = _resolve_load_under_tc(patched_load)

    def _load_quantized_module_comfy_only(
        module,
        super_load,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        load_extra_params=False,
    ):
        conf = decode_comfy_quant_conf(state_dict.get(f"{prefix}comfy_quant"))
        out = orig_load(
            module,
            super_load,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            load_extra_params=load_extra_params,
        )
        if is_nvfp4_conf(conf):
            _arm_convrot_after_stock_load(module, conf)
        else:
            _arm_int8_protect_convrot_after_stock_load(module, conf)
        return out

    _load_quantized_module_comfy_only._hswq_nvfp4_comfy_only = True  # type: ignore[attr-defined]
    _load_quantized_module_comfy_only._hswq_int8_protect_in_load = True  # type: ignore[attr-defined]
    # Bench marks full_load on the parity wrapper too; keep comfy_only distinct
    # so remember_nvfp4_tc_product_stack never stores this as SDXL TC.
    ops._load_quantized_module = _load_quantized_module_comfy_only

    _cur_mp = ops.mixed_precision_ops
    _base_install = _parity_mp_base(_cur_mp)

    def mixed_precision_ops_comfy_only(*args, **kwargs):
        mp = _base_install(*args, **kwargs)
        Lin = mp.Linear
        attach_nvfp4_linear_lora_bake(Lin)
        _ensure_single_parity_linear_forward(Lin)
        return mp

    mixed_precision_ops_comfy_only._hswq_nvfp4_comfy_only = True  # type: ignore[attr-defined]
    mixed_precision_ops_comfy_only._hswq_nvfp4_stack_ver = getattr(
        _base_install, "_hswq_nvfp4_stack_ver", 0
    )  # type: ignore[attr-defined]
    mixed_precision_ops_comfy_only._hswq_nvfp4_orig_mp = _base_install  # type: ignore[attr-defined]
    ops.mixed_precision_ops = mixed_precision_ops_comfy_only

    # Prove unwrap once at install; keep LoRA bake attached for product use.
    mp0 = _base_install()
    attach_nvfp4_linear_lora_bake(mp0.Linear)
    _ensure_single_parity_linear_forward(mp0.Linear)

    _PARITY_APPLIED = True
    _console(
        "[HSWQ NVFP4] comfy_parity ON: stock MixedPrecision GEMM + online act rotate "
        "(NVFP4 ConvRot + INT8 protect ConvRot) "
        "+ ConvRot Linear LoRA bake (Z Image; not HSWQ TC Linear.forward)"
    )
    return True
```

### `nodes/zimage_nvfp4/nvfp4_lora_bake.py`

```python
"""Z Image mixed-pack LoRA bake — Dynamic VRAM only (branch under zimage_nvfp4).

Problem (owner A/B + logs):
  Without LoRA, comfy_parity + act_rotate is fine.
  With LoRA, ModelPatcherDynamic attaches LowVramPatch (``180 patches``).

INT8 Dynamic bake (``patches/comfy_quant_int8.py``) often does **not** fire on this
hybrid pack (no INT8 bake dump in logs), so INT8-protect keys stay as
LowVramPatch. NVFP4 ConvRot bake alone leaves ``patches_left=60`` → broken.

v3: ENTER proved wrap fires; bake still silent (``nvfp4_convrot=False``).

v4 (owner: まだ駄目だ — ENTER patches=180 nvfp4_convrot=False):
  Root cause: kitchen ``QuantizedTensor`` inherits ``torch.Tensor.layout``
  (``torch.strided`` → type name ``\"layout\"``). Old ``_qt_layout_name``
  read ``qt.layout`` first and never saw ``_layout_cls``
  (``TensorCoreNVFP4Layout``), so ``_qt_is_nvfp4`` was always False → gate
  closed → no bake dump → LoRA LowVramPatch left attached.
  Fix: prefer ``_layout_cls`` / ``layout_cls``; gate on ConvRot **flag**
  (do not require QT on ``module.weight`` under Dynamic VRAM).

Does **not** edit ``nodes/nvfp4`` (SDXL).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_BAKE_HOOK_VER = 7
_STATUS_LOGS = 0
_STATUS_LOG_MAX = 24
_ENTER_LOGS = 0
_ENTER_LOG_MAX = 24
_SKIP_SAMPLE_LOGS = 0
_SKIP_SAMPLE_MAX = 6
_GPU_BAKE_INSTALLED = False


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def _qt_payload(weight, QuantizedTensor):
    if weight is None:
        return None
    if isinstance(weight, QuantizedTensor):
        return weight
    data = getattr(weight, "data", None)
    if data is not None and isinstance(data, QuantizedTensor):
        return data
    return None


def _qt_layout_name(qt) -> str:
    """Kitchen QT layout class name.

    Do **not** use ``qt.layout`` — that is ``torch.Tensor.layout``
    (``torch.strided``), whose type name is literally ``\"layout\"``.
    Real name lives in ``_layout_cls`` (str) / ``layout_cls`` (type).
    """
    if qt is None:
        return ""
    layout_cls = getattr(qt, "_layout_cls", None)
    if isinstance(layout_cls, str) and layout_cls:
        return layout_cls
    layout_cls_t = getattr(qt, "layout_cls", None)
    if layout_cls_t is not None and not isinstance(layout_cls_t, str):
        name = getattr(layout_cls_t, "__name__", "") or ""
        if name:
            return name
    # Legacy object layout (not torch.layout)
    legacy = getattr(qt, "_layout", None)
    if legacy is not None:
        name = type(legacy).__name__ or ""
        if name and name != "layout":
            return name
    return ""


def _qt_is_nvfp4(weight, QuantizedTensor) -> bool:
    qt = _qt_payload(weight, QuantizedTensor)
    if qt is None:
        return False
    name = _qt_layout_name(qt)
    return "NVFP4" in name or "nvfp4" in name.lower()


def _qt_is_int8_tensorwise(weight, QuantizedTensor) -> bool:
    """INT8 detect including ``_layout_cls`` string (kitchen / protect packs)."""
    qt = _qt_payload(weight, QuantizedTensor)
    if qt is None:
        return False
    name = _qt_layout_name(qt)
    return "TensorWiseINT8" in name or "int8_tensorwise" in name.lower()


def _module_is_nvfp4_convrot(module) -> bool:
    return bool(
        getattr(module, "_hswq_nvfp4_convrot", False)
        or getattr(module, "_hswq_nvfp4_convrot_parity", False)
    )


def _get_baked_key_set(model) -> set:
    keys = getattr(model, "_hswq_zi_nvfp4_baked_keys", None)
    if keys is None:
        keys = set()
        model._hswq_zi_nvfp4_baked_keys = keys
    return keys


def _nvfp4_convrot_diag(model) -> dict:
    """Count ConvRot-armed modules and how many still expose NVFP4 on ``.weight``."""
    out = {"flagged": 0, "qt_on_weight": 0, "has": False}
    if model is None:
        return out
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        QuantizedTensor = None
    for _name, module in model.named_modules():
        if not _module_is_nvfp4_convrot(module):
            continue
        out["flagged"] += 1
        if QuantizedTensor is None:
            continue
        w = getattr(module, "weight", None)
        if _qt_is_nvfp4(w, QuantizedTensor):
            out["qt_on_weight"] += 1
    out["has"] = out["flagged"] > 0
    return out


def _model_has_nvfp4_convrot(model) -> bool:
    """True if any module was armed with ConvRot NVFP4 (``_hswq_nvfp4_convrot``).

    Do **not** require QT on ``module.weight``: under Dynamic VRAM / LowVramPatch
    the QT often lives behind ``get_key_weight``, while the flag remains on the
    module (act_rotate still hits). v3 gate required both → always False.
    """
    return bool(_nvfp4_convrot_diag(model)["has"])


def _resolve_module(model, module_path: str):
    try:
        import comfy.utils as cu

        return cu.get_attr(model, module_path)
    except Exception:
        return None


def _bake_keys_on_module(patcher, module, keys_to_bake, device_to, already) -> int:
    """Clear LowVramPatch, patch_weight_to_device, drop backup+patches. Keep ``_v``."""
    baked = 0
    for param_key, _key in keys_to_bake:
        if hasattr(module, param_key + "_lowvram_function"):
            setattr(module, param_key + "_lowvram_function", None)
    for _param_key, key in keys_to_bake:
        patcher.patch_weight_to_device(key, device_to=device_to)
        if key in patcher.backup:
            try:
                del patcher.backup[key]
            except KeyError:
                pass
        try:
            del patcher.patches[key]
        except KeyError:
            pass
        already.add(key)
        baked += 1
    return baked


def _iter_patch_weight_keys(patcher):
    """Yield (key, module_path, param_key, module) for weight/bias patches."""
    patches = getattr(patcher, "patches", None) or {}
    model = getattr(patcher, "model", None)
    if model is None or not patches:
        return
    for key in list(patches.keys()):
        if not (key.endswith(".weight") or key.endswith(".bias")):
            continue
        module_path, param_key = key.rsplit(".", 1)
        module = _resolve_module(model, module_path)
        if module is None:
            continue
        yield key, module_path, param_key, module


def bake_nvfp4_convrot_patches_on_dynamic_patcher(patcher, device_to) -> dict:
    """Bake LoRA into ConvRot NVFP4 Linears after ModelPatcherDynamic.load."""
    stats = {
        "baked_nvfp4": 0,
        "candidates": 0,
        "skipped_no_set": 0,
        "skipped_not_nvfp4": 0,
        "skipped_not_convrot": 0,
        "cleared_already": 0,
        "unresolved": 0,
        "sample_nvfp4_keys": [],
    }
    if not getattr(patcher, "patches", None):
        return stats
    try:
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return stats

    global _SKIP_SAMPLE_LOGS
    already = _get_baked_key_set(patcher.model)
    uuid = getattr(patcher, "patches_uuid", None)
    prev_uuid = getattr(patcher.model, "_hswq_zi_nvfp4_baked_uuid", None)
    if prev_uuid is not None and prev_uuid != uuid:
        already.clear()

    # Group keys by module so LowVramPatch clear happens once per module.
    by_module: dict[str, list] = {}
    modules: dict[str, object] = {}
    for key, module_path, param_key, module in _iter_patch_weight_keys(patcher):
        stats["candidates"] += 1
        if key in already:
            attr = param_key + "_lowvram_function"
            if getattr(module, attr, None) is not None:
                setattr(module, attr, None)
            try:
                del patcher.patches[key]
            except KeyError:
                pass
            stats["cleared_already"] += 1
            continue
        if not _module_is_nvfp4_convrot(module):
            stats["skipped_not_convrot"] += 1
            if _SKIP_SAMPLE_LOGS < _SKIP_SAMPLE_MAX:
                w, _, _ = mp.get_key_weight(patcher.model, key)
                qt = _qt_payload(w, QuantizedTensor)
                _SKIP_SAMPLE_LOGS += 1
                params = getattr(qt, "_params", None) if qt is not None else None
                params_convrot = bool(getattr(params, "convrot", False)) if params else False
                _console(
                    f"[HSWQ ZI NVFP4 LoRA] nv_pass_defer_int8_rem sample "
                    f"#{_SKIP_SAMPLE_LOGS}: {key} layout={_qt_layout_name(qt)!r} "
                    f"nvfp4_convrot={getattr(module, '_hswq_nvfp4_convrot', False)} "
                    f"int8_convrot={getattr(module, '_hswq_int8_convrot', False)} "
                    f"params_convrot={params_convrot} "
                    f"(not a failure — baked in INT8 rem pass)"
                )
            continue
        weight, set_func, _convert_func = mp.get_key_weight(patcher.model, key)
        if weight is None:
            continue
        if not _qt_is_nvfp4(weight, QuantizedTensor):
            stats["skipped_not_nvfp4"] += 1
            continue
        if set_func is None:
            stats["skipped_no_set"] += 1
            _console(
                f"[HSWQ ZI NVFP4 LoRA] WARN cannot bake {key}: "
                "NVFP4 QT but no set_weight"
            )
            continue
        by_module.setdefault(module_path, []).append((param_key, key))
        modules[module_path] = module

    for module_path, keys_to_bake in by_module.items():
        n = _bake_keys_on_module(
            patcher, modules[module_path], keys_to_bake, device_to, already
        )
        stats["baked_nvfp4"] += n
        if n > 0 and len(stats["sample_nvfp4_keys"]) < 3:
            for _pk, full_key in keys_to_bake:
                if full_key not in stats["sample_nvfp4_keys"]:
                    stats["sample_nvfp4_keys"].append(full_key)
                if len(stats["sample_nvfp4_keys"]) >= 3:
                    break

    if stats["baked_nvfp4"] > 0:
        patcher.model._hswq_zi_nvfp4_baked_uuid = uuid

    return stats


def bake_remaining_quant_patches_on_dynamic_patcher(patcher, device_to) -> dict:
    """Bake leftover QT LoRA (ConvRot INT8 protect etc.) that NVFP4 pass skipped.

    Hybrid packs: NVFP4 ConvRot is baked first; INT8 protect ConvRot Linears
    use ``_hswq_int8_convrot`` + cleared Params (Conv2d twin) via
    ``Linear.convert_weight`` / ``set_weight`` (``_NVFP4_LORA_BAKE_VER`` >= 5).
    """
    stats = {
        "baked_int8": 0,
        "baked_other_qt": 0,
        "candidates": 0,
        "skipped_no_set": 0,
        "skipped_not_qt": 0,
        "cleared_already": 0,
        "sample_int8_keys": [],
    }
    if not getattr(patcher, "patches", None):
        return stats
    try:
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return stats

    already = _get_baked_key_set(patcher.model)
    uuid = getattr(patcher, "patches_uuid", None)

    by_module: dict[str, list] = {}
    modules: dict[str, object] = {}
    kinds: dict[str, str] = {}
    for key, module_path, param_key, module in _iter_patch_weight_keys(patcher):
        stats["candidates"] += 1
        if key in already:
            attr = param_key + "_lowvram_function"
            if getattr(module, attr, None) is not None:
                setattr(module, attr, None)
            try:
                del patcher.patches[key]
            except KeyError:
                pass
            stats["cleared_already"] += 1
            continue
        weight, set_func, _convert_func = mp.get_key_weight(patcher.model, key)
        if weight is None:
            continue
        qt = _qt_payload(weight, QuantizedTensor)
        if qt is None:
            stats["skipped_not_qt"] += 1
            continue
        if set_func is None:
            stats["skipped_no_set"] += 1
            _console(
                f"[HSWQ ZI NVFP4 LoRA] WARN cannot bake leftover {key}: "
                f"QT layout={_qt_layout_name(qt)!r} but no set_weight"
            )
            continue
        if module_path not in kinds:
            if _qt_is_int8_tensorwise(weight, QuantizedTensor):
                kinds[module_path] = "int8"
            elif _qt_is_nvfp4(weight, QuantizedTensor):
                kinds[module_path] = "nvfp4"
            else:
                kinds[module_path] = "other"
        by_module.setdefault(module_path, []).append((param_key, key))
        modules[module_path] = module

    for module_path, keys_to_bake in by_module.items():
        n = _bake_keys_on_module(
            patcher, modules[module_path], keys_to_bake, device_to, already
        )
        if kinds.get(module_path) == "int8":
            stats["baked_int8"] += n
            if len(stats["sample_int8_keys"]) < 3:
                for _pk, full_key in keys_to_bake:
                    if full_key not in stats["sample_int8_keys"]:
                        stats["sample_int8_keys"].append(full_key)
                    if len(stats["sample_int8_keys"]) >= 3:
                        break
        else:
            stats["baked_other_qt"] += n

    if stats["baked_int8"] > 0 or stats["baked_other_qt"] > 0:
        patcher.model._hswq_zi_nvfp4_baked_uuid = uuid
        if stats["baked_int8"] > 0:
            patcher.model._hswq_int8_baked_uuid = uuid

    return stats


def _dump_bake_status(
    nv_stats: dict,
    rem_stats: dict,
    patcher,
    reason: str,
    counters_before: dict | None = None,
) -> None:
    global _STATUS_LOGS
    nv_n = int(nv_stats.get("baked_nvfp4", 0) or 0)
    i8 = int(rem_stats.get("baked_int8", 0) or 0)
    # Empty re-bake / VAE: do not spam status or stale EVIDENCE.
    if nv_n == 0 and i8 == 0 and int(rem_stats.get("baked_other_qt", 0) or 0) == 0:
        return
    if _STATUS_LOGS >= _STATUS_LOG_MAX:
        return
    _STATUS_LOGS += 1
    left = len(getattr(patcher, "patches", None) or {})
    skip_i8_in_nv_pass = int(nv_stats.get("skipped_not_convrot", 0) or 0)
    uuid = getattr(patcher, "patches_uuid", None)
    uuid_s = f"{uuid}"[:8] if uuid is not None else "-"
    _console(
        "[HSWQ ZI NVFP4 LoRA] Dynamic.load bake "
        f"#{_STATUS_LOGS} ({reason}): "
        f"nvfp4_baked={nv_n} "
        f"int8_baked={i8} "
        f"other_qt_baked={rem_stats.get('baked_other_qt', 0)} "
        f"nv_candidates={nv_stats.get('candidates', 0)} "
        f"rem_candidates={rem_stats.get('candidates', 0)} "
        f"nv_pass_skip_int8_rem={skip_i8_in_nv_pass} "
        f"(INT8 rem baked separately as int8_baked) "
        f"patches_left={left} patches_uuid={uuid_s}"
    )
    try:
        from .zi_nvfp4_forward import log_nvfp4_lora_bake_evidence

        log_nvfp4_lora_bake_evidence(
            tag=f"bake#{_STATUS_LOGS}/{reason}",
            before=counters_before,
            nvfp4_baked=nv_n,
            int8_baked=i8,
            sample_nvfp4_keys=list(nv_stats.get("sample_nvfp4_keys") or []),
            sample_int8_keys=list(rem_stats.get("sample_int8_keys") or []),
        )
    except Exception as e:
        _console(f"[HSWQ ConvRot LoRA] EVIDENCE log failed: {e}")
    if left > 0:
        sample = list((getattr(patcher, "patches", None) or {}).keys())[:4]
        _console(
            f"[HSWQ ZI NVFP4 LoRA] WARN patches_left={left} after bake "
            f"sample_keys={sample}"
        )


def _patcher_has_quant_via_keys(patcher) -> bool:
    """True if any LoRA patch key resolves to NVFP4/INT8 QT via get_key_weight."""
    if not getattr(patcher, "patches", None):
        return False
    try:
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    for key, _module_path, _param_key, _module in _iter_patch_weight_keys(patcher):
        weight, _set_func, _convert = mp.get_key_weight(patcher.model, key)
        if weight is None:
            continue
        if _qt_is_nvfp4(weight, QuantizedTensor) or _qt_is_int8_tensorwise(
            weight, QuantizedTensor
        ):
            return True
    return False


def run_zimage_nvfp4_lora_bake_on_patcher(patcher, device_to=None, reason: str = "wrap") -> bool:
    """Bake NVFP4 ConvRot + leftover QT if this patcher is a ZI NVFP4 pack with LoRA."""
    model = getattr(patcher, "model", None)
    if model is None:
        return False
    diag = _nvfp4_convrot_diag(model)
    has_flag = bool(diag["has"])
    has_baked = bool(getattr(model, "_hswq_zi_nvfp4_baked_keys", None))
    n_patches = len(getattr(patcher, "patches", None) or {})
    if not has_flag and not has_baked:
        # Fallback: patches present and QT visible via get_key_weight
        if n_patches == 0 or not _patcher_has_quant_via_keys(patcher):
            return False
    if n_patches == 0 and not has_baked:
        return False
    if device_to is None:
        device_to = getattr(patcher, "load_device", None)
    try:
        from .zi_nvfp4_forward import snapshot_nvfp4_lora_bake_counters

        counters_before = snapshot_nvfp4_lora_bake_counters()
    except Exception:
        counters_before = None
    nv_stats = bake_nvfp4_convrot_patches_on_dynamic_patcher(patcher, device_to=device_to)
    rem_stats = bake_remaining_quant_patches_on_dynamic_patcher(
        patcher, device_to=device_to
    )
    _dump_bake_status(
        nv_stats, rem_stats, patcher, reason=reason, counters_before=counters_before
    )
    return True


def _unwrap_to_non_zi_load(load_fn):
    """Walk past our ZI wraps so reinstall does not nest ZI→ZI."""
    cur = load_fn
    seen = set()
    while (
        cur is not None
        and id(cur) not in seen
        and getattr(cur, "_hswq_zi_nvfp4_lora_bake", False)
    ):
        seen.add(id(cur))
        nxt = getattr(cur, "_hswq_zi_nvfp4_prev_dynamic_load", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def _chain_has_zi_dynamic_load(load_fn) -> bool:
    """True if any ZI wrap remains in prev / ``_hswq_orig_dynamic_load`` chain."""
    cur = load_fn
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if getattr(cur, "_hswq_zi_nvfp4_lora_bake", False):
            return True
        nxt = getattr(cur, "_hswq_zi_nvfp4_prev_dynamic_load", None)
        if nxt is None:
            nxt = getattr(cur, "_hswq_orig_dynamic_load", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return False


def _deep_clean_dynamic_load(load_fn):
    """Peel ZI wraps and discard INT8 wraps that closed over a ZI ``true_orig``.

    Owner log after peel-only uninstall: ``Dynamic.load bake hook OFF`` then still
    ``[HSWQ ZI NVFP4 LoRA] Dynamic.load ENTER … model=SDXL``. Cause: while ZI was
    outermost, INT8 re-patched with ``true_orig = ZI_wrap`` (closure). Peeling the
    outer ZI left INT8→ZI→… so ENTER still fires and SDXL LoRA strength is wrong.

    Returns ``(cleaned_load, discarded_contaminated_int8)``.
    """
    cur = load_fn
    discarded_int8 = False
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if getattr(cur, "_hswq_zi_nvfp4_lora_bake", False):
            nxt = getattr(cur, "_hswq_zi_nvfp4_prev_dynamic_load", None)
            if nxt is None or nxt is cur:
                break
            cur = nxt
            continue
        if getattr(cur, "_hswq_int8_lora_bake", False):
            captured = getattr(cur, "_hswq_orig_dynamic_load", None)
            if _chain_has_zi_dynamic_load(captured):
                # Closure already bound to ZI — attribute rewrite cannot fix it.
                discarded_int8 = True
                cur = _unwrap_to_non_zi_load(captured)
                continue
            break
        break
    return cur, discarded_int8


def uninstall_zimage_nvfp4_lora_bake() -> bool:
    """Remove Z Image Dynamic / load_models_gpu bake hooks (SDXL must not inherit them).

    After Z Image ConvRot NVFP4, these hooks stay on ``ModelPatcherDynamic.load``
    and bake SDXL LoRA with the ZI path (INT8 protect / other_qt) → noise.
    Also discards INT8 Dynamic.load wraps that captured ZI as ``true_orig``
    (logs still show ENTER on SDXL after OFF; LoRA keys apply but strength is weak).
    Call from SDXL loaders before applying the SDXL TC / INT8 stacks.
    """
    global _GPU_BAKE_INSTALLED
    removed = False
    need_int8_repatch = False
    try:
        import comfy.model_patcher as mp
    except ImportError:
        mp = None
    if mp is not None:
        Dynamic = getattr(mp, "ModelPatcherDynamic", None)
        if Dynamic is not None:
            cur = getattr(Dynamic, "load", None)
            cleaned, discarded_int8 = _deep_clean_dynamic_load(cur)
            if (
                cleaned is not cur
                or discarded_int8
                or getattr(cur, "_hswq_zi_nvfp4_lora_bake", False)
            ):
                if cleaned is not None:
                    Dynamic.load = cleaned
                removed = True
                need_int8_repatch = bool(discarded_int8)
                _console(
                    "[HSWQ ZI NVFP4 LoRA] Dynamic.load bake hook OFF "
                    "(restored for SDXL; Z Image path no longer wraps bake)"
                    + (
                        "; discarded INT8 wrap that captured ZI true_orig"
                        if discarded_int8
                        else ""
                    )
                )
    try:
        import comfy.model_management as mm
    except ImportError:
        mm = None
    if mm is not None:
        cur_gpu = getattr(mm, "load_models_gpu", None)
        if getattr(cur_gpu, "_hswq_zi_nvfp4_gpu_bake", False):
            mm.load_models_gpu = _unwrap_to_non_zi_load_models_gpu(cur_gpu)
            _GPU_BAKE_INSTALLED = False
            removed = True
            _console(
                "[HSWQ ZI NVFP4 LoRA] load_models_gpu bake hook OFF "
                "(restored for SDXL)"
            )
    if need_int8_repatch:
        try:
            from ...patches.comfy_quant_int8 import (
                _patch_model_patcher_dynamic_int8_lora_bake,
            )

            _patch_model_patcher_dynamic_int8_lora_bake()
            _console(
                "[HSWQ ZI NVFP4 LoRA] Reinstalled clean INT8 Dynamic.load bake "
                "(after discarding ZI-contaminated wrap)"
            )
        except Exception as e:
            logger.warning(
                "[HSWQ ZI NVFP4 LoRA] INT8 Dynamic.load re-patch after ZI clean "
                "failed: %s",
                e,
            )
    # Also strip VER=8 convert/set left on mp0.Linear by comfy_parity install.
    try:
        import comfy.ops as ops
        from ..nvfp4.nvfp4_forward import peel_all_nvfp4_linear_lora_bake

        if peel_all_nvfp4_linear_lora_bake(ops.mixed_precision_ops().Linear):
            removed = True
            _console(
                "[HSWQ ZI NVFP4 LoRA] peeled Linear convert/set bake wraps "
                "(int8_protect) for SDXL"
            )
    except Exception as e:
        logger.warning(
            "[HSWQ ZI NVFP4 LoRA] peel Linear bake wraps failed: %s", e
        )
    return removed


def install_zimage_nvfp4_lora_bake(force: bool = False) -> bool:
    """Wrap ModelPatcherDynamic.load: NVFP4 ConvRot bake + leftover INT8 QT bake."""
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return False

    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    if Dynamic is None:
        _console("[HSWQ ZI NVFP4 LoRA] ModelPatcherDynamic missing — bake hook skipped")
        return False
    original = getattr(Dynamic, "load", None)
    if original is None:
        return False
    if (
        not force
        and getattr(original, "_hswq_zi_nvfp4_lora_bake", False)
        and getattr(original, "_hswq_zi_nvfp4_lora_bake_ver", 0) >= _BAKE_HOOK_VER
    ):
        install_load_models_gpu_bake_hook(force=False)
        return True

    # Prefer chaining under current outer wrap (INT8 / stock), never nest ZI→ZI.
    prev_load = _unwrap_to_non_zi_load(original)

    def load(
        self,
        device_to=None,
        lowvram_model_memory=0,
        force_patch_weights=False,
        full_load=False,
        dirty=False,
    ):
        global _ENTER_LOGS
        if _ENTER_LOGS < _ENTER_LOG_MAX:
            _ENTER_LOGS += 1
            n_patches = len(getattr(self, "patches", None) or {})
            model = getattr(self, "model", None)
            diag = _nvfp4_convrot_diag(model)
            _console(
                f"[HSWQ ZI NVFP4 LoRA] Dynamic.load ENTER #{_ENTER_LOGS}: "
                f"patches={n_patches} "
                f"nvfp4_convrot={diag['has']} "
                f"flagged={diag['flagged']} "
                f"qt_on_weight={diag['qt_on_weight']} "
                f"model={type(model).__name__ if model is not None else None}"
            )
        result = prev_load(
            self,
            device_to=device_to,
            lowvram_model_memory=lowvram_model_memory,
            force_patch_weights=force_patch_weights,
            full_load=full_load,
            dirty=dirty,
        )
        run_zimage_nvfp4_lora_bake_on_patcher(
            self, device_to=device_to, reason="Dynamic.load"
        )
        return result

    load._hswq_zi_nvfp4_lora_bake = True  # type: ignore[attr-defined]
    load._hswq_zi_nvfp4_lora_bake_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
    load._hswq_zi_nvfp4_prev_dynamic_load = prev_load  # type: ignore[attr-defined]
    Dynamic.load = load
    _console(
        f"[HSWQ ZI NVFP4 LoRA] Dynamic.load bake hook ON v{_BAKE_HOOK_VER} "
        "(NVFP4 ConvRot bake + leftover INT8/QT bake + load_models_gpu)"
    )
    install_load_models_gpu_bake_hook(force=True)
    return True


def _unwrap_to_non_zi_load_models_gpu(fn):
    cur = fn
    seen = set()
    while (
        cur is not None
        and id(cur) not in seen
        and getattr(cur, "_hswq_zi_nvfp4_gpu_bake", False)
    ):
        seen.add(id(cur))
        nxt = getattr(cur, "_hswq_zi_nvfp4_prev_load_models_gpu", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def install_load_models_gpu_bake_hook(force: bool = False) -> bool:
    """After MultiGPU/stock load_models_gpu, bake any remaining ZI NVFP4 LoRA patches."""
    global _GPU_BAKE_INSTALLED
    try:
        import comfy.model_management as mm
    except ImportError:
        return False
    original = mm.load_models_gpu
    if (
        not force
        and getattr(original, "_hswq_zi_nvfp4_gpu_bake", False)
        and getattr(original, "_hswq_zi_nvfp4_gpu_bake_ver", 0) >= _BAKE_HOOK_VER
    ):
        _GPU_BAKE_INSTALLED = True
        return True
    prev = _unwrap_to_non_zi_load_models_gpu(original)

    def load_models_gpu(*args, **kwargs):
        result = prev(*args, **kwargs)
        try:
            for loaded in list(getattr(mm, "current_loaded_models", []) or []):
                patcher = getattr(loaded, "model", None)
                if patcher is None:
                    continue
                try:
                    if not bool(patcher.is_dynamic()):
                        continue
                except Exception:
                    continue
                if not getattr(patcher, "patches", None):
                    # Still try if bake keys exist (LowVram cleared but leftover)
                    if not getattr(
                        getattr(patcher, "model", None),
                        "_hswq_zi_nvfp4_baked_keys",
                        None,
                    ):
                        continue
                run_zimage_nvfp4_lora_bake_on_patcher(
                    patcher,
                    device_to=getattr(patcher, "load_device", None),
                    reason="load_models_gpu",
                )
        except Exception as exc:
            _console(f"[HSWQ ZI NVFP4 LoRA] load_models_gpu bake error: {exc!r}")
        return result

    load_models_gpu._hswq_zi_nvfp4_gpu_bake = True  # type: ignore[attr-defined]
    load_models_gpu._hswq_zi_nvfp4_gpu_bake_ver = _BAKE_HOOK_VER  # type: ignore[attr-defined]
    load_models_gpu._hswq_zi_nvfp4_prev_load_models_gpu = prev  # type: ignore[attr-defined]
    mm.load_models_gpu = load_models_gpu
    _GPU_BAKE_INSTALLED = True
    _console(
        f"[HSWQ ZI NVFP4 LoRA] load_models_gpu bake hook ON v{_BAKE_HOOK_VER}"
    )
    return True


def reset_zimage_nvfp4_lora_bake_log_counters() -> None:
    global _STATUS_LOGS, _SKIP_SAMPLE_LOGS, _ENTER_LOGS
    _STATUS_LOGS = 0
    _SKIP_SAMPLE_LOGS = 0
    _ENTER_LOGS = 0
```

### `nodes/zimage_nvfp4/nvfp4_tc_gate.py`

```python
"""NVFP4 TensorCore availability gate (shared by addmm patch + TC forward).

cuBLAS NVFP4 GEMM needs compute capability >= 10.0 (Blackwell). Cloud hosts are
often Ada / Hopper / Ampere — every ``scaled_mm_nvfp4`` then raises
``CUBLAS_STATUS_NOT_SUPPORTED`` and kitchen / addmm log WARNING per Linear.

This module:
  1) probes CC once
  2) after first NOT_SUPPORTED (or CC < 10.0), disables further TC attempts
  3) emits a single clear line; mutes kitchen nvfp4 WARNING spam
"""
from __future__ import annotations

import logging

_PROBED = False
_TC_OK: bool | None = None
_DISABLED = False
_WARNED = False
_DISABLE_REASON = ""

_KITCHEN_NVFP4_LOG = "comfy_kitchen.tensor.nvfp4"
_ADDMM_LOG = "nvfp4.nvfp4_addmm_patch"
_FORWARD_LOG = "nvfp4.nvfp4_forward"


def _mute_nvfp4_warning_spam() -> None:
    for name in (_KITCHEN_NVFP4_LOG, _ADDMM_LOG, _FORWARD_LOG):
        logging.getLogger(name).setLevel(logging.ERROR)


def probe_nvfp4_tc_support(device_index: int = 0) -> bool:
    """Return True if GPU CC looks NVFP4-TC capable (kitchen min is (10, 0))."""
    global _PROBED, _TC_OK
    if _PROBED and _TC_OK is not None:
        return bool(_TC_OK)
    _PROBED = True
    try:
        import torch

        if not torch.cuda.is_available():
            _TC_OK = False
            return False
        major, minor = torch.cuda.get_device_capability(device_index)
        # comfy_kitchen CUDA scaled_mm_nvfp4: min_compute_capability=(10, 0)
        _TC_OK = (int(major), int(minor)) >= (10, 0)
        return bool(_TC_OK)
    except Exception:
        _TC_OK = False
        return False


def nvfp4_tc_enabled() -> bool:
    if _DISABLED:
        return False
    return probe_nvfp4_tc_support()


def disable_nvfp4_tc(reason: str, *, announce: bool = True) -> None:
    """Permanent disable for this process; warn once then mute spam loggers."""
    global _DISABLED, _WARNED, _DISABLE_REASON
    _DISABLED = True
    _DISABLE_REASON = str(reason) if reason else "unknown"
    _mute_nvfp4_warning_spam()
    if announce and not _WARNED:
        _WARNED = True
        name = "?"
        cc = "?"
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                major, minor = torch.cuda.get_device_capability(0)
                cc = f"{major}.{minor}"
        except Exception:
            pass
        print(
            f"[HSWQ NVFP4] TensorCore scaled_mm disabled for this run "
            f"(GPU={name}, CC={cc}): {_DISABLE_REASON}. "
            f"Using dequant mm; further CUBLAS/kitchen WARNINGs suppressed.",
            flush=True,
        )


def note_scaled_mm_failure(exc: BaseException) -> bool:
    """If failure is permanent (NOT_SUPPORTED / unsupported), disable TC.

    Returns True if TC is now disabled (caller should dequant without retry storm).
    """
    msg = str(exc)
    permanent = (
        "CUBLAS_STATUS_NOT_SUPPORTED" in msg
        or "NOT_SUPPORTED" in msg
        or "not supported" in msg.lower()
    )
    if permanent:
        disable_nvfp4_tc(msg.split("\n", 1)[0][:240])
        return True
    return _DISABLED


def announce_tc_status_at_register() -> None:
    """One-line status when addmm / full stack is registered (cloud-visible)."""
    ok = probe_nvfp4_tc_support()
    try:
        import torch

        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            cc = f"{major}.{minor}"
        else:
            cc = "n/a"
    except Exception:
        name, cc = "?", "?"
    if ok:
        print(
            f"[HSWQ NVFP4] TC probe: GPU={name} CC={cc} — "
            f"scaled_mm_nvfp4 enabled (min CC 10.0)",
            flush=True,
        )
    else:
        disable_nvfp4_tc(
            f"compute capability {cc} < 10.0 (NVFP4 TensorCore requires Blackwell+)",
            announce=True,
        )
```

### `nodes/zimage_nvfp4/require_parity.py`

```python
"""Fail closed if ConvRot act-rotate forward is not armed."""


def require_convrot_parity_forward() -> None:
    """Fail if Linear.forward is not the ConvRot act-rotate wrapper."""
    import comfy.ops

    lin_fwd = comfy.ops.mixed_precision_ops().Linear.forward
    if getattr(lin_fwd, "_hswq_nvfp4_full_forward", False):
        raise RuntimeError(
            "Z Image ConvRot NVFP4: Linear.forward still has HSWQ TC wrap "
            "(_hswq_nvfp4_full_forward); quality would be destroyed"
        )
    if not getattr(lin_fwd, "_hswq_nvfp4_convrot_parity", False):
        raise RuntimeError(
            "Z Image ConvRot NVFP4: Linear.forward missing "
            "_hswq_nvfp4_convrot_parity (online act rotation required for "
            "offline W@H^T weights)"
        )
```

### `nodes/zimage_nvfp4/zi_comfy_quant_nvfp4.py`

```python
"""
Z Image arm for NVFP4 detect/load/LoRA bake — branch-only.

Owns the ZI delta that must not live in ``nodes/nvfp4`` (SDXL TC product):
  - walk stack_ver through INT8 / comfy_parity wraps
  - stamp stack_ver instead of false TC "upgrade" over ConvRot parity
  - never wrap TC Linear.forward over ``_hswq_nvfp4_convrot_parity``

Detect/load helpers stay under ``nodes/nvfp4``. Forward/bake come from
``zi_nvfp4_forward`` (hybrid). Call ``apply_nvfp4_comfy_parity`` after this.
"""
from __future__ import annotations

import logging

from ..nvfp4.nvfp4_conf import (
    fix_unet_config_packed_dims,
    is_nvfp4_conf,
    logical_linear_in_features,
)
from ..nvfp4.nvfp4_load import load_nvfp4_linear_module, peek_nvfp4_conf
from .zi_nvfp4_forward import (
    attach_nvfp4_linear_lora_bake,
    make_nvfp4_linear_forward,
)

logger = logging.getLogger(__name__)
_PATCHES_APPLIED = False
# Same contract bump as SDXL product; ZI reads through wrap chain.
_NVFP4_STACK_VER = 2

__all__ = [
    "apply_comfy_quant_nvfp4_patches",
]


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def _effective_nvfp4_stack_ver(mp_fn) -> int:
    """Read stack_ver through INT8 / comfy_parity wraps (attrs may live on inner)."""
    cur = mp_fn
    seen: set[int] = set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            return 0
        seen.add(id(cur))
        v = int(getattr(cur, "_hswq_nvfp4_stack_ver", 0) or 0)
        if v > 0:
            return v
        if getattr(cur, "_hswq_int8_conv_patched", False):
            cur = getattr(cur, "_hswq_orig_mixed_precision_ops", None)
            continue
        cur = getattr(cur, "_hswq_nvfp4_orig_mp", None)
    return 0


def _mp_chain_has_comfy_only(mp_fn) -> bool:
    cur = mp_fn
    seen: set[int] = set()
    for _ in range(8):
        if cur is None or id(cur) in seen:
            return False
        seen.add(id(cur))
        if getattr(cur, "_hswq_nvfp4_comfy_only", False):
            return True
        if getattr(cur, "_hswq_int8_conv_patched", False):
            cur = getattr(cur, "_hswq_orig_mixed_precision_ops", None)
            continue
        cur = getattr(cur, "_hswq_nvfp4_orig_mp", None)
    return False


def apply_comfy_quant_nvfp4_patches() -> bool:
    """ZI: NVFP4 detect/load + hybrid LoRA bake; skip TC over ConvRot parity."""
    global _PATCHES_APPLIED
    try:
        import comfy.model_detection as model_detection
        import comfy.ops as ops
    except Exception as e:
        logger.warning("[HSWQ NVFP4] comfy import failed: %s", e)
        return False

    mp_fn = getattr(ops, "mixed_precision_ops", None)
    stack_ver = _effective_nvfp4_stack_ver(mp_fn)
    if (
        _PATCHES_APPLIED
        and getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False)
        and stack_ver >= _NVFP4_STACK_VER
    ):
        return True

    # Already patched detect/load but LoRA bake missing: re-wrap mixed_precision_ops only.
    if getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False) and stack_ver < _NVFP4_STACK_VER:
        # Z Image: INT8 wrap used to drop _hswq_nvfp4_stack_ver → false "upgrade"
        # that wrapped TC over ConvRot parity → double online rotate after refresh.
        if _mp_chain_has_comfy_only(mp_fn) or (
            _PATCHES_APPLIED
            and stack_ver == 0
            and getattr(mp_fn, "_hswq_int8_conv_patched", False)
        ):
            try:
                if mp_fn is not None:
                    mp_fn._hswq_nvfp4_stack_ver = _NVFP4_STACK_VER  # type: ignore[attr-defined]
            except Exception:
                pass
            _PATCHES_APPLIED = True
            _console(
                "[HSWQ NVFP4] stack ver stamped "
                "(skip TC upgrade; comfy_parity / INT8 chain intact)"
            )
            return True

        _orig_mp = getattr(mp_fn, "_hswq_nvfp4_orig_mp", None)
        if _orig_mp is None:
            _orig_mp = mp_fn

        def mixed_precision_ops_upgraded(*args, **kwargs):
            mp = _orig_mp(*args, **kwargs)
            Lin = mp.Linear
            # Never wrap TC over ConvRot parity (Z Image double-rotate / noise).
            if getattr(Lin.forward, "_hswq_nvfp4_convrot_parity", False):
                attach_nvfp4_linear_lora_bake(Lin)
                return mp
            if not getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
                Lin.forward = make_nvfp4_linear_forward(Lin.forward)
            attach_nvfp4_linear_lora_bake(Lin)
            return mp

        mixed_precision_ops_upgraded._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]
        mixed_precision_ops_upgraded._hswq_nvfp4_stack_ver = _NVFP4_STACK_VER  # type: ignore[attr-defined]
        mixed_precision_ops_upgraded._hswq_nvfp4_orig_mp = _orig_mp  # type: ignore[attr-defined]
        ops.mixed_precision_ops = mixed_precision_ops_upgraded
        _PATCHES_APPLIED = True
        _console(
            "[HSWQ NVFP4] upgraded stack ver=%s "
            "(ConvRot Linear LoRA bake: convert_weight unrotate + set_weight re-rotate)"
            % _NVFP4_STACK_VER
        )
        return True

    _orig_detect = model_detection.detect_unet_config
    _orig_calc = model_detection.calculate_transformer_depth
    _orig_load = ops._load_quantized_module
    _orig_mp = ops.mixed_precision_ops

    def calculate_transformer_depth_patched(prefix, state_dict_keys, state_dict):
        out = _orig_calc(prefix, state_dict_keys, state_dict)
        if out is None:
            return None
        depth, context_dim, use_linear, time_stack, time_stack_cross = out
        k = f"{prefix}1.transformer_blocks.0.attn2.to_k.weight"
        if k in state_dict:
            try:
                context_dim = logical_linear_in_features(state_dict, k)
            except Exception as e:
                logger.warning("[HSWQ NVFP4] transformer context_dim fix skipped: %s", e)
        return depth, context_dim, use_linear, time_stack, time_stack_cross

    def detect_unet_config_patched(state_dict, key_prefix, metadata=None):
        unet_config = _orig_detect(state_dict, key_prefix, metadata=metadata)
        if unet_config is None:
            return None
        return fix_unet_config_packed_dims(unet_config, state_dict, key_prefix)

    def model_config_from_unet_patched(
        state_dict, unet_key_prefix, use_base_if_no_match=False, metadata=None
    ):
        import comfy.supported_models_base
        import comfy.utils

        unet_config = model_detection.detect_unet_config(
            state_dict, unet_key_prefix, metadata=metadata
        )
        if unet_config is None:
            return None
        model_config = model_detection.model_config_from_unet_config(
            unet_config, state_dict, unet_key_prefix
        )
        if model_config is None and use_base_if_no_match:
            model_config = comfy.supported_models_base.BASE(unet_config)

        quant_config = comfy.utils.detect_layer_quantization(
            state_dict, unet_key_prefix
        )
        if quant_config:
            if model_config is None:
                logging.error(
                    "[HSWQ NVFP4] model_config is None with quant_config present "
                    "(packed NVFP4 dims still unmatched?). prefix=%r config=%s",
                    unet_key_prefix,
                    unet_config,
                )
                return None
            model_config.quant_config = quant_config
            logging.info("Detected mixed precision quantization")
        return model_config

    def _load_quantized_module_patched(
        module,
        super_load,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        load_extra_params=False,
    ):
        conf = peek_nvfp4_conf(state_dict, prefix)
        if is_nvfp4_conf(conf):
            load_nvfp4_linear_module(
                module,
                super_load,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
                load_extra_params=load_extra_params,
            )
            return
        _orig_load(
            module,
            super_load,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            load_extra_params=load_extra_params,
        )

    def mixed_precision_ops_patched(*args, **kwargs):
        mp = _orig_mp(*args, **kwargs)
        Lin = mp.Linear
        if getattr(Lin.forward, "_hswq_nvfp4_convrot_parity", False):
            attach_nvfp4_linear_lora_bake(Lin)
            return mp
        if not getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
            Lin.forward = make_nvfp4_linear_forward(Lin.forward)
        attach_nvfp4_linear_lora_bake(Lin)
        return mp

    model_detection.calculate_transformer_depth = calculate_transformer_depth_patched
    model_detection.detect_unet_config = detect_unet_config_patched
    model_detection.model_config_from_unet = model_config_from_unet_patched
    ops._load_quantized_module = _load_quantized_module_patched
    ops.mixed_precision_ops = mixed_precision_ops_patched

    detect_unet_config_patched._hswq_nvfp4_packed_dims = True  # type: ignore[attr-defined]
    calculate_transformer_depth_patched._hswq_nvfp4_packed_dims = True  # type: ignore[attr-defined]
    model_config_from_unet_patched._hswq_nvfp4_packed_dims = True  # type: ignore[attr-defined]
    _load_quantized_module_patched._hswq_nvfp4_full_load = True  # type: ignore[attr-defined]
    mixed_precision_ops_patched._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]
    mixed_precision_ops_patched._hswq_nvfp4_stack_ver = _NVFP4_STACK_VER  # type: ignore[attr-defined]
    mixed_precision_ops_patched._hswq_nvfp4_orig_mp = _orig_mp  # type: ignore[attr-defined]

    _PATCHES_APPLIED = True
    _console(
        "[HSWQ NVFP4] Z Image stack applied "
        "(detect packed K + nvfp4_load + hybrid LoRA bake; "
        "TC skipped over ConvRot parity; ComfyUI-master untouched)"
    )
    return True
```

### `nodes/zimage_nvfp4/zi_nvfp4_conf.py`

```python
"""Z Image INT8 protect ConvRot conf helpers (branch-only; not in nodes/nvfp4)."""
from __future__ import annotations

from typing import Optional


def is_int8_tensorwise_conf(conf: Optional[dict]) -> bool:
    return isinstance(conf, dict) and str(conf.get("format") or "").lower() == "int8_tensorwise"


def int8_convrot_flags_from_conf(conf: Optional[dict]) -> tuple[bool, int]:
    """Return (enabled, groupsize) for INT8 protect ConvRot comfy_quant.

    Do **not** reuse ``convrot_flags_from_conf`` — that helper is NVFP4-only and
    always returns False for ``int8_tensorwise``. Used by load arm to set
    ``_hswq_int8_convrot`` and clear kitchen ``Params.convrot`` (Conv2d twin).
    """
    if not is_int8_tensorwise_conf(conf):
        return False, 256
    params_conf = conf.get("params", {})
    if not isinstance(params_conf, dict):
        params_conf = {}
    enabled = bool(conf.get("convrot", False)) or bool(params_conf.get("convrot", False))
    if not enabled:
        return False, 256
    gs = int(conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)) or 256)
    return True, gs
```

### `nodes/zimage_nvfp4/zi_nvfp4_forward.py`

```python
"""
Z Image hybrid ConvRot LoRA bake + NVFP4 Linear forward helpers (branch-only).

Owns the ZI delta that must not live in ``nodes/nvfp4`` (SDXL TC product):
  - per-kind LoRA bake counters / EVIDENCE (NVFP4 + INT8 protect)
  - ``_hswq_int8_convrot`` bake arm / Params.convrot clear
  - ``attach_nvfp4_linear_lora_bake`` ver >= 8 (hybrid)

Hadamard lives in ``zi_nvfp4_hadamard`` (Distorch-safe cache). Runtime stays under ``nodes/nvfp4``.
"""
from __future__ import annotations

import logging
import os

from .zi_nvfp4_hadamard import (
    build_hadamard,
    rotate_weight_linear,
    unrotate_weight_linear,
)
from ..nvfp4.nvfp4_runtime import (
    ensure_act_scale,
    clear_nvfp4_cudagraphs,
    nvfp4_quant_mm_cudagraph,
    quantize_nvfp4_act_pooled,
    rotate_last_dim_pooled,
    scaled_mm_nvfp4_pooled,
    _GRAPH_MAX_M,
)

logger = logging.getLogger(__name__)

# Counters for bench / diagnostics (reset per run if needed)
_TC_HITS = 0
_DEQUANT_FALLBACKS = 0
# Per-kind totals (always incremented) + per-kind sample log caps.
# Shared max of 8 hid all int8_protect samples (nvfp4 filled the quota first).
_LORA_CONVERT_TOTAL = {"nvfp4": 0, "int8_protect": 0}
_LORA_SET_TOTAL = {"nvfp4": 0, "int8_protect": 0}
_LORA_CONVERT_LOGGED = {"nvfp4": 0, "int8_protect": 0}
_LORA_SET_LOGGED = {"nvfp4": 0, "int8_protect": 0}
_LORA_KIND_LOG_MAX = 4
# Bump when convert_weight / set_weight ConvRot LoRA bake changes.
# v2: also unrotate/re-rotate INT8 protect ConvRot (``_hswq_int8_convrot``).
# Hybrid ZI packs = ConvRot NVFP4 + ConvRot INT8 protect — both need bake basis.
# v3: bake-time fallback if load arm missed and Params.convrot still True on INT8 QT.
# v4: (reverted) bake-only with Params.convrot — WRONG: kitchen dequant already
#     unrotates when Params.convrot=True → double unrotate → LoRA dead.
# v5: Conv2d twin — arm flag + clear Params; after set_weight keep Params=False
#     (noise was requant restoring Params while parity still rotated).
# v6: per-kind LoRA bake counters + EVIDENCE log (int8_protect must be visible).
# v7: pass-delta EVIDENCE only (no stale OK spam on empty re-bake / VAE load).
# v8: peer NVFP4_LORA_BAKE_* verdict + sample_nvfp4_keys (same weight as INT8).
_NVFP4_LORA_BAKE_VER = 8


def reset_nvfp4_lora_log_counters() -> None:
    for d in (
        _LORA_CONVERT_TOTAL,
        _LORA_SET_TOTAL,
        _LORA_CONVERT_LOGGED,
        _LORA_SET_LOGGED,
    ):
        for k in d:
            d[k] = 0


def reset_nvfp4_forward_stats() -> None:
    global _TC_HITS, _DEQUANT_FALLBACKS
    _TC_HITS = 0
    _DEQUANT_FALLBACKS = 0


def _lora_bake_kind(module) -> str:
    if getattr(module, "_hswq_nvfp4_convrot", False):
        return "nvfp4"
    return "int8_protect"


def nvfp4_lora_bake_counters() -> dict:
    """Totals for convert unrotate / set re-rotate by ConvRot kind."""
    return {
        "convert_unrotate_nvfp4": int(_LORA_CONVERT_TOTAL.get("nvfp4", 0)),
        "convert_unrotate_int8_protect": int(_LORA_CONVERT_TOTAL.get("int8_protect", 0)),
        "set_rerotate_nvfp4": int(_LORA_SET_TOTAL.get("nvfp4", 0)),
        "set_rerotate_int8_protect": int(_LORA_SET_TOTAL.get("int8_protect", 0)),
    }


def snapshot_nvfp4_lora_bake_counters() -> dict:
    """Copy of totals for pass-delta EVIDENCE (before bake → after bake)."""
    return dict(nvfp4_lora_bake_counters())


def _counter_delta(before: dict | None, after: dict | None) -> dict:
    b = before or {}
    a = after or nvfp4_lora_bake_counters()
    keys = (
        "convert_unrotate_nvfp4",
        "convert_unrotate_int8_protect",
        "set_rerotate_nvfp4",
        "set_rerotate_int8_protect",
    )
    return {k: int(a.get(k, 0)) - int(b.get(k, 0)) for k in keys}


def _lora_bake_side_verdict(prefix: str, baked: int, convert_n: int, set_n: int) -> str:
    """Peer verdict for one ConvRot kind (NVFP4 or INT8 protect)."""
    match = convert_n == set_n == int(baked)
    if int(baked) > 0 and match and convert_n > 0:
        return f"{prefix}_OK"
    if int(baked) > 0 and not match:
        return f"{prefix}_MISMATCH"
    if int(baked) == 0 and convert_n == 0:
        return f"{prefix}_N/A"
    return f"{prefix}_MISSING"


def _fmt_sample_keys(label: str, keys: list | None) -> str:
    if not keys:
        return ""
    shown = ", ".join(str(k) for k in keys[:3])
    return f" {label}=[{shown}]"


def log_nvfp4_lora_bake_evidence(
    tag: str = "",
    *,
    before: dict | None = None,
    nvfp4_baked: int = 0,
    int8_baked: int = 0,
    sample_nvfp4_keys: list | None = None,
    sample_int8_keys: list | None = None,
    force: bool = False,
) -> str | None:
    """Emit pass-scoped EVIDENCE only when this bake pass actually ran hooks.

    NVFP4 and INT8 protect are peer sides (same verdict shape + key samples).
    Returns the message if emitted, else None (silent skip for empty re-bake).
    """
    after = nvfp4_lora_bake_counters()
    d = _counter_delta(before, after)
    i8c = d["convert_unrotate_int8_protect"]
    i8s = d["set_rerotate_int8_protect"]
    nvc = d["convert_unrotate_nvfp4"]
    nvs = d["set_rerotate_nvfp4"]
    this_pass_hooks = (i8c + i8s + nvc + nvs) > 0
    this_pass_layer = (int(nvfp4_baked) + int(int8_baked)) > 0
    if not force and not this_pass_hooks and not this_pass_layer:
        return None

    nv_verdict = _lora_bake_side_verdict(
        "NVFP4_LORA_BAKE", int(nvfp4_baked), nvc, nvs
    )
    i8_verdict = _lora_bake_side_verdict(
        "INT8_PROTECT_LORA_BAKE", int(int8_baked), i8c, i8s
    )

    suffix = f" ({tag})" if tag else ""
    nv_samples = _fmt_sample_keys("sample_nvfp4_keys", sample_nvfp4_keys)
    i8_samples = _fmt_sample_keys("sample_int8_keys", sample_int8_keys)
    msg = (
        f"[HSWQ ConvRot LoRA] EVIDENCE{suffix}: {nv_verdict} {i8_verdict} "
        f"this_pass | "
        f"nvfp4 convert_unrotate={nvc} set_rerotate={nvs} "
        f"nvfp4_baked={int(nvfp4_baked)}{nv_samples} | "
        f"int8_protect convert_unrotate={i8c} set_rerotate={i8s} "
        f"int8_baked={int(int8_baked)}{i8_samples} | "
        f"session_total nv_c/s="
        f"{after['convert_unrotate_nvfp4']}/"
        f"{after['set_rerotate_nvfp4']} "
        f"int8_c/s="
        f"{after['convert_unrotate_int8_protect']}/"
        f"{after['set_rerotate_int8_protect']}"
    )
    # Single emit path (caller may also _console — prefer logger+print once here).
    logger.info(msg)
    print(msg, flush=True)
    return msg


def _clear_int8_qt_params_convrot(module) -> bool:
    """Force Params.convrot=False on INT8 QT (must stay False after requant)."""
    import dataclasses

    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    w = getattr(module, "weight", None)
    qt = w if isinstance(w, QuantizedTensor) else getattr(w, "data", None)
    if qt is None or not isinstance(qt, QuantizedTensor):
        return False
    params = getattr(qt, "_params", None)
    if params is None or not bool(getattr(params, "convrot", False)):
        return False
    new_params = dataclasses.replace(params, convrot=False)
    try:
        object.__setattr__(qt, "_params", new_params)
        return True
    except Exception:
        pass
    try:
        qt._params = new_params
        return True
    except Exception:
        return False


def _linear_convrot_lora_groupsize(module) -> int | None:
    """Groupsize for offline ConvRot Linear LoRA bake, or None if not ConvRot.

    Hybrid Z Image packs:
      - NVFP4: ``_hswq_nvfp4_convrot`` (Params cleared; parity rotates).
      - INT8 protect: ``_hswq_int8_convrot`` (Params cleared; parity rotates).
        Kitchen dequant with Params.convrot=True already unrotates — bake must
        see Params=False so convert unrotates rotated-basis float once.
    """
    if getattr(module, "_hswq_nvfp4_convrot", False):
        return int(getattr(module, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
    if getattr(module, "_hswq_int8_convrot", False):
        return int(getattr(module, "_hswq_int8_convrot_groupsize", 256) or 256)
    # Late arm if load missed: Params.convrot still True on INT8 QT.
    try:
        from comfy.quant_ops import QuantizedTensor

        w = getattr(module, "weight", None)
        qt = w if isinstance(w, QuantizedTensor) else getattr(w, "data", None)
        if qt is None or not isinstance(qt, QuantizedTensor):
            return None
        layout_cls = getattr(qt, "_layout_cls", None) or ""
        if isinstance(layout_cls, type):
            layout_cls = getattr(layout_cls, "__name__", "") or ""
        if str(layout_cls) != "TensorWiseINT8Layout":
            return None
        params = getattr(qt, "_params", None)
        if params is None or not bool(getattr(params, "convrot", False)):
            return None
        gs = int(getattr(params, "convrot_groupsize", 256) or 256)
        module._hswq_int8_convrot = True
        module._hswq_int8_convrot_groupsize = gs
        _clear_int8_qt_params_convrot(module)
        return gs
    except Exception:
        return None


def nvfp4_forward_stats() -> dict:
    return {"scaled_mm_hits": _TC_HITS, "dequant_fallbacks": _DEQUANT_FALLBACKS}


def _slice_nvfp4_mm_out(result, orig_m: int, orig_n: int):
    if result.shape[0] != orig_m or result.shape[1] != orig_n:
        return result[:orig_m, :orig_n]
    return result


def scaled_mm_nvfp4_linear(input_qt, weight_qt, bias):
    """Kitchen / tritant NVFP4 linear (QT path; used as fallback)."""
    global _TC_HITS, _DEQUANT_FALLBACKS
    import torch
    import torch.nn.functional as F
    import comfy_kitchen as ck
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    if not (
        isinstance(input_qt, QuantizedTensor)
        and isinstance(weight_qt, QuantizedTensor)
        and input_qt._layout_cls == "TensorCoreNVFP4Layout"
        and weight_qt._layout_cls == "TensorCoreNVFP4Layout"
    ):
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)
    if input_qt._qdata.dim() != 2:
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)
    if getattr(input_qt._params, "transposed", False) or getattr(
        weight_qt._params, "transposed", False
    ):
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()

    a_qdata, scale_a, block_scale_a = TensorCoreNVFP4Layout.get_plain_tensors(input_qt)
    w_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(weight_qt)
    out_dtype = input_qt._params.orig_dtype
    try:
        result = ck.scaled_mm_nvfp4(
            a_qdata,
            w_qdata,
            tensor_scale_a=scale_a,
            tensor_scale_b=scale_b,
            block_scale_a=block_scale_a,
            block_scale_b=block_scale_b,
            bias=bias,
            out_dtype=out_dtype,
        )
        orig_m = input_qt._params.orig_shape[0]
        orig_n = weight_qt._params.orig_shape[0]  # (out, in)
        _TC_HITS += 1
        return _slice_nvfp4_mm_out(result, orig_m, orig_n)
    except (RuntimeError, TypeError) as e:
        logger.warning("[HSWQ NVFP4] scaled_mm_nvfp4 failed: %s — F.linear dequant", e)
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)


def _plain_weight_cached(module, weight_qt):
    """Cache get_plain_tensors on the module (weight QT identity stable after load)."""
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    cached = getattr(module, "_hswq_nvfp4_w_plain", None)
    if cached is not None and cached[0] is weight_qt._qdata:
        return cached[1], cached[2], cached[3], cached[4]
    w_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(weight_qt)
    orig_n = int(weight_qt._params.orig_shape[0])
    module._hswq_nvfp4_w_plain = (
        weight_qt._qdata,
        w_qdata,
        scale_b,
        block_scale_b,
        orig_n,
    )
    return w_qdata, scale_b, block_scale_b, orig_n


def _tc_forward_pooled(module, input_2d, weight_qt, bias, act_scale, out_dtype):
    """Act float → pooled NVFP4 quant → pooled cuBLAS mm (no QT alloc).

    Prefers CUDA Graph (quantize+mm) after first capture per shape/weight; falls
    back to eager pooled kernels if capture/replay fails.
    """
    global _TC_HITS, _DEQUANT_FALLBACKS
    import torch
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    if not (
        isinstance(weight_qt, QuantizedTensor)
        and weight_qt._layout_cls == "TensorCoreNVFP4Layout"
    ):
        _DEQUANT_FALLBACKS += 1
        return None
    if getattr(weight_qt._params, "transposed", False):
        _DEQUANT_FALLBACKS += 1
        return None

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()

    orig_m, orig_k = int(input_2d.shape[0]), int(input_2d.shape[1])
    needs_padding = TensorCoreNVFP4Layout.get_padded_shape((orig_m, orig_k)) != (
        orig_m,
        orig_k,
    )

    scale_a = ensure_act_scale(input_2d, act_scale)
    try:
        w_qdata, scale_b, block_scale_b, orig_n = _plain_weight_cached(module, weight_qt)

        # Calib input_scale and placeholder ones are static — always cache
        # alpha. Recomputing scale_a*scale_b every Linear (~18k/sample) was
        # pure waste on FULL ConvRot (every layer has input_scale).
        cached_alpha = getattr(module, "_hswq_nvfp4_alpha", None)
        if cached_alpha is None:
            alpha = scale_a * scale_b
            if alpha.dtype != torch.float32:
                alpha = alpha.to(dtype=torch.float32)
            if alpha.dim() == 0:
                alpha = alpha.reshape(1)
            module._hswq_nvfp4_alpha = alpha
        else:
            alpha = cached_alpha

        # CUDA Graph is OFF by default: shape-shared replay copies full weight
        # every call and was slower than eager (13.05s vs ~11.8s). Opt-in:
        # HSWQ_NVFP4_CUDAGRAPH=1
        use_cg = (
            os.environ.get("HSWQ_NVFP4_CUDAGRAPH", "").strip() == "1"
            and orig_m <= _GRAPH_MAX_M
            and not getattr(module, "_hswq_nvfp4_no_cudagraph", False)
        )
        if use_cg:
            try:
                result = nvfp4_quant_mm_cudagraph(
                    input_2d,
                    w_qdata=w_qdata,
                    weight_scale=scale_b,
                    block_scale_w=block_scale_b,
                    scale_a=scale_a,
                    bias=bias,
                    out_dtype=out_dtype,
                    alpha=alpha,
                    pad_16x=needs_padding,
                    orig_n=orig_n,
                )
                _TC_HITS += 1
                return result
            except torch.cuda.OutOfMemoryError:
                clear_nvfp4_cudagraphs()
                torch.cuda.empty_cache()
                logger.warning(
                    "[HSWQ NVFP4] CUDA Graph OOM — cache cleared; eager pooled"
                )
            except (RuntimeError, TypeError, ValueError) as e:
                if "out of memory" in str(e).lower():
                    clear_nvfp4_cudagraphs()
                    torch.cuda.empty_cache()
                    logger.warning(
                        "[HSWQ NVFP4] CUDA Graph OOM (%s); eager pooled", e
                    )
                else:
                    module._hswq_nvfp4_no_cudagraph = True
                    logger.warning(
                        "[HSWQ NVFP4] CUDA Graph disabled for module (%s); eager pooled",
                        e,
                    )

        a_qdata, block_scale_a, _pr, _pc = quantize_nvfp4_act_pooled(
            input_2d, scale_a, pad_16x=needs_padding
        )
        result = scaled_mm_nvfp4_pooled(
            a_qdata,
            w_qdata,
            tensor_scale_a=scale_a,
            tensor_scale_b=scale_b,
            block_scale_a=block_scale_a,
            block_scale_b=block_scale_b,
            bias=bias,
            out_dtype=out_dtype,
            alpha=alpha,
            orig_m=orig_m,
            orig_n=orig_n,
        )
        _TC_HITS += 1
        return result
    except (RuntimeError, TypeError, ValueError) as e:
        logger.warning("[HSWQ NVFP4] pooled TC path failed: %s", e)
        _DEQUANT_FALLBACKS += 1
        return None


def make_nvfp4_linear_forward(stock_forward):
    """
    Return a Linear.forward replacement.

    For modules flagged ``_hswq_nvfp4`` (set at load), run the HSWQ TC path.
    All other layers keep stock_forward unchanged.
    """
    import torch
    import comfy.model_management
    from comfy.ops import cast_bias_weight, run_every_op, uncast_bias_weight

    def forward_nvfp4(self, input, *args, **kwargs):
        if not getattr(self, "_hswq_nvfp4", False) or getattr(self, "_full_precision_mm", False):
            return stock_forward(self, input, *args, **kwargs)

        # Training / forced cast: fall back to stock
        if input.requires_grad or getattr(self, "comfy_force_cast_weights", False):
            return stock_forward(self, input, *args, **kwargs)
        # LoRA weight_function: stay on HSWQ path (act ConvRot + cast_bias_weight
        # with want_requant). Stock forward would skip act rotate → ConvRot break.

        run_every_op()
        input_shape = input.shape
        compute_dtype = input.dtype

        # 1) Reshape ≥3D → 2D first (same last-dim math; cheaper than rotating ND)
        reshaped_nd = input.ndim >= 3
        input_2d = input.reshape(-1, input_shape[-1]) if reshaped_nd else input
        if input_2d.ndim != 2:
            return stock_forward(self, input, *args, **kwargs)

        # 2) FULL ConvRot: dense Hadamard GEMM (gs=256 butterfly is ~15x slower)
        if getattr(self, "_hswq_nvfp4_convrot", False):
            gs = int(getattr(self, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
            h = getattr(self, "_hswq_nvfp4_H", None)
            if h is None or h.device != input_2d.device or h.dtype != input_2d.dtype:
                h = build_hadamard(gs, device=input_2d.device, dtype=input_2d.dtype)
                self._hswq_nvfp4_H = h
            input_2d = rotate_last_dim_pooled(input_2d, h, gs)

        # 3) Weight / bias: skip cast_bias_weight when already on-device QT
        #    (cast+sync every Linear was a major share of NVFP4 > FP16 wall time).
        #    Always cast when LoRA weight/bias_function present (need bake apply).
        offload_stream = None
        weight = self.weight
        if isinstance(weight, torch.nn.Parameter):
            weight = weight.data
        bias = self.bias.data if self.bias is not None else None
        has_wf = len(getattr(self, "weight_function", []) or []) or len(
            getattr(self, "bias_function", []) or []
        )
        need_cast = weight.device != input_2d.device or (
            bias is not None and bias.device != input_2d.device
        )
        if has_wf or need_cast or hasattr(self, "_v"):
            weight, bias, offload_stream = cast_bias_weight(
                self,
                input_2d,
                offloadable=True,
                compute_dtype=compute_dtype,
                want_requant=True,
            )

        scale = getattr(self, "input_scale", None)
        if scale is not None:
            if isinstance(scale, torch.nn.Parameter):
                scale = scale.data
            if scale.device != input.device:
                scale = comfy.model_management.cast_to_device(scale, input.device, None)

        layout = getattr(self, "layout_type", None)
        if layout is None:
            if offload_stream is not None:
                uncast_bias_weight(self, weight, bias, offload_stream)
            return stock_forward(self, input, *args, **kwargs)

        # 4) Pooled Tensor Core path (no QuantizedTensor.from_float alloc)
        out_2d = _tc_forward_pooled(
            self, input_2d, weight, bias, scale, compute_dtype
        )
        if out_2d is None:
            # Fallback: stock QT path
            from comfy.quant_ops import QuantizedTensor

            q_input = QuantizedTensor.from_float(input_2d, layout, scale=scale)
            out_2d = scaled_mm_nvfp4_linear(q_input, weight, bias)

        # 5) Restore rank with logical out_features (never QT storage shape[0])
        if reshaped_nd:
            out = out_2d.reshape((*input_shape[:-1], int(self.out_features)))
        else:
            out = out_2d

        if offload_stream is not None:
            uncast_bias_weight(self, weight, bias, offload_stream)
        return out

    forward_nvfp4._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]
    return forward_nvfp4


def make_nvfp4_linear_convert_weight(stock_convert_weight):
    """Wrap Linear.convert_weight: dequant then unrotate ConvRot weights for LoRA bake.

    Handles ConvRot NVFP4 **and** ConvRot INT8 protect (hybrid Z Image packs).
    Clear INT8 Params.convrot **before** stock dequant — kitchen already
    unrotates when Params.convrot=True (would double-unrotate with bake).
    """
    import torch
    from comfy.quant_ops import QuantizedTensor

    def convert_weight(self, weight, inplace=False, **kwargs):
        # Arm / clear Params before dequant (Conv2d twin).
        gs = _linear_convrot_lora_groupsize(self)
        if callable(stock_convert_weight):
            out = stock_convert_weight(self, weight, inplace=inplace, **kwargs)
        elif isinstance(weight, QuantizedTensor):
            out = weight.dequantize()
        else:
            out = weight
        if gs is None:
            gs = _linear_convrot_lora_groupsize(self)
        if gs is not None and out is not None and getattr(out, "ndim", 0) == 2:
            h = build_hadamard(gs, device="cpu", dtype=torch.float32)
            out = unrotate_weight_linear(out, h, gs)
            kind = _lora_bake_kind(self)
            _LORA_CONVERT_TOTAL[kind] = int(_LORA_CONVERT_TOTAL.get(kind, 0)) + 1
            if int(_LORA_CONVERT_LOGGED.get(kind, 0)) < _LORA_KIND_LOG_MAX:
                _LORA_CONVERT_LOGGED[kind] = int(_LORA_CONVERT_LOGGED.get(kind, 0)) + 1
                logger.info(
                    "[HSWQ ConvRot LoRA] Linear.convert_weight #%s (%s): unrotate "
                    "gs=%s in=%s/%s -> out=%s/%s",
                    _LORA_CONVERT_TOTAL[kind],
                    kind,
                    gs,
                    type(weight).__name__,
                    getattr(weight, "dtype", None),
                    type(out).__name__,
                    getattr(out, "dtype", None),
                )
        return out

    convert_weight._hswq_nvfp4_lora_bake_ver = _NVFP4_LORA_BAKE_VER  # type: ignore[attr-defined]
    convert_weight._hswq_nvfp4_lora_bake_stock = stock_convert_weight  # type: ignore[attr-defined]
    return convert_weight


def make_nvfp4_linear_set_weight(stock_set_weight):
    """Wrap Linear.set_weight: re-rotate ConvRot float weights before requant.

    Handles ConvRot NVFP4 **and** ConvRot INT8 protect (hybrid Z Image packs).
    After INT8 requant, force Params.convrot=False (parity rotates acts;
    kitchen must not also rotate).
    """
    import torch

    def set_weight(
        self,
        weight,
        inplace_update=False,
        seed=None,
        return_weight=False,
        **kwargs,
    ):
        gs = _linear_convrot_lora_groupsize(self)
        if gs is not None and getattr(weight, "ndim", 0) == 2:
            h = build_hadamard(gs, device="cpu", dtype=torch.float32)
            weight = rotate_weight_linear(weight, h, gs)
            kind = _lora_bake_kind(self)
            _LORA_SET_TOTAL[kind] = int(_LORA_SET_TOTAL.get(kind, 0)) + 1
            if int(_LORA_SET_LOGGED.get(kind, 0)) < _LORA_KIND_LOG_MAX:
                _LORA_SET_LOGGED[kind] = int(_LORA_SET_LOGGED.get(kind, 0)) + 1
                logger.info(
                    "[HSWQ ConvRot LoRA] Linear.set_weight #%s (%s): re-rotate "
                    "gs=%s shape=%s layout=%s",
                    _LORA_SET_TOTAL[kind],
                    kind,
                    gs,
                    tuple(weight.shape) if hasattr(weight, "shape") else "?",
                    getattr(self, "layout_type", None),
                )
        out = stock_set_weight(
            self,
            weight,
            inplace_update=inplace_update,
            seed=seed,
            return_weight=return_weight,
            **kwargs,
        )
        # Requant may restore Params.convrot — keep cleared for INT8 protect.
        if getattr(self, "_hswq_int8_convrot", False):
            _clear_int8_qt_params_convrot(self)
        return out

    set_weight._hswq_nvfp4_lora_bake_ver = _NVFP4_LORA_BAKE_VER  # type: ignore[attr-defined]
    set_weight._hswq_nvfp4_lora_bake_stock = stock_set_weight  # type: ignore[attr-defined]
    return set_weight


def _peel_lora_bake_wrap(fn):
    """Unwrap nested HSWQ convert/set wraps to true stock.

    After #2 split, ``nodes/nvfp4`` (3.3.0) may attach VER=1 first; ZI must not
    wrap that as stock (double unrotate / re-rotate → dead LoRA). Same as
    3.3.4 single-module attach: one hybrid wrap over stock only.
    """
    cur = fn
    for _ in range(8):
        if not callable(cur):
            return cur
        if int(getattr(cur, "_hswq_nvfp4_lora_bake_ver", 0) or 0) <= 0:
            return cur
        stock = getattr(cur, "_hswq_nvfp4_lora_bake_stock", None)
        if stock is not None and stock is not cur:
            cur = stock
            continue
        closure = getattr(cur, "__closure__", None)
        code = getattr(cur, "__code__", None)
        if closure is None or code is None:
            return cur
        names = code.co_freevars
        nxt = None
        for i, name in enumerate(names):
            if name in ("stock_convert_weight", "stock_set_weight"):
                nxt = closure[i].cell_contents
                break
        if nxt is None or nxt is cur:
            return cur
        cur = nxt
    return cur


def attach_nvfp4_linear_lora_bake(Lin) -> bool:
    """Ensure MixedPrecision Linear has hybrid ConvRot LoRA wraps (one layer).

    Peels any prior HSWQ bake wrap (e.g. SDXL ``nodes/nvfp4`` VER=1) so ZI
    hybrid VER never nests — nesting double-unrotates NVFP4 and kills LoRA.
    """
    applied = False
    cvt = getattr(Lin, "convert_weight", None)
    if callable(cvt):
        ver = int(getattr(cvt, "_hswq_nvfp4_lora_bake_ver", 0) or 0)
        if ver != _NVFP4_LORA_BAKE_VER:
            stock = _peel_lora_bake_wrap(cvt) if ver > 0 else cvt
            Lin.convert_weight = make_nvfp4_linear_convert_weight(stock)
            applied = True
    sw = getattr(Lin, "set_weight", None)
    if callable(sw):
        ver = int(getattr(sw, "_hswq_nvfp4_lora_bake_ver", 0) or 0)
        if ver != _NVFP4_LORA_BAKE_VER:
            stock = _peel_lora_bake_wrap(sw) if ver > 0 else sw
            Lin.set_weight = make_nvfp4_linear_set_weight(stock)
            applied = True
    return applied
```

### `nodes/zimage_nvfp4/zi_nvfp4_hadamard.py`

```python
"""Z Image Hadamard helpers (Distorch-safe cache) for ConvRot + act rotation.

Owns the ZI delta that must not live in ``nodes/nvfp4`` (SDXL TC product):
  - ``_tensor_storage_ok`` / poisoned-storage rejection
  - ``clear_hadamard_global_caches`` (Method 2c / parity clear)
  - storage-ok gating inside ``build_hadamard`` / ``_h4``

SDXL TC uses the 3.3.0 stock module ``nodes/nvfp4/nvfp4_hadamard.py``.
"""
from __future__ import annotations

import math

_HADAMARD_CACHE: dict = {}
_H4_CACHE: dict = {}


def _tensor_storage_ok(t) -> bool:
    """False after Distorch nuclear kill / empty-storage reuse (UAF risk)."""
    if t is None:
        return False
    try:
        if int(getattr(t, "numel", lambda: 0)()) <= 0:
            return False
        st = t.untyped_storage() if hasattr(t, "untyped_storage") else t.storage()
        if int(st.nbytes()) <= 0:
            return False
        # Shape must match a square Hadamard (or 4x4 h4); reject emptied shells.
        if getattr(t, "ndim", 0) == 2:
            if int(t.shape[0]) != int(t.shape[1]) or int(t.shape[0]) < 4:
                return False
        return True
    except Exception:
        return False


def clear_hadamard_global_caches() -> int:
    """Drop module-level Hadamard caches (Distorch Method 2c / parity clear).

    Method 3 may ``t.data = empty`` on tensors still referenced by these dicts.
    Returning them on the next gen rotates with dead/garbage ``H`` and quality
    decays as CUDA reallocates the freed region (2nd→3rd→4th gen worse).
    """
    n = len(_HADAMARD_CACHE) + len(_H4_CACHE)
    _HADAMARD_CACHE.clear()
    _H4_CACHE.clear()
    return n


def build_hadamard(size: int, device="cpu", dtype=None):
    """Build (and cache) a normalized Hadamard matrix.

    Always construct in float32 (CPU or GPU), then cast to ``dtype``.
    Building the Kronecker product directly in float16 destroys ConvRot
    orthonormality and collapses NVFP4 quality.
    """
    import torch

    if dtype is None:
        dtype = torch.float32
    device = torch.device(device) if not isinstance(device, torch.device) else device
    cache_key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(cache_key)
    if cached is not None and _tensor_storage_ok(cached):
        return cached
    if cached is not None:
        _HADAMARD_CACHE.pop(cache_key, None)
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    master_key = (size, str(device), torch.float32)
    h_matrix = _HADAMARD_CACHE.get(master_key)
    if h_matrix is None or not _tensor_storage_ok(h_matrix):
        if h_matrix is not None:
            _HADAMARD_CACHE.pop(master_key, None)
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=torch.float32,
            device=device,
        )
        h_matrix = h4
        current_size = 4
        while current_size < size:
            h_matrix = torch.kron(h_matrix, h4)
            current_size *= 4
        h_matrix = h_matrix / (size**0.5)
        _HADAMARD_CACHE[master_key] = h_matrix
    if dtype != torch.float32:
        h_matrix = h_matrix.to(dtype=dtype)
    _HADAMARD_CACHE[cache_key] = h_matrix
    return h_matrix


def _h4(device, dtype):
    import torch

    key = (str(device), dtype)
    h = _H4_CACHE.get(key)
    if h is None or not _tensor_storage_ok(h):
        if h is not None:
            _H4_CACHE.pop(key, None)
        h = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=dtype,
            device=device,
        )
        _H4_CACHE[key] = h
    return h


def _apply_kron_h4_unnorm(x2d, size: int):
    """Right-multiply by unnormalized Kronecker power of h4 (same as build_hadamard).

    ``x2d`` shape ``(M, size)`` with ``size == 4**k``. Equivalent to
    ``x2d @ kron_power(h4)`` before the ``/sqrt(size)`` normalization.
    """
    import torch

    if size == 4:
        return torch.matmul(x2d, _h4(x2d.device, x2d.dtype))
    p = size // 4
    # H_size = H_p ⊗ h4  (left-associated kron growth in build_hadamard)
    x = x2d.reshape(-1, p, 4)
    y = torch.matmul(x, _h4(x2d.device, x2d.dtype))  # apply h4 on last dim
    # apply H_p on the middle dim: for each of 4 cols, (M,p) @ H_p
    yt = y.transpose(1, 2).reshape(-1, p)
    yt = _apply_kron_h4_unnorm(yt, p)
    z = yt.reshape(-1, 4, p).transpose(1, 2)
    return z.reshape(-1, size)


def rotate_last_dim(x, h_matrix, group_size: int):
    import torch

    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    group_count = features // group_size
    x_grouped = x.reshape(-1, group_count, group_size)
    if h_matrix.device == x.device and h_matrix.dtype == x.dtype:
        h = h_matrix
    else:
        h = h_matrix.to(dtype=x.dtype, device=x.device)
    return torch.matmul(x_grouped, h).reshape(orig_shape)


def rotate_weight_linear(weight, h_matrix, group_size: int):
    """Offline Linear: W_rot = W @ H^T (group-wise along in_features)."""
    import torch

    if getattr(weight, "ndim", 0) != 2:
        raise ValueError(f"Linear weight must be 2D, got ndim={getattr(weight, 'ndim', None)}")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    h_t = h_matrix.T.to(dtype=weight.dtype, device=weight.device)
    return torch.matmul(weight_grouped, h_t).reshape(weight.shape)


def unrotate_weight_linear(weight, h_matrix, group_size: int):
    """Inverse of rotate_weight_linear: W = W_rot @ H (for LoRA float space)."""
    import torch

    if getattr(weight, "ndim", 0) != 2:
        raise ValueError(f"Linear weight must be 2D, got ndim={getattr(weight, 'ndim', None)}")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    h = h_matrix.to(dtype=weight.dtype, device=weight.device)
    return torch.matmul(weight_grouped, h).reshape(weight.shape)


def rotate_last_dim_fast(x, group_size: int):
    """Same math as ``rotate_last_dim`` + ``build_hadamard``, O(n log n) butterflies.

    Avoids materializing the dense ``group_size x group_size`` Hadamard and the
    large GEMM that dominates online FULL ConvRot act rotation.
    """
    import torch

    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    flat = x.reshape(-1, group_size)
    y = _apply_kron_h4_unnorm(flat, group_size)
    y = y * (group_size**-0.5)
    return y.reshape(orig_shape)
```

### `prestartup_script.py`

```python
"""Wire Z Image UNet ConvRot NVFP4 without regressing the product loader.

ComfyUI runs this before the custom-node ``__init__.py``. We keep a reference to
the *original* ``comfy_quant_nvfp4.load_unet_nvfp4_weight_dtype`` (INT8 protect +
disable_dynamic + LoRA bake + stock GEMM + act rotate), then optionally rebind
the module attribute to ``nodes.zimage_nvfp4.load_unet`` which *delegates* to that
saved original — never to the rebound name (that would recurse).

SDXL ``load_checkpoint_sdxl_nvfp4_weight_dtype`` is left unchanged.

Do NOT insert this package root onto ``sys.path``. That shadows ComfyUI's top-level
``nodes`` module and crashes startup with::

    AttributeError: module 'nodes' has no attribute 'init_extra_nodes'
"""
from __future__ import annotations

import builtins
import importlib
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

_PATCHED = False
_ORIG_IMPORT = builtins.__import__
_PRODUCT_LOAD_UNET = None


def _zimage_load_module():
    """Resolve zimage load only via the already-imported HSWQ package prefix."""
    for name in list(sys.modules):
        if not name.endswith("nodes.nvfp4.comfy_quant_nvfp4"):
            continue
        pkg = name[: -len(".nodes.nvfp4.comfy_quant_nvfp4")]
        if not pkg:
            continue
        return importlib.import_module(f"{pkg}.nodes.zimage_nvfp4.load_unet")
    raise ImportError(
        "comfy_quant_nvfp4 not in sys.modules yet "
        "(cannot import nodes.zimage_nvfp4 without shadowing ComfyUI nodes)"
    )


def _try_patch() -> bool:
    global _PATCHED, _PRODUCT_LOAD_UNET
    if _PATCHED:
        return True
    try:
        zl = _zimage_load_module()
    except Exception as e:
        print(f"[HSWQ NVFP4] Z Image load import deferred: {e}", flush=True)
        return False
    for name, mod in list(sys.modules.items()):
        if not (
            name.endswith("nodes.nvfp4.comfy_quant_nvfp4")
            or name.endswith(".comfy_quant_nvfp4")
            or name == "comfy_quant_nvfp4"
        ):
            continue
        if not hasattr(mod, "load_unet_nvfp4_weight_dtype"):
            continue
        # Save product implementation *before* rebind (avoid recursion).
        _PRODUCT_LOAD_UNET = mod.load_unet_nvfp4_weight_dtype
        zl._PRODUCT_LOAD_UNET = _PRODUCT_LOAD_UNET
        mod.load_unet_nvfp4_weight_dtype = zl.load_unet_nvfp4_weight_dtype
        _PATCHED = True
        print(
            "[HSWQ NVFP4] UNet ConvRot NVFP4 -> nodes.zimage_nvfp4 "
            "(delegates to saved product: GEMM + act rotate + int8 + LoRA bake + "
            "disable_dynamic)",
            flush=True,
        )
        return True
    return False


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _ORIG_IMPORT(name, globals, locals, fromlist, level)
    if not _PATCHED and "comfy_quant_nvfp4" in str(name):
        _try_patch()
    elif not _PATCHED and fromlist:
        if any("comfy_quant_nvfp4" in str(x) for x in fromlist):
            _try_patch()
    return mod


builtins.__import__ = _import
print(
    "[HSWQ NVFP4] prestartup: Z Image ConvRot NVFP4 product path armed",
    flush=True,
)
_try_patch()
```

### `nodes/nvfp4/comfy_quant_nvfp4.py`

```python
"""
ComfyUI runtime monkey-patches for HSWQ comfy_quant NVFP4 (FULL ConvRot).

Runtime only — never permanently edit ComfyUI-master.

Owns (via sibling modules under nodes/nvfp4/):
  - packed-K UNet detection (logical in_features)
  - full NVFP4 Linear load (scales, QT, ConvRot flags, storage validation)
  - full Tensor Core forward (act ConvRot → NVFP4 quant → scaled_mm_nvfp4)
  - ConvRot NVFP4 Linear LoRA bake (convert_weight unrotate → set_weight re-rotate)

This is not an INT8/FP8 “small tweak”: load + forward are HSWQ-owned stacks.
"""
from __future__ import annotations

import logging

from .nvfp4_conf import (
    checkpoint_looks_like_comfy_quant_nvfp4,
    decode_comfy_quant_conf,
    fix_unet_config_packed_dims,
    is_nvfp4_conf,
    logical_linear_in_features,
)
from .nvfp4_forward import (
    attach_nvfp4_linear_lora_bake,
    make_nvfp4_linear_forward,
    nvfp4_forward_stats,
    peel_all_nvfp4_linear_lora_bake,
    reset_nvfp4_forward_stats,
    reset_nvfp4_lora_log_counters,
)
from .nvfp4_load import load_nvfp4_linear_module, peek_nvfp4_conf

logger = logging.getLogger(__name__)
_PATCHES_APPLIED = False
# Bump when NVFP4 stack contract changes (forces re-wire of mixed_precision_ops).
_NVFP4_STACK_VER = 2

# Re-export for benches / callers
__all__ = [
    "NVFP4_WEIGHT_DTYPE",
    "apply_comfy_quant_nvfp4_patches",
    "checkpoint_looks_like_comfy_quant_nvfp4",
    "decode_comfy_quant_conf",
    "install_nvfp4_option_dispatch",
    "is_nvfp4_conf",
    "load_checkpoint_sdxl_nvfp4_weight_dtype",
    "logical_linear_in_features",
    "nvfp4_forward_stats",
    "reset_nvfp4_forward_stats",
    "reset_nvfp4_lora_log_counters",
]


def _console(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def _clear_zimage_parity_contamination_for_sdxl() -> None:
    """Peel Z Image comfy_parity + ZI bake hooks before SDXL TC / INT8 load.

    Owner log (SDXL → Z Image → SDXL): after Z Image, ``comfy_parity`` stays on
    ``ops._load_quantized_module`` / ``mixed_precision_ops`` and ZI Dynamic.load
    bake hijacks SDXL → ``arm INT8 protect`` on SDXL NVFP4, ``nvfp4_baked=0``,
    salt-pepper. Later: ZI VER=8 ``[HSWQ ConvRot LoRA] int8_protect`` on SDXL
    INT8 → LoRA falls off on the 3rd prompt. Restore product TC (or peel to
    stock) and uninstall ZI bake hooks.
    """
    try:
        from ..zimage_nvfp4.nvfp4_comfy_parity import (
            peel_non_product_nvfp4_ops,
            restore_nvfp4_tc_product_stack,
        )

        restore_nvfp4_tc_product_stack()
        try:
            import comfy.ops as ops

            peel_non_product_nvfp4_ops(ops)
        except Exception:
            pass
    except Exception as e:
        logger.warning("[HSWQ NVFP4] restore TC stack for SDXL failed: %s", e)
    try:
        from ..zimage_nvfp4.nvfp4_lora_bake import uninstall_zimage_nvfp4_lora_bake

        uninstall_zimage_nvfp4_lora_bake()
    except Exception as e:
        logger.warning("[HSWQ NVFP4] uninstall ZI bake hooks for SDXL failed: %s", e)
    # Z Image mutates mp0.Linear in place; peel ops wrappers alone leaves VER=8.
    try:
        import comfy.ops as ops

        Lin = ops.mixed_precision_ops().Linear
        peeled = peel_all_nvfp4_linear_lora_bake(Lin)
        mp_fn = ops.mixed_precision_ops
        if getattr(mp_fn, "_hswq_nvfp4_product_tc", False):
            if attach_nvfp4_linear_lora_bake(Lin) or peeled:
                _console(
                    "[HSWQ NVFP4] SDXL product Linear LoRA bake VER=1 on live Linear"
                )
        elif peeled:
            _console(
                "[HSWQ NVFP4] peeled Z Image Linear LoRA bake (int8_protect) "
                "off live Linear — SDXL INT8/stock safe"
            )
    except Exception as e:
        logger.warning(
            "[HSWQ NVFP4] peel live Linear LoRA bake for SDXL failed: %s", e
        )


def apply_comfy_quant_nvfp4_patches() -> bool:
    """Install NVFP4 detection + full load + TC Linear forward + ConvRot LoRA bake."""
    global _PATCHES_APPLIED
    try:
        import comfy.model_detection as model_detection
        import comfy.ops as ops
    except Exception as e:
        logger.warning("[HSWQ NVFP4] comfy import failed: %s", e)
        return False

    # Always peel Z Image parity before touching / early-returning the SDXL stack.
    _clear_zimage_parity_contamination_for_sdxl()

    mp_fn = getattr(ops, "mixed_precision_ops", None)
    load_fn = getattr(ops, "_load_quantized_module", None)
    stack_ver = int(getattr(mp_fn, "_hswq_nvfp4_stack_ver", 0) or 0) if mp_fn else 0
    parity_still = bool(
        getattr(mp_fn, "_hswq_nvfp4_comfy_only", False)
        or getattr(load_fn, "_hswq_nvfp4_comfy_only", False)
    )
    # Early return only when the live ops are SDXL product TC (stamped), not Z Image.
    if (
        _PATCHES_APPLIED
        and not parity_still
        and getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False)
        and stack_ver >= _NVFP4_STACK_VER
        and getattr(mp_fn, "_hswq_nvfp4_full_forward", False)
        and getattr(mp_fn, "_hswq_nvfp4_product_tc", False)
        and getattr(load_fn, "_hswq_nvfp4_full_load", False)
        and getattr(load_fn, "_hswq_nvfp4_product_tc", False)
    ):
        try:
            from ..zimage_nvfp4.nvfp4_comfy_parity import remember_nvfp4_tc_product_stack

            remember_nvfp4_tc_product_stack(load_fn, mp_fn)
        except Exception:
            pass
        return True

    # Already patched detect/load but LoRA bake missing: re-wrap mixed_precision_ops only.
    # Never upgrade while comfy_parity is still live (parity copies stack_ver from TC base).
    if (
        not parity_still
        and getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False)
        and stack_ver < _NVFP4_STACK_VER
    ):
        _orig_mp = getattr(mp_fn, "_hswq_nvfp4_orig_mp", mp_fn)

        def mixed_precision_ops_upgraded(*args, **kwargs):
            mp = _orig_mp(*args, **kwargs)
            Lin = mp.Linear
            if not getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
                Lin.forward = make_nvfp4_linear_forward(Lin.forward)
            attach_nvfp4_linear_lora_bake(Lin)
            return mp

        mixed_precision_ops_upgraded._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]
        mixed_precision_ops_upgraded._hswq_nvfp4_stack_ver = _NVFP4_STACK_VER  # type: ignore[attr-defined]
        mixed_precision_ops_upgraded._hswq_nvfp4_orig_mp = _orig_mp  # type: ignore[attr-defined]
        mixed_precision_ops_upgraded._hswq_nvfp4_product_tc = True  # type: ignore[attr-defined]
        # Stamp load if it is already SDXL product TC (may lack stamp from older session).
        cur_load = ops._load_quantized_module
        if getattr(cur_load, "_hswq_nvfp4_full_load", False) and not getattr(
            cur_load, "_hswq_nvfp4_comfy_only", False
        ):
            try:
                cur_load._hswq_nvfp4_product_tc = True  # type: ignore[attr-defined]
            except Exception:
                pass
        ops.mixed_precision_ops = mixed_precision_ops_upgraded
        _PATCHES_APPLIED = True
        try:
            from ..zimage_nvfp4.nvfp4_comfy_parity import remember_nvfp4_tc_product_stack

            remember_nvfp4_tc_product_stack(
                ops._load_quantized_module, mixed_precision_ops_upgraded
            )
        except Exception:
            pass
        _console(
            "[HSWQ NVFP4] upgraded stack ver=%s "
            "(ConvRot Linear LoRA bake: convert_weight unrotate + set_weight re-rotate)"
            % _NVFP4_STACK_VER
        )
        return True

    # Refuse wrapping TC on top of leftover comfy_parity (would bake SDXL as INT8 protect).
    if parity_still:
        _clear_zimage_parity_contamination_for_sdxl()
        mp_fn = getattr(ops, "mixed_precision_ops", None)
        load_fn = getattr(ops, "_load_quantized_module", None)
        parity_still = bool(
            getattr(mp_fn, "_hswq_nvfp4_comfy_only", False)
            or getattr(load_fn, "_hswq_nvfp4_comfy_only", False)
        )
        if parity_still:
            logger.error(
                "[HSWQ NVFP4] comfy_parity still on ops after restore — "
                "refusing full TC reinstall on top of parity (would corrupt SDXL)"
            )
            return False
        # Restored product TC: early-return path if already at current stack ver.
        stack_ver = int(getattr(mp_fn, "_hswq_nvfp4_stack_ver", 0) or 0) if mp_fn else 0
        if (
            _PATCHES_APPLIED
            and getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False)
            and stack_ver >= _NVFP4_STACK_VER
            and getattr(mp_fn, "_hswq_nvfp4_full_forward", False)
            and getattr(mp_fn, "_hswq_nvfp4_product_tc", False)
            and getattr(load_fn, "_hswq_nvfp4_full_load", False)
            and getattr(load_fn, "_hswq_nvfp4_product_tc", False)
        ):
            try:
                from ..zimage_nvfp4.nvfp4_comfy_parity import remember_nvfp4_tc_product_stack

                remember_nvfp4_tc_product_stack(load_fn, mp_fn)
            except Exception:
                pass
            return True

    _orig_detect = model_detection.detect_unet_config
    _orig_calc = model_detection.calculate_transformer_depth
    _orig_load = ops._load_quantized_module
    _orig_mp = ops.mixed_precision_ops

    def calculate_transformer_depth_patched(prefix, state_dict_keys, state_dict):
        out = _orig_calc(prefix, state_dict_keys, state_dict)
        if out is None:
            return None
        depth, context_dim, use_linear, time_stack, time_stack_cross = out
        k = f"{prefix}1.transformer_blocks.0.attn2.to_k.weight"
        if k in state_dict:
            try:
                context_dim = logical_linear_in_features(state_dict, k)
            except Exception as e:
                logger.warning("[HSWQ NVFP4] transformer context_dim fix skipped: %s", e)
        return depth, context_dim, use_linear, time_stack, time_stack_cross

    def detect_unet_config_patched(state_dict, key_prefix, metadata=None):
        unet_config = _orig_detect(state_dict, key_prefix, metadata=metadata)
        if unet_config is None:
            return None
        return fix_unet_config_packed_dims(unet_config, state_dict, key_prefix)

    def model_config_from_unet_patched(
        state_dict, unet_key_prefix, use_base_if_no_match=False, metadata=None
    ):
        import comfy.supported_models_base
        import comfy.utils

        unet_config = model_detection.detect_unet_config(
            state_dict, unet_key_prefix, metadata=metadata
        )
        if unet_config is None:
            return None
        model_config = model_detection.model_config_from_unet_config(
            unet_config, state_dict, unet_key_prefix
        )
        if model_config is None and use_base_if_no_match:
            model_config = comfy.supported_models_base.BASE(unet_config)

        quant_config = comfy.utils.detect_layer_quantization(
            state_dict, unet_key_prefix
        )
        if quant_config:
            if model_config is None:
                logging.error(
                    "[HSWQ NVFP4] model_config is None with quant_config present "
                    "(packed NVFP4 dims still unmatched?). prefix=%r config=%s",
                    unet_key_prefix,
                    unet_config,
                )
                return None
            model_config.quant_config = quant_config
            logging.info("Detected mixed precision quantization")
        return model_config

    def _load_quantized_module_patched(
        module,
        super_load,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        load_extra_params=False,
    ):
        conf = peek_nvfp4_conf(state_dict, prefix)
        if is_nvfp4_conf(conf):
            load_nvfp4_linear_module(
                module,
                super_load,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
                load_extra_params=load_extra_params,
            )
            return
        _orig_load(
            module,
            super_load,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            load_extra_params=load_extra_params,
        )
        # Non-nvfp4 path: leave stock. (INT8 ConvRot etc. stay on stock/int8 patches.)

    def mixed_precision_ops_patched(*args, **kwargs):
        mp = _orig_mp(*args, **kwargs)
        Lin = mp.Linear
        if not getattr(Lin.forward, "_hswq_nvfp4_full_forward", False):
            Lin.forward = make_nvfp4_linear_forward(Lin.forward)
        attach_nvfp4_linear_lora_bake(Lin)
        return mp

    model_detection.calculate_transformer_depth = calculate_transformer_depth_patched
    model_detection.detect_unet_config = detect_unet_config_patched
    model_detection.model_config_from_unet = model_config_from_unet_patched
    ops._load_quantized_module = _load_quantized_module_patched
    ops.mixed_precision_ops = mixed_precision_ops_patched

    detect_unet_config_patched._hswq_nvfp4_packed_dims = True  # type: ignore[attr-defined]
    calculate_transformer_depth_patched._hswq_nvfp4_packed_dims = True  # type: ignore[attr-defined]
    model_config_from_unet_patched._hswq_nvfp4_packed_dims = True  # type: ignore[attr-defined]
    _load_quantized_module_patched._hswq_nvfp4_full_load = True  # type: ignore[attr-defined]
    _load_quantized_module_patched._hswq_nvfp4_product_tc = True  # type: ignore[attr-defined]
    mixed_precision_ops_patched._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]
    mixed_precision_ops_patched._hswq_nvfp4_stack_ver = _NVFP4_STACK_VER  # type: ignore[attr-defined]
    mixed_precision_ops_patched._hswq_nvfp4_orig_mp = _orig_mp  # type: ignore[attr-defined]
    mixed_precision_ops_patched._hswq_nvfp4_product_tc = True  # type: ignore[attr-defined]

    _PATCHES_APPLIED = True
    try:
        from ..zimage_nvfp4.nvfp4_comfy_parity import remember_nvfp4_tc_product_stack

        remember_nvfp4_tc_product_stack(
            _load_quantized_module_patched, mixed_precision_ops_patched
        )
    except Exception:
        pass
    _console(
        "[HSWQ NVFP4] full stack applied "
        "(detect packed K + nvfp4_load + TC forward + ConvRot act + "
        "ConvRot Linear LoRA bake; ComfyUI-master untouched)"
    )
    return True


# UI / dispatch value — HSWQ Checkpoint Loader (SDXL) dropdown ONLY.
# Z Image / Krea UNet uses ZI_NVFP4_WEIGHT_DTYPE == "Z Image ConvRot NVFP4"
# (separate being — never the SDXL string below).
NVFP4_WEIGHT_DTYPE = "ConvRot NVFP4"


def load_checkpoint_sdxl_nvfp4_weight_dtype(ckpt_name, weight_dtype, device=None):
    """Load SDXL checkpoint with HSWQ NVFP4 Linear (+ INT8 Conv2d ConvRot) stack."""
    import sys

    import folder_paths
    import comfy.sd

    # Package root = ComfyUI-nunchaku-unofficial-loader
    pkg = sys.modules[__name__.rsplit(".", 3)[0]]
    get_current_device = pkg.get_current_device
    set_current_device = pkg.set_current_device
    sdxl_logger = pkg.sdxl_logger

    from ...patches.comfy_quant_int8 import (
        _int8_quant_conv_scope,
        apply_comfy_quant_int8_patches,
        reset_int8_lora_log_counters,
        summarize_int8_lora_capability,
    )

    original_device = get_current_device()
    if device is not None:
        set_current_device(device)
    try:
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        apply_comfy_quant_nvfp4_patches()
        # Mixed pack: Linear=nvfp4, Conv2d=int8_tensorwise (+ ConvRot) — same as bench.
        apply_comfy_quant_int8_patches()
        reset_int8_lora_log_counters()
        reset_nvfp4_lora_log_counters()
        sdxl_logger.info(
            "[SDXL NVFP4] Loading checkpoint via MixedPrecisionOps "
            "(nvfp4 Linear + int8 Conv / ConvRot + ConvRot Linear LoRA bake): "
            "%s (weight_dtype=%s)",
            ckpt_name,
            weight_dtype,
        )
        with _int8_quant_conv_scope():
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=False,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                model_options={},
            )
        model, clip, _v = out[:3]
        summarize_int8_lora_capability(model)
        return (model, clip)
    finally:
        set_current_device(original_device)


def install_nvfp4_option_dispatch(node_class_mappings) -> bool:
    """Wrap SDXL loader so ConvRot NVFP4 uses nodes/nvfp4 (bench) stack.

    Must run *after* ``install_int8_option_dispatch``: NVFP4 checkpoints also
    contain ``int8_tensorwise`` Conv layers, so INT8-only auto-detect would
    otherwise steal the load path without NVFP4 Linear patches.
    """
    if not isinstance(node_class_mappings, dict):
        return False

    _FP8_WEIGHT_DTYPES = frozenset({"fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"})

    sdxl_cls = node_class_mappings.get("HSWQCheckpointLoaderSDXL")
    if sdxl_cls is None:
        return False

    _prev_load_checkpoint = sdxl_cls.load_checkpoint

    def load_checkpoint(self, ckpt_name, weight_dtype, device=None):
        if weight_dtype in _FP8_WEIGHT_DTYPES:
            return _prev_load_checkpoint(self, ckpt_name, weight_dtype, device=device)
        if weight_dtype == NVFP4_WEIGHT_DTYPE:
            return load_checkpoint_sdxl_nvfp4_weight_dtype(
                ckpt_name, weight_dtype, device=device
            )
        import folder_paths

        # default (and any non-FP8 path): NVFP4 markers beat INT8-only auto-detect.
        # Mixed packs also have int8_tensorwise Conv layers.
        if weight_dtype == "default":
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            if checkpoint_looks_like_comfy_quant_nvfp4(ckpt_path):
                return load_checkpoint_sdxl_nvfp4_weight_dtype(
                    ckpt_name, weight_dtype, device=device
                )
        return _prev_load_checkpoint(self, ckpt_name, weight_dtype, device=device)

    sdxl_cls.load_checkpoint = load_checkpoint
    _console(
        "[HSWQ NVFP4] install_nvfp4_option_dispatch: "
        f"SDXL weight_dtype includes {NVFP4_WEIGHT_DTYPE!r}"
    )
    return True
```

### `nodes/nvfp4/nvfp4_forward.py`

```python
"""
HSWQ-owned NVFP4 Linear Tensor Core forward path.

Stock MixedPrecision Linear inference does:
  reshape → QuantizedTensor.from_float(act) → F.linear → often aten.addmm
  → unregistered → full dequant (slow), or wrong reshape via weight.shape[0]
  (QT storage dim).

This module owns the full inference path for HSWQ NVFP4 (+ optional ConvRot):
  1) reshape act to 2D
  2) FULL ConvRot via dense Hadamard GEMM (butterfly is slower for gs=256)
  3) cast weight/bias when off-device
  4) pooled CUDA quantize_nvfp4 (no per-call alloc)
  5) pooled cuBLAS scaled_mm_nvfp4
  6) reshape with module.out_features (never QT storage shape[0])

Never edits ComfyUI-master; installed via monkey-patch on MixedPrecision Linear.
"""
from __future__ import annotations

import logging
import os

from .nvfp4_hadamard import (
    build_hadamard,
    rotate_weight_linear,
    unrotate_weight_linear,
)
from .nvfp4_runtime import (
    ensure_act_scale,
    clear_nvfp4_cudagraphs,
    nvfp4_quant_mm_cudagraph,
    quantize_nvfp4_act_pooled,
    rotate_last_dim_pooled,
    scaled_mm_nvfp4_pooled,
    _GRAPH_MAX_M,
)

logger = logging.getLogger(__name__)

# Counters for bench / diagnostics (reset per run if needed)
_TC_HITS = 0
_DEQUANT_FALLBACKS = 0
_LORA_CONVERT_LOGS = 0
_LORA_SET_LOGS = 0
_LORA_LOG_MAX = 8
# Bump when convert_weight / set_weight ConvRot LoRA bake changes.
_NVFP4_LORA_BAKE_VER = 1


def reset_nvfp4_lora_log_counters() -> None:
    global _LORA_CONVERT_LOGS, _LORA_SET_LOGS
    _LORA_CONVERT_LOGS = 0
    _LORA_SET_LOGS = 0


def reset_nvfp4_forward_stats() -> None:
    global _TC_HITS, _DEQUANT_FALLBACKS
    _TC_HITS = 0
    _DEQUANT_FALLBACKS = 0


def nvfp4_forward_stats() -> dict:
    return {"scaled_mm_hits": _TC_HITS, "dequant_fallbacks": _DEQUANT_FALLBACKS}


def _slice_nvfp4_mm_out(result, orig_m: int, orig_n: int):
    if result.shape[0] != orig_m or result.shape[1] != orig_n:
        return result[:orig_m, :orig_n]
    return result


def scaled_mm_nvfp4_linear(input_qt, weight_qt, bias):
    """Kitchen / tritant NVFP4 linear (QT path; used as fallback)."""
    global _TC_HITS, _DEQUANT_FALLBACKS
    import torch
    import torch.nn.functional as F
    import comfy_kitchen as ck
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    if not (
        isinstance(input_qt, QuantizedTensor)
        and isinstance(weight_qt, QuantizedTensor)
        and input_qt._layout_cls == "TensorCoreNVFP4Layout"
        and weight_qt._layout_cls == "TensorCoreNVFP4Layout"
    ):
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)
    if input_qt._qdata.dim() != 2:
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)
    if getattr(input_qt._params, "transposed", False) or getattr(
        weight_qt._params, "transposed", False
    ):
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()

    a_qdata, scale_a, block_scale_a = TensorCoreNVFP4Layout.get_plain_tensors(input_qt)
    w_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(weight_qt)
    out_dtype = input_qt._params.orig_dtype
    try:
        result = ck.scaled_mm_nvfp4(
            a_qdata,
            w_qdata,
            tensor_scale_a=scale_a,
            tensor_scale_b=scale_b,
            block_scale_a=block_scale_a,
            block_scale_b=block_scale_b,
            bias=bias,
            out_dtype=out_dtype,
        )
        orig_m = input_qt._params.orig_shape[0]
        orig_n = weight_qt._params.orig_shape[0]  # (out, in)
        _TC_HITS += 1
        return _slice_nvfp4_mm_out(result, orig_m, orig_n)
    except (RuntimeError, TypeError) as e:
        logger.warning("[HSWQ NVFP4] scaled_mm_nvfp4 failed: %s — F.linear dequant", e)
        _DEQUANT_FALLBACKS += 1
        return F.linear(input_qt, weight_qt, bias)


def _plain_weight_cached(module, weight_qt):
    """Cache get_plain_tensors on the module (weight QT identity stable after load)."""
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    cached = getattr(module, "_hswq_nvfp4_w_plain", None)
    if cached is not None and cached[0] is weight_qt._qdata:
        return cached[1], cached[2], cached[3], cached[4]
    w_qdata, scale_b, block_scale_b = TensorCoreNVFP4Layout.get_plain_tensors(weight_qt)
    orig_n = int(weight_qt._params.orig_shape[0])
    module._hswq_nvfp4_w_plain = (
        weight_qt._qdata,
        w_qdata,
        scale_b,
        block_scale_b,
        orig_n,
    )
    return w_qdata, scale_b, block_scale_b, orig_n


def _tc_forward_pooled(module, input_2d, weight_qt, bias, act_scale, out_dtype):
    """Act float → pooled NVFP4 quant → pooled cuBLAS mm (no QT alloc).

    Prefers CUDA Graph (quantize+mm) after first capture per shape/weight; falls
    back to eager pooled kernels if capture/replay fails.
    """
    global _TC_HITS, _DEQUANT_FALLBACKS
    import torch
    from comfy_kitchen.tensor.base import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    if not (
        isinstance(weight_qt, QuantizedTensor)
        and weight_qt._layout_cls == "TensorCoreNVFP4Layout"
    ):
        _DEQUANT_FALLBACKS += 1
        return None
    if getattr(weight_qt._params, "transposed", False):
        _DEQUANT_FALLBACKS += 1
        return None

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()

    orig_m, orig_k = int(input_2d.shape[0]), int(input_2d.shape[1])
    needs_padding = TensorCoreNVFP4Layout.get_padded_shape((orig_m, orig_k)) != (
        orig_m,
        orig_k,
    )

    scale_a = ensure_act_scale(input_2d, act_scale)
    try:
        w_qdata, scale_b, block_scale_b, orig_n = _plain_weight_cached(module, weight_qt)

        # Calib input_scale and placeholder ones are static — always cache
        # alpha. Recomputing scale_a*scale_b every Linear (~18k/sample) was
        # pure waste on FULL ConvRot (every layer has input_scale).
        cached_alpha = getattr(module, "_hswq_nvfp4_alpha", None)
        if cached_alpha is None:
            alpha = scale_a * scale_b
            if alpha.dtype != torch.float32:
                alpha = alpha.to(dtype=torch.float32)
            if alpha.dim() == 0:
                alpha = alpha.reshape(1)
            module._hswq_nvfp4_alpha = alpha
        else:
            alpha = cached_alpha

        # CUDA Graph is OFF by default: shape-shared replay copies full weight
        # every call and was slower than eager (13.05s vs ~11.8s). Opt-in:
        # HSWQ_NVFP4_CUDAGRAPH=1
        use_cg = (
            os.environ.get("HSWQ_NVFP4_CUDAGRAPH", "").strip() == "1"
            and orig_m <= _GRAPH_MAX_M
            and not getattr(module, "_hswq_nvfp4_no_cudagraph", False)
        )
        if use_cg:
            try:
                result = nvfp4_quant_mm_cudagraph(
                    input_2d,
                    w_qdata=w_qdata,
                    weight_scale=scale_b,
                    block_scale_w=block_scale_b,
                    scale_a=scale_a,
                    bias=bias,
                    out_dtype=out_dtype,
                    alpha=alpha,
                    pad_16x=needs_padding,
                    orig_n=orig_n,
                )
                _TC_HITS += 1
                return result
            except torch.cuda.OutOfMemoryError:
                clear_nvfp4_cudagraphs()
                torch.cuda.empty_cache()
                logger.warning(
                    "[HSWQ NVFP4] CUDA Graph OOM — cache cleared; eager pooled"
                )
            except (RuntimeError, TypeError, ValueError) as e:
                if "out of memory" in str(e).lower():
                    clear_nvfp4_cudagraphs()
                    torch.cuda.empty_cache()
                    logger.warning(
                        "[HSWQ NVFP4] CUDA Graph OOM (%s); eager pooled", e
                    )
                else:
                    module._hswq_nvfp4_no_cudagraph = True
                    logger.warning(
                        "[HSWQ NVFP4] CUDA Graph disabled for module (%s); eager pooled",
                        e,
                    )

        a_qdata, block_scale_a, _pr, _pc = quantize_nvfp4_act_pooled(
            input_2d, scale_a, pad_16x=needs_padding
        )
        result = scaled_mm_nvfp4_pooled(
            a_qdata,
            w_qdata,
            tensor_scale_a=scale_a,
            tensor_scale_b=scale_b,
            block_scale_a=block_scale_a,
            block_scale_b=block_scale_b,
            bias=bias,
            out_dtype=out_dtype,
            alpha=alpha,
            orig_m=orig_m,
            orig_n=orig_n,
        )
        _TC_HITS += 1
        return result
    except (RuntimeError, TypeError, ValueError) as e:
        logger.warning("[HSWQ NVFP4] pooled TC path failed: %s", e)
        _DEQUANT_FALLBACKS += 1
        return None


def make_nvfp4_linear_forward(stock_forward):
    """
    Return a Linear.forward replacement.

    For modules flagged ``_hswq_nvfp4`` (set at load), run the HSWQ TC path.
    All other layers keep stock_forward unchanged.
    """
    import torch
    import comfy.model_management
    from comfy.ops import cast_bias_weight, run_every_op, uncast_bias_weight

    def forward_nvfp4(self, input, *args, **kwargs):
        if not getattr(self, "_hswq_nvfp4", False) or getattr(self, "_full_precision_mm", False):
            return stock_forward(self, input, *args, **kwargs)

        # Training / forced cast: fall back to stock
        if input.requires_grad or getattr(self, "comfy_force_cast_weights", False):
            return stock_forward(self, input, *args, **kwargs)
        # LoRA weight_function: stay on HSWQ path (act ConvRot + cast_bias_weight
        # with want_requant). Stock forward would skip act rotate → ConvRot break.

        run_every_op()
        input_shape = input.shape
        compute_dtype = input.dtype

        # 1) Reshape ≥3D → 2D first (same last-dim math; cheaper than rotating ND)
        reshaped_nd = input.ndim >= 3
        input_2d = input.reshape(-1, input_shape[-1]) if reshaped_nd else input
        if input_2d.ndim != 2:
            return stock_forward(self, input, *args, **kwargs)

        # 2) FULL ConvRot: dense Hadamard GEMM (gs=256 butterfly is ~15x slower)
        if getattr(self, "_hswq_nvfp4_convrot", False):
            gs = int(getattr(self, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
            h = getattr(self, "_hswq_nvfp4_H", None)
            if h is None or h.device != input_2d.device or h.dtype != input_2d.dtype:
                h = build_hadamard(gs, device=input_2d.device, dtype=input_2d.dtype)
                self._hswq_nvfp4_H = h
            input_2d = rotate_last_dim_pooled(input_2d, h, gs)

        # 3) Weight / bias: skip cast_bias_weight when already on-device QT
        #    (cast+sync every Linear was a major share of NVFP4 > FP16 wall time).
        #    Always cast when LoRA weight/bias_function present (need bake apply).
        offload_stream = None
        weight = self.weight
        if isinstance(weight, torch.nn.Parameter):
            weight = weight.data
        bias = self.bias.data if self.bias is not None else None
        has_wf = len(getattr(self, "weight_function", []) or []) or len(
            getattr(self, "bias_function", []) or []
        )
        need_cast = weight.device != input_2d.device or (
            bias is not None and bias.device != input_2d.device
        )
        if has_wf or need_cast or hasattr(self, "_v"):
            weight, bias, offload_stream = cast_bias_weight(
                self,
                input_2d,
                offloadable=True,
                compute_dtype=compute_dtype,
                want_requant=True,
            )

        scale = getattr(self, "input_scale", None)
        if scale is not None:
            if isinstance(scale, torch.nn.Parameter):
                scale = scale.data
            if scale.device != input.device:
                scale = comfy.model_management.cast_to_device(scale, input.device, None)

        layout = getattr(self, "layout_type", None)
        if layout is None:
            if offload_stream is not None:
                uncast_bias_weight(self, weight, bias, offload_stream)
            return stock_forward(self, input, *args, **kwargs)

        # 4) Pooled Tensor Core path (no QuantizedTensor.from_float alloc)
        out_2d = _tc_forward_pooled(
            self, input_2d, weight, bias, scale, compute_dtype
        )
        if out_2d is None:
            # Fallback: stock QT path
            from comfy.quant_ops import QuantizedTensor

            q_input = QuantizedTensor.from_float(input_2d, layout, scale=scale)
            out_2d = scaled_mm_nvfp4_linear(q_input, weight, bias)

        # 5) Restore rank with logical out_features (never QT storage shape[0])
        if reshaped_nd:
            out = out_2d.reshape((*input_shape[:-1], int(self.out_features)))
        else:
            out = out_2d

        if offload_stream is not None:
            uncast_bias_weight(self, weight, bias, offload_stream)
        return out

    forward_nvfp4._hswq_nvfp4_full_forward = True  # type: ignore[attr-defined]
    return forward_nvfp4


def make_nvfp4_linear_convert_weight(stock_convert_weight):
    """Wrap Linear.convert_weight: dequant then unrotate ConvRot weights for LoRA bake."""
    import torch
    from comfy.quant_ops import QuantizedTensor

    def convert_weight(self, weight, inplace=False, **kwargs):
        global _LORA_CONVERT_LOGS
        if callable(stock_convert_weight):
            out = stock_convert_weight(self, weight, inplace=inplace, **kwargs)
        elif isinstance(weight, QuantizedTensor):
            out = weight.dequantize()
        else:
            out = weight
        if (
            getattr(self, "_hswq_nvfp4_convrot", False)
            and out is not None
            and getattr(out, "ndim", 0) == 2
        ):
            gs = int(getattr(self, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
            h = build_hadamard(gs, device="cpu", dtype=torch.float32)
            out = unrotate_weight_linear(out, h, gs)
        if _LORA_CONVERT_LOGS < _LORA_LOG_MAX and getattr(
            self, "_hswq_nvfp4_convrot", False
        ):
            _LORA_CONVERT_LOGS += 1
            logger.info(
                "[HSWQ NVFP4 LoRA] Linear.convert_weight #%s: unrotate ConvRot "
                "in=%s/%s -> out=%s/%s",
                _LORA_CONVERT_LOGS,
                type(weight).__name__,
                getattr(weight, "dtype", None),
                type(out).__name__,
                getattr(out, "dtype", None),
            )
        return out

    convert_weight._hswq_nvfp4_lora_bake_ver = _NVFP4_LORA_BAKE_VER  # type: ignore[attr-defined]
    convert_weight._hswq_nvfp4_lora_bake_stock = stock_convert_weight  # type: ignore[attr-defined]
    return convert_weight


def make_nvfp4_linear_set_weight(stock_set_weight):
    """Wrap Linear.set_weight: re-rotate ConvRot float weights before requant."""
    import torch

    def set_weight(
        self,
        weight,
        inplace_update=False,
        seed=None,
        return_weight=False,
        **kwargs,
    ):
        global _LORA_SET_LOGS
        if (
            getattr(self, "_hswq_nvfp4_convrot", False)
            and getattr(weight, "ndim", 0) == 2
        ):
            gs = int(getattr(self, "_hswq_nvfp4_convrot_groupsize", 256) or 256)
            h = build_hadamard(gs, device="cpu", dtype=torch.float32)
            weight = rotate_weight_linear(weight, h, gs)
            if _LORA_SET_LOGS < _LORA_LOG_MAX:
                _LORA_SET_LOGS += 1
                logger.info(
                    "[HSWQ NVFP4 LoRA] Linear.set_weight #%s: re-rotate ConvRot "
                    "shape=%s layout=%s",
                    _LORA_SET_LOGS,
                    tuple(weight.shape) if hasattr(weight, "shape") else "?",
                    getattr(self, "layout_type", None),
                )
        return stock_set_weight(
            self,
            weight,
            inplace_update=inplace_update,
            seed=seed,
            return_weight=return_weight,
            **kwargs,
        )

    set_weight._hswq_nvfp4_lora_bake_ver = _NVFP4_LORA_BAKE_VER  # type: ignore[attr-defined]
    set_weight._hswq_nvfp4_lora_bake_stock = stock_set_weight  # type: ignore[attr-defined]
    return set_weight


def _peel_lora_bake_wrap(fn):
    """Unwrap nested HSWQ convert/set wraps to true stock.

    Z Image attaches VER=8 (``[HSWQ ConvRot LoRA] int8_protect``). SDXL product
    is VER=1; ``ver < 1`` never replaced VER=8 → LoRA fell off on the 3rd prompt.
    Peel any foreign bake wrap before attaching VER=1.
    """
    cur = fn
    for _ in range(8):
        if not callable(cur):
            return cur
        if int(getattr(cur, "_hswq_nvfp4_lora_bake_ver", 0) or 0) <= 0:
            return cur
        stock = getattr(cur, "_hswq_nvfp4_lora_bake_stock", None)
        if stock is not None and stock is not cur:
            cur = stock
            continue
        closure = getattr(cur, "__closure__", None)
        code = getattr(cur, "__code__", None)
        if closure is None or code is None:
            return cur
        names = code.co_freevars
        nxt = None
        for i, name in enumerate(names):
            if name in ("stock_convert_weight", "stock_set_weight"):
                nxt = closure[i].cell_contents
                break
        if nxt is None or nxt is cur:
            return cur
        cur = nxt
    return cur


def peel_all_nvfp4_linear_lora_bake(Lin) -> bool:
    """Strip every HSWQ Linear bake wrap down to stock convert/set.

    Z Image ``install_nvfp4_comfy_parity`` mutates ``mp0.Linear`` in place
    (VER=8 ``[HSWQ ConvRot LoRA] int8_protect``). Peeling
    ``ops.mixed_precision_ops`` alone does not undo that class mutation, so
    SDXL INT8 after Z Image still bakes through ZI wraps and LoRA falls off
    on the 3rd prompt. Call this from SDXL clear / ZI uninstall.
    """
    changed = False
    for attr in ("convert_weight", "set_weight"):
        fn = getattr(Lin, attr, None)
        if not callable(fn):
            continue
        if int(getattr(fn, "_hswq_nvfp4_lora_bake_ver", 0) or 0) <= 0:
            continue
        stock = _peel_lora_bake_wrap(fn)
        if stock is not fn:
            setattr(Lin, attr, stock)
            changed = True
    return changed


def attach_nvfp4_linear_lora_bake(Lin) -> bool:
    """Ensure MixedPrecision Linear has SDXL product ConvRot LoRA wraps (VER=1).

    Peels Z Image VER=8 (or any other HSWQ bake) so SDXL never nests / keeps
    ``int8_protect`` convert/set on INT8 ConvRot Linears.
    """
    applied = False
    cvt = getattr(Lin, "convert_weight", None)
    if callable(cvt):
        ver = int(getattr(cvt, "_hswq_nvfp4_lora_bake_ver", 0) or 0)
        if ver != _NVFP4_LORA_BAKE_VER:
            stock = _peel_lora_bake_wrap(cvt) if ver > 0 else cvt
            Lin.convert_weight = make_nvfp4_linear_convert_weight(stock)
            applied = True
    sw = getattr(Lin, "set_weight", None)
    if callable(sw):
        ver = int(getattr(sw, "_hswq_nvfp4_lora_bake_ver", 0) or 0)
        if ver != _NVFP4_LORA_BAKE_VER:
            stock = _peel_lora_bake_wrap(sw) if ver > 0 else sw
            Lin.set_weight = make_nvfp4_linear_set_weight(stock)
            applied = True
    return applied
```

### `nodes/nvfp4/nvfp4_conf.py`

```python
"""NVFP4 comfy_quant config helpers (HSWQ-owned; never edit ComfyUI-master)."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Packed E2M1 nibble pairs along K: storage_K = logical_padded_K // 2
_NVFP4_PACK_FACTOR = 2


def decode_comfy_quant_conf(raw: Any) -> Optional[dict]:
    """Decode a comfy_quant marker into a dict layer config."""
    import torch

    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if torch.is_tensor(raw):
        conf = json.loads(raw.detach().cpu().numpy().tobytes())
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        conf = json.loads(bytes(raw))
    elif isinstance(raw, str):
        conf = raw
    else:
        conf = raw

    while isinstance(conf, str):
        try:
            parsed = json.loads(conf)
        except (TypeError, json.JSONDecodeError):
            return {"format": conf}
        if parsed is conf:
            return {"format": conf}
        conf = parsed

    if isinstance(conf, dict):
        return conf
    raise TypeError(
        f"comfy_quant config must be a dict or format string, got {type(conf).__name__}"
    )


def comfy_quant_key_for_weight(weight_key: str) -> str:
    if weight_key.endswith(".weight"):
        return weight_key[: -len("weight")] + "comfy_quant"
    if weight_key.endswith("weight"):
        return weight_key[: -len("weight")] + "comfy_quant"
    return weight_key + ".comfy_quant"


def is_nvfp4_conf(conf: Optional[dict]) -> bool:
    return isinstance(conf, dict) and conf.get("format") == "nvfp4"


def convrot_flags_from_conf(conf: Optional[dict]) -> tuple[bool, int]:
    """Return (enabled, groupsize) from an nvfp4 comfy_quant dict."""
    if not is_nvfp4_conf(conf):
        return False, 256
    if not bool(conf.get("convrot", False)):
        return False, 256
    params_conf = conf.get("params", {})
    if not isinstance(params_conf, dict):
        params_conf = {}
    gs = int(conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)) or 256)
    return True, gs


def logical_linear_in_features(state_dict: dict, weight_key: str) -> int:
    """Return logical in_features for a Linear weight (expand packed NVFP4 K)."""
    import torch

    weight = state_dict[weight_key]
    if not torch.is_tensor(weight) or weight.ndim < 2:
        raise ValueError(
            f"{weight_key}: expected 2D+ tensor, got {type(weight)} "
            f"ndim={getattr(weight, 'ndim', None)}"
        )

    packed_in = int(weight.shape[1])
    cq_key = comfy_quant_key_for_weight(weight_key)
    conf = decode_comfy_quant_conf(state_dict.get(cq_key))
    if is_nvfp4_conf(conf) and weight.ndim == 2:
        return packed_in * _NVFP4_PACK_FACTOR
    return packed_in


def checkpoint_looks_like_comfy_quant_nvfp4(state_dict_or_path) -> bool:
    """True if checkpoint has at least one nvfp4 comfy_quant marker."""
    import torch

    if isinstance(state_dict_or_path, (str, os.PathLike)):
        return _probe_path_comfy_quant_nvfp4(str(state_dict_or_path))

    state_dict = state_dict_or_path
    for key, value in state_dict.items():
        if not key.endswith(".comfy_quant"):
            continue
        if not torch.is_tensor(value):
            continue
        conf = decode_comfy_quant_conf(value)
        if is_nvfp4_conf(conf):
            return True
    return False


def _probe_path_comfy_quant_nvfp4(path: str) -> bool:
    try:
        from safetensors import safe_open
    except ImportError:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            comfy_keys = [k for k in keys if k.endswith(".comfy_quant")]
            for ck in comfy_keys[:64]:
                conf = decode_comfy_quant_conf(f.get_tensor(ck))
                if is_nvfp4_conf(conf):
                    return True
    except Exception as e:
        logger.debug("NVFP4 probe failed for %s: %s", path, e)
        return False
    return False


def fix_unet_config_packed_dims(unet_config: dict, state_dict: dict, key_prefix: str) -> dict:
    """Rewrite context_dim / adm_in_channels using logical NVFP4 in_features."""
    if not isinstance(unet_config, dict):
        return unet_config

    y_input = f"{key_prefix}label_emb.0.0.weight"
    if y_input in state_dict and unet_config.get("adm_in_channels") is not None:
        try:
            unet_config["adm_in_channels"] = logical_linear_in_features(state_dict, y_input)
        except Exception as e:
            logger.warning("[HSWQ NVFP4] adm_in_channels fix skipped: %s", e)

    if unet_config.get("context_dim") is not None:
        attn_k = None
        suffix = "attn2.to_k.weight"
        for k in state_dict.keys():
            if k.startswith(key_prefix) and k.endswith(suffix):
                attn_k = k
                break
        if attn_k is not None:
            try:
                unet_config["context_dim"] = logical_linear_in_features(state_dict, attn_k)
            except Exception as e:
                logger.warning("[HSWQ NVFP4] context_dim fix skipped: %s", e)

    return unet_config
```

### `nodes/nvfp4/nvfp4_hadamard.py`

```python
"""Hadamard helpers for FULL offline ConvRot + online act rotation (HSWQ)."""
from __future__ import annotations

import math

_HADAMARD_CACHE: dict = {}
_H4_CACHE: dict = {}


def build_hadamard(size: int, device="cpu", dtype=None):
    """Build (and cache) a normalized Hadamard matrix.

    Always construct in float32 (CPU or GPU), then cast to ``dtype``.
    Building the Kronecker product directly in float16 destroys ConvRot
    orthonormality and collapses NVFP4 quality.
    """
    import torch

    if dtype is None:
        dtype = torch.float32
    device = torch.device(device) if not isinstance(device, torch.device) else device
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    master_key = (size, str(device), torch.float32)
    if master_key not in _HADAMARD_CACHE:
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=torch.float32,
            device=device,
        )
        h_matrix = h4
        current_size = 4
        while current_size < size:
            h_matrix = torch.kron(h_matrix, h4)
            current_size *= 4
        h_matrix = h_matrix / (size**0.5)
        _HADAMARD_CACHE[master_key] = h_matrix
    h_matrix = _HADAMARD_CACHE[master_key]
    if dtype != torch.float32:
        h_matrix = h_matrix.to(dtype=dtype)
    _HADAMARD_CACHE[cache_key] = h_matrix
    return h_matrix


def _h4(device, dtype):
    import torch

    key = (str(device), dtype)
    h = _H4_CACHE.get(key)
    if h is None:
        h = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=dtype,
            device=device,
        )
        _H4_CACHE[key] = h
    return h


def _apply_kron_h4_unnorm(x2d, size: int):
    """Right-multiply by unnormalized Kronecker power of h4 (same as build_hadamard).

    ``x2d`` shape ``(M, size)`` with ``size == 4**k``. Equivalent to
    ``x2d @ kron_power(h4)`` before the ``/sqrt(size)`` normalization.
    """
    import torch

    if size == 4:
        return torch.matmul(x2d, _h4(x2d.device, x2d.dtype))
    p = size // 4
    # H_size = H_p ⊗ h4  (left-associated kron growth in build_hadamard)
    x = x2d.reshape(-1, p, 4)
    y = torch.matmul(x, _h4(x2d.device, x2d.dtype))  # apply h4 on last dim
    # apply H_p on the middle dim: for each of 4 cols, (M,p) @ H_p
    yt = y.transpose(1, 2).reshape(-1, p)
    yt = _apply_kron_h4_unnorm(yt, p)
    z = yt.reshape(-1, 4, p).transpose(1, 2)
    return z.reshape(-1, size)


def rotate_last_dim(x, h_matrix, group_size: int):
    import torch

    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    group_count = features // group_size
    x_grouped = x.reshape(-1, group_count, group_size)
    if h_matrix.device == x.device and h_matrix.dtype == x.dtype:
        h = h_matrix
    else:
        h = h_matrix.to(dtype=x.dtype, device=x.device)
    return torch.matmul(x_grouped, h).reshape(orig_shape)


def rotate_weight_linear(weight, h_matrix, group_size: int):
    """Offline Linear: W_rot = W @ H^T (group-wise along in_features)."""
    import torch

    if getattr(weight, "ndim", 0) != 2:
        raise ValueError(f"Linear weight must be 2D, got ndim={getattr(weight, 'ndim', None)}")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    h_t = h_matrix.T.to(dtype=weight.dtype, device=weight.device)
    return torch.matmul(weight_grouped, h_t).reshape(weight.shape)


def unrotate_weight_linear(weight, h_matrix, group_size: int):
    """Inverse of rotate_weight_linear: W = W_rot @ H (for LoRA float space)."""
    import torch

    if getattr(weight, "ndim", 0) != 2:
        raise ValueError(f"Linear weight must be 2D, got ndim={getattr(weight, 'ndim', None)}")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    h = h_matrix.to(dtype=weight.dtype, device=weight.device)
    return torch.matmul(weight_grouped, h).reshape(weight.shape)


def rotate_last_dim_fast(x, group_size: int):
    """Same math as ``rotate_last_dim`` + ``build_hadamard``, O(n log n) butterflies.

    Avoids materializing the dense ``group_size x group_size`` Hadamard and the
    large GEMM that dominates online FULL ConvRot act rotation.
    """
    import torch

    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    flat = x.reshape(-1, group_size)
    y = _apply_kron_h4_unnorm(flat, group_size)
    y = y * (group_size**-0.5)
    return y.reshape(orig_shape)
```

### `nodes/nvfp4/nvfp4_load.py`

```python
"""
HSWQ-owned NVFP4 Linear load path.

Stock Comfy ``_load_quantized_module`` loads NVFP4 weights/scales into a
QuantizedTensor, but does **not**:
  - stamp FULL ConvRot flags (Params have no convrot for NVFP4)
  - validate storage shape against logical (out, in)
  - mark the module for the HSWQ Tensor Core forward path

This module owns that load logic entirely. It never edits ComfyUI-master;
callers monkey-patch ``ops._load_quantized_module`` to route nvfp4 here.
"""
from __future__ import annotations

import logging
from typing import Optional

from .nvfp4_conf import convrot_flags_from_conf, decode_comfy_quant_conf, is_nvfp4_conf

logger = logging.getLogger(__name__)


def peek_nvfp4_conf(state_dict, prefix: str) -> Optional[dict]:
    """Read comfy_quant without popping (for routing before stock load)."""
    return decode_comfy_quant_conf(state_dict.get(f"{prefix}comfy_quant"))


def arm_nvfp4_module(module, conf: Optional[dict]) -> None:
    """Attach HSWQ NVFP4 runtime flags after weight QT is in place."""
    if not is_nvfp4_conf(conf):
        return
    import torch

    module._hswq_nvfp4 = True
    enabled, gs = convrot_flags_from_conf(conf)
    module._hswq_nvfp4_convrot = bool(enabled)
    module._hswq_nvfp4_convrot_groupsize = int(gs)
    # Checkpoints often omit input_scale (0 keys in test.safetensors).
    # Placeholder ones(1) + flag → one amax per module then freeze (not every Linear).
    if getattr(module, "input_scale", None) is None:
        device = module.factory_kwargs.get("device", "cpu")
        module.register_parameter(
            "input_scale",
            torch.nn.Parameter(
                torch.ones(1, device=device, dtype=torch.float32),
                requires_grad=False,
            ),
        )
        module._hswq_nvfp4_scale_placeholder = True
    else:
        module._hswq_nvfp4_scale_from_ckpt = True
        module._hswq_nvfp4_scale_placeholder = False
    if enabled:
        logger.debug(
            "[HSWQ NVFP4] ConvRot armed groupsize=%s on %s",
            gs,
            getattr(module, "in_features", "?"),
        )


def validate_nvfp4_weight_storage(module, weight) -> None:
    """Ensure packed uint8 storage matches TensorCoreNVFP4Layout for _orig_shape."""
    import torch
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout

    orig = getattr(module, "_orig_shape", None)
    if orig is None or not torch.is_tensor(weight) or weight.ndim != 2:
        return
    expected = TensorCoreNVFP4Layout.get_storage_shape(tuple(int(x) for x in orig))
    got = tuple(int(x) for x in weight.shape)
    if got != expected:
        raise ValueError(
            f"[HSWQ NVFP4] weight storage shape mismatch: got {got}, "
            f"expected {expected} for orig_shape={tuple(orig)}"
        )


def load_nvfp4_linear_module(
    module,
    super_load,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
    load_extra_params: bool = True,
) -> None:
    """
    Full NVFP4 Linear ``_load_from_state_dict`` body (HSWQ).

    Mirrors stock scale/QT construction, then:
      - validates storage vs logical shape
      - loads input_scale when present
      - arms ConvRot + HSWQ TC forward flags
    """
    import torch
    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class

    device = module.factory_kwargs["device"]
    compute_dtype = module.factory_kwargs["dtype"]
    disabled_formats = module._disabled_formats
    layer_name = prefix.rstrip(".")

    weight = state_dict.pop(f"{prefix}weight", None)
    if weight is None:
        logger.warning("Missing weight for layer %s", layer_name)
        module.weight = None
        return
    manually_loaded_keys = [f"{prefix}weight"]

    def pop_scale(name, dtype=None):
        key = f"{prefix}{name}"
        v = state_dict.pop(key, None)
        if v is not None:
            v = v.to(device=device)
            if dtype is not None:
                v = v.view(dtype=dtype)
            manually_loaded_keys.append(key)
        return v

    layer_conf_raw = state_dict.pop(f"{prefix}comfy_quant", None)
    layer_conf = decode_comfy_quant_conf(layer_conf_raw)
    if layer_conf_raw is not None:
        manually_loaded_keys.append(f"{prefix}comfy_quant")

    if not is_nvfp4_conf(layer_conf):
        raise ValueError(
            f"[HSWQ NVFP4] load_nvfp4_linear_module called for non-nvfp4 "
            f"layer {layer_name}: {layer_conf}"
        )

    module.quant_format = "nvfp4"
    module._full_precision_mm_config = bool(layer_conf.get("full_precision_matrix_mult", False))
    if not module._full_precision_mm:
        module._full_precision_mm = module._full_precision_mm_config
    if module.quant_format in disabled_formats:
        module._full_precision_mm = True

    qconfig = QUANT_ALGOS["nvfp4"]
    module.layout_type = qconfig["comfy_tensor_layout"]
    layout_cls = get_layout_class(module.layout_type)

    ts = pop_scale("weight_scale_2")
    bs = pop_scale("weight_scale", torch.float8_e4m3fn)
    if ts is None or bs is None:
        raise ValueError(f"Missing NVFP4 scales for layer {layer_name}")

    validate_nvfp4_weight_storage(module, weight)

    params = layout_cls.Params(
        scale=ts,
        block_scale=bs,
        orig_dtype=compute_dtype,
        orig_shape=module._orig_shape,
    )
    module.weight = torch.nn.Parameter(
        QuantizedTensor(
            weight.to(device=device, dtype=qconfig["storage_t"]),
            module.layout_type,
            params,
        ),
        requires_grad=False,
    )

    if load_extra_params:
        for param_name in qconfig["parameters"]:
            if param_name in {"weight_scale", "weight_scale_2"}:
                continue
            param_key = f"{prefix}{param_name}"
            _v = state_dict.pop(param_key, None)
            if _v is None:
                continue
            module.register_parameter(
                param_name, torch.nn.Parameter(_v.to(device=device), requires_grad=False)
            )
            manually_loaded_keys.append(param_key)

    arm_nvfp4_module(module, layer_conf)

    super_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    for key in manually_loaded_keys:
        if key in missing_keys:
            missing_keys.remove(key)
```

### `patches/comfy_quant_int8.py`

```python
"""
ComfyUI core-safe patches for native comfy_quant INT8 (int8_tensorwise).

Upstream MixedPrecisionOps only quant-loads Linear / Embedding / MoE.
SD UNet INT8 checkpoints also store Conv2d weights as int8 + comfy_quant, which
fails with: Only Tensors of floating point and complex dtype can require gradients.

Also normalizes bare-string / double-encoded comfy_quant JSON some exporters write.

LoRA: native Linear already has convert_weight + set_weight (dequant → bake →
requant, same idea as BobJohnson24/ComfyUI-INT8-Fast). Injected Conv2d must
mirror that set_weight; without it ModelPatcher falls back to rounding into
int8 and LoRA deltas on Conv layers vanish.

Applied from ComfyUI-HSWQ-Loader-and-Tools so ComfyUI core updates do not wipe it.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)
_PATCHES_APPLIED = False

# LoRA bake path logs (rate-limited so console stays readable)
_LORA_CONVERT_LOG_MAX = 0  # quiet; Status dump is enough
_LORA_SET_LOG_MAX = 0
_LORA_PATCHER_LOG_MAX = 0  # per-key bake lines off; Status dump is enough
_lora_convert_logs = 0
_lora_set_logs = 0
_lora_patcher_logs = 0
_lora_patcher_stats = {
    "calls": 0,
    "with_set_func": 0,
    "without_set_func": 0,
    "with_convert_func": 0,
}

# LoRA key attach / skip accounting (last load_lora_for_models call)
_lora_attach_last = {
    "lora_name": "",
    "strength_model": None,
    "strength_clip": None,
    "lora_file_keys": 0,
    "mapped_keys": 0,
    "applied_unet": 0,
    "applied_clip": 0,
    "applied_unet_keys": [],
    "applied_clip_keys": [],
    "not_mapped": [],
    "mapped_but_not_attached": [],
    "add_patches_skipped_unet": [],
}
# One entry per load_lora_for_models call (stacked loaders → multiple entries)
_lora_attach_history = []
# key -> "requant" | "int8_round" recorded during bake
_lora_bake_by_key = {}
# Set by LoraLoader.load_lora wrap (and cleared after attach)
_current_lora_name = None
_current_lora_strength_model = None
_current_lora_strength_clip = None
_lora_shape_skips = []  # list of (lora_name, key, reason)
_LORA_SKIP_PRINT_MAX = 40


def _console(msg: str) -> None:
    """Always visible in ComfyUI console (print + INFO)."""
    print(msg, flush=True)
    logger.info(msg)


def record_lora_shape_skip(lora_name: str, key: str, reason: str) -> None:
    """Called from LoraDiff reshape/numel skip path."""
    _lora_shape_skips.append((str(lora_name), str(key), str(reason)))


def _basename_lora(name: str) -> str:
    if not name:
        return name
    return os.path.basename(str(name).replace("\\", "/"))


# WeightAdapterBase class attrs — NOT filenames (was the lora=lora bug)
_ADAPTER_TYPE_NAMES = frozenset({"lora", "loha", "lokr", "oft", "boft", "glora"})


def _looks_like_lora_filename(s) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s or s.lower() in _ADAPTER_TYPE_NAMES:
        return False
    low = s.lower()
    if low.endswith((".safetensors", ".pt", ".ckpt", ".bin", ".sft")):
        return True
    if "/" in s or "\\" in s:
        return True
    # Short folder-relative names without extension still count as filenames
    if len(s) >= 2 and not s.startswith("diffusion_model"):
        return True
    return False


def _lora_line(msg: str) -> None:
    """One visible console line (print only — no print+logger twin)."""
    print(msg, flush=True)


def _slot_skip_count(entry: dict) -> int:
    return len(entry.get("not_mapped") or []) + len(
        entry.get("mapped_but_not_attached") or []
    )


def _slot_applied_count(entry: dict) -> int:
    return int(entry.get("applied_unet") or 0) + int(entry.get("applied_clip") or 0)


def _format_lora_slot_line(slot_i: int, entry: dict, include_bake: bool = False) -> str:
    """lora_name / applied_keys / skipped_keys — always present."""
    name = entry.get("lora_name") or "(unknown)"
    sm = entry.get("strength_model")
    sc = entry.get("strength_clip")
    u = int(entry.get("applied_unet") or 0)
    c = int(entry.get("applied_clip") or 0)
    applied = u + c
    skip = _slot_skip_count(entry)
    parts = [
        f"Slot {slot_i}:",
        f"lora_name='{name}'",
        f"applied_keys={applied} (unet={u} clip={c})",
        f"skipped_keys={skip}",
    ]
    if sm is not None:
        parts.append(f"strength_model={sm}")
    if sc is not None:
        parts.append(f"strength_clip={sc}")
    if include_bake:
        verdict, rq, ir, nb = _per_lora_bake_verdict(entry)
        parts.append(f"bake rq={rq} ir={ir} nb={nb}")
        if verdict == "OK_requant":
            parts.append("→ APPLIED ✓")
        elif verdict == "BROKEN_int8_round":
            parts.append("→ BROKEN ✗")
        elif verdict == "N/A_CLIP_only":
            parts.append("→ CLIP_only ✓")
        else:
            parts.append(f"→ {verdict}")
    else:
        if applied > 0:
            parts.append("→ APPLIED ✓")
        else:
            parts.append("→ SKIPPED ✗")
    return f"[HSWQ LoRA Status] {' | '.join(parts)}"


def _log_lora_slot_attach(entry: dict) -> None:
    """Emit one Status line immediately when a LoRA is attached (any loader)."""
    n = len(_lora_attach_history)
    if n == 1:
        _lora_line("[HSWQ LoRA Status] Processing LoRA slot(s):")
    _lora_line(_format_lora_slot_line(n, entry, include_bake=False))
    _lora_line(
        f"[HSWQ LoRA Status]   file_keys={entry.get('lora_file_keys', 0)} "
        f"mapped={entry.get('mapped_keys', 0)} "
        f"not_mapped={len(entry.get('not_mapped') or [])} "
        f"mapped_not_attached={len(entry.get('mapped_but_not_attached') or [])}"
    )


def _set_current_lora_name(name, strength_model=None, strength_clip=None) -> None:
    """Store real filename/UI name; never store adapter type 'lora'."""
    global _current_lora_name, _current_lora_strength_model, _current_lora_strength_clip
    if _looks_like_lora_filename(name):
        _current_lora_name = _basename_lora(name)
    if strength_model is not None:
        _current_lora_strength_model = strength_model
    if strength_clip is not None:
        _current_lora_strength_clip = strength_clip


def _path_is_under_loras_dir(path: str) -> bool:
    """True if path is inside any registered loras/ folder (any loader)."""
    if not path:
        return False
    try:
        import folder_paths

        bases = folder_paths.get_folder_paths("loras") or []
    except Exception:
        bases = []
    norm = os.path.normcase(os.path.abspath(str(path)))
    for base in bases:
        try:
            b = os.path.normcase(os.path.abspath(str(base)))
            if norm == b or norm.startswith(b + os.sep):
                return True
        except Exception:
            continue
    # Fallback when folder list not ready yet
    low = str(path).replace("\\", "/").lower()
    return "/loras/" in low or low.endswith("/loras")


def _resolve_lora_name(loaded_patches=None) -> str:
    """Filename for the LoRA currently being attached (any loader → common hooks)."""
    global _current_lora_name
    if _looks_like_lora_filename(_current_lora_name):
        return _basename_lora(_current_lora_name)

    try:
        import inspect

        # Common local names used by many LoRA loader nodes / helpers
        keys = (
            "lora_name",
            "lora_path",
            "lora",
            "path",
            "filename",
            "file_path",
            "lora_file",
            "name",
        )
        for frame in inspect.stack()[1:24]:
            loc = frame.frame.f_locals
            for key in keys:
                cand = loc.get(key)
                if _looks_like_lora_filename(cand):
                    return _basename_lora(cand)
            # Widget-style dicts: {'lora': '<file>', 'on': True, 'strength': ...}
            for cand in loc.values():
                if not isinstance(cand, dict):
                    continue
                ui = cand.get("lora")
                if _looks_like_lora_filename(ui) and (
                    "strength" in cand or "on" in cand or "strengthTwo" in cand
                ):
                    return _basename_lora(ui)
    except Exception:
        pass

    return f"unknown_lora#{len(_lora_attach_history) + 1}"


def reset_int8_lora_log_counters() -> None:
    global _lora_convert_logs, _lora_set_logs, _lora_patcher_logs
    global _current_lora_name, _current_lora_strength_model, _current_lora_strength_clip
    _lora_convert_logs = 0
    _lora_set_logs = 0
    _lora_patcher_logs = 0
    _lora_patcher_stats.update(
        calls=0, with_set_func=0, without_set_func=0, with_convert_func=0
    )
    _lora_shape_skips.clear()
    _lora_attach_history.clear()
    _lora_bake_by_key.clear()
    _current_lora_name = None
    _current_lora_strength_model = None
    _current_lora_strength_clip = None
    _lora_attach_last.update(
        lora_name="",
        strength_model=None,
        strength_clip=None,
        lora_file_keys=0,
        mapped_keys=0,
        applied_unet=0,
        applied_clip=0,
        applied_unet_keys=[],
        applied_clip_keys=[],
        not_mapped=[],
        mapped_but_not_attached=[],
        add_patches_skipped_unet=[],
    )
    dump_int8_lora_bake_stats._dumped_this_load = False


def summarize_int8_lora_capability(model) -> dict:
    """Scan loaded MODEL / diffusion_model and print LoRA hook readiness."""
    try:
        from comfy.ops import QuantizedTensor
    except ImportError:
        QuantizedTensor = type(None)

    diffusion = model
    # ModelPatcher -> BaseModel -> diffusion_model
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        diffusion = model.model.diffusion_model
    elif hasattr(model, "diffusion_model"):
        diffusion = model.diffusion_model

    n_lin = n_conv = 0
    lin_set = conv_set = 0
    lin_cvt = conv_cvt = 0
    lin_q = conv_q = 0
    sample_missing = []

    for name, mod in diffusion.named_modules():
        cls = type(mod).__name__
        is_lin = "Linear" in cls
        is_conv = "Conv2d" in cls
        if not is_lin and not is_conv:
            continue
        has_set = callable(getattr(mod, "set_weight", None))
        has_cvt = callable(getattr(mod, "convert_weight", None))
        w = getattr(mod, "weight", None)
        is_q = False
        if QuantizedTensor is not type(None):
            is_q = isinstance(w, QuantizedTensor) or isinstance(
                getattr(w, "data", None), QuantizedTensor
            )
        layout = getattr(mod, "layout_type", None)
        if is_lin:
            n_lin += 1
            lin_set += int(has_set)
            lin_cvt += int(has_cvt)
            lin_q += int(is_q or layout is not None)
        else:
            n_conv += 1
            conv_set += int(has_set)
            conv_cvt += int(has_cvt)
            conv_q += int(is_q or layout is not None)
            if (not has_set or not has_cvt) and len(sample_missing) < 5:
                sample_missing.append(
                    f"{name} set={has_set} convert={has_cvt} layout={layout}"
                )

    _lora_line("[HSWQ INT8 LoRA] ===== load summary =====")
    _lora_line(
        f"[HSWQ INT8 LoRA] Linear: {n_lin}  set_weight={lin_set}  convert_weight={lin_cvt}  quantized/layout={lin_q}"
    )
    _lora_line(
        f"[HSWQ INT8 LoRA] Conv2d: {n_conv}  set_weight={conv_set}  convert_weight={conv_cvt}  quantized/layout={conv_q}"
    )
    if conv_set < n_conv or conv_cvt < n_conv:
        _lora_line(
            "[HSWQ INT8 LoRA] WARN: some Conv2d lack set/convert — LoRA on those layers will round into int8 and die"
        )
        for s in sample_missing:
            _lora_line(f"[HSWQ INT8 LoRA]   missing: {s}")
    else:
        _lora_line(
            "[HSWQ INT8 LoRA] OK: Conv2d has set_weight+convert_weight (dequant -> bake -> requant)"
        )
    _lora_line("[HSWQ INT8 LoRA] =========================")
    return {
        "linear": n_lin,
        "conv2d": n_conv,
        "linear_set_weight": lin_set,
        "conv_set_weight": conv_set,
    }


def decode_comfy_quant_conf(raw):
    """Decode a comfy_quant marker into a dict layer config."""
    import torch

    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if torch.is_tensor(raw):
        conf = json.loads(raw.numpy().tobytes())
    elif isinstance(raw, (bytes, bytearray, memoryview)):
        conf = json.loads(bytes(raw))
    elif isinstance(raw, str):
        conf = raw
    else:
        conf = raw

    while isinstance(conf, str):
        try:
            parsed = json.loads(conf)
        except (TypeError, json.JSONDecodeError):
            return {"format": conf}
        if parsed is conf:
            return {"format": conf}
        conf = parsed

    if isinstance(conf, dict):
        return conf
    raise TypeError(f"comfy_quant config must be a dict or format string, got {type(conf).__name__}")


def checkpoint_looks_like_comfy_quant_int8(state_dict_or_path) -> bool:
    """True if checkpoint has comfy_quant INT8 markers (native MixedPrecisionOps path).

    Accepts a loaded state_dict, or a filesystem path (probes via safetensors without full load).
    """
    import torch

    if isinstance(state_dict_or_path, (str, os.PathLike)):
        return _probe_path_comfy_quant_int8(str(state_dict_or_path))

    state_dict = state_dict_or_path
    has_marker = False
    has_int8 = False
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if key.endswith(".comfy_quant"):
            has_marker = True
            conf = decode_comfy_quant_conf(value)
            if isinstance(conf, dict) and conf.get("format") == "int8_tensorwise":
                return True
        if key.endswith(".weight") and value.dtype == torch.int8:
            has_int8 = True
    return has_marker and has_int8


def _probe_path_comfy_quant_int8(path: str) -> bool:
    """Lightweight safetensors probe for int8_tensorwise."""
    import torch

    try:
        from safetensors import safe_open
    except ImportError:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            comfy_keys = [k for k in keys if k.endswith(".comfy_quant")]
            for ck in comfy_keys[:16]:
                conf = decode_comfy_quant_conf(f.get_tensor(ck))
                if isinstance(conf, dict) and conf.get("format") == "int8_tensorwise":
                    return True
            if comfy_keys:
                for k in keys:
                    if not k.endswith(".weight"):
                        continue
                    if f.get_tensor(k).dtype == torch.int8:
                        return True
                    break
            meta = f.metadata() or {}
            if "_quantization_metadata" in meta:
                try:
                    qm = json.loads(meta["_quantization_metadata"])
                    layers = qm.get("layers", {}) if isinstance(qm, dict) else {}
                    for v in layers.values():
                        if isinstance(v, str) and v == "int8_tensorwise":
                            return True
                        if isinstance(v, dict) and v.get("format") == "int8_tensorwise":
                            return True
                except (TypeError, json.JSONDecodeError):
                    pass
    except Exception as e:
        logger.debug("[HSWQ INT8] probe failed for %s: %s", path, e)
        return False
    return False


def _comfy_quant_conf_has_convrot(conf) -> bool:
    if not isinstance(conf, dict):
        return False
    if conf.get("convrot") is True:
        return True
    params = conf.get("params")
    if isinstance(params, dict) and params.get("convrot") is True:
        return True
    return False


def checkpoint_looks_like_comfy_quant_convrot(state_dict_or_path) -> bool:
    """True if checkpoint marks int8_tensorwise layers with ConvRot (Hadamard)."""
    if isinstance(state_dict_or_path, (str, os.PathLike)):
        return _probe_path_comfy_quant_convrot(str(state_dict_or_path))

    state_dict = state_dict_or_path
    import torch

    for key, value in state_dict.items():
        if not key.endswith(".comfy_quant"):
            continue
        if not torch.is_tensor(value) and not isinstance(value, (dict, bytes, bytearray, str)):
            continue
        conf = decode_comfy_quant_conf(value)
        if _comfy_quant_conf_has_convrot(conf):
            return True
    return False


def checkpoint_needs_hswq_int8_conv2d(state_dict_or_path) -> bool:
    """True for SDXL/ZI-style UNets that need HSWQ INT8 Conv2d patches.

    Keyed off architecture (``input_blocks`` / ``middle_block`` / ``output_blocks``),
    not off ConvRot. DiT/Krea2 (``double_blocks`` / ``single_blocks``) returns False
    so ConvRot stock load stays free of our Conv2d inject (VRAM).
    """
    if isinstance(state_dict_or_path, (str, os.PathLike)):
        return _probe_path_needs_hswq_int8_conv2d(str(state_dict_or_path))

    keys = list(state_dict_or_path.keys())
    return _keys_need_hswq_int8_conv2d(keys)


def _keys_need_hswq_int8_conv2d(keys) -> bool:
    sdxl = False
    dit = False
    for k in keys:
        if (
            ".input_blocks." in k
            or ".middle_block." in k
            or ".output_blocks." in k
            or k.startswith("input_blocks.")
            or k.startswith("middle_block.")
            or k.startswith("output_blocks.")
        ):
            sdxl = True
        if (
            ".double_blocks." in k
            or ".single_blocks." in k
            or ".joint_blocks." in k
            or k.startswith("double_blocks.")
            or k.startswith("single_blocks.")
            or k.startswith("joint_blocks.")
        ):
            dit = True
        if sdxl and dit:
            break
    # Prefer SDXL Conv2d path when UNet blocks exist; DiT-only → no inject.
    if sdxl:
        return True
    return False


def _probe_path_needs_hswq_int8_conv2d(path: str) -> bool:
    try:
        from safetensors import safe_open
    except ImportError:
        # Filename heuristics only as last resort.
        base = os.path.basename(path).lower()
        if "krea" in base or "dit" in base:
            return False
        return True
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            return _keys_need_hswq_int8_conv2d(list(f.keys()))
    except Exception as e:
        logger.debug("[HSWQ INT8] SDXL/ZI Conv2d need probe failed for %s: %s", path, e)
        base = os.path.basename(path).lower()
        if "krea" in base or "convrot" in base or "int8convrot" in base:
            return False
        return True


def _probe_path_comfy_quant_convrot(path: str) -> bool:
    """Lightweight safetensors probe for comfy_quant.convrot=true."""
    try:
        from safetensors import safe_open
    except ImportError:
        return "convrot" in os.path.basename(path).lower()
    base = os.path.basename(path).lower()
    name_hint = "convrot" in base or "int8convrot" in base
    comfy_keys = []
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            comfy_keys = [k for k in keys if k.endswith(".comfy_quant")]
            for ck in comfy_keys[:32]:
                conf = decode_comfy_quant_conf(f.get_tensor(ck))
                if _comfy_quant_conf_has_convrot(conf):
                    return True
            meta = f.metadata() or {}
            if "_quantization_metadata" in meta:
                try:
                    qm = json.loads(meta["_quantization_metadata"])
                    layers = qm.get("layers", {}) if isinstance(qm, dict) else {}
                    for v in layers.values():
                        if isinstance(v, dict) and _comfy_quant_conf_has_convrot(v):
                            return True
                except (TypeError, json.JSONDecodeError):
                    pass
    except Exception as e:
        logger.debug("[HSWQ INT8] ConvRot probe failed for %s: %s", path, e)
        return name_hint
    # Filename alone is enough for *Int8Convrot* when markers were stripped/odd.
    return name_hint


def _normalize_comfy_quant_tensor(value):
    import torch

    conf = decode_comfy_quant_conf(value)
    if conf is None:
        return None
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _patch_convert_old_quants() -> bool:
    try:
        import torch
        import comfy.utils as utils_module
    except ImportError:
        return False

    original = getattr(utils_module, "convert_old_quants", None)
    if original is None or getattr(original, "_hswq_int8_patched", False):
        return False

    def convert_old_quants_pre(state_dict, model_prefix="", metadata=None):
        if metadata is None:
            metadata = {}
        # Normalize string layer configs in metadata before upstream json.dumps(v).
        if isinstance(metadata, dict) and "_quantization_metadata" in metadata:
            try:
                quant_meta = json.loads(metadata["_quantization_metadata"])
            except (TypeError, json.JSONDecodeError):
                quant_meta = None
            if isinstance(quant_meta, dict) and isinstance(quant_meta.get("layers"), dict):
                layers = quant_meta["layers"]
                changed = False
                for k, v in list(layers.items()):
                    if isinstance(v, str):
                        layers[k] = {"format": v}
                        changed = True
                    elif not isinstance(v, dict):
                        raise TypeError(
                            f"quantization layer config for {k} must be dict or format string, got {type(v).__name__}"
                        )
                if changed:
                    metadata = dict(metadata)
                    metadata["_quantization_metadata"] = json.dumps(quant_meta)

        state_dict, metadata = original(state_dict, model_prefix=model_prefix, metadata=metadata)

        # Re-normalize any .comfy_quant tensors (file-embedded or metadata-written).
        for key in list(state_dict.keys()):
            if not key.endswith(".comfy_quant"):
                continue
            normalized = _normalize_comfy_quant_tensor(state_dict[key])
            if normalized is None:
                state_dict.pop(key, None)
            else:
                state_dict[key] = normalized
        return state_dict, metadata

    convert_old_quants_pre._hswq_int8_patched = True
    utils_module.convert_old_quants = convert_old_quants_pre
    return True


def _quant_config_has_int8_tensorwise(quant_config) -> bool:
    """True if MixedPrecisionOps quant_config targets int8_tensorwise layers."""
    if not isinstance(quant_config, dict) or not quant_config:
        return False
    for v in quant_config.values():
        if isinstance(v, dict) and v.get("format") == "int8_tensorwise":
            return True
        if v == "int8_tensorwise":
            return True
    return False


# INT8 Conv2d inject must NOT run for FP MixedPrecisionOps.
# detect_layer_quantization() only returns {"mixed_ops": True} for both INT8 and FP8,
# so we gate Conv2d injection on this load-scoped flag (set only in INT8 load helpers).
_int8_quant_conv_tls = threading.local()


@contextlib.contextmanager
def _int8_quant_conv_scope():
    prev = getattr(_int8_quant_conv_tls, "active", False)
    _int8_quant_conv_tls.active = True
    try:
        yield
    finally:
        _int8_quant_conv_tls.active = prev


def _should_inject_int8_conv(quant_config) -> bool:
    # Only while an HSWQ INT8 UNet/Checkpoint load explicitly opens the scope.
    # Do NOT key off quant_config alone: once mixed_precision_ops is monkeypatched,
    # stock UNETLoader / Krea2 ConvRot loads also build MixedPrecisionOps with
    # int8_tensorwise config — injecting our Conv2d there is wrong for DiT/ConvRot
    # and can inflate VRAM vs stock.
    _ = quant_config
    return bool(getattr(_int8_quant_conv_tls, "active", False))


def _module_path_is_real_nunchaku_package(mod: str) -> bool:
    """True only for real Nunchaku package modules — never this unofficial-loader.

    INT8 Conv2d from this extension lives under a path containing ``nunchaku``;
    a bare ``\"nunchaku\" in path`` false-positive armed VRAM handoff on
    non-SVDQ loads (SDXL INT8 and any other architecture using those Conv2d)
    and destroyed normal generation. Substring match is forbidden.
    """
    mod_l = (mod or "").lower().replace("\\", "/")
    if not mod_l:
        return False
    # This extension / INT8 patch path must never count as SVDQ.
    if (
        "unofficial" in mod_l
        or "comfy_quant_int8" in mod_l
        or "nunchaku-unofficial" in mod_l
        or "nunchaku_unofficial" in mod_l
    ):
        return False
    if mod_l == "nunchaku" or mod_l.startswith("nunchaku."):
        return True
    if ".nunchaku." in mod_l:
        return True
    return False


def _model_is_nunchaku_svdq(model) -> bool:
    """True only when the graph carries real Nunchaku SVDQ modules.

    ComfyUI registers Z-Image as ``Lumina2`` — classname checks for
    ``Nunchaku`` / ``ZImage`` miss that. Any SVDQ / ComfyNunchaku module means
    never run comfy_quant INT8 Dynamic LoRA bake.

    Branch: everything that is not real SVDQ (SDXL, Flux, ZIT, native INT8,
    FP, …) returns False. Module-path checks must not match this
    unofficial-loader package (see ``_module_path_is_real_nunchaku_package``).
    """
    if model is None:
        return False
    roots = [model]
    dm = getattr(model, "diffusion_model", None)
    if dm is not None:
        roots.append(dm)
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        roots.append(inner)
        dm2 = getattr(inner, "diffusion_model", None)
        if dm2 is not None:
            roots.append(dm2)
    seen = set()
    for root in roots:
        rid = id(root)
        if rid in seen:
            continue
        seen.add(rid)
        try:
            named = root.named_modules()
        except Exception:
            continue
        for _, module in named:
            cls_name = type(module).__name__
            if (
                "SVDQ" in cls_name
                or "Nunchaku" in cls_name
                or cls_name.startswith("ComfyNunchaku")
            ):
                return True
            mod = getattr(type(module), "__module__", "") or ""
            if _module_path_is_real_nunchaku_package(mod):
                return True
    return False


def _model_is_zimage_nvfp4_parity(model) -> bool:
    """True for Z Image / Krea ConvRot NVFP4 parity packs (not SDXL TC).

    Absolute branch: shared INT8 Dynamic bake must not own these models.
    Z Image bake lives under ``nodes/zimage_nvfp4/nvfp4_lora_bake.py``.
    """
    if model is None:
        return False
    for _, module in model.named_modules():
        if getattr(module, "_hswq_nvfp4_convrot_parity", False):
            return True
        fwd = getattr(module, "forward", None)
        if fwd is not None and getattr(fwd, "_hswq_nvfp4_convrot_parity", False):
            return True
    return False


def _model_has_int8_quantized_weights(model) -> bool:
    """True for native comfy_quant QuantizedTensor (INT8 / NVFP4 / FP8 QT).

    Must NOT treat bare ``torch.int8`` weights as comfy_quant INT8.
    Nunchaku SVDQ / Z-Image / Lumina2 modules often use int8 storage; a false
    positive here arms Dynamic.load INT8 LoRA bake and can Abort those paths.

    SDXL ConvRot NVFP4 (Checkpoint Loader dropdown ``ConvRot NVFP4``):
      bake must see NVFP4 Linear QT so ``patch_weight_to_device`` runs
      nodes/nvfp4 ConvRot convert/set_weight (3.3.0 SDXL path).

    Z Image ConvRot NVFP4 (UNet dropdown ``Z Image ConvRot NVFP4``):
      do not treat as the SDXL bake path — see ``_model_is_zimage_nvfp4_parity``.
    """
    if _model_is_nunchaku_svdq(model):
        return False
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    for _, module in model.named_modules():
        cls_name = type(module).__name__
        if "SVDQ" in cls_name or "Nunchaku" in cls_name:
            continue
        w = getattr(module, "weight", None)
        if w is None:
            continue
        if isinstance(w, QuantizedTensor):
            return True
    return False


def _load_native_convert_int8_helpers():
    """Lazy-load Hadamard / rotate helpers from sibling native_convert_int8.py."""
    import importlib.util

    global _NATIVE_CONVERT_INT8_MOD
    if _NATIVE_CONVERT_INT8_MOD is not None:
        return _NATIVE_CONVERT_INT8_MOD
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "native_convert_int8.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"native_convert_int8.py not found: {path}")
    name = "native_convert_int8_for_hswq_conv2d"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _NATIVE_CONVERT_INT8_MOD = mod
    return mod


_NATIVE_CONVERT_INT8_MOD = None


def _qt_payload(weight, QuantizedTensor):
    """Unwrap Parameter → QuantizedTensor if needed."""
    if weight is None:
        return None
    if isinstance(weight, QuantizedTensor):
        return weight
    data = getattr(weight, "data", None)
    if isinstance(data, QuantizedTensor):
        return data
    return None


def _arm_hswq_conv2d_convrot(module, QuantizedTensor):
    """Full ConvRot on Conv2d: keep online rotate on module; clear kitchen Params.convrot.

    Kitchen dequantize_int8_convrot_* is 2D-only. Stamping Params.convrot=True on
    4D weights and calling .dequantize() crashes. Weights stay in rotated basis;
    forward rotates NCHW activations; LoRA convert_weight unrotates to float space.
    """
    import dataclasses

    import torch

    qt = _qt_payload(getattr(module, "weight", None), QuantizedTensor)
    if qt is None:
        return
    params = getattr(qt, "_params", None)
    qdata = getattr(qt, "_qdata", None)
    if params is None or qdata is None:
        return
    if getattr(qdata, "ndim", None) != 4:
        return
    if not bool(getattr(params, "convrot", False)):
        return

    gs = int(getattr(params, "convrot_groupsize", 256) or 256)
    module._hswq_convrot = True
    module._hswq_convrot_groupsize = gs
    new_params = dataclasses.replace(params, convrot=False)
    # Prefer in-place params swap. Reconstructing QT needs layout *string*
    # (_layout_cls), not a layout object — wrong arg → empty AssertionError.
    try:
        object.__setattr__(qt, "_params", new_params)
        return
    except Exception:
        pass
    try:
        qt._params = new_params
        return
    except Exception:
        pass
    layout_cls = getattr(qt, "_layout_cls", None)
    if not isinstance(layout_cls, str):
        layout_cls = getattr(module, "layout_type", None)
    if not isinstance(layout_cls, str):
        return
    new_qt = type(qt)(qdata, layout_cls, new_params)
    module.weight = torch.nn.Parameter(new_qt, requires_grad=False)


def _make_quantized_conv2d(ops_module, MixedPrecisionOps, disabled):
    """Build MixedPrecisionOps.Conv2d class using current comfy.ops helpers."""
    import torch

    CastWeightBiasOp = ops_module.CastWeightBiasOp
    QuantizedTensor = ops_module.QuantizedTensor
    cast_bias_weight = ops_module.cast_bias_weight
    uncast_bias_weight = ops_module.uncast_bias_weight
    run_every_op = ops_module.run_every_op
    _load_quantized_module = ops_module._load_quantized_module
    _quantized_weight_state_dict = ops_module._quantized_weight_state_dict
    _quantized_apply = ops_module._quantized_apply

    class Conv2d(torch.nn.Module, CastWeightBiasOp):
        _disabled_formats = disabled
        _hswq_quant_conv2d = True

        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
        ):
            super().__init__()
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            if isinstance(stride, int):
                stride = (stride, stride)
            if isinstance(padding, int):
                padding = (padding, padding)
            if isinstance(dilation, int):
                dilation = (dilation, dilation)

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.padding_mode = padding_mode
            self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}
            self._orig_shape = (out_channels, in_channels // groups, kernel_size[0], kernel_size[1])

            if bias:
                self.bias = torch.nn.Parameter(
                    torch.empty(out_channels, **self.factory_kwargs), requires_grad=False
                )
            else:
                self.register_parameter("bias", None)

            self.weight = None
            self.quant_format = None
            self.layout_type = None
            self._full_precision_mm = MixedPrecisionOps._full_precision_mm
            self._full_precision_mm_config = False
            self._hswq_convrot = False
            self._hswq_convrot_groupsize = 256

        def reset_parameters(self):
            return None

        def _load_from_state_dict(self, *args):
            _load_quantized_module(self, super()._load_from_state_dict, *args, load_extra_params=False)
            _arm_hswq_conv2d_convrot(self, QuantizedTensor)

        def state_dict(self, *args, destination=None, prefix="", **kwargs):
            sd = destination if destination is not None else {}
            sd = _quantized_weight_state_dict(self, sd, prefix)
            # Re-stamp ConvRot on export (Params.convrot cleared for safe 4D dequant).
            if getattr(self, "_hswq_convrot", False):
                cq_key = f"{prefix}comfy_quant"
                conf = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(
                        getattr(self, "_hswq_convrot_groupsize", 256) or 256
                    ),
                }
                sd[cq_key] = torch.tensor(
                    list(json.dumps(conf, separators=(",", ":")).encode("utf-8")),
                    dtype=torch.uint8,
                )
            return sd

        def _conv_forward(self, input, weight, bias):
            if self.padding_mode != "zeros":
                return torch.nn.functional.conv2d(
                    torch.nn.functional.pad(
                        input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                    ),
                    weight,
                    bias,
                    self.stride,
                    (0, 0),
                    self.dilation,
                    self.groups,
                )
            return torch.nn.functional.conv2d(
                input, weight, bias, self.stride, self.padding, self.dilation, self.groups
            )

        def forward_comfy_cast_weights(self, input):
            # Mirror MixedPrecision Linear: when weight is QuantizedTensor and
            # Dynamic VRAM uses weight_lowvram_function, want_requant=True so
            # post_cast dequant → LoRA → requant (want_requant=False left QT
            # in the resident path after the first step and killed LoRA).
            if getattr(self, "_hswq_convrot", False):
                nc = _load_native_convert_int8_helpers()
                gs = int(getattr(self, "_hswq_convrot_groupsize", 256) or 256)
                h = nc.build_hadamard(gs, device="cpu", dtype=torch.float32)
                input = nc.rotate_activation_nchw(input, h, gs)
            want_requant = isinstance(getattr(self, "weight", None), QuantizedTensor)
            weight, bias, offload_stream = cast_bias_weight(
                self,
                input,
                offloadable=True,
                compute_dtype=getattr(input, "dtype", None),
                want_requant=want_requant,
            )
            x = self._conv_forward(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return x

        def forward(self, input, *args, **kwargs):
            run_every_op()
            return self.forward_comfy_cast_weights(input)

        def convert_weight(self, weight, inplace=False, **kwargs):
            # Same contract as MixedPrecisionOps.Linear: LoRA / ModelPatcher
            # dequant → calculate_weight → set_weight (see ComfyUI-INT8-Fast bake path).
            # ConvRot weights are stored rotated; unrotate to original float basis for LoRA.
            # LowVRAM may re-materialize QT with Params.convrot still True — clear
            # before dequantize (kitchen ConvRot dequant is 2D-only).
            global _lora_convert_logs
            if isinstance(weight, QuantizedTensor):
                _arm_hswq_conv2d_convrot(self, QuantizedTensor)
                qt = _qt_payload(weight, QuantizedTensor)
                if qt is not None:
                    params = getattr(qt, "_params", None)
                    qdata = getattr(qt, "_qdata", None)
                    if (
                        params is not None
                        and qdata is not None
                        and getattr(qdata, "ndim", None) == 4
                        and bool(getattr(params, "convrot", False))
                    ):
                        import dataclasses

                        gs = int(getattr(params, "convrot_groupsize", 256) or 256)
                        self._hswq_convrot = True
                        self._hswq_convrot_groupsize = gs
                        new_params = dataclasses.replace(params, convrot=False)
                        try:
                            object.__setattr__(qt, "_params", new_params)
                        except Exception:
                            qt._params = new_params
                out = weight.dequantize()
            else:
                out = weight
            if getattr(self, "_hswq_convrot", False) and out is not None and getattr(out, "ndim", 0) == 4:
                nc = _load_native_convert_int8_helpers()
                gs = int(getattr(self, "_hswq_convrot_groupsize", 256) or 256)
                h = nc.build_hadamard(gs, device="cpu", dtype=torch.float32)
                out = nc.unrotate_weight_conv2d(out, h, gs)
            if _lora_convert_logs < _LORA_CONVERT_LOG_MAX:
                _lora_convert_logs += 1
                wdtype = getattr(weight, "dtype", None)
                odtype = getattr(out, "dtype", None)
                _console(
                    f"[HSWQ INT8 LoRA] Conv2d.convert_weight #{_lora_convert_logs}: "
                    f"in={type(weight).__name__}/{wdtype} -> out={type(out).__name__}/{odtype} "
                    f"layout={getattr(self, 'layout_type', None)} "
                    f"convrot={getattr(self, '_hswq_convrot', False)}"
                )
            return out

        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            # Mirror MixedPrecisionOps.Linear.set_weight so Conv2d LoRA bake
            # does not fall through to stochastic_rounding(..., int8), which
            # destroys float LoRA deltas (INT8-Fast: normal LoRA loader works).
            # ConvRot: convert_weight returned unrotated float; re-rotate before requant.
            global _lora_set_logs
            layout = getattr(self, "layout_type", None)
            path = "requant" if layout is not None else "cast_only"
            if getattr(self, "_hswq_convrot", False) and getattr(weight, "ndim", 0) == 4:
                nc = _load_native_convert_int8_helpers()
                gs = int(getattr(self, "_hswq_convrot_groupsize", 256) or 256)
                h = nc.build_hadamard(gs, device="cpu", dtype=torch.float32)
                weight = nc.rotate_weight_conv2d(weight, h, gs)
            if _lora_set_logs < _LORA_SET_LOG_MAX:
                _lora_set_logs += 1
                _console(
                    f"[HSWQ INT8 LoRA] Conv2d.set_weight #{_lora_set_logs}: "
                    f"path={path} float_in={getattr(weight, 'dtype', None)} "
                    f"shape={tuple(weight.shape) if hasattr(weight, 'shape') else '?'} "
                    f"seed={seed} layout={layout} "
                    f"convrot={getattr(self, '_hswq_convrot', False)}"
                )
            if layout is not None:
                weight = self.weight.requantize_from_float(
                    weight,
                    scale="recalculate",
                    stochastic_rounding=seed,
                    inplace_ops=True,
                ).to(self.weight.dtype)
            else:
                weight = weight.to(self.weight.dtype)
            if return_weight:
                return weight

            assert inplace_update is False
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

        def _apply(self, fn, recurse=True):
            return _quantized_apply(self, fn, recurse)

        @property
        def _reversed_padding_repeated_twice(self):
            return tuple(x for x in reversed(self.padding) for _ in range(2))

    return Conv2d


def _patch_ops_decode_and_conv() -> bool:
    try:
        import comfy.ops as ops_module
    except ImportError:
        return False

    ops_module._decode_comfy_quant_conf = decode_comfy_quant_conf

    original_load = getattr(ops_module, "_load_quantized_module", None)
    if original_load is None:
        return False

    if not getattr(original_load, "_hswq_int8_decode_patched", False):

        def _load_quantized_module_patched(
            module,
            super_load,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            load_extra_params=False,
        ):
            key = f"{prefix}comfy_quant"
            if key in state_dict:
                normalized = _normalize_comfy_quant_tensor(state_dict[key])
                if normalized is None:
                    state_dict.pop(key, None)
                else:
                    state_dict[key] = normalized
            return original_load(
                module,
                super_load,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
                load_extra_params=load_extra_params,
            )

        _load_quantized_module_patched._hswq_int8_decode_patched = True
        ops_module._load_quantized_module = _load_quantized_module_patched

    # Also normalize Embedding's direct json.loads path by wrapping Embedding._load_from_state_dict
    # is covered if convert_old_quants + file markers are normalized; keep load wrapper as safety.

    original_mp = getattr(ops_module, "mixed_precision_ops", None)
    if original_mp is None or not callable(original_mp):
        return False
    _OPS_PATCH_VER = 4  # Conv2d full ConvRot: online act rotate + safe 4D dequant
    true_orig = getattr(original_mp, "_hswq_orig_mixed_precision_ops", original_mp)
    if (
        getattr(original_mp, "_hswq_int8_ops_ver", 0) >= _OPS_PATCH_VER
        and getattr(original_mp, "_hswq_int8_conv_patched", False)
    ):
        # Heal older INT8 wraps that dropped NVFP4 stack markers (false upgrade).
        if int(getattr(original_mp, "_hswq_nvfp4_stack_ver", 0) or 0) <= 0:
            inner = true_orig
            for _ in range(6):
                if inner is None:
                    break
                v = int(getattr(inner, "_hswq_nvfp4_stack_ver", 0) or 0)
                if v > 0:
                    try:
                        original_mp._hswq_nvfp4_stack_ver = v
                    except Exception:
                        pass
                    break
                if getattr(inner, "_hswq_nvfp4_comfy_only", False):
                    try:
                        original_mp._hswq_nvfp4_comfy_only = True
                        if hasattr(inner, "_hswq_nvfp4_orig_mp"):
                            original_mp._hswq_nvfp4_orig_mp = getattr(
                                inner, "_hswq_nvfp4_orig_mp"
                            )
                    except Exception:
                        pass
                inner = getattr(inner, "_hswq_nvfp4_orig_mp", None) or getattr(
                    inner, "_hswq_orig_mixed_precision_ops", None
                )
        return True

    def mixed_precision_ops_force_conv(
        quant_config=None, compute_dtype=None, full_precision_mm=False, disabled=None
    ):
        if quant_config is None:
            quant_config = {}
        if compute_dtype is None:
            import torch

            compute_dtype = torch.bfloat16
        if disabled is None:
            disabled = []
        result = true_orig(
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            full_precision_mm=full_precision_mm,
            disabled=disabled,
        )
        # Inject Quantized Conv2d only during HSWQ INT8 load scope
        # (_int8_quant_conv_scope). Never from quant_config alone — that would
        # also hit stock UNETLoader / Krea2 ConvRot MixedPrecision builds.
        if _should_inject_int8_conv(quant_config):
            result.Conv2d = _make_quantized_conv2d(ops_module, result, disabled)
        return result

    mixed_precision_ops_force_conv._hswq_orig_mixed_precision_ops = true_orig
    mixed_precision_ops_force_conv._hswq_int8_conv_patched = True
    mixed_precision_ops_force_conv._hswq_int8_ops_ver = _OPS_PATCH_VER
    # Preserve NVFP4 markers through INT8 wrap so Z Image early-return works
    # (lost stack_ver → false "upgraded stack" → TC over parity → noise).
    for _attr in (
        "_hswq_nvfp4_stack_ver",
        "_hswq_nvfp4_comfy_only",
        "_hswq_nvfp4_orig_mp",
    ):
        if hasattr(original_mp, _attr):
            try:
                setattr(
                    mixed_precision_ops_force_conv,
                    _attr,
                    getattr(original_mp, _attr),
                )
            except Exception:
                pass
    ops_module.mixed_precision_ops = mixed_precision_ops_force_conv
    return True


def _patch_lowvram_patch_float_intermediate() -> bool:
    """Fix LowVramPatch intermediate_dtype for comfy_quant QuantizedTensor only.

    Upstream LowVramPatch passes intermediate_dtype=weight.dtype. When the
    weight is still a QuantizedTensor (int8 storage), LoRA matmul casts to
    int8 and either errors or silently produces a no-op delta — same bug as
    BobJohnson24/ComfyUI-INT8-Fast#76.

    Must NOT divert bare ``torch.int8`` tensors. Nunchaku SVDQ / Lumina2 use
    int8 storage; grabbing them here corrupts fused CUDA (Abort in
    ``_forward_silu_gating``) even when VRAM handoff already freed GPU memory.
    """
    try:
        import torch
        import comfy.lora
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False

    LowVramPatch = getattr(mp, "LowVramPatch", None)
    if LowVramPatch is None:
        return False
    original = getattr(LowVramPatch, "__call__", None)
    _LV_VER = 3
    if original is None or getattr(original, "_hswq_int8_lora_dtype_ver", 0) >= _LV_VER:
        return getattr(original, "_hswq_int8_lora_dtype", False)
    true_orig = getattr(original, "_hswq_orig_lowvram_call", original)

    def __call__(self, weight):
        # QuantizedTensor only. Bare int8 / float / None → upstream unchanged.
        if weight is None or not isinstance(weight, QuantizedTensor):
            return true_orig(self, weight)
        patches = (
            self.prepared_patches
            if self.prepared_patches is not None
            else self.patches[self.key]
        )
        w = weight.dequantize()
        dtype = getattr(w, "dtype", None)
        if dtype is not None and hasattr(dtype, "is_floating_point") and dtype.is_floating_point:
            idtype = dtype
        else:
            idtype = torch.float32
        return comfy.lora.calculate_weight(patches, w, self.key, intermediate_dtype=idtype)

    __call__._hswq_int8_lora_dtype = True
    __call__._hswq_int8_lora_dtype_ver = _LV_VER
    __call__._hswq_orig_lowvram_call = true_orig
    LowVramPatch.__call__ = __call__
    return True


def _get_baked_key_set(model) -> set:
    s = getattr(model, "_hswq_int8_baked_keys", None)
    if s is None:
        s = set()
        model._hswq_int8_baked_keys = s
    return s


def _maybe_invalidate_baked_keys(patcher) -> None:
    """If patches_uuid changed (new LoRA), allow those keys to be baked again."""
    model = patcher.model
    baked_uuid = getattr(model, "_hswq_int8_baked_uuid", None)
    cur = getattr(patcher, "patches_uuid", None)
    if baked_uuid is None or cur is None:
        return
    if baked_uuid != cur and patcher.patches:
        _get_baked_key_set(model).clear()
        model._hswq_int8_baked_uuid = None


def _strip_lowvram_for_baked_keys(patcher) -> int:
    """Dynamic.load re-attaches LowVramPatch; clear it for already-baked keys.

    Shared modules keep their VBAR ``_v`` across loads. Re-attaching LoRA on
    top of baked INT8 weights would double-apply; clearing lowvram avoids that.
    """
    _maybe_invalidate_baked_keys(patcher)
    baked = getattr(patcher.model, "_hswq_int8_baked_keys", None)
    if not baked:
        return 0
    cleared = 0
    for name, module in patcher.model.named_modules():
        for param_key in ("weight", "bias"):
            key = f"{name}.{param_key}"
            if key not in baked:
                continue
            attr = param_key + "_lowvram_function"
            if getattr(module, attr, None) is not None:
                setattr(module, attr, None)
                cleared += 1
            # Drop from this patcher's dict so later loads do not re-attach
            if key in patcher.patches:
                try:
                    del patcher.patches[key]
                except KeyError:
                    pass
    return cleared


def _bake_int8_patches_on_dynamic_patcher(patcher, device_to) -> int:
    """Bake LoRA into INT8 modules after ModelPatcherDynamic.load.

    Dynamic VRAM attaches LowVramPatch on weight_lowvram_function and asserts
    force_patch_weights=False. For comfy_quant INT8 that path often leaves
    LoRA attached in the patcher dict but visually inert (keys count OK,
    bake logs absent). We bake via convert_weight/set_weight (requant).

    Critical VBAR rule (2nd-gen FaceDetailer OOM):
      ModelVBAR.alloc is a bump allocator (offset only grows). Deleting
      module._v after bake makes the next load call alloc() again → VBAR OOM.
      Keep ``_v``. Clear LowVramPatch, bake, then pop patches + drop the
      pre-bake backup entry so restore_loaded_backups does not undo bake.
    """
    if _model_is_nunchaku_svdq(getattr(patcher, "model", None)):
        return 0
    # Absolute branch: Z Image / Krea parity packs use zimage_nvfp4 bake only.
    # Never apply the SDXL (3.3.0 nodes/nvfp4) Dynamic bake to them.
    if _model_is_zimage_nvfp4_parity(getattr(patcher, "model", None)):
        return 0
    if not getattr(patcher, "patches", None):
        return 0
    try:
        import comfy.model_patcher as mp
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return 0

    _maybe_invalidate_baked_keys(patcher)
    already = _get_baked_key_set(patcher.model)
    baked = 0
    for name, module in patcher.model.named_modules():
        keys_to_bake = []
        for param_key in ("weight", "bias"):
            key = f"{name}.{param_key}"
            if key not in patcher.patches:
                continue
            if key in already:
                # Already baked under this patches_uuid; clear re-attached LowVramPatch
                attr = param_key + "_lowvram_function"
                if getattr(module, attr, None) is not None:
                    setattr(module, attr, None)
                try:
                    del patcher.patches[key]
                except KeyError:
                    pass
                continue
            weight, set_func, convert_func = mp.get_key_weight(patcher.model, key)
            if weight is None:
                continue
            # SDXL path (3.3.0): bake all comfy_quant QuantizedTensor — never bare int8.
            # NVFP4 Linear uses nodes/nvfp4 ConvRot convert/set_weight.
            # Z Image path never reaches here (parity early-return above).
            if not isinstance(weight, QuantizedTensor):
                continue
            if set_func is None:
                _console(
                    f"[HSWQ INT8 LoRA] WARN cannot bake {key}: "
                    "QuantizedTensor but no set_weight (int8_round risk)"
                )
                continue
            keys_to_bake.append((param_key, key))

        if not keys_to_bake:
            continue

        # Clear LowVramPatch so bake uses Parameter + set_weight, not lazy patch.
        # Do NOT unpin/delete module._v — that causes 2nd-load VBAR OOM.
        for param_key, _key in keys_to_bake:
            if hasattr(module, param_key + "_lowvram_function"):
                setattr(module, param_key + "_lowvram_function", None)

        for _param_key, key in keys_to_bake:
            patcher.patch_weight_to_device(key, device_to=device_to)
            # Drop pre-bake backup so the next Dynamic.load restore keeps baked weights
            if key in patcher.backup:
                try:
                    del patcher.backup[key]
                except KeyError:
                    pass
            try:
                del patcher.patches[key]
            except KeyError:
                pass
            already.add(key)
            baked += 1

    if baked > 0:
        patcher.model._hswq_int8_baked_uuid = getattr(patcher, "patches_uuid", None)

    return baked


def _patch_model_patcher_dynamic_int8_lora_bake() -> bool:
    """After ModelPatcherDynamic.load, bake INT8 LoRA via set_weight."""
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return False

    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    if Dynamic is None:
        return False
    original = getattr(Dynamic, "load", None)
    if original is None:
        return False
    _DYN_VER = 8
    # Never wrap over live Z Image bake. Capturing ZI as true_orig leaves
    # ``Dynamic.load ENTER … model=SDXL`` after ZI uninstall (closure), and SDXL
    # LoRA bake strength breaks even when Status shows APPLIED.
    if getattr(original, "_hswq_zi_nvfp4_lora_bake", False):
        return True
    if getattr(original, "_hswq_int8_lora_bake_ver", 0) >= _DYN_VER:
        return True
    true_orig = getattr(original, "_hswq_orig_dynamic_load", original)

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False, dirty=False):
        result = true_orig(
            self,
            device_to=device_to,
            lowvram_model_memory=lowvram_model_memory,
            force_patch_weights=force_patch_weights,
            full_load=full_load,
            dirty=dirty,
        )
        # INT8 LoRA bake only — never touch Nunchaku SVDQ (class is often Lumina2).
        if _model_is_nunchaku_svdq(self.model):
            return result
        if not _model_has_int8_quantized_weights(self.model) and not getattr(
            self.model, "_hswq_int8_baked_keys", None
        ):
            return result
        # Load re-attaches LowVramPatch for any keys still in patches / clones
        _strip_lowvram_for_baked_keys(self)
        if self.patches:
            n = _bake_int8_patches_on_dynamic_patcher(self, device_to=device_to)
            if n > 0 or _lora_attach_history or (_lora_attach_last.get("mapped_keys") or 0) > 0:
                dump_int8_lora_bake_stats(force=True)
        elif _lora_attach_history or (_lora_attach_last.get("mapped_keys") or 0) > 0:
            # Patches already consumed by a prior bake; still emit Status once if needed
            dump_int8_lora_bake_stats(force=False)
        return result

    load._hswq_int8_lora_bake = True
    load._hswq_int8_lora_bake_ver = _DYN_VER
    load._hswq_orig_dynamic_load = true_orig
    Dynamic.load = load
    return True


def _force_detach_int8_dynamic_models(device=None, keep_patchers=None) -> int:
    """Offload INT8 Dynamic VRAM (VBAR + hostbufs) without destroying QT weights.

    free_memory often stops after partially_unload and leaves HostBuffers, so
    Nunchaku sees ``0.00 MB usable`` and Aborts. We must fully offload INT8
    Dynamic models before SVDQ load.

    Critical: use ``unpatch_weights=False`` / ``detach(unpatch_all=False)``.
    ``unpatch_all=True`` unpatches INT8 QuantizedTensor + baked LoRA and causes
    black / noise on the next normal SDXL INT8 KSampler.
    """
    try:
        import comfy.model_management as mm
    except ImportError:
        return 0

    keep_ids = {id(p) for p in (keep_patchers or []) if p is not None}
    unloaded = 0
    i = 0
    while i < len(mm.current_loaded_models):
        lm = mm.current_loaded_models[i]
        patcher = lm.model
        if patcher is None:
            i += 1
            continue
        if id(patcher) in keep_ids:
            i += 1
            continue
        if device is not None and getattr(lm, "device", None) is not None:
            try:
                if str(lm.device) != str(device):
                    i += 1
                    continue
            except Exception:
                pass
        is_dyn = False
        try:
            is_dyn = bool(patcher.is_dynamic())
        except Exception:
            is_dyn = False
        if not is_dyn:
            i += 1
            continue
        base = getattr(patcher, "model", None)
        if base is None or not _model_has_int8_quantized_weights(base):
            i += 1
            continue
        # Preserve QT + baked LoRA; only free GPU / VBAR occupancy.
        try:
            lm.model_unload(unpatch_weights=False)
        except TypeError:
            try:
                patcher.detach(unpatch_all=False)
            except TypeError:
                try:
                    patcher.detach(False)
                except Exception as exc:
                    _console(f"[HSWQ INT8→Nunchaku] detach(False) failed: {exc!r}")
            except Exception as exc:
                _console(f"[HSWQ INT8→Nunchaku] detach(unpatch_all=False) failed: {exc!r}")
            try:
                fin = getattr(lm, "model_finalizer", None)
                if fin is not None:
                    fin.detach()
            except Exception:
                pass
            try:
                lm.model_finalizer = None
                lm.real_model = None
            except Exception:
                pass
        except Exception as exc:
            _console(f"[HSWQ INT8→Nunchaku] model_unload(False) failed: {exc!r}")
            i += 1
            continue
        mm.current_loaded_models.pop(i)
        unloaded += 1
    if unloaded > 0:
        try:
            mm.soft_empty_cache()
        except Exception:
            pass
    return unloaded


def _patch_load_models_gpu_int8_nunchaku_handoff() -> bool:
    """Before Nunchaku SVDQ load, offload INT8 Dynamic VRAM without unpatch."""
    try:
        import comfy.model_management as mm
    except ImportError:
        return False

    original = getattr(mm, "load_models_gpu", None)
    if original is None:
        return False
    # v10 = handoff arms ONLY for real Nunchaku SVDQ. All other loads
    # (SDXL / Flux / ZIT / native INT8 / FP / …) pass through untouched.
    # free_memory → model_unload(unpatch_weights=True) kills non-SVDQ INT8.
    _VER = 10
    if getattr(original, "_hswq_int8_nunchaku_handoff_ver", 0) >= _VER:
        return True
    true_orig = getattr(original, "_hswq_orig_load_models_gpu", original)

    def load_models_gpu(
        models,
        memory_required=0,
        force_patch_weights=False,
        minimum_memory_required=None,
        force_full_load=False,
    ):
        keep = []
        need_handoff = False
        device = None
        for m in models or []:
            keep.append(m)
            for mm_extra in getattr(m, "model_patches_models", lambda: [])() or []:
                keep.append(mm_extra)
            base = getattr(m, "model", None)
            # Branch A: native comfy_quant INT8 (any architecture) — never handoff.
            if base is not None and _model_has_int8_quantized_weights(base):
                continue
            # Branch B: only real Nunchaku SVDQ on the BaseModel arms handoff.
            # Do not probe the ModelPatcher itself (false positives).
            if base is not None and _model_is_nunchaku_svdq(base):
                need_handoff = True
                if device is None:
                    device = getattr(m, "load_device", None)
        if need_handoff:
            n = _force_detach_int8_dynamic_models(device=device, keep_patchers=keep)
            # Second pass: any INT8 Dynamic still listed (missed first pass) —
            # never leave them for free_memory(unpatch=True).
            n2 = _force_detach_int8_dynamic_models(device=None, keep_patchers=keep)
            try:
                mm.soft_empty_cache()
            except Exception as exc:
                _console(f"[HSWQ INT8→Nunchaku] soft_empty_cache failed: {exc!r}")
            _console(
                f"[HSWQ INT8→Nunchaku] VRAM handoff before SVDQ load "
                f"(INT8 Dynamic offload keep-weights={n + n2}, no free_memory unpatch)"
            )
        return true_orig(
            models,
            memory_required=memory_required,
            force_patch_weights=force_patch_weights,
            minimum_memory_required=minimum_memory_required,
            # Full load after handoff avoids Nunchaku 0.00 MB usable Abort.
            # Non-SVDQ loads never set need_handoff (branches A/B above).
            force_full_load=True if need_handoff else force_full_load,
        )

    load_models_gpu._hswq_int8_nunchaku_handoff = True
    load_models_gpu._hswq_int8_nunchaku_handoff_ver = _VER
    load_models_gpu._hswq_orig_load_models_gpu = true_orig
    mm.load_models_gpu = load_models_gpu
    return True


def _patch_model_patcher_lora_logs() -> bool:
    """Log whether LoRA bake uses set_weight (requant) or int8_round fallback."""
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return False

    original = getattr(mp.ModelPatcher, "patch_weight_to_device", None)
    if original is None or getattr(original, "_hswq_int8_lora_log", False):
        return getattr(original, "_hswq_int8_lora_log", False)

    def patch_weight_to_device_logged(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        global _lora_patcher_logs
        weight, set_func, convert_func = mp.get_key_weight(self.model, key)
        if key in self.patches:
            _lora_patcher_stats["calls"] += 1
            if set_func is not None:
                _lora_patcher_stats["with_set_func"] += 1
            else:
                _lora_patcher_stats["without_set_func"] += 1
            if convert_func is not None:
                _lora_patcher_stats["with_convert_func"] += 1

            path = "requant" if set_func is not None else "int8_round"
            _lora_bake_by_key[key] = path
            if _lora_patcher_logs < _LORA_PATCHER_LOG_MAX:
                _lora_patcher_logs += 1
                wdtype = getattr(weight, "dtype", None)
                warn = ""
                if set_func is None and wdtype is not None and str(wdtype) in ("torch.int8", "int8"):
                    warn = "  << BROKEN for INT8 (LoRA delta will be destroyed)"
                owners = [
                    e["lora_name"]
                    for e in _lora_attach_history
                    if key in (e.get("applied_unet_keys") or [])
                ]
                owner_s = ",".join(owners[:3]) if owners else "-"
                if len(owners) > 3:
                    owner_s += f"+{len(owners) - 3}"
                _console(
                    f"[HSWQ INT8 LoRA] bake #{_lora_patcher_logs}: key={key} "
                    f"path={path} lora={owner_s} weight_dtype={wdtype} "
                    f"convert={'yes' if convert_func else 'no'} "
                    f"set={'yes' if set_func else 'no'}{warn}"
                )
            # After stacked UNet keys are baked, dump per-LoRA summary once
            target = sum(int(e.get("applied_unet") or 0) for e in _lora_attach_history)
            if target <= 0:
                target = int(_lora_attach_last.get("applied_unet") or 0)
            # Unique baked keys may be less than sum (shared keys across LoRAs)
            unique_target = len(
                {
                    k
                    for e in _lora_attach_history
                    for k in (e.get("applied_unet_keys") or [])
                }
            ) or target
            if (
                unique_target > 0
                and _lora_patcher_stats["calls"] >= unique_target
                and not getattr(dump_int8_lora_bake_stats, "_dumped_this_load", False)
            ):
                # Do NOT set the flag before dump (that made dump a no-op).
                dump_int8_lora_bake_stats(force=False)


        return original(
            self,
            key,
            device_to=device_to,
            inplace_update=inplace_update,
            return_weight=return_weight,
            force_cast=force_cast,
        )

    patch_weight_to_device_logged._hswq_int8_lora_log = True
    mp.ModelPatcher.patch_weight_to_device = patch_weight_to_device_logged
    return True


def _per_lora_bake_verdict(entry: dict) -> tuple[str, int, int, int]:
    """Return (verdict, requant, int8_round, not_baked) for one LoRA attach entry."""
    unet_keys = entry.get("applied_unet_keys") or []
    clip_n = int(entry.get("applied_clip") or 0)
    unet_n = int(entry.get("applied_unet") or 0)
    if unet_n == 0 and clip_n > 0:
        return ("N/A_CLIP_only", 0, 0, 0)
    if unet_n == 0:
        return ("SKIP_no_keys", 0, 0, 0)
    requant = 0
    int8_round = 0
    not_baked = 0
    for k in unet_keys:
        path = _lora_bake_by_key.get(k)
        if path == "requant":
            requant += 1
        elif path == "int8_round":
            int8_round += 1
        else:
            not_baked += 1
    if int8_round > 0:
        return ("BROKEN_int8_round", requant, int8_round, not_baked)
    if requant == 0 and not_baked == unet_n:
        return ("WARN_not_baked_yet", requant, int8_round, not_baked)
    if requant > 0 and int8_round == 0:
        return ("OK_requant", requant, int8_round, not_baked)
    return ("PARTIAL", requant, int8_round, not_baked)


def dump_int8_lora_bake_stats(force: bool = False) -> None:
    """Full Status dump: lora_name / applied_keys / skipped_keys (+ bake if any)."""
    if not force and getattr(dump_int8_lora_bake_stats, "_dumped_this_load", False):
        return
    dump_int8_lora_bake_stats._dumped_this_load = True

    history = list(_lora_attach_history) if _lora_attach_history else []
    if not history and (_lora_attach_last.get("mapped_keys") or 0) > 0:
        history = [dict(_lora_attach_last)]

    n = len(history)
    _lora_line(f"[HSWQ LoRA Status] ===== bake summary ({n} slot(s)) =====")
    if not history:
        _lora_line(
            "[HSWQ LoRA Status] Slot -: | lora_name='(none)' | applied_keys=0 | skipped_keys=0 | → SKIPPED ✗"
        )
    ok_n = 0
    for i, a in enumerate(history, 1):
        line = _format_lora_slot_line(i, a, include_bake=True)
        _lora_line(line)
        verdict, _rq, _ir, _nb = _per_lora_bake_verdict(a)
        if verdict in ("OK_requant", "N/A_CLIP_only") or _slot_applied_count(a) > 0:
            if verdict != "BROKEN_int8_round":
                ok_n += 1
    _lora_line(
        f"[HSWQ LoRA Status] Summary: {ok_n}/{n} LoRA(s) with applied keys"
    )

    s = _lora_patcher_stats
    if s["calls"] == 0:
        _lora_line("[HSWQ LoRA Bake] not yet (model not on GPU)")
        return
    _lora_line(
        f"[HSWQ LoRA Bake] total={s['calls']} requant={s['with_set_func']} "
        f"int8_round={s['without_set_func']} shape_skip={len(_lora_shape_skips)}"
    )
    if s["without_set_func"] > 0:
        _lora_line(
            "[HSWQ LoRA Bake] WARNING: int8_round used — those layers are broken"
        )
    else:
        _lora_line("[HSWQ LoRA Bake] path OK (all requant)")
    if _lora_shape_skips:
        for name, key, reason in _lora_shape_skips[:_LORA_SKIP_PRINT_MAX]:
            _lora_line(
                f"[HSWQ LoRA Bake] shape_skip | '{name}' | {key} | {reason}"
            )


def _patch_lora_loader_name_context() -> bool:
    """Capture name from nodes.LoraLoader when any node calls it."""
    try:
        import nodes as nodes_mod
    except ImportError:
        return False

    LoraLoader = getattr(nodes_mod, "LoraLoader", None)
    if LoraLoader is None:
        return False
    original = getattr(LoraLoader, "load_lora", None)
    if original is None:
        return False
    _NAME_VER = 6
    if getattr(original, "_hswq_lora_name_ctx_ver", 0) >= _NAME_VER:
        return True
    true_orig = getattr(original, "_hswq_orig_load_lora", original)

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        global _current_lora_name, _current_lora_strength_model, _current_lora_strength_clip
        prev = (
            _current_lora_name,
            _current_lora_strength_model,
            _current_lora_strength_clip,
        )
        _set_current_lora_name(lora_name, strength_model, strength_clip)
        try:
            return true_orig(self, model, clip, lora_name, strength_model, strength_clip)
        finally:
            (
                _current_lora_name,
                _current_lora_strength_model,
                _current_lora_strength_clip,
            ) = prev

    load_lora._hswq_lora_name_ctx = True
    load_lora._hswq_lora_name_ctx_ver = _NAME_VER
    load_lora._hswq_orig_load_lora = true_orig
    LoraLoader.load_lora = load_lora
    return True


def _patch_loras_folder_path_name() -> bool:
    """Any loader that resolves folder_paths 'loras' → capture filename."""
    try:
        import folder_paths
    except ImportError:
        return False

    _PATH_VER = 3
    ok = False

    for fname in ("get_full_path", "get_full_path_or_raise"):
        original = getattr(folder_paths, fname, None)
        if original is None:
            continue
        if getattr(original, "_hswq_lora_path_name_ver", 0) >= _PATH_VER:
            ok = True
            continue
        true_orig = getattr(original, "_hswq_orig_get_full_path", original)

        def _make(orig):
            def wrapped(folder_name, filename):
                if folder_name == "loras":
                    _set_current_lora_name(filename)
                return orig(folder_name, filename)

            wrapped._hswq_lora_path_name_ver = _PATH_VER
            wrapped._hswq_orig_get_full_path = orig
            return wrapped

        setattr(folder_paths, fname, _make(true_orig))
        ok = True
    return ok


def _patch_load_torch_file_lora_name() -> bool:
    """Any loader that load_torch_file(lora_path) → capture basename."""
    try:
        import comfy.utils as utils_mod
    except ImportError:
        return False
    original = getattr(utils_mod, "load_torch_file", None)
    if original is None:
        return False
    _TORCH_VER = 1
    if getattr(original, "_hswq_lora_torch_name_ver", 0) >= _TORCH_VER:
        return True
    true_orig = getattr(original, "_hswq_orig_load_torch_file", original)

    def load_torch_file(ckpt, *args, **kwargs):
        if isinstance(ckpt, (str, os.PathLike)):
            p = str(ckpt)
            if _path_is_under_loras_dir(p):
                _set_current_lora_name(p)
        return true_orig(ckpt, *args, **kwargs)

    load_torch_file._hswq_lora_torch_name_ver = _TORCH_VER
    load_torch_file._hswq_orig_load_torch_file = true_orig
    utils_mod.load_torch_file = load_torch_file
    return True


def _patch_load_lora_key_counts() -> bool:
    """Wrap load_lora + load_lora_for_models for applied/skipped key counts."""
    try:
        import comfy.lora as lora_mod
        import comfy.sd as sd_mod
        import comfy.weight_adapter as weight_adapter
    except ImportError:
        return False

    orig_load_lora = getattr(lora_mod, "load_lora", None)
    orig_for_models = getattr(sd_mod, "load_lora_for_models", None)
    if orig_load_lora is None or orig_for_models is None:
        return False

    _KEY_VER = 6
    if getattr(orig_for_models, "_hswq_lora_key_count_ver", 0) >= _KEY_VER:
        _patch_lora_loader_name_context()
        _patch_loras_folder_path_name()
        _patch_load_torch_file_lora_name()
        return True

    if getattr(orig_for_models, "_hswq_lora_key_count", False):
        orig_for_models = getattr(
            orig_for_models, "_hswq_orig_for_models", orig_for_models
        )
    if getattr(orig_load_lora, "_hswq_lora_key_count", False):
        orig_load_lora = getattr(orig_load_lora, "_hswq_orig_load_lora", orig_load_lora)

    _ctx = {"patch_dict": {}, "not_mapped": [], "file_keys": 0}

    def load_lora_counted(lora, to_load, log_missing=True):
        patch_dict = {}
        loaded_keys = set()
        for x in to_load:
            alpha_name = "{}.alpha".format(x)
            alpha = None
            if alpha_name in lora.keys():
                alpha = lora[alpha_name].item()
                loaded_keys.add(alpha_name)

            dora_scale_name = "{}.dora_scale".format(x)
            dora_scale = None
            if dora_scale_name in lora.keys():
                dora_scale = lora[dora_scale_name]
                loaded_keys.add(dora_scale_name)

            for adapter_cls in weight_adapter.adapters:
                adapter = adapter_cls.load(x, lora, alpha, dora_scale, loaded_keys)
                if adapter is not None:
                    patch_dict[to_load[x]] = adapter
                    loaded_keys.update(adapter.loaded_keys)
                    continue

            w_norm_name = "{}.w_norm".format(x)
            b_norm_name = "{}.b_norm".format(x)
            w_norm = lora.get(w_norm_name, None)
            b_norm = lora.get(b_norm_name, None)

            if w_norm is not None:
                loaded_keys.add(w_norm_name)
                patch_dict[to_load[x]] = ("diff", (w_norm,))
                if b_norm is not None:
                    loaded_keys.add(b_norm_name)
                    patch_dict["{}.bias".format(to_load[x][: -len(".weight")])] = (
                        "diff",
                        (b_norm,),
                    )

            diff_name = "{}.diff".format(x)
            diff_weight = lora.get(diff_name, None)
            if diff_weight is not None:
                patch_dict[to_load[x]] = ("diff", (diff_weight,))
                loaded_keys.add(diff_name)

            diff_bias_name = "{}.diff_b".format(x)
            diff_bias = lora.get(diff_bias_name, None)
            if diff_bias is not None:
                patch_dict["{}.bias".format(to_load[x][: -len(".weight")])] = (
                    "diff",
                    (diff_bias,),
                )
                loaded_keys.add(diff_bias_name)

            set_weight_name = "{}.set_weight".format(x)
            set_weight = lora.get(set_weight_name, None)
            if set_weight is not None:
                patch_dict[to_load[x]] = ("set", (set_weight,))
                loaded_keys.add(set_weight_name)

        not_mapped = [x for x in lora.keys() if x not in loaded_keys]
        _ctx["patch_dict"] = patch_dict
        _ctx["not_mapped"] = not_mapped
        _ctx["file_keys"] = len(lora) if hasattr(lora, "keys") else 0

        if log_missing:
            for x in not_mapped:
                logging.warning("lora key not loaded: {}".format(x))

        return patch_dict

    def load_lora_for_models_counted(
        model, clip, lora, strength_model, strength_clip, lora_metadata=None
    ):
        new_model, new_clip = orig_for_models(
            model, clip, lora, strength_model, strength_clip, lora_metadata
        )
        loaded = _ctx.get("patch_dict") or {}
        not_mapped = list(_ctx.get("not_mapped") or [])
        file_key_count = int(_ctx.get("file_keys") or 0)
        lora_name = _resolve_lora_name(loaded)

        unet_keys = set(new_model.patches.keys()) if new_model is not None else set()
        if new_clip is not None and hasattr(new_clip, "patcher"):
            clip_keys = set(new_clip.patcher.patches.keys())
        else:
            clip_keys = set()

        applied_unet_keys = []
        applied_clip_keys = []
        mapped_but_not = []
        add_patches_miss = []
        for x in loaded:
            key = x if isinstance(x, str) else x[0]
            in_u = key in unet_keys
            in_c = key in clip_keys
            if in_u:
                applied_unet_keys.append(key)
            if in_c:
                applied_clip_keys.append(key)
            if not in_u and not in_c:
                mapped_but_not.append(x)
                add_patches_miss.append(x)

        applied_unet = len(applied_unet_keys)
        applied_clip = len(applied_clip_keys)

        entry = {
            "lora_name": lora_name,
            "strength_model": strength_model,
            "strength_clip": strength_clip,
            "lora_file_keys": file_key_count,
            "mapped_keys": len(loaded),
            "applied_unet": applied_unet,
            "applied_clip": applied_clip,
            "applied_unet_keys": list(applied_unet_keys),
            "applied_clip_keys": list(applied_clip_keys),
            "not_mapped": sorted(str(x) for x in not_mapped),
            "mapped_but_not_attached": list(mapped_but_not),
            "add_patches_skipped_unet": list(add_patches_miss),
        }
        _lora_attach_last.update(entry)
        _lora_attach_history.append(dict(entry))
        _log_lora_slot_attach(entry)
        return (new_model, new_clip)

    load_lora_counted._hswq_lora_key_count = True
    load_lora_counted._hswq_orig_load_lora = orig_load_lora
    load_lora_for_models_counted._hswq_lora_key_count = True
    load_lora_for_models_counted._hswq_lora_key_count_ver = _KEY_VER
    load_lora_for_models_counted._hswq_orig_for_models = orig_for_models
    lora_mod.load_lora = load_lora_counted
    sd_mod.load_lora_for_models = load_lora_for_models_counted
    _patch_lora_loader_name_context()
    _patch_loras_folder_path_name()
    _patch_load_torch_file_lora_name()
    return True


def _patch_controllora_int8_dequant() -> bool:
    """Dequantize borrowed base-UNet quantized weights in ControlLora.pre_run.

    LoRA-type ControlNets (``lora_controlnet`` marker, e.g. anytest) build a
    control_model that BORROWS the base UNet's own weights via
    ``diffusion_model.state_dict()`` and injects them with ``set_attr_param``.
    The control_model uses ``ControlLoraOps`` (plain float ops); its forward
    calls ``comfy.ops.cast_bias_weight``, which cannot reconstruct a quantized
    weight without its scale.

    Root cause (confirmed from logs + comfy/ops.py):
    ``MixedPrecisionOps.state_dict`` (``_quantized_weight_state_dict``) does NOT
    emit ``QuantizedTensor`` objects. It FLATTENS each quantized ``weight`` into
    separate tensors:
      * ``X.weight``        -> raw int8 qdata      (torch.int8)
      * ``X.weight_scale``  -> per-tensor scale    (torch.float32)
      * ``X.comfy_quant``   -> JSON metadata       (torch.uint8)
      * ``X.input_scale`` / ``X.weight_scale_2`` -> extra params (fp8/nvfp4)
    So ``ControlLora.pre_run`` injects the RAW int8 ``X.weight`` (no scale) into
    the float control_model, and forward feeds int8 straight into
    ``F.linear`` / ``conv2d`` -> NaN / black output. FP8 avoids this only
    because its dtype differs from the compute dtype.

    Fix: wrap ``diffusion_model.state_dict`` during ``ControlLora.pre_run`` and
    return a DEQUANTIZED state dict: for every module whose ``.weight`` is a
    ``QuantizedTensor``, replace ``X.weight`` with ``weight.dequantize()`` (a
    real float tensor) and drop the now-meaningless sidecar keys
    (``X.weight_scale``, ``X.weight_scale_2``, ``X.comfy_quant``,
    ``X.input_scale``). All non-quant weights, biases and buffers pass through
    unchanged. Full-weight ControlNets (Canny) never enter
    ``ControlLora.pre_run`` and are unaffected; the real anytest LoRA weights
    (``.up`` / ``.down``) are plain fp16 and are not touched.
    """
    try:
        import comfy.controlnet as cn
        import comfy.utils
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False

    ControlLora = getattr(cn, "ControlLora", None)
    if ControlLora is None:
        return False
    original = getattr(ControlLora, "pre_run", None)
    _CL_VER = 2
    if original is None or getattr(original, "_hswq_int8_controllora_ver", 0) >= _CL_VER:
        return getattr(original, "_hswq_int8_controllora", False)
    true_orig = getattr(original, "_hswq_orig_controllora_pre_run", original)

    def _dequantized_state_dict(diffusion_model, orig_sd):
        """Return diffusion_model.state_dict() with quantized weights turned
        back into float tensors and their scale/metadata sidecars removed.

        ``orig_sd`` is the ORIGINAL bound ``state_dict`` method captured before
        we replaced ``diffusion_model.state_dict``. It MUST be used here instead
        of ``diffusion_model.state_dict()`` to avoid re-entering our wrapper
        (which caused ``RecursionError: maximum recursion depth exceeded``)."""
        full = orig_sd()

        # Collect the state-dict prefix of every quantized weight.
        quant_weight_keys = {}
        for name, module in diffusion_model.named_modules():
            w = getattr(module, "weight", None)
            if isinstance(w, QuantizedTensor):
                key = (name + "." if name else "") + "weight"
                quant_weight_keys[key] = w

        out = {}
        n_dequant = 0
        n_drop = 0
        for k, v in full.items():
            replaced = False
            dropped = False
            for wk, qt in quant_weight_keys.items():
                if k == wk:
                    # raw int8 qdata -> real float weight
                    try:
                        out[k] = qt.dequantize()
                        n_dequant += 1
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "[HSWQ INT8] ControlLora: dequantize failed for %s: %s",
                            k, e,
                        )
                        out[k] = v
                    replaced = True
                    break
                base = wk[: -len("weight")]  # "X."
                if (
                    (k.startswith(wk) and k != wk)      # X.weight_scale / weight_scale_2
                    or k == base + "comfy_quant"        # uint8 JSON metadata
                    or k == base + "input_scale"        # fp8 extra param
                ):
                    dropped = True
                    break
            if replaced:
                continue
            if dropped:
                n_drop += 1
                continue
            out[k] = v

        print(
            f"[HSWQ INT8][ControlLora] dequantized state_dict: "
            f"weights dequantized(int8->float)={n_dequant}, "
            f"sidecar keys dropped(scale/comfy_quant/input_scale)={n_drop}, "
            f"total keys out={len(out)}",
            flush=True,
        )
        return out

    def pre_run(self, model, percent_to_timestep_function):
        diffusion_model = getattr(model, "diffusion_model", None)
        patched = False
        orig_sd = None
        if diffusion_model is not None:
            orig_sd = diffusion_model.state_dict

            def dequant_state_dict(*a, **kw):
                # Only intercept the argument-less borrow call ControlLora makes;
                # fall back to the original for any keyword/destination usage.
                if a or kw:
                    return orig_sd(*a, **kw)
                return _dequantized_state_dict(diffusion_model, orig_sd)

            print(
                "[HSWQ INT8][ControlLora] pre_run ENTER "
                "(LoRA-type ControlNet / lora_controlnet path) "
                "-> wrapping diffusion_model.state_dict for INT8 base-weight dequant",
                flush=True,
            )
            diffusion_model.state_dict = dequant_state_dict
            patched = True
        else:
            print(
                "[HSWQ INT8][ControlLora] pre_run ENTER but model has no "
                "diffusion_model; running unpatched",
                flush=True,
            )

        try:
            result = true_orig(self, model, percent_to_timestep_function)
        finally:
            if patched:
                # Remove the instance-level override so the class method is used again.
                try:
                    del diffusion_model.state_dict
                except AttributeError:
                    diffusion_model.state_dict = orig_sd

        print(
            "[HSWQ INT8][ControlLora] pre_run EXIT (base weights injected as float)",
            flush=True,
        )
        logger.info(
            "[HSWQ INT8] ControlLora: injected dequantized base UNet weights "
            "(anytest / lora_controlnet black-output fix)"
        )
        return result

    pre_run._hswq_int8_controllora = True
    pre_run._hswq_int8_controllora_ver = _CL_VER
    pre_run._hswq_orig_controllora_pre_run = true_orig
    ControlLora.pre_run = pre_run
    print(
        "[HSWQ INT8][ControlLora] pre_run patch INSTALLED "
        "(v%d): borrowed INT8 base weights dequantized via state_dict wrap "
        "for LoRA-type ControlNet (anytest fix)" % _CL_VER,
        flush=True,
    )
    return True


def apply_comfy_quant_int8_patches() -> bool:
    """Install INT8 comfy_quant patches once. Returns True if applied (or already applied)."""
    global _PATCHES_APPLIED
    ok_keys = _patch_load_lora_key_counts()
    ok_name = _patch_lora_loader_name_context()
    ok_path = _patch_loras_folder_path_name()
    ok_torch = _patch_load_torch_file_lora_name()
    ok_lowvram = _patch_lowvram_patch_float_intermediate()
    ok_dyn_bake = _patch_model_patcher_dynamic_int8_lora_bake()
    ok_handoff = _patch_load_models_gpu_int8_nunchaku_handoff()
    ok_controllora = _patch_controllora_int8_dequant()
    # Re-apply ops when patch version bumps (e.g. Conv2d inject gate change).
    ok_ops = _patch_ops_decode_and_conv()
    if _PATCHES_APPLIED:
        return True
    ok_utils = _patch_convert_old_quants()
    ok_lora_log = _patch_model_patcher_lora_logs()
    if ok_ops:
        _PATCHES_APPLIED = True
        _console(
            "[HSWQ INT8] comfy_quant patches applied "
            f"(Conv2d quant load + decode"
            f"{' + convert_old_quants' if ok_utils else ''}"
            f"{' + LoRA bake logs' if ok_lora_log else ''}"
            f"{' + LoRA key counts' if ok_keys else ''}"
            f"{' + LoRA name' if ok_name or ok_path or ok_torch else ''}"
            f"{' + LowVramPatch float dtype' if ok_lowvram else ''}"
            f"{' + Dynamic INT8 LoRA bake' if ok_dyn_bake else ''}"
            f"{' + INT8→Nunchaku VRAM handoff' if ok_handoff else ''}"
            f"{' + ControlLora INT8 dequant' if ok_controllora else ''})"
        )
        return True
    logger.warning(
        "[HSWQ INT8] Failed to apply comfy_quant patches (ops=%s utils=%s)",
        ok_ops,
        ok_utils,
    )
    return False


def load_unet_hswq_weight_dtype(unet_name, weight_dtype):
    import logging
    import torch
    import folder_paths
    import comfy.sd

    # INT8 Conv2d patches: SDXL/ZI UNet (architecture), even if Linear has ConvRot.
    # Krea2/DiT ConvRot: stock-equivalent load — Conv2d inject inflates VRAM vs stock.
    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    is_convrot = checkpoint_looks_like_comfy_quant_convrot(unet_path)
    is_int8 = weight_dtype == "int8_tensorwise" or checkpoint_looks_like_comfy_quant_int8(unet_path)
    needs_conv2d = checkpoint_needs_hswq_int8_conv2d(unet_path)

    if is_int8 and is_convrot and not needs_conv2d:
        model_options = {}
        logging.info(
            "[HSWQ INT8] DiT/Krea2 ConvRot — stock-equivalent load "
            "(no INT8 Conv2d patches): %s",
            unet_name,
        )
        print(
            f"[HSWQ INT8] ConvRot DiT/Krea2 stock-equivalent load: {unet_name}",
            flush=True,
        )
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
    elif is_int8:
        apply_comfy_quant_int8_patches()
        model_options = {}
        reset_int8_lora_log_counters()
        if is_convrot and needs_conv2d:
            logging.info(
                "[HSWQ INT8] SDXL/ZI + ConvRot FULL — MixedPrecision + INT8 Conv2d "
                "(Linear: kitchen online; Conv2d: HSWQ online act rotate): %s",
                unet_name,
            )
            print(
                f"[HSWQ INT8] SDXL/ZI ConvRot FULL (Linear+Conv2d) load: {unet_name}",
                flush=True,
            )
        else:
            logging.info(
                "[HSWQ INT8] Loading UNet via MixedPrecisionOps (int8_tensorwise / comfy_quant)"
            )
            print(f"[HSWQ INT8] Loading UNet: {unet_name}", flush=True)
        with _int8_quant_conv_scope():
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        summarize_int8_lora_capability(model)
    else:
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

    return (model,)


def load_checkpoint_sdxl_hswq_weight_dtype(ckpt_name, weight_dtype, device=None):
    import sys
    import torch
    import folder_paths
    import comfy.sd

    pkg = sys.modules[__name__.rsplit(".", 2)[0]]
    get_current_device = pkg.get_current_device
    set_current_device = pkg.set_current_device
    sdxl_logger = pkg.sdxl_logger

    original_device = get_current_device()
    if device is not None:
        set_current_device(device)
    try:
        # INT8 Conv2d + comfy_quant decode only when checkpoint is INT8.
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        # Auto-detect native comfy_quant INT8; do not force float8 dtype over int8 weights.
        is_int8 = weight_dtype == "int8_tensorwise" or checkpoint_looks_like_comfy_quant_int8(ckpt_path)

        model_options = {}
        if is_int8:
            # Z Image comfy_parity + ZI Dynamic bake must not wrap SDXL INT8.
            try:
                from ..nodes.nvfp4.comfy_quant_nvfp4 import (
                    _clear_zimage_parity_contamination_for_sdxl,
                )

                _clear_zimage_parity_contamination_for_sdxl()
            except Exception as e:
                sdxl_logger.warning(
                    "[SDXL INT8] clear Z Image NVFP4 contamination failed: %s", e
                )
            apply_comfy_quant_int8_patches()
            reset_int8_lora_log_counters()
            sdxl_logger.info(
                "[SDXL INT8] Loading checkpoint via MixedPrecisionOps "
                "(int8_tensorwise / comfy_quant): %s",
                ckpt_name,
            )
            with _int8_quant_conv_scope():
                out = comfy.sd.load_checkpoint_guess_config(
                    ckpt_path,
                    output_vae=False,
                    output_clip=True,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    model_options=model_options,
                )
            model, clip, _v = out[:3]
            summarize_int8_lora_capability(model)
            return (model, clip)

        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=False,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options=model_options,
        )
        model, clip, _v = out[:3]
        return (model, clip)
    finally:
        set_current_device(original_device)


def install_int8_option_dispatch(node_class_mappings) -> bool:
    if not isinstance(node_class_mappings, dict):
        return False

    # Do NOT apply INT8 patches at node registration / import.
    # Patches install only inside load_unet_hswq_weight_dtype /
    # load_checkpoint_sdxl_hswq_weight_dtype when INT8 is actually loaded.

    _FP8_WEIGHT_DTYPES = frozenset({"fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"})

    unet_cls = node_class_mappings.get("HSWQFP8E4M3UNetLoader")
    if unet_cls is not None:
        _orig_load_unet = unet_cls.load_unet

        def load_unet(self, unet_name, weight_dtype):
            # Explicit FP8 choices stay on the original FP loader body — never INT8 helper.
            if weight_dtype in _FP8_WEIGHT_DTYPES:
                return _orig_load_unet(self, unet_name, weight_dtype)
            if weight_dtype == "int8_tensorwise":
                return load_unet_hswq_weight_dtype(unet_name, weight_dtype)
            # default: auto-detect INT8 checkpoints only; otherwise original FP path.
            import folder_paths

            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
            if checkpoint_looks_like_comfy_quant_int8(unet_path):
                return load_unet_hswq_weight_dtype(unet_name, weight_dtype)
            return _orig_load_unet(self, unet_name, weight_dtype)

        unet_cls.load_unet = load_unet

    sdxl_cls = node_class_mappings.get("HSWQCheckpointLoaderSDXL")
    if sdxl_cls is not None:
        _orig_load_checkpoint = sdxl_cls.load_checkpoint

        def load_checkpoint(self, ckpt_name, weight_dtype, device=None):
            if weight_dtype in _FP8_WEIGHT_DTYPES:
                return _orig_load_checkpoint(self, ckpt_name, weight_dtype, device=device)
            if weight_dtype == "int8_tensorwise":
                return load_checkpoint_sdxl_hswq_weight_dtype(
                    ckpt_name, weight_dtype, device=device
                )
            import folder_paths

            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            if checkpoint_looks_like_comfy_quant_int8(ckpt_path):
                return load_checkpoint_sdxl_hswq_weight_dtype(
                    ckpt_name, weight_dtype, device=device
                )
            return _orig_load_checkpoint(self, ckpt_name, weight_dtype, device=device)

        sdxl_cls.load_checkpoint = load_checkpoint

    return True
```

---

## 结尾

本指南对策模块的 **③** 义务由 **附录 A** 满足（完整文件正文；所列 18 个模块已与磁盘字符级校验一致）。

上文 **④** 对每个新增/修改文件以及每个关键符号写明了 **文件含义** 与 **代码含义** — 不只是戳记口号。

污染类操作者复测见 **P7**。英文孪生：`md/HSWQ_FROM_a9d372_PROBLEM_COUNTERMEASURES_GUIDE.md`。
