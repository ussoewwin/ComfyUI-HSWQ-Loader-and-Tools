# 更新日志

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../changelog.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

## Version 3.4.0

- **新增**：**SDXL ConvRot NVFP4 Blackwell Tensor Boost** — 仅在 `nodes/nvfp4/` 内对 SM >= 100（B200 / GB200、RTX 5090 / SM120）启用 Per-Weight CUDA Graph 自动分发（不影响 Z Image / INT8 / FP8 / 标准路径）。回放时消除 shape-shared 的权重 `.copy_()`；自适应 `M` 上限 16384；捕获 / 命中控制台日志与 `nvfp4_forward_stats()`（`blackwell_graph_hits`、`blackwell_tensor_boost_active`）。**HSWQ Sampler** 与 **HSWQ Ultimate SD Upscale** 上独立的 **`tensor_boost` BOOLEAN**（默认 OFF；Loader 无开关），经 `HSWQ_NVFP4_TENSORBOOST` / `HSWQ_NVFP4_CUDAGRAPH` 控制，OFF 时调用 `clear_nvfp4_cudagraphs()`，避免 USDU 分块时显存暴涨。**开启会使显存增加数 GB**（CUDA Graph arena）——放大 / Tensor Boost 余量推荐 **RTX 5090 32 GB+**；采样器路径 **16 GB+**。文档：`md/HSWQ_SDXL_NVFP4_BLACKWELL_ACCELERATION_GUIDE.md`。
- 详情见 [发布说明 v3.4.0](v3.4.0.md)。

## Version 3.3.9

- **ComfyUI 0.30.2 兼容性 & Krea2 parity 污染修复**（commits `21792a8`..`ecd6bc0`）：
  - **性能**: Krea2 ConvRot INT8 GPU 缓存 Hadamard 矩阵（`native_convert_int8.get_hadamard_on_device`）、全模型 INT8/SVDQ 扫描 200 模块提前退出、`mixed_precision_ops` 重入守卫、`disabled` set 归一化。
  - **性能 / 显存**: ZI NVFP4 `load_models_gpu` bake 钩子快速跳过（无 patches + 无 baked keys -> 跳过；非 dynamic 模型 -> 跳过），降低每次 GPU load 时全量诊断导致的显存压力。
  - **性能**: Krea2 ConvRot INT8 多次运行逐步恶化（1 次 ~4s/step，2 次 4s->16s->22s->26s/step）**已修复**。根因：Z Image `comfy_parity` 包装器残留在 `mixed_precision_ops` / `_load_quantized_module` 上，导致 Krea2 INT8 ConvRot 层被标记 `_hswq_int8_convrot` 并在每个 Linear 上安装 `forward_parity`（在线 Hadamard act rotate）-> 每步不必要的旋转 -> CUDA 碎片逐次累积。修复：在 Krea2 纯 stock 加载前调用 `_clear_zimage_parity_contamination_for_sdxl()`（与 SDXL 路径一致）。
  - **兼容**: `Parameter.data` 解包适配 ComfyUI 0.30.2 延迟权重表示、`comfy.weight_adapter.lora` 导入回退、`calculate_weight` `intermediate_dtype` 默认 = `torch.float32`、`LowVramPatch.__call__` `original_weights` 参数、`state_dict` `extra_quant_params`。
  - **文档**: 技术解说 `md/HSWQ_COMFYUI_0_30_2_COMPATIBILITY_FIX_GUIDE.md` 覆盖所有根因、修复与验证。
- 详情见 [发布说明 v3.3.9](v3.3.9.md)。

## Version 3.3.8

- **新增**：**HSWQ Sampler** `clip_perfect_offload (Krea2 only)` 开关 —— 在采样前释放 Krea2 文本编码器（从 `current_loaded_models` 丢弃其 patcher），在紧张显存的显卡上达到与基准一致的显存占用。双向限定 Krea2：通过 loader 标记 `_hswq_is_krea2` 与精确的 `comfy.text_encoders.krea2` 模块身份识别（不靠类名猜测）；默认关闭、严格布尔读取、绝不调用任何全局分配器操作，任何失败都会被捕获，运行绝不中断。UI 控件现显示 `(Krea2 only)` 范围标记。文档：中英 README 节点说明与新增 `md/HSWQ_KREA2_TE_OFFLOAD_GUIDE.md`。
- 详情见 [Release Notes v3.3.8](v3.3.8.md)。

## Version 3.3.7

- **修复 / 更改（许可与来源说明）**：清除残留的 Apache-2.0 表述，使本加载器仓库统一为 **GPL-3.0**；明确上游 **HSWQ**（[Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)）仍为 **AGPL-3.0**，与本包许可分离。重写 README / zhmd 中 **USDU**、**Torch Compile（KJNodes）**、**Batched Detailer（Impact Pack）** 的来源说明（去掉 “copy” 类措辞）。Batched Detailer 现于 `nodes/batched_detailer_lib/` 内嵌辅助代码，运行时**不需要**安装 Impact Pack，同时保留 GPL 归属声明。
- 详情见 [Release Notes v3.3.7](v3.3.7.md)。

## Version 3.3.6

- **新增 / 修复**：**HSWQ Torch Compile** 节点（`HSWQTorchCompileModel`）—— 使用 ComfyUI `set_torch_compile_wrapper`，不依赖 KJNodes；强制 `compile_threads=1` 与 `worker_start_method=subprocess`，避免 SeedVR2 / `utils.install_util` 的 spawn 崩溃；默认 inductor + `max-autotune-no-cudagraphs`。**ZI INT8 peel**：`peel_non_product_nvfp4_ops` 在 PRODUCT NVFP4 load 下层为外来 INT8 / ZI protect 时继续剥离，使 Z Image 之后 SDXL INT8 仍可存活。文档：中英 README 节点说明、技术指南、去掉 BETA 标记。
- 详情见 [Release Notes v3.3.6](v3.3.6.md)。

## Version 3.3.5

- **修复 / 更改**：v3.3.4 之后的 Z Image ConvRot NVFP4 大规模加固 —— 将 Z Image 剥离到专用 `nodes/zimage_nvfp4`（不再与 SDXL `nodes/nvfp4` Tensor Core 产品路径共有实现）；下拉项分离为 **`Z Image ConvRot NVFP4`** 与 SDXL **`ConvRot NVFP4`**，并据此分支 Dynamic VRAM LoRA bake；回到 SDXL INT8 / SDXL ConvRot NVFP4 时清除 Z Image 留下的 **comfy_parity** load overlay、就地 Linear bake（**VER=8**）以及 INT8-protect 武装残留，避免 SDXL → Z Image → SDXL 后的椒盐噪声、LoRA 失效与全噪声粘连。
- 详情见 [Release Notes v3.3.5](v3.3.5.md)。

## Version 3.3.4

- **修复**：Z Image / ZIT **ConvRot NVFP4** / INT8 protect —— **Distorch** purge 后，模块本地 `_hswq_nvfp4_parity_H` 的复用判定弱于全局 `_tensor_storage_ok` → **第 2 次及之后**画质劣化。parity 现共用 `_tensor_storage_ok`。
- 详情见 [Release Notes v3.3.4](v3.3.4.md)。

## Version 3.3.3

- **修复**：Z Image 混合包（**ConvRot NVFP4** + **ConvRot INT8 protect**）—— Dynamic VRAM 下 LoRA bake 现覆盖 **两系** Linear。INT8 protect 按 Conv2d 同型武装（清除 kitchen `Params.convrot`，requant 后保持 False）；二段 bake + pass-delta EVIDENCE（`NVFP4_LORA_BAKE_*` / `INT8_PROTECT_LORA_BAKE_*`），protect 层上残留的 LowVramPatch 不再导致 LoRA 无效或噪声。
- 详情见 [Release Notes v3.3.3](v3.3.3.md)。

## Version 3.3.2

- **修复**：Z Image / ZIT **ConvRot NVFP4** 在 **DistOrch VRAM purge 后的第 2 次生成**出现椒盐噪声。INT8 decode wrap 会丢掉 NVFP4 stack 标记，后续“upgrade”又把 Tensor Core 产品路径叠到 Comfy parity 之上；DistOrch refresh 只剥掉 TC 层，重载后留下 **双重在线 act rotate**。现于 INT8 wrap 中保留标记，parity refresh 不再二次武装 rotate。
- 详情见 [Release Notes v3.3.2](v3.3.2.md)。

## Version 3.3.1

- **新增**：Z Image / ZIT **ConvRot NVFP4** 支持，经 **HSWQ ConvRot INT8/ConvRot NVFP4 UNet Loader**（`weight_dtype`：`ConvRot NVFP4`，或带 NVFP4 自动检测的 `default`）。采用与 bench 对齐的 Comfy parity 路径（stock MixedPrecision GEMM + 在线 act rotate），实现位于 `nodes/zimage_nvfp4`；覆盖 NVFP4 + INT8 protect 混合包与 Dynamic VRAM LoRA bake，**不是** SDXL Checkpoint Loader 的 Tensor Core 产品路径。**仅支持经 [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization) 量化的模型。**
- 详情见 [Release Notes v3.3.1](v3.3.1.md)。

## Version 3.3.0

- **更改**：其余 ComfyUI 节点 class ID 由 Nunchaku 前缀统一为 HSWQ 前缀（`HSWQSaveImage`、`HSWQCheckpointLoaderSDXL`、`HSWQSDXLLoraStackV3`、`HSWQZImageDiTLoader`，以及相关 JS hooks）。
- 详情见 [Release Notes v3.3.0](v3.3.0.md)。

## Version 3.2.9

- **更改**：更新 `pyproject.toml` 的 `[project].name`，使其与新仓库身份一致，ComfyUI 注册表分类显示为 **comfyui-hswq-loader-and-tools**。
- **更改**：以更正后的项目名称向 ComfyUI 重新注册本节点包。
- 详情见 [Release Notes v3.2.9](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.9)。

## Version 3.2.8

- **更改**：仓库重命名为 **ComfyUI-HSWQ-Loader-and-Tools**。
- **更改**：节点由 **HSWQ&Nunchaku Ultimate SD Upscale** 重命名为 **HSWQ Ultimate SD Upscale**（包括类名、ID 与标题）。
- 详情见 [Release Notes v3.2.8](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.8)。

## Version 3.2.7

- **移除**：节点内 INT8 W8A8 Triton Linear 加速（Plan B）—— 融合内核、`install.py` 的 Triton 阶段以及 **Triton accelerate** UI 开关。INT8 Linear 速度改由 ComfyUI + `comfy_kitchen`（`int8_linear`：cuda → triton → eager）负责。本扩展仅保留 INT8 加载兼容补丁（Conv2d / LoRA / ControlLora / handoff）。
- 详情见 [Release Notes v3.2.7](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.7)。

## Version 3.2.6

- **新增**：面向 HSWQ INT8 加载器的公开 INT8 W8A8 Triton Linear 加速（Plan B）—— 融合的逐行激活量化 → INT8 GEMM → 反量化，无需依赖 Comfy `--enable-triton-backend`；`install.py` 中内置 Windows/Linux Triton 安装；UI 开关 **Triton accelerate**；分块逐行量化，使宽层（如 K=10240）仍可走融合路径。
- 详情见 [Release Notes v3.2.6](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.6)。

## Version 3.2.5

- **修复**：在过时的便携 / 内嵌 Python 环境下 `requirements.txt` 安装失败 —— 一个无 wheel 的传递性源码依赖（`facexlib` 拉取的 `filterpy`）强制进行源码构建，由于环境自带旧版 `setuptools`，在 Python 3.12+ 上因 `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'` 而崩溃。新增的 `install.py` 会在安装 `requirements.txt` 前升级 `pip` / `setuptools` / `wheel`，使 ComfyUI-Manager 的安装/更新先修复构建工具，旧源码构建得以成功。
- 详情见 [Release Notes v3.2.5](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.5)。

## Version 3.2.4

- **修复**：SDXL LoRA 型 ControlNet（如 `anytest`）在 INT8 量化下输出全黑 —— `ControlLora.pre_run` 通过 `diffusion_model.state_dict()` 借用 INT8 基础 UNet 权重，而该接口返回的是被扁平化的原始 `int8`/`uint8` 张量而非 `QuantizedTensor`，导致借用的权重未被反量化。补丁拦截该 `state_dict()` 并即时对 INT8 基础权重进行反量化（全权重 ControlNet 如 `canny` 不受影响；FP8 下不会出现该问题）。
- 详情见 [Release Notes v3.2.4](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.4)。

## Version 3.2.3

- **新增**：**HSWQ Sampler** —— 与标准 ComfyUI KSampler 行为完全一致的等效节点，但在安装了 [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) 时会自动加入其全部 samplers 与 schedulers。它复刻了 Forge 的动态 sampler 生成逻辑，使完整的 Runge-Kutta（`rk_beta`）sampler 家族在原生 ComfyUI 中保持可选且可运行。
- 详情见 [Release Notes v3.2.3](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.3)。

## Version 3.2.2

- **修复**：非 SVDQ 加载（包括 SDXL INT8 普通生成）时 INT8→Nunchaku VRAM handoff 误判 —— SVDQ 检测不再使用单纯的 `"nunchaku" in __module__`（本扩展的 INT8 Conv2d 路径包含该子串）；handoff `_VER = 10` 仅在 BaseModel 上存在真正的 Nunchaku SVDQ 时启用，原生 comfy_quant INT8（任意架构）从不启用 handoff。
- 详情见 [Release Notes v3.2.2](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.2)。

## Version 3.2.1

- **修复**：INT8 HSWQ（Dynamic VRAM）→ Nunchaku SVDQ 共存 Abort —— LowVramPatch 与 Dynamic LoRA bake 仅限于 `comfy.quant_ops.QuantizedTensor`（绝不针对裸 `torch.int8`）；在 SVDQ 加载前使用单向 VRAM handoff `detach(unpatch_all=True)`。
- **移除**：再次重新引入 **HSWQ Pin Buffer Cache**（Abort 修复并不需要；AIMDO HostBuffer 之后 Detailer 作用域的 pin 池化依然过时）。
- **文档**：重写 `md/HSWQ_INT8_NUNCHAKU_COEXISTENCE_GUIDE.md`，记录经核实的 Abort 原因与 PinCache 相关性。
- 详情见 [Release Notes v3.2.1](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.1)。

## Version 3.2.0

- **移除**：**HSWQ Pin Buffer Cache**（`nodes/hswq_pin_cache.py` 及 Detailer `hswq_pin_cache_scope`）—— 在 ComfyUI Dynamic VRAM / AIMDO `HostBuffer` 更新后已冗余（不存在 `unpin` 路径的抖动）。保留 Batched Detailer 三阶段流程；使用原生 ComfyUI pin 行为。
- **更改**：SDXL checkpoint 加载器节点的显示标题强制改为 **HSWQ Checkpoint Loader (SDXL)**。
- 详情见 [Release Notes v3.2.0](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.2.0)。

## Version 3.1.9

- **新增**：面向 SDXL 检查点的原生 **comfy_quant INT8**（`int8_tensorwise`）加载路径 —— **HSWQ FP8/INT8 Loader (VRAM Opt)** 自动检测 INT8 与 Scaled FP8；**HSWQ FP8 E4M3 UNet Loader** 增加 `int8_tensorwise` / 自动检测。扩展侧提供 Conv2d 量化支持以及 Dynamic VRAM 下的 INT8 安全 LoRA bake。
- 详情见 [Release Notes v3.1.9](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.1.9)。

## Version 3.1.8

- **新增**：**HSWQ Save Image**（`NunchakuSaveImage`）—— 将 `IMAGE` 输出保存为 PNG 或 JPG（选择 JPG 时可设置 JPEG 质量）。
- **新增**：**Nunchaku Ultimate SD Upscale** —— `upscale_by` 下拉框带有 **Auto** 模式与 `target_height`（默认 4320），可由输入高度推导放大倍率；固定倍率 0.05–4.00 仍然可用。
- 详情见 [Release Notes v3.1.8](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.1.8)。

## Version 3.1.7

- **修复**：关键性修复 —— 在与 Lumina/HunYuan-DiT 架构配合使用时，`NunchakuUltimateSDUpscale` 出现严重输出噪声与 `RuntimeError`。已修正 conditioning 张量切片逻辑，可从拼接张量中精确提取 T5/LLM 特征。
- 详情见 [Release Notes v3.1.7](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.1.7)。

## Version 3.1.3

- **修复**：针对 `NunchakuUltimateSDUpscale` 中 `RuntimeError` 的临时绕过方案 —— 近期 ComfyUI 核心变更会沿特征维度（例如由 2560 变为 7680）拼接多编码器 conditioning，影响基于 Lumina/HunYuan 的模型。已在采样前加入自动检测与截断。
- 详情见 [Release Notes v3.1.3](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.1.3)。

## Version 3.1.2

- **修复**：Pin Buffer Cache（对 `comfy.pinned_memory.pin_memory` / `unpin_memory` 的 monkey-patch）仅在运行 `HSWQ Batched Detailer (SEGS)` 时启用。在 Detailer SEGS 之外，扩展会回落到 ComfyUI 原生 pin/unpin 行为，避免对其他节点/工作流产生副作用。

## Version 3.1.1

- **修复**：Bug 修复与更正（加载器注册、zimage 模型处理、USDU crop 模型补丁）。
- 详情见 [Release Notes v3.1.1](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.1.1)。

## Version 3.1.0

- **新增** 两个节点：
  - **HSWQ FP8 E4M3 UNet Loader**（`HSWQFP8E4M3UNetLoader`）—— 面向 HSWQ FP8 E4M3 模型的标准 UNet 加载器；扩展还安装 Pin Buffer Cache，降低 Dynamic VRAM Loading 下的 `cudaHostRegister`/`cudaHostUnregister` 开销。
  - **HSWQ Batched Detailer (SEGS)** —— Detailer (SEGS) 风格节点，以三阶段运行 VAE 编码 → UNet 采样 → VAE 解码（先全部编码、再全部采样、最后全部解码），最大程度减少模型切换，提升 Dynamic VRAM Loading 下的性能。
- 详情见 [Release Notes v3.1.0](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/v3.1.0)。

## Version 3.0.2

- **README**：更新 FP8 (fp8e4m3) 与 torch.compile 小节 —— 用途（将本节点与 FP8 和 torch.compile 一起使用）以及补丁说明。
- 详情见 [Release Notes v3.0.2](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/3.0.2)。

## Version 3.0.0

- **破坏性**：与 SDXL SVDQ 弃用保持一致（见顶部 IMPORTANT NOTICE）。节点注册缩减为以下三个：
  - **Nunchaku-ussoewwin SDXL Integrated Loader**（Checkpoint Loader 风格：单个检查点）
  - **Nunchaku-ussoewwin SDXL DiT Loader (DualCLIP)**（UNet + CLIP 来自不同文件）
  - **Nunchaku Ultimate SD Upscale**
- 从注册中**移除**（不再出现在 ComfyUI 中）：
  - Nunchaku-ussoewwin Z-Image-Turbo DiT Loader
  - Nunchaku-ussoewwin SDXL LoRA Stack V3
  - Nunchaku Apply First Block Cache Patch Advanced
- 未来的 SDXL 工作流在适用时应使用 fp8e4m3 与标准 ComfyUI 加载器。

## Version 2.6.6

- **修复**：修复了导致 prompt 执行崩溃的 `AttributeError: 'Logger' object has no attribute 'mgpu_mm_log'` 错误。在 `model_management_mgpu.py`、`device_utils.py` 与 `wrappers.py` 中将所有 `logger.mgpu_mm_log()` 替换为 `logger.info()`。

## Version 2.6.3

- 新增 **Checkpoint Loader (SDXL)** 节点
  - 从标准 SDXL 检查点加载 MODEL 与 CLIP，可选设备选择，支持 FP8 精度
- Nunchaku SDXL SVDQ（4-bit）开发停止；更新仓库状态（见顶部 IMPORTANT NOTICE）
- 详情见 [Release Notes v2.6.3](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.6.3)

## Version 2.6.2

- 修复 NunchakuUltimateSDUpscale 在 Nunchaku 1.2.0 下的节点注册问题
  - 改进 INPUT_TYPES 的错误处理，防止节点注册失败
  - 节点独立运行：使用内置的 `usdu_bundle`，不依赖 ComfyUI_UltimateSDUpscale
  - 详情见 [Issue #2](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/issues/2)
- 详情见 [Release Notes v2.6.2](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.6.2)

## Version 2.6.1

- 优化 SDXL 模型的 LoRA 处理性能
- 详情见 [Release Notes v2.6.1](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.6.1)

## Version 2.6

- 修复 SDXL 模型的 ControlNet 支持（OpenPose、Depth、Canny 等）
- 详情见 [Release Notes v2.6](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.6)

## Version 2.5

- 新增 SDXL Integrated Loader 节点，用于统一检查点加载
  - 支持从单个检查点文件同时加载 UNet 和 CLIP
  - 内置 Flash Attention 2 支持（默认开启）
  - 从检查点键自动检测模型配置
- 重组节点文档顺序
- 更新 SDXL DiT Loader，加入面向高级用户的警告
- 详情见 [Release Notes v2.5](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.5)

## Version 2.4

- 为 SDXL DiT Loader 新增 Flash Attention 2 支持
  - 可选加速功能，默认开启
  - 自动对所有 attention 层应用 FA2（SDXL 模型中通常为 140 层）
  - 需要在环境中安装 Flash Attention 2
  - 如需要可通过 `enable_fa2` 参数关闭
- 更新 SDXL DiT Loader 节点截图
- 详情见 [Release Notes v2.4](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.4)

## Version 2.3

- 新增带有改进色彩归一化的 Nunchaku Ultimate SD Upscale 节点
- 改进 First Block Cache，加入残差注入以提升质量
- 修复 Nunchaku SDXL VAE 输出的 USDU 色彩归一化
- 修复模块引用分离，防止数据丢失
- 使用融合内核优化缓存相似度计算
- 为 SDXL DiT Loader 新增 Flash Attention 2 支持（可选，默认开启）
- 详情见 [Release Notes v2.3](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.3)

## Version 2.2

- 为 Nunchaku SDXL 模型新增 First Block Cache 功能
- 详情见 [Release Notes v2.2](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/releases/tag/2.2)

## Version 2.1

- 发布 LoRA Loader 技术文档
- 详情见 [Release Notes v2.1](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-z-image-turbo-loader/releases/tag/2.1)

## Version 2.0

- 新增 SDXL DIT Loader 支持
- 新增 SDXL LoRA 支持
- 新增 SDXL 模型的 ControlNet 支持
- 详情见 [Release Notes v2.0](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-z-image-turbo-loader/releases/tag/2.0)

## Version 1.1

- 为 Z-Image-Turbo 模型新增 Diffsynth ControlNet 支持
  - 注意：无法与标准 model patch loader 配合工作。需要作者开发的自定义节点。
- 详情见 [Release Notes v1.1](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-z-image-turbo-loader/releases/tag/1.1)

## 2025-12-25

- 通过改进带更好路径解析的替代导入方式，修复 `NunchakuZImageDiTLoader` 节点的导入错误（见 [Issue #1](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools/issues/1)）
