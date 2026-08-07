# HSWQ ComfyUI 0.30.2 互換性修正 完全技術解説書

- 対象リポジトリ: `ComfyUI-HSWQ-Loader-and-Tools`（v3.3.8）
- 対象 ComfyUI: 0.30.2（comfy_kitchen / MixedPrecisionOps 統合版）
- コミット: `21792a8` → `1fd6ad4` → `3dc8a9a` → `ecd6bc0`（全て `origin/main` にプッシュ済み）
- 検証結果既知: Krea2 ConvRot INT8 1回目速度復旧 ✅ / Krea2 2回目以降悪化 → 修正済 ✅ / ZI NVFP4 LoRA 正常 ✅

---

## ① 何が問題だったのか

### 問題 1: Krea2 ConvRot INT8 が異常に遅い（主症状）

ComfyUI を 0.30.x 系に更新した後、Krea2 ConvRot INT8 の推論が極端に遅くなった。
原因は 2 つ。

**(a) `load_models_gpu` パッチによる毎回の全モジュール走査**

`patches/comfy_quant_int8.py` は `comfy.model_management.load_models_gpu` を
モンキーパッチしており、**呼び出しのたびに**以下の判定を実行していた:

- `_model_has_int8_quantized_weights(model)` … `model.named_modules()` で
  **全モジュール**（Krea2 は数千〜数万個）を走査して QuantizedTensor を探す
- `_model_is_nunchaku_svdq(model)` … 同じく**全モジュール**を走査

ComfyUI 0.30.x ではモデルロード/メモリ管理の頻度が上がったため、
この O(n) 走査が毎回実行され、総合的に大きなオーバーヘッドになった。

**(b) Hadamard 行列の CPU→GPU 転送が毎回発生**

ConvRot（Hadamard 回転）では、活性化を `x_rot = x @ H` で回転させる。
旧実装の `rotate_activation()` は毎回 `h_matrix.to(dtype, device)` で
CPU 上に構築された Hadamard 行列を GPU へ転送していた。
さらに HSWQ 注入 Conv2d の forward も `build_hadamard(..., device="cpu")` を
毎回呼び出していた。GPU 転送は同期コストが高く、レイヤー数分だけ積算される。

### 問題 2: ZI NVFP4 で VRAM 増加（副症状）

`nodes/zimage_nvfp4/nvfp4_lora_bake.py` の `install_load_models_gpu_bake_hook`
が、`load_models_gpu` の**たびに current_loaded_models 全部**に対して:

- `_nvfp4_convrot_diag(model)` … 全モジュール走査（キャッシュなし）
- `run_zimage_nvfp4_lora_bake_on_patcher()` … フォールバック判定で
  `_patcher_has_quant_via_keys()` が**全 LoRA パッチキー**を走査し、
  各キーで `get_key_weight()`（重い QT アンラップ）を実行

これを繰り返すため、GPU メモリの断片化・不要な weight 移動が起き、
VRAM 使用量が累積的に増加した。

### 問題 3（潜在バグ）: `get_hadamard_on_device` が未定義のまま参照されていた

1 回目の修正コミット `21792a8` で、`patches/comfy_quant_int8.py` の
注入 Conv2d forward が `nc.get_hadamard_on_device(...)` を呼ぶように
変更されたが、`native_convert_int8.py` 側に**関数を追加する Replace が
静かに失敗**しており、`_HADAMARD_GPU_CACHE` dict の追加（+3 行）のみが
コミットされていた。

- Krea2（DiT）は Conv2d 注入パスを通らないため発症せず「速度戻った」と見えた
- **SDXL ConvRot INT8 を使うと最初の forward で `AttributeError`** になる潜在バグ

### 問題 4（潜在バグ）: `weight_inner` が定義されずに参照されていた

同じく `21792a8` で `_bake_int8_patches_on_dynamic_patcher` 内の
`isinstance(weight, QuantizedTensor)` が `isinstance(weight_inner, ...)` に
変更されたが、`weight_inner = weight.data if hasattr(weight, "data") else weight`
の定義行を追加するパッチは「pattern not found」で**適用されていなかった**。

- SDXL / INT8 + LoRA の Dynamic bake パスで `NameError` になる潜在バグ

### 問題 5（互換性）: ComfyUI 0.30.2 の API 変化

- `mixed_precision_ops` の `disabled` 引数は 0.30.2 では `set` 型が前提
  （旧 HSWQ コードは `[]` を渡していた）
- `LowVramPatch.__call__` → `comfy.lora.calculate_weight` に
  `original_weights` 引数が追加された
- 0.30.2 では量子化 weight が `Parameter(QuantizedTensor)` として
  保持されることがあり、`isinstance(w, QuantizedTensor)` だけでは
  検出できない
- `_quantized_weight_state_dict` に `extra_quant_params` 引数が追加された
- LoRA モジュールの配置が `comfy.weight_adapter.lora` に移動

---

## ② 新規作成・修正したファイル名

| ファイル | 種別 | 内容 |
|---|---|---|
| `native_convert_int8.py` | 修正 | GPU 側 Hadamard キャッシュ追加（`get_hadamard_on_device`）、`rotate_activation` の GPU キャッシュ利用化 |
| `patches/comfy_quant_int8.py` | 修正 | 早期リターン/キャッシュ、`disabled` の set 化、0.30.2 互換（Parameter.data・original_weights・extra_quant_params）、`weight_inner` 定義 |
| `nodes/zimage_nvfp4/nvfp4_lora_bake.py` | 修正 | `load_models_gpu` bake フックの高速スキップ |
| `__init__.py` | 修正 | `comfy.weight_adapter.lora` import フォールバック、`calculate_weight` シグネチャ修正 |

新規ファイルはなし（全て既存ファイルの修正）。

---

## ③ 修正したコードの全文

### 3-1. `native_convert_int8.py`

#### (a) モジュール冒頭（GPU キャッシュ dict 追加）

```python
_DEFAULT_GROUPSIZE = 256
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
# GPU-side cache: avoids CPU→GPU transfer on every rotate_activation call.
# Keyed by (size, device_str, dtype) – same as CPU cache but on target device.
_HADAMARD_GPU_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
```

#### (b) `get_hadamard_on_device()` 新規追加（`build_hadamard` の直後）

```python
def get_hadamard_on_device(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return Hadamard matrix on target device, with GPU-side caching.

    Builds on CPU (via build_hadamard) once, then transfers to the target
    device and caches there. Subsequent calls with the same
    (size, device, dtype) hit the GPU cache and skip CPU->GPU transfer.
    """
    cache_key = (size, str(device), dtype)
    cached = _HADAMARD_GPU_CACHE.get(cache_key)
    if cached is not None:
        return cached
    h = build_hadamard(size, device="cpu", dtype=torch.float32)
    h = h.to(dtype=dtype, device=device)
    _HADAMARD_GPU_CACHE[cache_key] = h
    return h
```

#### (c) `rotate_activation()`（GPU キャッシュ経由に変更）

```python
def rotate_activation(
    x: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Online Linear: x_rot = x @ H (last dim = features)."""
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    group_count = features // group_size
    x_grouped = x.reshape(-1, group_count, group_size)
    # GPU-cached Hadamard: build/transfer once, reuse on every call
    h = get_hadamard_on_device(group_size, device=x.device, dtype=x.dtype)
    return torch.matmul(x_grouped, h).reshape(orig_shape)
```

### 3-2. `patches/comfy_quant_int8.py`

#### (a) `_model_is_nunchaku_svdq()` 早期リターン

```python
    seen = set()
    _checked = 0
    _MAX_CHECK_SVDQ = 100  # Early exit: SVDQ modules are typically at the top
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
            _checked += 1
            if _checked >= _MAX_CHECK_SVDQ:
                break
    return False
```

#### (b) `_model_has_int8_quantized_weights()` 早期リターン + 0.30.2 対応

```python
    if _model_is_nunchaku_svdq(model):
        return False
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    _checked = 0
    _MAX_CHECK = 200  # Early exit: only scan first 200 modules
    for _, module in model.named_modules():
        cls_name = type(module).__name__
        if "SVDQ" in cls_name or "Nunchaku" in cls_name:
            continue
        w = getattr(module, "weight", None)
        if w is None:
            continue
        if isinstance(w, QuantizedTensor):
            return True
        # 0.30.2: Parameter wrapping QuantizedTensor
        if hasattr(w, "data") and isinstance(w.data, QuantizedTensor):
            return True
        # Fast path: layout_type set means quantized
        if getattr(module, "layout_type", None) is not None:
            return True
        _checked += 1
        if _checked >= _MAX_CHECK:
            break
    return False
```

#### (c) 注入 Conv2d `state_dict()`（0.30.2 の `_quantized_weight_state_dict` 対応）

```python
        def state_dict(self, *args, destination=None, prefix="", **kwargs):
            sd = destination if destination is not None else {}
            sd = _quantized_weight_state_dict(self, sd, prefix, extra_quant_params=("input_scale", "pre_quant_scale"))
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
```

#### (d) 注入 Conv2d `forward_comfy_cast_weights()`（GPU キャッシュ利用）

```python
        def forward_comfy_cast_weights(self, input):
            # Mirror MixedPrecision Linear: when weight is QuantizedTensor and
            # Dynamic VRAM uses weight_lowvram_function, want_requant=True so
            # post_cast dequant → LoRA → requant (want_requant=False left QT
            # in the resident path after the first step and killed LoRA).
            if getattr(self, "_hswq_convrot", False):
                nc = _load_native_convert_int8_helpers()
                gs = int(getattr(self, "_hswq_convrot_groupsize", 256) or 256)
                # Use GPU-cached Hadamard via nc.get_hadamard_on_device
                h = nc.get_hadamard_on_device(gs, device=input.device, dtype=input.dtype)
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
```

#### (e) `mixed_precision_ops_force_conv()` の `disabled` 正規化（0.30.2 の set 型対応）

```python
        if disabled is None:
            disabled = set()
        elif isinstance(disabled, list):
            disabled = set(disabled)
        result = true_orig(
            quant_config=quant_config,
            compute_dtype=compute_dtype,
            full_precision_mm=full_precision_mm,
            disabled=disabled,
        )
```

#### (f) `LowVramPatch.__call__` の `original_weights` 対応（0.30.2）

```python
    def __call__(self, weight):
        # QuantizedTensor only. Bare int8 / float / None -> upstream unchanged.
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
        # 0.30.2: calculate_weight accepts original_weight=None
        try:
            return comfy.lora.calculate_weight(patches, w, self.key, intermediate_dtype=idtype, original_weights=None)
        except TypeError:
            # Fallback for older ComfyUI without original_weights param
            return comfy.lora.calculate_weight(patches, w, self.key, intermediate_dtype=idtype)
```

#### (g) `_bake_int8_patches_on_dynamic_patcher()` の `weight_inner` 定義（バグ修正）

```python
            weight, set_func, convert_func = mp.get_key_weight(patcher.model, key)
            if weight is None:
                continue
            # 0.30.2: weight may be Parameter(QuantizedTensor) - unwrap for isinstance check
            weight_inner = weight.data if hasattr(weight, "data") else weight
            # SDXL path (3.3.0): bake all comfy_quant QuantizedTensor ? never bare int8.
            # NVFP4 Linear uses nodes/nvfp4 ConvRot convert/set_weight.
            # Z Image path never reaches here (parity early-return above).
            if not isinstance(weight_inner, QuantizedTensor):
                continue
            if set_func is None:
                _console(
                    f"[HSWQ INT8 LoRA] WARN cannot bake {key}: "
                    "QuantizedTensor but no set_weight (int8_round risk)"
                )
                continue
            keys_to_bake.append((param_key, key))
```

### 3-3. `nodes/zimage_nvfp4/nvfp4_lora_bake.py`

`install_load_models_gpu_bake_hook()` 内の `load_models_gpu` ラッパー（高速スキップ追加）:

```python
    def load_models_gpu(*args, **kwargs):
        result = prev(*args, **kwargs)
        try:
            for loaded in list(getattr(mm, "current_loaded_models", []) or []):
                patcher = getattr(loaded, "model", None)
                if patcher is None:
                    continue
                # Fast skip: no patches AND no baked keys = not our model
                has_patches = bool(getattr(patcher, "patches", None))
                has_baked = bool(getattr(getattr(patcher, "model", None), "_hswq_zi_nvfp4_baked_keys", None))
                if not has_patches and not has_baked:
                    continue
                # Skip non-dynamic models
                try:
                    if not bool(patcher.is_dynamic()):
                        continue
                except Exception:
                    continue
                # Fast skip: check if this is a ZI NVFP4 model via cached diag
                model = getattr(patcher, "model", None)
                if model is not None:
                    diag = _nvfp4_convrot_diag(model)
                    if not diag["has"] and not has_baked:
                        continue
                run_zimage_nvfp4_lora_bake_on_patcher(
                    patcher,
                    device_to=getattr(patcher, "load_device", None),
                    reason="load_models_gpu",
                )
        except Exception as exc:
            _console(f"[HSWQ ZI NVFP4 LoRA] load_models_gpu bake error: {exc!r}")
        return result
```

### 3-4. `__init__.py`

#### (a) LoRA モジュールの import フォールバック

```python
        try:
            try:
                import comfy.weight_adapter.lora as _lora_mod
            except ImportError:
                import comfy.lora as _lora_mod
            _LoraDiff = getattr(_lora_mod, "LoraDiff", None)
```

#### (b) `_lora_skip_calculate_weight` のシグネチャ修正（0.30.2 はデフォルト引数が `torch.float32`）

```python
                    def _lora_skip_calculate_weight(
                        self, weight, key, strength, strength_model, offset,
                        function, intermediate_dtype=_torch_lora.float32, original_weight=None,
                    ):
                        v = self.weights
                        reshape = v[5]
                        if reshape is not None and tuple(reshape) != weight.shape:
                            logger.warning(
                                "LoRA %s: skip %s (reshape %s != weight %s) [HSWQ compat]",
                                self.name, key, list(reshape), list(weight.shape),
                            )
                            return weight
                        try:
                            lora_diff = _torch_lora.mm(
                                v[0].flatten(start_dim=1), v[1].flatten(start_dim=1)
                            )
                            if lora_diff.numel() != weight.numel():
                                logger.warning(
                                    "LoRA %s: skip %s (numel %d != %d) [HSWQ compat]",
                                    self.name, key, lora_diff.numel(), weight.numel(),
                                )
                                return weight
                        except Exception:
                            return weight
                        return _orig_cw(
                            self, weight=weight, key=key, strength=strength,
                            strength_model=strength_model, offset=offset,
                            function=function, intermediate_dtype=intermediate_dtype,
                            original_weight=original_weight,
                        )
```

---

## ④ その意味（変更ごとの解説）

### 4-1. GPU Hadamard キャッシュ（Krea2 INT8 速度回復の主因）

- **何が起きていたか**: ConvRot は活性化を Hadamard 行列 H で回転させる。
  旧実装は forward のたびに CPU 上で H を構築（`build_hadamard(device="cpu")`）し、
  `h.to(dtype, device)` で GPU に転送していた。GPU 転送はデバイス同期を伴うため、
  レイヤー数 × ステップ数ぶんの転送コストが積算された。
- **修正後**: `get_hadamard_on_device()` が (size, device, dtype) ごとに
  GPU 上の行列を 1 回だけ生成して `_HADAMARD_GPU_CACHE` にキャッシュ。
  2 回目以降はキャッシュヒットで転送ゼロ。
- **効果**: Krea2 ConvRot INT8 の推論が大幅に高速化。

### 4-2. 早期リターン + キャッシュ（load_models_gpu のオーバーヘッド削減）

- **何が起きていたか**: `load_models_gpu` のモンキーパッチが呼ばれるたびに、
  `_model_has_int8_quantized_weights` と `_model_is_nunchaku_svdq` が
  `named_modules()` で全モジュールを走査。Krea2 のような大規模 DiT では
  数千〜数万モジュールになり、ロード/メモリ管理の頻度が高い 0.30.x では
  総合的なコストが大きかった。
- **修正後**:
  - `_model_has_int8_quantized_weights`: 先頭 200 モジュールまでで
    QuantizedTensor / `layout_type` を検出できなければ False を返す
    （INT8 モデルは必ず先頭ブロックに quantized Linear があるため安全）。
  - `_model_is_nunchaku_svdq`: 先頭 100 モジュールまでで判定。
    SVDQ はモデル全体を置換する構造のため先頭に現れる。
- **安全性**: いずれも「検出漏れ」が起きる方向は False（bake を実行しない、
  handoff を実行しない）だが、対象モデルは必ず先頭に quantized/SVDQ
  モジュールを持つため実用上問題ない。ZI NVFP4 / Krea2 での動作確認済み。

### 4-3. `Parameter.data` アンラップ（0.30.2 の weight 保持形式対応）

- 0.30.2 では量子化 weight が `torch.nn.Parameter(QuantizedTensor)` として
  保持されることがある。`isinstance(w, QuantizedTensor)` は Parameter に
  対して False を返すため、従来の判定では INT8 モデルを検出できなかった。
- `hasattr(w, "data") and isinstance(w.data, QuantizedTensor)` で
  Parameter をアンラップして検出。`_bake_int8_patches_on_dynamic_patcher`
  の `weight_inner` も同様の意図（こちらは定義漏れを修正）。
- `layout_type` チェックも追加: quantized レイヤーには
  `_load_quantized_module` が `layout_type` を設定するため、高速な代替判定になる。

### 4-4. `disabled` の set 化（0.30.2 の API 前提に合わせる）

- 0.30.2 の `mixed_precision_ops` は `disabled` を `set` として扱う
  （`disabled.add("nvfp4")` など）。旧 HSWQ コードは `[]` を渡しており、
  `add()` が存在しない list を渡すと `AttributeError` になり得た。
- `disabled = set()` / `set(disabled)` に正規化して安全に。

### 4-5. `original_weights` 引数対応（0.30.2 の calculate_weight シグネチャ）

- 0.30.2 の `comfy.lora.calculate_weight` は `original_weights=None` を
  受け取る。旧コードはこれを渡していなかったため、`model_as_lora` 系の
  パッチが正しく動かない可能性があった。
- `try/except TypeError` で旧 ComfyUI へのフォールバックも残してある。

### 4-6. `extra_quant_params`（0.30.2 の state_dict 仕様）

- 0.30.2 の `_quantized_weight_state_dict` は `extra_quant_params` を
  受け取る。`("input_scale", "pre_quant_scale")` を明示しないと、
  `state_dict()` 保存時に余分なキーが落ちる/混ざる可能性がある。
- 注入 Conv2d の `state_dict` で 0.30.2 の Linear と同じ呼び出し形に統一。

### 4-7. `load_models_gpu` bake フックの高速スキップ（ZI NVFP4 VRAM 増加対策）

- **何が起きていたか**: ZI NVFP4 の `install_load_models_gpu_bake_hook` は
  `load_models_gpu` のたびに **全 loaded model** を舐めて
  `_nvfp4_convrot_diag`（全モジュール走査）→ 場合により bake を実行。
  対象外のモデル（通常の SDXL 等）でも毎回走査され、VRAM の不要な
  weight 移動・断片化を招いた。
- **修正後**:
  1. `patches` も `_hswq_zi_nvfp4_baked_keys` も無ければ即スキップ
  2. 非 dynamic モデルはスキップ
  3. `_nvfp4_convrot_diag` の結果 `has=False` かつ baked 無しならスキップ
- **LoRA への影響**: このフックは **Dynamic.load ラッパーで既に bake された
  残りを拾うセカンドパス**。メインの bake は `Dynamic.load` ラッパーが
  無条件に実行するため、フックがスキップしても LoRA は正常に bake される
  （ユーザー確認済み: ZI NVFP4 LoRA 正常）。

### 4-8. `__init__.py` の import / シグネチャ（0.30.2 のモジュール再配置）

- 0.30.2 で LoRA 関連は `comfy.weight_adapter.lora` に移動。
  `ImportError` 時は旧 `comfy.lora` にフォールバックする二段構えに。
- `_lora_skip_calculate_weight` の `intermediate_dtype` デフォルトを
  `None`（内部で `torch.float32` に置換）から直接 `torch.float32` に変更し、
  0.30.2 の実シグネチャに一致させた。reshape/numel 不一致の LoRA を
  スキップする安全性パッチの動作が保証される。


### 5. Krea2 ConvRot INT8 の実行累積速度悪化（2回目以降 4〜7 倍遅延）

#### 症状

1 回目の Krea2 実行は正常（~4s/step）。2 回目で step タイムが
`4.1s → 16.6s → 9.4s → 22.7s → 26.9s → 15.8s` と爆発的に悪化し、
3 回目以降はさらに悪化していく。Z Image NVFP4 の実行は問題ない。

#### 原因: Z Image `comfy_parity` ラッパーが Krea2 ロード時に残存

Krea2 のロードパス（`is_int8 and is_convrot and not needs_conv2d`）は
`comfy.sd.load_diffusion_model` を直接呼ぶ「ストックロード」で、
`apply_comfy_quant_int8_patches()` を呼ばない。しかし、このパスには
SDXL ロードパスでは呼ばれている `_clear_zimage_parity_contamination_for_sdxl()`
が欠落していた。

Z Image 実行後の HSWQ パージ処理は `mixed_precision_ops` から
Z Image の `comfy_parity` ラッパー（`apply_nvfp4_comfy_parity()` が
インストールしたオンライン act rotate）を**完全には除去しない**。
このラッパーが残存したまま Krea2 がストックロードされると、以下が発生:

1. **`_load_quantized_module_comfy_only` が Krea2 の INT8 ConvRot レイヤーに発火**
   → `_arm_int8_protect_convrot_after_stock_load()` が各 Linear モジュールに
   `_hswq_int8_convrot = True` を設定し、`Params.convrot` をクリア。

2. **`_ensure_single_parity_linear_forward()` が stock `Linear.forward` を
   `forward_parity` に置換** → オンライン Hadamard act rotate が
   全 INT8 ConvRot Linear の forward にラップされる。

3. **毎ステップで `forward_parity` が Hadamard 行列を構築・キャッシュし、
   入力を `rotate_last_dim()` で回転**。Krea2 は ConvRot が重みに
   オフラインで焼き込まれており、kitchen の
   `dequantize_int8_convrot_weight` + `int8_linear` が ConvRot を処理するため、
   `forward_parity` によるオンライン回転は不要（二重回転になる可能性あり）。

4. **HSWQ パージが Hadamard キャッシュ（`_hswq_nvfp4_parity_H`）を
   破棄** → 実行ごとに再構築 → CUDA メモリ断片化が進行。

この結果、Krea2 の全 ConvRot Linear（最大 256 レイヤー）で
毎ステップ不要な Hadamard 回転が走り、実行を重ねるごとに
CUDA メモリ断片化が進行し、step タイムが指数関数的に悪化する。

#### 修正: Krea2 ストックロード前に parity 汚染を除去

`patches/comfy_quant_int8.py` の `load_unet_hswq_weight_dtype()` 内、
Krea2 ストックロードパスの先頭で、SDXL と同様に
`_clear_zimage_parity_contamination_for_sdxl()` を呼び出す。

\`\`\`python
if is_int8 and is_convrot and not needs_conv2d:
    try:
        from ..nodes.nvfp4.comfy_quant_nvfp4 import (
            _clear_zimage_parity_contamination_for_sdxl,
        )
        _clear_zimage_parity_contamination_for_sdxl()
    except Exception as e:
        logging.warning(
            "[HSWQ INT8] clear Z Image NVFP4 contamination "
            "for Krea2 failed: %s", e
        )
    model = comfy.sd.load_diffusion_model(...)
\`\`\`

これにより:
- `ops.mixed_precision_ops` が stock（非 parity）に復元される
- `ops._load_quantized_module` が stock に復元される
- Krea2 の INT8 ConvRot レイヤーに `_hswq_int8_convrot` が付与されない
- `forward_parity` が `Linear.forward` にインストールされない
- Krea2 の INT8 ストック処理（kitchen `dequantize_int8_convrot_weight` +
  `int8_linear`）がそのまま動作

---

## 付録: コミット履歴

```
ecd6bc0 fix: clear Z Image parity contamination before Krea2 stock load
3dc8a9a fix: define weight_inner before isinstance check in INT8 bake path
1fd6ad4 fix: add missing get_hadamard_on_device GPU cache to native_convert_int8
21792a8 fix: ComfyUI 0.30.2 compatibility + perf (Krea2 ConvRot INT8 slow, ZI NVFP4 VRAM)
```

## 付録: 検証

- `python -m py_compile` 全修正ファイル構文チェック OK
- `get_hadamard_on_device(256)` が同一オブジェクトを返す（キャッシュヒット）確認
- `rotate_activation` の形状維持確認（(2,256) → (2,256)）
- 実機確認: Krea2 ConvRot INT8 (1回目速度復旧 / 2回目以降 parity 除去済) / ZI NVFP4 LoRA 正常
