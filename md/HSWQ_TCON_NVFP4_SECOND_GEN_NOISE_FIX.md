# HSWQ tcon NVFP4 2回目ノイズ問題 完全解説書

**日付:** 2026-08-27
**基準コミット:** `1156f00`（ComfyUI-HSWQ-Loader-and-Tools / docs: add v3.4.4 Chinese release notes）
**対象リポジトリ:**
- ComfyUI-HSWQ-Loader-and-Tools（`1156f00` → `fdc60bc`）
- ComfyUI-DistorchMemoryManager（`62c7cbd` → `9854848`）

---

## ① 何が問題だったのか

### 症状

tcon（Z Image TC/W4A4）NVFP4 モデルを使用したワークフローで、**1回目の生成は正常**だが、**DisTorch の HSWQ パージ実行後の2回目生成が完全にノイズ化**する。

### ログでの確証（ベイク結果の対比）

| 世代 | ベイク結果 | 状態 |
|------|-----------|------|
| 1回目 | `nvfp4_baked=86 int8_baked=94` / `NVFP4_LORA_BAKE_OK` | 正常 |
| 2回目 | `nvfp4_baked=0 other_qt_baked=83` / `NVFP4_LORA_BAKE_N/A` | NVFP4 レイヤーが `other_qt` に誤分類され、ConvRot なしで誤ベイク → ノイズ |

2回目のログで `nvfp4_baked=0` は、**NVFP4 ConvRot レイヤーが1つも LoRA ベイクされなかった**ことを意味する。NVFP4 レイヤーは重みが Hadamard 回転（ConvRot）された状態で保存されており、LoRA をベイクする際は「逆回転（unrotate）→ LoRA 適用 → 再回転（re-rotate）」の手順が必要。これが実行されないと、回転済み重みに生の LoRA 差分が加算され、重みが壊れてノイズになる。

---

## ② 本質的な原因

### 原因の連鎖（4ステップ）

1. **パージが HSWQ のロードラップを剥がす**
   DisTorch の HSWQ パージは `uninstall_zimage_nvfp4_lora_bake()` を呼び、`ops._load_quantized_module` のラップ（HSWQ が量子化モジュールをロードする際に NVFP4 フラグを武装する処理）を剥がす。このとき `_hswq_nvfp4_full_load` スタンプが失われる。

2. **`apply_comfy_quant_nvfp4_patches()` が早期リターンする**
   この関数は `_PATCHES_APPLIED` フラグと `stack_ver` だけで「適用済み」と判断し、`_load_quantized_module` のラップが**実際に剥がれているか**を確認していなかった。そのためパージ後も「適用済み」のまま早期リターンし、ラップを再適用しなかった。

3. **再ロード時に NVFP4 フラグが武装されない**
   `_load_quantized_module` ラップがないと、モデル再ロード時に `arm_nvfp4_module()` が呼ばれず、各 Linear モジュールに `_hswq_nvfp4_convrot` フラグが設定されない。

4. **ベイク関数が NVFP4 を検出できず誤ベイク**
   LoRA ベイクの際、`_module_is_nvfp4_convrot()` がフラグを確認するが、フラグがないため NVFP4 レイヤーを検出できない。結果として `other_qt`（その他量子化）として処理され、ConvRot の unrotate/re-rotate が実行されず重みが壊れる → ノイズ。

### さらに踏み込んだ背景

- **ComfyUI はローダーノードの出力（MODEL オブジェクト）をキャッシュする**。パージで `current_loaded_models` からモデルが除去されても、ローダーノード自体は再実行されない。そのため「パージ → 次生成」で `load_unet` が走らず、フックの再インストールも起こらなかった。
- パージの `uninstall_zimage_nvfp4_lora_bake()` は `Dynamic.load` のラップを剥がすが、これは「SDXL に切り替える前に ZI のフックを掃除する」という正しい設計。問題は**剥がした後の再アーム手段がなかった**こと。

---

## ③ 追加・修正したファイル名

### ComfyUI-HSWQ-Loader-and-Tools（`1156f00` → `fdc60bc`）

| ファイル | コミット | 内容 |
|---------|---------|------|
| `nodes/zimage_nvfp4/load_unet.py` | `d97bb5b` | `_install_permanent_dynamic_load_guard()` を追加。パージで剥がされない恒久ガードで `Dynamic.load` のベイクフックを自動再アーム |
| `nodes/zimage_nvfp4/zi_comfy_quant_nvfp4.py` | `fdc60bc` | `apply_comfy_quant_nvfp4_patches()` の早期リターンに `_load_wrap_ok` 条件を追加。パージでロードラップが剥がれていたら完全再適用 |

### ComfyUI-DistorchMemoryManager（`62c7cbd` → `9854848`）

| ファイル | コミット | 内容 |
|---------|---------|------|
| `nodes/purge_vram.py`（本流。`__init__.py` が優先ロード） | `9854848` | HSWQ パージ完了後に `unload_models` + `free_memory` キューフラグを設定し、ComfyUI のローダーノード出力キャッシュを破棄 → 次生成でローダーが再実行され TC スタックが再構築される |

※ `purge_vram.py`（ルート、レガシーフォールバック）にも `2936341` で同様の修正が入っているが、`__init__.py` は `nodes/purge_vram.py` を優先するため、本流はそちら。

---

## ④ 追加・修正したコードの全文

### 4-1. `nodes/zimage_nvfp4/load_unet.py`（`d97bb5b` で追加）

`_ensure_dynamic_load_bake_wrap()` の直後に追加された関数:

```python
def _install_permanent_dynamic_load_guard() -> None:
    """Install an outer ModelPatcherDynamic.load guard the purge cannot peel.

    The purge uninstall_zimage_nvfp4_lora_bake walks the chain of wraps stamped
    ``_hswq_zi_nvfp4_lora_bake`` and restores the unwrapped Dynamic.load. After that,
    nothing re-installs the bake hook because ComfyUI caches the loader-node output
    and never re-runs load_unet. The 2nd generation then runs without the NVFP4
    ConvRot LoRA bake and produces noise.

    This guard is a separate, permanent wrap (NOT stamped ``_hswq_zi_nvfp4_lora_bake``),
    so the purge deep-clean walks past it. On every Dynamic.load it ensures the bake
    hook is installed via ``_ensure_dynamic_load_bake_wrap()`` (a no-op when the hook
    is already armed at the current version).
    """
    try:
        import comfy.model_patcher as mp
    except ImportError:
        return
    Dynamic = getattr(mp, "ModelPatcherDynamic", None)
    if Dynamic is None:
        return
    cur = getattr(Dynamic, "load", None)
    if cur is None or getattr(cur, "_hswq_zi_rearm_guard", False):
        return

    def _guarded_load(self, *args, **kwargs):
        try:
            _ensure_dynamic_load_bake_wrap()
        except Exception:
            pass
        return cur(self, *args, **kwargs)

    _guarded_load._hswq_zi_rearm_guard = True  # type: ignore[attr-defined]
    _guarded_load._hswq_zi_rearm_guard_prev = cur  # type: ignore[attr-defined]
    Dynamic.load = _guarded_load
```

呼び出し追加（2箇所）:

```python
# load_unet_nvfp4_weight_dtype() 内（install_zimage_nvfp4_lora_bake の直後）
    _ensure_dynamic_load_bake_wrap()
    _install_permanent_dynamic_load_guard()   # ← 追加
    reset_int8_lora_log_counters()
    reset_nvfp4_lora_log_counters()
    reset_zimage_nvfp4_lora_bake_log_counters()
```

```python
# install_zimage_nvfp4_unet_dispatch() 内の load_unet ラッパー
    def load_unet(self, unet_name, weight_dtype):
        _ensure_dynamic_load_bake_wrap()
        _install_permanent_dynamic_load_guard()   # ← 追加
        if weight_dtype in _fp8:
            return _prev(self, unet_name, weight_dtype)
        if weight_dtype == ZI_NVFP4_WEIGHT_DTYPE:
            return load_unet_nvfp4_weight_dtype(unet_name, weight_dtype)
```

### 4-2. `nodes/zimage_nvfp4/zi_comfy_quant_nvfp4.py`（`fdc60bc` で修正）

```python
    mp_fn = getattr(ops, "mixed_precision_ops", None)
    stack_ver = _effective_nvfp4_stack_ver(mp_fn)
    # The HSWQ purge peels the ops._load_quantized_module wrap (drops the
    # _hswq_nvfp4_full_load stamp) while leaving _PATCHES_APPLIED and the
    # detect_unet_config stamp intact. If the load wrap is gone, the arming
    # (arm_nvfp4_module -> _hswq_nvfp4_convrot) never fires on reload, so the
    # 2nd generation bakes NVFP4 layers as other_qt -> noise. Only early-return
    # when the load wrap is still armed; otherwise fall through to re-apply.
    _load_wrap_ok = bool(
        getattr(ops._load_quantized_module, "_hswq_nvfp4_full_load", False)
    )
    if (
        _PATCHES_APPLIED
        and _load_wrap_ok
        and getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False)
        and stack_ver >= _NVFP4_STACK_VER
    ):
        return True

    # Already patched detect/load but LoRA bake missing: re-wrap mixed_precision_ops only.
    if _load_wrap_ok and getattr(model_detection.detect_unet_config, "_hswq_nvfp4_packed_dims", False) and stack_ver < _NVFP4_STACK_VER:
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
```

※ 変更点は「`_load_wrap_ok` の定義追加」「1つ目の `if` に `and _load_wrap_ok` 追加」「2つ目の `if` の先頭に `_load_wrap_ok and ` 追加」の3点。以降の完全再適用パス（`_orig_detect` 以降）は変更なし。

### 4-3. `nodes/purge_vram.py`（`9854848` で追加）

`HSWQ INT8/NVFP4: Done ? cleared ...` の print 直後（`except Exception as e:` の直前）に追加:

```python
                # tcon NVFP4 support: after the full HSWQ reset, force ComfyUI to re-run
                # the loader node on the next prompt. The purge unloads the model from
                # current_loaded_models, but ComfyUI still caches the loader node output
                # (the MODEL object). Without re-running the loader, the TC (W4A4) stack
                # patches applied in load_unet are not re-installed and the 2nd generation
                # produces noise. Setting the unload_models + free_memory queue flags makes
                # the executor drop cached outputs so the loader re-arms the TC stack.
                try:
                    import server as _srv
                    _ps = getattr(_srv.PromptServer, "instance", None)
                    if _ps is not None and getattr(_ps, "prompt_queue", None) is not None:
                        _pq = _ps.prompt_queue
                        if not getattr(_pq, "currently_running", False):
                            _pq.set_flag("unload_models", True)
                            _pq.set_flag("free_memory", True)
                            print("HSWQ INT8/NVFP4: tcon NVFP4 - queued model unload/cache reset for next prompt (loader will re-arm TC stack)")
                except Exception as _e:
                    print(f"HSWQ INT8/NVFP4: tcon cache-reset flag skipped: {_e}")
```

---

## ⑤ コードの意味

### 5-1. `_install_permanent_dynamic_load_guard()`（load_unet.py）

**目的:** パージが剥がせない「最後の砦」のガードを `ModelPatcherDynamic.load` の外側に設置する。

- `_hswq_zi_rearm_guard` スタンプを付けるが、**`_hswq_zi_nvfp4_lora_bake` スタンプは付けない**。パージの `_deep_clean_dynamic_load()` は `_hswq_zi_nvfp4_lora_bake` スタンプを持つラップだけを辿って剥がすため、このガードは素通りする。
- `_guarded_load` は `Dynamic.load` が呼ばれるたびに、まず `_ensure_dynamic_load_bake_wrap()` を実行する。これは「ベイクフックが既にアーム済みなら no-op、剥がれていれば再インストール」という関数なので、毎回呼んでもコストはほぼゼロ。
- その後、元の `cur`（剥がされた後の素の `Dynamic.load`）を呼ぶ。つまり「ガード → 再アーム確認 → 実際のロード」という順序になり、**ベイクフックが剥がれた状態で Dynamic.load が走ることは二度とない**。
- `_guarded_load._hswq_zi_rearm_guard_prev = cur` は、将来のデバッグ用に元の関数を保持しているだけ。

### 5-2. `_load_wrap_ok` ガード（zi_comfy_quant_nvfp4.py）

**目的:** 「パッチ適用済み」の判定を、フラグだけでなく**実際にラップが生きているか**で行う。

- `_load_wrap_ok = bool(getattr(ops._load_quantized_module, "_hswq_nvfp4_full_load", False))` は、`_load_quantized_module` が HSWQ のラップ（`_hswq_nvfp4_full_load` スタンプ付き）のままかを確認する。
- パージがラップを剥がすと `_hswq_nvfp4_full_load` が消え、`_load_wrap_ok = False` になる。
- 1つ目の `if`（早期リターン）に `and _load_wrap_ok` を追加したことで、**ラップが剥がれていたら早期リターンしない**。完全再適用パス（`_orig_detect` 以降）に進み、`_load_quantized_module` を再ラップする。
- 2つ目の `if`（`stack_ver < _NVFP4_STACK_VER` の再ラップパス）にも `_load_wrap_ok` を追加。ここは `mixed_precision_ops` のみ再ラップして早期リターンするパスで、`_load_quantized_module` は再ラップしないため、ラップが剥がれている場合はこのパスもスキップして完全再適用に進む必要がある。
- 結果: 2回目のロードで `arm_nvfp4_module()` が再実行され、`_hswq_nvfp4_convrot` フラグが全モジュールに再設定される → ベイク関数が NVFP4 レイヤーを正しく検出 → `nvfp4_baked=86` が復活。

### 5-3. パージ後のキャッシュリセット（nodes/purge_vram.py）

**目的:** パージ後、ComfyUI に「ローダーノードを再実行しろ」と伝える。

- パージは `current_loaded_models` から全モデルを除去するが、ComfyUI の**ローダーノード出力キャッシュには MODEL オブジェクトが残る**。このまま次生成すると、キャッシュされた MODEL が再利用され、`load_unet` が再実行されない（＝TC スタック再構築が起きない）。
- `prompt_queue.set_flag("unload_models", True)` と `set_flag("free_memory", True)` は、ComfyUI の executor に「次プロンプト前に全モデルをアンロードし、ノード出力キャッシュを破棄せよ」というフラグを立てる。
- これにより次生成でローダーノードが再実行され、`load_unet_nvfp4_weight_dtype()` → `apply_comfy_quant_nvfp4_patches()` → `install_zimage_nvfp4_lora_bake()` → `_install_permanent_dynamic_load_guard()` の一連の再構築が走り、TC (W4A4) スタックが完全に再武装される。
- `currently_running` チェックは、実行中（プロンプト処理中）にフラグを立てて誤動作するのを防ぐ。`except` はサーバー未初期化などの環境差を吸収する。

---

## 修正の全体像（3層防御）

| 層 | ファイル | 役割 |
|----|---------|------|
| 1 | `nodes/purge_vram.py` | パージ後に ComfyUI キャッシュをリセットし、**ローダー再実行を保証** |
| 2 | `zi_comfy_quant_nvfp4.py` | 再実行されたローダーが**確実に完全再適用**（ラップ剥がれを検出） |
| 3 | `load_unet.py` | 以後、**Dynamic.load が走るたびにベイクフックを自動再アーム** |

1回目の生成は従来どおり正常。パージ後の2回目以降も、この3層で常に正しい NVFP4 ConvRot LoRA ベイクが保証される。
