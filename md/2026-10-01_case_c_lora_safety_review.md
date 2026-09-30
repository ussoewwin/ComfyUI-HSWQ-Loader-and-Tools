# C案実装前のLoRA整合性チェック — 実測確認結果（設計ドキュメント）

Date: 2026-10-01
Status: 実装設計確定前の実測メモ（コード確認済み、実行未検証）

## 1. 確認したLoRA保護機構（実測）

C案（NVFP4フォールバック時のINT8再量子化）が壊しうるLoRA経路は次の4つ。すべて実コードを確認した。

### 1.1 ランク分解residual経路（`nvfp4_lora_bake.py::_extract_nvfp4_lora_residual`）
- LoRAパッチが `WeightAdapterBase` の単純な A(down)/B(up)/alpha 構成の場合、**packed QTを保持したまま** `module._hswq_krea2_lora_res = [(mat_dn, mat_up, scale)]` を保存
- forward側は `_add_krea2_lora_residual` がstock forward出力に `inp @ A^T @ B^T * scale` を加算
- **重要**: residualはpacked重みに加算せずforward後加算のため、**重みがNVFP4からINT8に変わってもresidual加算はそのまま有効**（`_add_krea2_lora_residual` は `stock_forward` の出力に乗る。INT8 fallback経路の `stock_forward` は `F.linear` dequant出力であり、basis は回転空間だがresidualは「出力に対する加算」なので数学的に不変）

### 1.2 requantize bake経路（rank分解不可能なLoRA: DoRA/mid/reshape/strength≠1）
- `_extract_nvfp4_lora_residual` がNoneを返すと `patcher.patch_weight_to_device` → `convert_weight`（dequant+unrotate）→ `set_weight`（re-rotate→`requantize_from_float`）の流れ
- **`requantize_from_float` は `TensorCoreNVFP4Layout` に存在しない**（kitchen `nvfp4.py` のメソッド一覧実測: quantize/dequantize/get_plain_tensors/state_dict_tensors/get_padded_shape/get_storage_shape/get_logical_shape_from_storage のみ）。INT8の `TensorWiseINT8Layout` にも存在しない（実測0件）
- 現行は `set_weight` → `self.weight.requantize_from_float(...)` を QuantizedTensor 基底（`base.py` に1件実測）が捌いている
- **C案でweightがINT8 QTに置き換わった場合も、基底の `requantize_from_float` はlayoutの `quantize` を呼ぶため動作する**（INT8 layoutの `quantize` は実測で存在）。ただし重みが「NVFP4 packed→INT8 packed」に変わった直後にLoRAを当てる場合、`convert_weight` が呼うのはlayout非依存の `dequantize()` なので INT8 QT でも float に戻る → unrotate → re-rotate → requantize の流れは壊れない

### 1.3 baked_keys無効化機構（`_maybe_invalidate_baked_keys` / `_hswq_int8_baked_keys`）
- `patches_uuid` 変更時にbake集合をクリア。C案で新しく「INT8化した層」をbake済み集合に登録するかは設計次第だが、**登録しないのが正しい**（bake済み集合は「LoRAが焼かれた層」の管理であり、重み形式変換は別概念）

### 1.4 backup/patches削除（`_bake_nvfp4_residual_keys_on_module` 内）
- bake後に `patcher.backup[key]` と `patcher.patches[key]` を削除する。C案の再量子化が発生するタイミングは「フォワード中のフォールバック」＝bake処理後である可能性がある。ここで `patcher.backup` が既に削除済みでも、C案は「重みテンソルの置き換え」のみでpatcherの状態を触らないため競合しない

## 2. C案の安全設計（上記を踏まえた確定版）

1. **変換は重みテンソルの置き換えに限定**: `module.weight` を NVFP4 QT → INT8 QT（`TensorWiseINT8Layout.quantize` 経由）に置き換える。patcher/backup/patches/LoRA residual 属性には一切触れない
2. **residual属性は温存**: `_hswq_krea2_lora_res` / `_hswq_krea2_lora_res_gpu` はそのまま。forward側の `_add_krea2_lora_residual` は INT8 経路でも呼ばれる（stock_forward出力への加算のためbasis不変）
3. **convert_weight/set_weight のwarpは温存**: INT8化後のLoRA bakeは `dequantize()`（layout非依存）→ unrotate → re-rotate → `requantize_from_float`（基底実装がINT8 layoutの `quantize` を呼ぶ）で完結する。`_hswq_nvfp4_convrot` フラグは保持（unrotate/re-rotateの判定に使用）
4. **スケール保持の禁止**: NVFP4の `input_scale` はINT8経路では不要。削除せず保持（state_dict出力への影響を避ける。`_hswq_nvfp4_scale_placeholder` フラグも保持）
5. **片方向変換**: NVFP4→INT8は一方向。同一セッション内でTC復帰しても自動では戻さない（戻す場合は明示的な再ロード）。これによりbake状態と重み形式の不整合を構造的に排除
6. **baked_keys登録なし**: 重み形式変換はLoRA bakeではないため `_hswq_int8_baked_keys` に登録しない。`patches_uuid` が変わったときのbake再実行は通常通り働く
7. **1回だけ変換**: 変換済みは `module._hswq_realquant_backend == "int8_fallback"` タグで判定し、二重変換を禁止

## 3. 検証計画（LoRA破壊の絶対禁止を検証する項目）

1. ランク分解residual LoRA装着 → フォールバック誘発（TC無効化）→ 出力のcos-sim比較（INT8化前のresidual有り出力 vs INT8化+residual有り出力、FP16 GT比）
2. DoRA / mid / strength_model≠1 / offset付きpatch → requantize bake経路がINT8 QT上で完走すること
3. 変換後にLoRA強度を変更（再bake誘発）→ 正しく再bakeされること
4. 変換後にLoRA除去（patcher restore）→ 重みがINT8に戻ること（backup由来のfloatに戻らない。backup削除済みの場合の挙動も確認）
5. 既存SSIM/s/itプロトコル（12ステップ同seed）
