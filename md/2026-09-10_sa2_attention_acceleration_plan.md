# SA2 (SageAttention 2) 転用計画書 — HSWQ 量子化 Linear との併用による推論加速

- 作成日: 2026-09-10
- 状態: 計画（Phase 0 の核心部分は実測済み）
- 対象: HSWQ NVFP4 / INT8 Linear + SA2 attention の併用加速
- 情報源: `D:\USERFILES\fp8e4m3\SageAttention\sageattention\` ソース実読 + インストール済みパッケージ実測
- 前身: `2026-09-09_sa3_nvfp4_hswq_acceleration_plan.md`（SA3 は改良 A/C/D 全滅、B 未着手）
- 参考: `implementation_plan.md`（SA2 計画ドラフト。本計画はこれを土台に実測値で再構成したもの）

---

## 0. 背景 — SA3 全滅と SA2 の位置づけ

### 0.1 前計画（SA3）の実測結果

| 改良 | 結果 | 原因 |
|---|---|---|
| A (mean shift, Linear) | **不採用** | 実モデルの DC 極大層で delta_y が出力 DC を 42〜64% ずらす（seed42: 0.615 vs OFF 0.984） |
| C (block-only) | **実装失敗** | 既定パスを破壊（既存 0.974 → 0.924 相当）→ 全 revert |
| D (SA3 attention) | **不採用** | SA3 FP4 attention の固有誤差 ~1.8%/層が多層蓄積（NVFP4+SA3 = 0.0556、INT8+SA3 = 0.0909） |
| B (rotate+quantize 融合 / butterfly) | **未着手** | 独立して有効。本計画と並行可能 |

**Phase 0 プロファイル実測**（1024×1024、6step、5060 Ti）: softmax/attention 26.4%（3.613s）／ rotate 16.9%（2.306s）／ cuBLAS FP4 GEMM 19.6%（2.680s）／ quantize 1.8%（0.246s）／ その他。**attention が最大の単一ボトルネック**であり、SA2 が効けば最大の速度改善になる。

### 0.2 SA3 vs SA2 の構造差

| 項目 | SA3 (FP4 attention) | SA2 (INT8/FP8 attention) |
|---|---|---|
| Q/K 量子化 | FP4 e2m1（仮数 1bit = 8 値） | **INT8**（256 値、per-warp/per-block/per-thread） |
| V 量子化 | FP4 e2m1 | **FP8 E4M3**（仮数 3bit = 16 値） |
| P (softmax 出力) | FP4 e2m1 | FP16 → FP8 E4M3（`S_FP8_OFFSET=8.807` で精度維持） |
| GQA/MQA | **非対応** | **対応**（`nqheads // nkheads` 自動展開） |
| head_dim | 64/128 のみ | **64/128/256**（パディング対応、>256 は ValueError） |
| Blackwell | sm120/121 のみ | **sm100/120/121 対応**（SM89 カーネルで動作） |
| SDPA 比較精度（実測） | **0.98192** | **0.999258** ← 本計画の核心データ |

### 0.3 本計画の核心（Phase 0 で実測済み）

**SA2 単体の SDPA 比較精度を実測した結果、cos = 0.999258（誤差 0.074%）。**

| 手法 | seq=4096 | seq=4128 |
|---|---|---|
| SA3 (`sageattn3_blackwell`) | 0.98192 | 0.98175 |
| **SA2 (`sageattn` → sm120 パス)** | **0.999258** | **0.999254** |
| SA2 fp16 per_warp | カーネル無し（sm120 未提供: `no kernel image`） |

→ **SA2 は SA3 の約 24 倍高精度**（誤差 1.8% → 0.074%）。量子化 Linear（NVFP4 誤差 ~2.6% / INT8 ~1.4%）と併用しても、誤差の加算が実質 Linear 分に留まる可能性が高い。

---

## 1. SA2 コード実読（検証済み事実）

### 1.1 アーキテクチャ別カーネルマップ（`core.py`）

| アーキ | Q/K | V | PV アキュムレータ | カーネル |
|---|---|---|---|---|
| SM75 | INT8 | FP16 | — | `sageattn_qk_int8_pv_fp16_triton` |
| SM80/86/87 | INT8 | FP16 | fp32 | `sageattn_qk_int8_pv_fp16_cuda` |
| SM89 (Ada) | INT8 | FP8 E4M3 | fp32 / fp32+fp32 / **fp32+fp16** | `sageattn_qk_int8_pv_fp8_cuda` |
| SM90 (Hopper) | INT8 | FP8 E4M3 | fp32+fp32 | `sageattn_qk_int8_pv_fp8_cuda_sm90` |
| **SM100/120/121 (Blackwell)** | INT8 | FP8 E4M3 | fp32 / **fp32+fp16** | `sageattn_qk_int8_pv_fp8_cuda` + `qk_quant_gran="per_warp"` |

### 1.2 Blackwell ディスパッチ（`core.py` L171-178、実測確認済み）

```python
elif arch in {"sm100", "sm120", "sm121"}:
    if get_cuda_version() < (12, 8):
        pv_accum_dtype = "fp32"        # 安全モード
    else:
        pv_accum_dtype = "fp32+fp16"   # SA2++（本環境は CUDA 13.2 → こちら）
    return sageattn_qk_int8_pv_fp8_cuda(..., qk_quant_gran="per_warp", pv_accum_dtype=pv_accum_dtype)
```

実測: `sageattention.core._cuda_archs = ['sm120']`、`torch.version.cuda = 13.2` → **SA2++ パス（fp32+fp16, per_warp）が選択される**。`sageattn()` の戻り値は SDPA 比較 cos 0.999258（前述）で、当該パスが正常動作することを確認済み。

### 1.3 インストール済み環境（実測）

| 項目 | 値 |
|---|---|
| パッケージ | `sageattention-2.2.0.post6+cu132torch2.14.0` |
| コンパイル済みモジュール | `_fused`, `_qattn_sm80`, `_qattn_sm89`（sm89 に sm100/120/121 を含む） |
| SM89/SM90 有効フラグ | `SM89_ENABLED=True`, `SM90_ENABLED=True` |
| CUDA / arch | 13.2 / sm120（RTX 5060 Ti 16GB） |
| fp16 per_warp (sm120) | **カーネル未提供**（`no kernel image is available`）→ **fp8 パスのみ実用** |

### 1.4 smooth_k（K mean subtraction、attention 内で完結）

- `sageattn_qk_int8_pv_fp8_cuda(..., smooth_k=True)` が**既定**（実測確認）
- K の sequence 方向平均を引いてから INT8 量子化 → 動的レンジ縮小で量子化誤差低減
- GQA では `repeat_interleave` で Q head 数に展開
- `quant_per_block_int8_fuse_sub_mean_cuda` で mean 減算と量子化を 1 カーネルに融合
- **SA3 の mean shift（改良 A）と同概念だが、attention 内部に閉じており Linear 前処理には適用しない**（= DC 極大層の問題を構造的に回避）

### 1.5 FP8 V 量子化 + softmax offset トリック

- V: `per_channel_fp8`（head_dim 方向 per-channel amax、`scale_max=448` / SA2++ では `2.25`）
- P: `S_FP8_OFFSET = 8.807`（log2 空間で P の最大値 1.0 を E4M3 の 448 に写像、最終正規化で自動キャンセル）

### 1.6 head_dim パディング規則（`core.py` L75-89、実測確認済み）

| 元 head_dim | パディング先 |
|---|---|
| < 64 | 64 |
| 65〜127 | 128 |
| 129〜255 | 256 |
| > 256 | **ValueError**（→ SDPA フォールバック必須） |

### 1.7 ビルド資産

- ビルド済み whl: `sageattention-2.2.0+cu132torch2.12.0`（cp312/cp313）
- MSVC SAL 回避: `fused.cu` 冒頭で `#undef __in/__out/__inout`（SA3 と同方式）
- CUDA 要件: SM89 ≥ 12.4 / SM90 ≥ 12.3 / **SM120 ≥ 12.8**

---

## 2. 検証計画

### Phase 0: SA2 単体精度・ディスパッチ確認（**核心は実測済み**）

- [x] パッケージ確認: `sageattention 2.2.0.post6+cu132torch2.14.0` / `_qattn_sm89` 存在
- [x] arch ディスパッチ確認: `sm120` → `sageattn_qk_int8_pv_fp8_cuda(per_warp, fp32+fp16)`
- [x] **単体精度実測**: seq=4096/4128、head_dim 128、fp16、HND、SDPA 比較 → **cos 0.999258 / 0.999254**
- [ ] 追加条件: bf16 入力 / smooth_k=False / head_dim 64・256 での確認
- [ ] `pv_accum_dtype="fp32+fp32"` との精度比較（SA2++ の妥当性確認）

**判定基準**: cos ≥ 0.999（SA3 は 0.982 で不採用になった）
**現状**: **0.999258 で基準クリア。Phase 1 へ進む**

### Phase 1: 実モデル検証 — 量子化 Linear + SA2 attention（1 日）

SA3 で破綻した組合せが SA2 で成立するかを検証。

- [ ] **NVFP4 Linear (nv100) + SA2**: `zi_convrot_nvfp4_traj_compare.py --attention sage2`（12 step × seed42）
  - 判定基準: final-cos ≥ 0.96（現行 NVFP4 単体 0.974 から劣化 ≤ 0.015）
- [ ] **INT8 Linear + SA2**: `zi_int8_bench.py --attention sage2`（25 step × seed42）
  - 判定基準: latent-cos ≥ 0.97（現行 INT8 単体 0.986 から劣化 ≤ 0.015）
- [ ] **FP16 Linear + SA2**: FP16 + SA2 vs FP16 + SDPA（SA2 固有誤差の多層蓄積を直接計測）
- [ ] **速度計測**: SA2 ON/OFF の wall time（3 回中央値）
  - 参考: SA3 では INT8+SA3 が 30% 高速（62.5s → 43.3s）。SA2 も同程度を期待

**完了条件**: 4 条件の精度/速度表 + 採否判定

### Phase 2: 20 シード統計検証（Phase 1 合格後のみ、1 日）

- [ ] 正規 10 桁シード 20 個 × 12 ステップで cosine mean/std/95%CI
- [ ] moodyProMix / moodyRealMix nv100 のリグレッション（SA2 OFF の既定パス）
- [ ] 最終ベンチ表（FP16 / NVFP4 / NVFP4+SA2 / INT8 / INT8+SA2 + 速度）
- [ ] SDPA フォールバック条件の整理（head_dim > 256、mask あり、import 失敗）

**完了条件**: 20 シード summary + 最終ベンチ表

---

## 3. 実装設計

**設計原則（SA3 の失敗から確立した鉄則）**
1. **既定パス（env／フラグなし）の動作を一切変えない**
2. 実装直後に既存スコア（既定パス）を再計測し、非劣化を確認してから新機能を評価
3. 変更は追加のみ（既存コードの書き換え禁止）。専用フラグで明示的に有効化

### 3.1 統合方式

ベンチスクリプトの `--attention {sdpa,sage2}` 引数で選択。**既定は sdpa（従来動作）**。
量子化モデル側にのみ attention override を適用し、FP16 ベースラインは stock attention のまま（比較の基準を維持）。

### 3.2 API 呼び出し

```python
# sageattn() は GPU アーキテクチャを自動検出（sm120 → INT8 QK + FP8 PV, per_warp）
out = sageattn(q, k, v, tensor_layout="HND", is_causal=False, smooth_k=True)
```

- Z-Image の attention は `optimized_attention_masked(mask, skip_reshape=True)` 経由 → **mask がある場合は SDPA フォールバック**（SA3 実装と同じ扱い）
- `smooth_k=True`（既定）で K mean subtraction が有効

### 3.3 SDPA フォールバック条件

| 条件 | 動作 |
|---|---|
| `mask is not None` | SDPA |
| `head_dim > 256` | SDPA（SA2 が ValueError） |
| `import sageattention` 失敗 | SDPA |
| 実行時例外 | SDPA（reason をログ） |

---

## 4. 検証プロトコル（SA3 計画書から継承 + 教訓）

| 項目 | 手順 |
|---|---|
| 精度 | 20 シード × 12 ステップ TC(W4A4) cosine。**全行 + summary + GEMM MODE を全文出力** |
| 速度 | 同一シード・同一設定で 3 回実行の中央値。FP16 baseline との比率 |
| parity | `nvfp4_comfy_parity` 既存フロー。SA2 ON/OFF で実施 |
| **既存非劣化（リグレッション）** | **変更後、既定パス（sdpa）のスコアを再計測し非劣化を確認**（SA3 の失敗の直接原因への対策） |
| シード | **正規シード（10 桁 20 個）を使用**（独自シードは mean が最大 0.05 変動） |
| 環境差 | ローカル（5060 Ti）とクラウドで mean ±0.005 変動。**0.97 台で同等と判定** |
| 評価対象の照合 | 計測前に artifact（正規品か）とシード（正規か）を照合 |
| 失敗時 | 未達の生成物は即削除。FAIL 原因を 1 行記録してから削除 |

---

## 5. リスクと回避策

| リスク | 影響 | 回避策 |
|---|---|---|
| SA2 でも多層蓄積で劣化 | SA3 と同じ破綻 | Phase 0 は実測済み（0.9993）。Phase 1 の FP16+SA2 で蓄積挙動を直接確認 |
| Blackwell でのカーネル動作不良 | 未知のバグ | Phase 0 でディスパッチ＋単体精度を実測済み（正常） |
| fp16 カーネルが sm120 に無い | fp16 経路が使えない | fp8（SA2++）パスのみ使用。実測で問題なし |
| SA2++ (fp32+fp16) の精度不足 | `scale_max=2.25` が Z-Image の V 分布に不適合 | Phase 1 で `pv_accum_dtype="fp32+fp32"` と比較 |
| mask ありパスが多く SDPA に落ちる | 速度改善が出ない | Phase 1 で SA2 実走率をログ計測（override 呼び出し回数とフォールバック回数） |
| 速度改善が僅少 | 導入意義が薄い | Phase 1 で速度計測。≤10% 改善なら不採用 |
| 既定パスの破壊 | スコア劣化（SA3 で実例） | `--attention` フラグ制御・既定 sdpa・実装後の既存スコア再計測 |

---

## 6. 対象ファイルマップ

### 読む（参照実装）
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention\core.py` — ディスパッチ・attention 実装（L171-178 が Blackwell 分岐）
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention\quant.py` — INT8/FP8 量子化・fused 操作
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention\sm89_compile.py` — Blackwell バインディング
- `D:\USERFILES\fp8e4m3\SageAttention\csrc\qattn\` — SM89/Blackwell カーネル
- `D:\USERFILES\fp8e4m3\SageAttention\csrc\fused\fused.cu` — 融合量子化カーネル

### 変更する（HSWQ 側・追加のみ）
- `benchmark/zi_convrot_nvfp4_traj_compare.py` — `--attention {sdpa,sage2}` 追加（既定 sdpa）
- `benchmark/zi_int8_bench.py` — 同上
- （採用後）`ComfyUI-HSWQ-Loader-and-Tools` の該当ノードへ移植

### 変更しない
- HSWQ Linear forward（NVFP4 / INT8 パス）— 一切触らない
- 既存 parity / bench スクリプトの既定動作

---

## 7. SA3 計画書との関係

本計画は SA3 計画書（`2026-09-09_sa3_nvfp4_hswq_acceleration_plan.md`）の**改良 D の後継**。

| SA3 計画の改良 | 状態 | 本計画との関係 |
|---|---|---|
| A (mean shift) | 不採用 | SA2 では attention 内の `smooth_k` として同等機能が内蔵（Linear には適用しない） |
| B (融合カーネル / butterfly) | 未着手 | **独立して有効**。SA2 と併用可能（Linear 側の速度改善） |
| C (block-only) | 実装失敗 | SA2 とは無関係 |
| D (SA3 attention) | 不採用 | **本計画で SA2 attention に置換** |
