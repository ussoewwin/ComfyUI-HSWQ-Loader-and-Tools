# SA2 (SageAttention 2) Attention 加速計画書

- 作成日: 2026-09-10
- 状態: 計画（未実装）
- 対象: HSWQ NVFP4 / INT8 Linear + SA2 attention の併用による推論加速
- 情報源: `D:\USERFILES\fp8e4m3\SageAttention\sageattention\` ソース全行実読
- 前提: **SA3 は全滅**（改良 A/C/D すべて不採用。詳細は SA3 計画書の実測結果参照）

---

## 0. SA3 全滅の原因分析 → SA2 で回避可能か

### SA3 不採用の根本原因

| 改良 | 失敗原因 | 本質 |
|---|---|---|
| D (SA3 attention) | SA3 FP4 attention の固有誤差 ~1.8%/layer が多層蓄積で破綻（NVFP4+SA3 で cos 0.0556） | **FP4 の量子化粒度が粗すぎる**（e2m1 = 仮数1bit）。Q/K/V/P すべて FP4 |
| A (mean shift) | 実モデルの DC 極大層で delta_y 加算が出力を 42~64% ずらす | Linear 前処理への mean shift は DC が大きいモデルでは逆効果 |
| C (block-only) | 既定パスを壊して revert | 実装上のミス（設計問題ではない） |

### SA2 の設計が SA3 と根本的に異なる点

| 項目 | SA3 (FP4 attention) | SA2 (INT8/FP8 attention) |
|---|---|---|
| **Q/K 量子化** | FP4 (e2m1, 仮数1bit) | **INT8**（対称、per-block/per-warp/per-thread） |
| **V 量子化** | FP4 (e2m1) | **FP8 E4M3**（仮数3bit、SM89+）or **FP16**（SM80） |
| **P (softmax出力)** | FP4 (e2m1) | FP16 → FP8 E4M3 変換（`S_FP8_OFFSET` で精度維持） |
| **ビット精度** | 仮数 1bit（8値） | INT8: 256値 / FP8: 仮数3bit（16値） |
| **smooth_k** | K mean shift + delta_s 復元 | K mean subtraction（同じ概念、attention 限定） |
| **GQA/MQA** | **非対応** | **対応**（`num_qo_heads // num_kv_heads` で自動展開） |
| **head_dim** | 64/128 のみ | **64/128/256**（パディング対応） |
| **Blackwell** | sm120/121 のみ（FP4 MMA 専用） | **sm100/120/121 対応**（SM89 カーネルで INT8+FP8 MMA 使用） |
| **SDPA 比較精度** | ~0.982（常時 ~1.8% 誤差） | **論文: ほぼロスレス**（INT8 の 256 段階は attention score に対して十分） |

### SA2 が SA3 の失敗を回避できる構造的理由

1. **量子化ビット幅の差**: SA3 の FP4 (e2m1) は仮数 1 bit = 8 値しか表現できない。SA2 の INT8 は 256 値、FP8 E4M3 は仮数 3 bit = 16 値。attention score の分布に対して SA2 のほうが十分な粒度
2. **P の FP8 変換**: SA2 は softmax 出力を FP8 E4M3 に変換する際、`S_FP8_OFFSET = 8.807` で log2 空間を 448 倍シフトし、attention probability の (0, 1] 範囲を E4M3 の最大表現範囲にマッピング。これにより下位ビットの損失を最小化
3. **SageAttention2++** (CUDA ≥ 12.8): `fp32+fp16` 混合アキュムレータ + `scale_max = 2.25` で V の動的レンジを絞り、FP16 アキュムレータの精度限界を回避しつつスループット 2 倍

### 核心的な疑問（本計画で検証する）

> **SA3 の SDPA 比較 cos ~0.982 に対し、SA2 は SDPA 比較でどの程度の cos を出すか？**
> SA2 が 0.999+ を出すなら、量子化 Linear との組み合わせで多層蓄積しても破綻しない可能性が高い

---

## 1. SA2 の実装分析 — ソース実読

### 1.1 アーキテクチャ別カーネルマップ

| アーキ | Q/K | V | PV アキュムレータ | MMA 命令 | カーネル |
|---|---|---|---|---|---|
| SM75 (Turing) | INT8 | FP16 | — | Triton | `sageattn_qk_int8_pv_fp16_triton` |
| SM80/86/87 (Ampere) | INT8 | FP16 | fp32 / fp16 / fp16+fp32 | `mma.sync.m16n8k32.s8s8s32` + `m16n8k16.f16f16f32` | `sageattn_qk_int8_pv_fp16_cuda` |
| SM89 (Ada) | INT8 | FP8 E4M3 | fp32 / fp32+fp32 / **fp32+fp16 (SA2++)** | `mma.sync.m16n8k32.s8s8s32` + `m16n8k32.e4m3.e4m3.f32` | `sageattn_qk_int8_pv_fp8_cuda` |
| SM90 (Hopper) | INT8 | FP8 E4M3 | fp32+fp32 | `wgmma.m64n64k32.s8s8s32` + `wgmma.m64n64k32.e4m3.e4m3.f32` | `sageattn_qk_int8_pv_fp8_cuda_sm90` |
| **SM100/120/121 (Blackwell)** | INT8 | FP8 E4M3 | fp32 / **fp32+fp16 (SA2++)** | SM89 と同一（`mma.sync` 系） | `sageattn_qk_int8_pv_fp8_cuda` + `qk_quant_gran="per_warp"` |

### 1.2 Blackwell (RTX 5060 Ti / 5090) での SA2 ディスパッチ

`core.py` L172-178:
```python
elif arch in ["sm100", "sm120", "sm121"]:
    if get_cuda_version() < (12, 8):
        pv_accum_dtype = "fp32"     # 安全モード
    else:
        pv_accum_dtype = "fp32+fp16" # SA2++ (CUDA 13.2 なので確実にこちら)
    return sageattn_qk_int8_pv_fp8_cuda(
        ..., pv_accum_dtype=pv_accum_dtype, qk_quant_gran="per_warp")
```

- **SM89 用カーネル** (`_qattn_sm89`) で動作。ビルド時 `-gencode arch=compute_120a,code=sm_120a` で Blackwell ネイティブコンパイル
- `per_warp` 量子化: Q を warp (16/32 トークン) 単位で量子化（per-block より高粒度）
- `fp32+fp16` アキュムレータ: FP8 Tensor Core を FP16 アキュムレーション（2x スループット）→ CUDA core で FP32 に展開・蓄積

### 1.3 smooth_k の実装（SA2 版）

```python
# core.py L283-292
km = k.mean(dim=seq_dim, keepdim=True)    # K の sequence 方向平均
# GQA の場合、km を Q の head 数に展開
q_per_kv_heads = nqheads // nkheads
if q_per_kv_heads > 1:
    km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
```

- SA3 の mean shift と同じ概念だが、**attention 内部に閉じている**（Linear 前処理に適用しない）
- softmax の shift 不変性 `softmax(QK^T) = softmax(Q(K-Km)^T)` を利用
- K から mean を引くことで動的レンジを縮小 → INT8 量子化誤差を低減
- `quant_per_block_int8_fuse_sub_mean_cuda`: mean 減算と INT8 量子化を 1 カーネルに融合

### 1.4 FP8 V 量子化パイプライン

`quant.py` `per_channel_fp8`:
1. V を転置: `[B, H, D, padded_L]`（head_dim と seq_len を入れ替え）
2. seq_len を 64 の倍数にパディング
3. FP8 MMA のフラグメントレイアウトに合わせた permute: `[0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15]`
4. per-channel (= per head_dim スライス) で amax → scale → FP8 E4M3 量子化
5. `scale_max = 448.0`（通常）or `2.25`（SA2++）

### 1.5 FP8 softmax offset トリック

`S_FP8_OFFSET = 8.807`（`2^8.807 ≈ 448 = FP8 E4M3 最大値`）

softmax 確率 P ∈ (0, 1] を FP8 E4M3 に変換する際:
- log2 空間で `+8.807` シフト → P の最大値 1.0 が E4M3 の 448 にマッピング
- 微小な確率も E4M3 の表現範囲内に収まる
- 最終的な正規化 `O /= d` でオフセットが自動的にキャンセルされる（追加演算なし）

### 1.6 既存ビルド資産

| 項目 | 詳細 |
|---|---|
| ビルドスクリプト | `build_sa2_dvenv.cmd`（MSVC 2022 + CUDA 13.2） |
| ターゲットアーキ | `TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0;10.0;12.0;12.1` |
| ビルド済み whl | `sageattention-2.2.0+cu132torch2.12.0` (cp312/cp313) |
| 現在ロード中のバイナリ | `w211` バイナリ（sm80 単一 arch、sm89 に sm100/120/121 込み） |
| MSVC SAL 回避 | `fused.cu` L18-22 で `#undef __in/__out/__inout`（SA3 と同じ方式） |
| CUDA バージョン要件 | SM89: CUDA ≥ 12.4 / SM90: ≥ 12.3 / SM120 (Blackwell): ≥ 12.8 |

---

## 2. 検証計画

### Phase 0: SA2 単体精度テスト（最優先、半日）

SA3 で致命的だった「SDPA 比較精度」を SA2 で計測し、採否を即決する。

- [ ] SA2 whl のインストール確認（既存 `sageattention` パッケージ、sm120/121 ビルド済み）
- [ ] **SA2 単体テスト**: ランダム q/k/v、fp16/bf16、HND、head_dim 128、SDPA 比較
  - 条件: seq=4096, smooth_k=True/False, is_causal=False
  - **判定基準**: cos ≥ 0.999 なら Phase 1 へ進む。0.99 未満なら SA2 も不採用
  - （参考: SA3 はこのテストで 0.982 → 不採用の根拠になった）
- [ ] SA2 の Blackwell ディスパッチ確認: `sageattn()` が `sageattn_qk_int8_pv_fp8_cuda` + `per_warp` + `fp32+fp16` を選択していることをログで確認

**完了条件**: SA2 単体精度表 + 採否判定

### Phase 1: 実モデル検証 — 量子化 Linear + SA2 attention（1日）

SA3 で cos 0.0556 に破綻した組み合わせが SA2 で成立するか検証。

- [ ] **NVFP4 Linear (nv100) + SA2**: `zi_convrot_nvfp4_traj_compare.py --attention sage2`
  - 12 step × seed 42、final-cos を計測
  - **判定基準**: cos ≥ 0.96（既存 NVFP4 単体 0.974 からの劣化 ≤ 0.015）
- [ ] **INT8 Linear + SA2**: `zi_int8_bench.py --attention sage2`
  - 25 step × seed 42、latent-cos を計測
  - **判定基準**: cos ≥ 0.97（既存 INT8 単体 0.986 からの劣化 ≤ 0.015）
- [ ] **速度計測**: SA2 ON/OFF で wall time 比較（3 回中央値）
  - SA3 では INT8+SA3 で 30% 高速化の実績あり（品質は壊滅だったが）
- [ ] **FP16 baseline + SA2**: FP16 Linear + SA2 attention → SDPA baseline との cos
  - SA2 の attention 固有誤差が多層蓄積でどう振る舞うかの直接計測

**完了条件**: 4 条件の精度/速度表 + 採否判定

### Phase 2: 20 シード統計検証（採用決定後のみ、1日）

Phase 1 で cos が基準を満たした場合のみ実施。

- [ ] 正規 10 桁シード 20 個 × 12 ステップで cosine mean/std を計測
- [ ] moodyProMix nv100 リグレッション
- [ ] 最終ベンチ表（FP16 / NVFP4 / NVFP4+SA2 / INT8 / INT8+SA2）
- [ ] SDPA フォールバック条件の整理（head_dim > 256 等）

**完了条件**: 20 シード summary + 最終ベンチ表

---

## 3. 実装設計

> 設計原則（SA3 の失敗から学んだ鉄則）:
> 1. **既定パス（env なし）の動作を一切変えない**
> 2. 実装直後に既存スコアの再計測で非劣化を確認
> 3. 専用フラグ/引数で明示的に有効化

### 3.1 統合方式

ベンチスクリプトの `--attention` 引数で SA2 を選択可能にする（SA3 と同じ方式）。
既定は `sdpa`（変更なし）。

### 3.2 SA2 の API 呼び出し

```python
# sageattn() は GPU アーキテクチャを自動検出して最適カーネルを選択
output = sageattn(
    q, k, v,
    tensor_layout="HND",  # Z-Image は [B, H, N, D]
    is_causal=False,
    smooth_k=True,         # K mean subtraction（精度向上）
)
```

- `sageattn()` が自動で `sageattn_qk_int8_pv_fp8_cuda` (Blackwell) を選択
- `smooth_k=True` がデフォルト（K の動的レンジ縮小）
- head_dim は 64/128/256 を自動パディング

### 3.3 SDPA フォールバック条件

SA2 が対応しない条件では SDPA にフォールバック:
- `head_dim > 256` → SA2 が ValueError → SDPA
- SA2 未インストール環境 → import 失敗 → SDPA

---

## 4. 検証プロトコル（SA3 計画書から継承 + 教訓追加）

| 項目 | 手順 |
|---|---|
| 精度 | 20 シード × 12 ステップ TC(W4A4) cosine。**結果は全行 + summary + GEMM MODE を全文出力** |
| 速度 | 同一シード・同一設定で 3 回実行の中央値。FP16 baseline との比率で報告 |
| parity | `nvfp4_comfy_parity` 既存フロー。SA2 ON/OFF で実施 |
| **既存非劣化（リグレッション）** | **変更後、既定パス（sdpa）のスコアを再計測し、非劣化を確認してから SA2 を評価**（SA3 で学んだ鉄則） |
| シード | **正規シード（10桁 20個）を使用** |
| 環境差 | ローカル（5060 Ti）とクラウドで mean ±0.005 変動。0.97 台で同等と判定 |
| 失敗時 | 未達の生成物は即削除。FAIL の原因を 1 行でも記録してから削除 |

---

## 5. リスクと回避策

| リスク | 影響 | 回避策 |
|---|---|---|
| SA2 の INT8 量子化でも多層蓄積で精度劣化 | SA3 と同じ破綻パターン | Phase 0 の単体テストで即判定。cos < 0.999 なら中止 |
| Blackwell での SA2 カーネル動作不良 | sm120/121 での未知のバグ | Phase 0 でディスパッチ確認 + 単体テスト |
| SA2++ (fp32+fp16) の精度不足 | `scale_max = 2.25` が Z-Image の V 分布に合わない | Phase 1 で `pv_accum_dtype="fp32+fp32"` との比較も実施 |
| SA2 whl のバージョン不一致 | import エラー / silent corruption | `sageattention.__version__` と `_qattn_sm89` の存在を確認 |
| 速度改善が僅少 | SA2 導入の意義が薄い | Phase 1 で速度計測。≤ 10% 改善なら不採用 |
| 既定パスの破壊（SA3 で経験） | スコア劣化 | `--attention` フラグ制御、既定は sdpa、実装後に既存スコア再計測 |

---

## 6. 対象ファイルマップ

### 読む（参照実装）
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention\core.py` — メインディスパッチ・attention 実装
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention\quant.py` — INT8/FP8 量子化・fused 操作
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention\sm89_compile.py` — Blackwell カーネルバインディング
- `D:\USERFILES\fp8e4m3\SageAttention\csrc\qattn\qk_int_sv_f8_cuda_sm89.cuh` — SM89/Blackwell 注意カーネル
- `D:\USERFILES\fp8e4m3\SageAttention\csrc\fused\fused.cu` — 融合量子化カーネル

### 変更する（HSWQ 側・最小限）
- ベンチスクリプト: `--attention sage2` オプション追加
- attention 差し替えフック（SA3 と同じ方式）

### 変更しない
- HSWQ Linear forward（NVFP4 / INT8 パス）— 一切触らない
- 既存 parity / bench スクリプトの既定動作

---

## 7. SA3 計画書との関係

本計画は SA3 計画書（`2026-09-09_sa3_nvfp4_hswq_acceleration_plan.md`）の **改良 D の後継**。
SA3 計画書の改良 B（融合カーネル）は SA2 とは独立であり、引き続き有効な速度改善候補。

| SA3 計画の改良 | 状態 | SA2 計画との関係 |
|---|---|---|
| A (mean shift) | 不採用 | SA2 では attention 内の smooth_k として同等機能が内蔵（Linear には適用しない） |
| B (融合カーネル) | 未着手 | **独立して有効**。SA2 と組み合わせ可能（Linear 速度 + attention 速度の両方を改善） |
| C (block-only) | 実装失敗 | SA2 とは無関係 |
| D (SA3 attention) | 不採用 | **本計画で SA2 attention に置き換え** |
