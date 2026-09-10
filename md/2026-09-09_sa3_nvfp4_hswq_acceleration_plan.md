# SA3 NVFP4 → HSWQ NVFP4 加速転用計画書

- 作成日: 2026-09-09
- 状態: 計画（未実装）
- 対象: HSWQ NVFP4 (Z-Image / ConvRot TC W4A4) の精度・速度改善
- 情報源: `D:\USERFILES\fp8e4m3\SageAttention` ソース実読（本計画書の記載はすべてソース確認済み）

---

## 0. 目標と成功条件

| 項目 | 現状 | 目標 |
|---|---|---|
| 精度 (moodyRealMix_xhsEdition TC W4A4) | **nv100 確定: mean 0.97366**（10桁シード20個、クラウド計測） | 維持（劣化させない） |
| 精度 (moodyProMix_collectorsEdition) | nv100 確定 (0.96033) | 維持（劣化させない） |
| 速度 (NVFP4 Linear 前処理) | rotate GEMM → quantize → GEMM の 3 パス | rotate+quantize 融合で前処理 1 カーネル化 |
| calib input_scale 依存 | convrot 系は calib 必須 | block-scale-only モードで calib 不要化の可否判定 |

**総合成功条件**: 上記のうち最低 1 つを達成し、既存 parity 検証（nvfp4_comfy_parity）を壊さないこと。

### 実測結果サマリ（2026-09-10、本計画の初回実施）

| 項目 | 結果 |
|---|---|
| 改良 A（mean shift） | **不採用**: 実モデルで破綻（seed42 で 0.615、OFF は 0.984）。単層では改善するが多層では悪化 |
| 改良 C（block-only） | **実装失敗**: 既定パス（env なし）を壊し、既存スコア 0.974 → 0.924 相当に低下 → 全変更 revert |
| 改良 B/D | 未着手 |
| 基準スコア再計測 | 2015feb 復元後、既存 nv100 = mean 0.96849（10桁シード、ローカル）。クラウド 0.97366 との差 -0.005 は環境差 |

**最重要の教訓**: 変更は「既定パス（env なし）の動作を一切変えない」ことが絶対条件。
実装直後に既存スコアの再計測（リグレッションテスト）を行い、非劣化を確認してから新機能を評価する。

---

## 1. SA3 (SageAttention3) の NVFP4 実装分析 — ソース実測

### 1.1 全体パイプライン (`sageattention3_blackwell/sageattn3/api.py`)

```
q, k, v (bf16/fp16, [B, H, L, D])
  ↓ preprocess_qkv (per_block_mean: bool = True)
    1. k -= k.mean(dim=-2)          # K のテンソル全体平均をシフト
    2. q,k,v を L→128 の倍数に pad (0 埋め。K,V も同様に pad_128)
    3. [per_block_mean=True] triton_group_mean: Q を 128 トークンごとの group に分割、
       qm[group] = mean、q -= qm  (per-block mean removal)
       [per_block_mean=False] qm = q.mean(dim=-2)、q -= qm  (global mean)
    4. delta_s = qm @ k^T  (fp32, [B,H,num_groups,L_k])
  ↓ scale_and_quant_fp4 (Q) / _permute (K) / _transpose (V)
    - 出力: uint8 パック FP4 (D/2) + e4m3 スケール (D/16)
  ↓ blockscaled_fp4_attn → fp4attn_cuda.fwd (Blackwell カーネル)
  ↓ 出力スライス [:, :, :QL, :]
```

### 1.2 量子化カーネル (`csrc/quantization/fp4_quantization_4d.cu`)

- `CVT_FP4_ELTS_PER_THREAD = 16`、`BLOCK_SIZE = 128`（トークン）
- **スケール粒度: 16 要素 / block**、`SF = amax / 6.0` → `e4m3` に丸め、逆数を適用後 `e2m1` パック
- FP4 変換は PTX `cvt.rn.satfinite.e2m1x2.f32`（ハードウェア命令）
- **SF のメモリレイアウト**: 128×4 タイル
  `offset = (col/4)*256 + (col%4) + (row/16)*4 + (row%16)*16`
  （これは cuBLAS blockwise FP4 / tcgen05 の SF レイアウトと同系統。HSWQ の
  `scaled_mm_nvfp4_pooled` の `(roundup_m 128, roundup_sk 4)` と互換）
- **permute 版 (K 用)**: 32 トークン単位で `[0,1,8,9,16,17,24,25,2,3,...]` に入れ替え。
  MMA 命令のオペランドレイアウトに事前に合わせるため
- **trans 版 (V 用)**: shared memory で転置してから同一の量子化処理
- 1 カーネルで pad 吸収 + amax + スケール + FP4 パック + レイアウト整列まで完結

### 1.3 Attention メインループ (`csrc/blackwell/mainloop_tma_ws.h`)

- MMA 命令: `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`
  （warp レベル block-scaled FP4 MMA。QK^T と PV の両方に使用）
- **`add_delta_s`**: Q@K^T アキュムレータに事前計算の `delta_s` を float4 加算。
  per-block mean を除去した分をソフトウェアで復元
- **`quantize` ラムダ（P のオンライン FP4 量子化）**:
  softmax 出力 P を absmax → `ue4m3` スケール → `packed_float_to_e2m1` で FP4 化し、
  そのまま PV MMA に投入。`__shfl_xor_sync` で quad 間の SF を結合
- `softmax_fused.online_softmax_with_quant`: softmax と P 量子化の融合
- 制約: sm120/sm121 のみ、head_dim 64/128 のみ（256 は `api.py` で SDPA fallback、
  `DISPATCH_HEAD_DIM` / `launch.h static_assert` に 256 分岐なし＝未実装）、
  GQA 非対応（`TORCH_CHECK(num_heads == num_heads_k)`）、Q/K/V すべて seqlen を 128 の倍数に pad

### 1.4 SA3 の設計思想（HSWQ への示唆）

1. **平滑化は「シフト」で行う**: 回転ではなく平均シフト（+ delta 復元）。
   回転（ConvRot/Hadamard）とシフトは**数学的に直交**する改善
2. **per-tensor scale を使わない**: 16 要素 block scale (e4m3) だけで成立
3. **前処理は全部 1 カーネルに融合**: pad/amax/scale/pack/permute を分離しない
4. **P も FP4 で保持**: attention 固有の最適化（GEMM 間中間テンソルの FP4 保持）

---

## 2. HSWQ NVFP4 現行パスの実測（ボトルネック特定）

### 2.1 現行 Linear forward (`nodes/zimage_nvfp4/zi_nvfp4_forward.py`)

```
input (bf16, ≥3D)
  ↓ reshape 2D
  ↓ [ConvRot の場合] rotate_last_dim_pooled(input, H, gs=256)
      ← dense Hadamard GEMM (256×256)。torch.matmul、出力は _ROT_OUT_POOL 再利用
  ↓ _tc_forward_pooled
      - ensure_act_scale (calib input_scale / amax フリーズ)
      - alpha = scale_a * scale_b キャッシュ
      - [opt-in CUDA Graph] nvfp4_quant_mm_cudagraph(_perweight)
      - eager: quantize_nvfp4_act_pooled  (comfy_kitchen _C.quantize_nvfp4)
               → scaled_mm_nvfp4_pooled   (_C.cublas_gemm_blockwise_fp4)
```

### 2.2 既知の問題点（コードコメント・memory からの実測事実）

| # | 問題 | 証拠 |
|---|---|---|
| P1 | rotate 用 dense Hadamard GEMM (gs=256) が高コスト。butterfly 版 `rotate_last_dim_fast` は実装済みだが**本番パス未接続** | `zi_nvfp4_forward.py` コメント「butterfly is ~15x slower に対し dense 使用」、`zi_nvfp4_hadamard.rotate_last_dim_fast` 未使用 |
| P2 | rotate → メモリ書き戻し → quantize カーネル読み直し → 書き戻し → GEMM と**グローバルメモリ往復が 2 回余計** | `nvfp4_runtime.py` の 2 段構成 |
| P3 | calib `input_scale` がない層は amax フリーズ（1 回目の amax を固定）→ 層ごとの分布変化に追従しない。**注: サブシステムごとに挙動が異なる**: (1) `nvfp4_runtime.py` は step-0 amax を `module._hswq_nvfp4_act_scale` にフリーズ、(2) Krea2 側は step-0 noise amax による 0.05 SSIM 劣化を発見済みで per-call online amax に変更済み、(3) Z-Image 側は `ensure_act_scale` を直接呼ぶが `alpha = scale_a * scale_b` を `module._hswq_nvfp4_alpha` にキャッシュして初回固定（暗黙のフリーズ） | `ensure_act_scale_cached`、`krea2_convrot_nvfp4/nvfp4_runtime.py` コメント |
| P4 | 小 M で NVFP4 GEMM が FP16 に負ける（量子化オーバーヘッドが勝つ） | `_GRAPH_MAX_M = 512` のコメント |
| P5 | 精度: nv90〜nv84 で 0.95 未達。scale 調整（nv 値）では頭打ち | memory/2026-08-26.md、K-search 結果 |

### 2.3 ボトルネックの構造

- **速度**: P1 + P2 は同一根（前処理が分離していること）。P4 も前処理コストが原因の一つ
- **精度**: P3 + per-tensor scale 方式の限界。16 要素 block scale の e4m3 は
  既に十分細かい → あとは**活性分布そのものの平滑化**（回転は実施済み、シフト未実施）

---

## 3. 改良項目の詳細設計

> 設計原則（Owner 哲学）: **各改良は専用コードパスに完全分離**。「共通処理 + 条件分岐で混ぜる」禁止。
> env フラグまたは別関数・別ファイルで明示的に切り替え、既定パスの挙動は変えない。

### 改良 A: per-block mean shift（精度狙い・Linear 版 delta_s）

**発想元**: SA3 `preprocess_qkv` の group mean + delta_s。

**数学**: Linear `y = W·x` に対し、トークン 128 個ごとの group g で

```
μ_g = mean(x ∈ g)                    # (num_groups, in_features)
x'  = x − μ_g                        # group 内でシフト（broadcast）
delta_y = μ_g @ W^T                   # (num_groups, out_features) の小 GEMM 1 回
y   = W·x' + delta_y                  # delta_y を group ごとに broadcast 加算
```

- `delta_y` は `(num_groups, out_features)` の小 GEMM 1 回（group 数 = M/128）。
  SA3 の `delta_s = qm @ k^T` と同じ `μ @ 重み^T` 形式で統一
- **ConvRot との順序**: Hadamard は group 内 (gs=256) の線形演算なので
  `(x − μ)H = xH − μH`。実装は **rotate 後に 128-token mean を取る**（＝SA3 と同じ
  「量子化直前の分布を平坦化する」位置。rotate 後分布に対するシフトになる）
- **input_scale との干渉**: シフトで amax が下がるため、calib scale と実測分布が
  ミスマッチする。→ 改良 C（block-only）と同時評価、またはシフト分布での再キャリブ
- **alpha キャッシュ干渉**: Z-Image 側は `alpha = scale_a * scale_b` を
  `module._hswq_nvfp4_alpha` にキャッシュして初回固定している。mean shift で活性分布が
  変わり scale_a が変動するため、**専用パスでは alpha キャッシュを使わず毎回再計算する**
- **専用パス**: `HSA_MEANSHIFT=1` env で有効化する別関数
  `_tc_forward_pooled_meanshift`（既存 `_tc_forward_pooled` を書き換えない）

**リスク**:
- 画像 latent を flatten した 2D の「128 トークン」は空間的に連続だが、
  head 次元・チャネルの並びによっては group が不自然な集合になる可能性 → 実測で判定
- `delta_y` 加算を忘れると完全に壊れる（parity で即検出可能）

**期待効果**: SA3 で video/image gen に有効だった平滑化。活性ダイナミックレンジ縮小
→ W4A4 量子化誤差低減 → **0.95 ライン到達の本命候補**。

**実測結果（2026-09-10）— 不採用**

| 構成（seed 42、1024x1024、12step、TC） | final-cos |
|---|---|
| mean shift OFF（新型 artifact） | 0.98432 |
| mean shift ON | **0.60676**（破綻） |
| mean shift ON + scale_a=1.0 | 0.93669 |
| mean shift ON + noise_refiner 層を除外 | 0.85668 |
| 単層ユニットテスト（理想条件） | 0.993 → 0.9995（改善） |

- **原因**: noise_refiner 等の層は入力の DC 成分が極大（mu/x 比 0.77〜0.998、出力 DC は 800 級）。
  delta（mu @ W^T）が出力の大半を占め、その微小な不正確さが出力 DC を 42〜64% ずらす。
- 単層では正しく機能する（量子化誤差を低減）が、実モデル（100 層の蓄積）では一貫して悪化する。
- なお、この実装段階で**既定パスを壊す副作用**も発生した（下記「既定パス破壊」リスク参照）。
- **判断**: このモデル（Qwen-Image / NextDiT）では不採用。将来 DC が小さいモデルで再検討の余地はある。

### 改良 B: rotate + quantize 融合カーネル（速度狙い・本命）

**発想元**: SA3 `fp4_quantization_4d.cu`（1 カーネル完結の量子化）。

**現行 3 パス**:
```
[bf16 act] → matmul (rotate, gs=256 dense H) → グローバル書き戻し
           → _C.quantize_nvfp4 (読み直し→FP4) → グローバル書き戻し
           → cublas_gemm_blockwise_fp4
```

**融合後 1 パス**:
```
[bf16 act] → fused_rotquant_kernel:
   1. タイルを global から 1 回読む (bf16)
   2. gs=256 Hadamard butterfly を shared memory / register で適用
      (log4(256) = 4 段 × h4。zi_nvfp4_hadamard.rotate_last_dim_fast と同一数学)
   3. 16 要素ごとに amax → SF = amax/6 → e4m3
   4. e2m1 パック (cvt.rn.satfinite.e2m1x2)
   5. SF を 128×4 レイアウトで store（cuBLAS 互換。SA3 の offset 式を流用）
   → cublas_gemm_blockwise_fp4
```

- グローバルメモリ往復 2 回分（bf16 act 1 回 + FP4 書き出しは残るが読み直し消滅）と
  カーネル起動 1 回分を削減
- **Hadamard 行列の実体化を回避**（256×256 bf16 = 128KB → butterfly 定数だけ）
- **非 ConvRot 層用の第二カーネル**（rotate なし quantize のみ）も同じ拡張内に
  **別カーネルとして**用意（混ぜない。SA3 のカーネルがほぼそのまま使える）
- 実装形態: 新規 CUDA 拡張 `hswq_fused_rotquant`（`nodes/zimage_nvfp4/csrc/` または独立 dir）。
  Windows ビルドは SA3 の `setup.py` 実績を踏襲。MSVC SAL マクロ回避
  （`#undef __in/__out/__inout`）は **`.cu` ソース内**で実施
  （`api.cu` L20–25、`fp4_quantization_4d.cu` L19–24 の実績。`setup.py` 側ではない）
- butterfly の CUDA 化: `zi_nvfp4_hadamard._apply_kron_h4_unnorm` は再帰的
  `torch.matmul` を log4(256) = 4 段重ねる Python 実装。CUDA 版は **shared memory 上で
  4 段の in-place butterfly 演算**に変換する必要がある（4×4 Hadamard 定数のみ使用）
- 検証（2 段階で分離）:
  1. **SF レイアウト**: 融合カーネル vs 現行で **ビット完全一致必須**。
     cos で逃げると SF の 1-bit ずれによる silent corruption を見逃す
  2. **FP4 qdata**: まず butterfly 単体 vs dense GEMM で rotate 出力の cos を測定し、
     float 演算順序差が FP4 丸め境界に影響する度合いを定量化。その結果に基づき閾値決定
     （shape: M ∈ {1, 16, 128, 1024, 4096} × K/N 実モデル主要 shape）

**専用パス**: `HSWQ_FUSED_ROTQUANT=1` で有効化。失敗時は自動フォールバックではなく
**例外停止 + ログ**（静かに旧パスへ落ちて計測を汚染させない）。

**期待効果**: P1+P2+P4（前処理系）を一掃。eager path のまま CUDA Graph なしで
FP16 に対して安定して速くなる line を狙う。

### 改良 C: block-scale-only モード（calib 脱却の可否判定）

**発想元**: SA3 は per-tensor scale なしで e4m3 block scale のみ。

**設計**:
- `HSWQ_NVFP4_BLOCKONLY=1` 時: `ensure_act_scale` を常時 1.0 に固定、
  `alpha = scale_b`（weight 側のみ）。**per-tensor scale 計算・キャリブ依存を排除**
- **alpha キャッシュの無効化**: Z-Image 側は `alpha = scale_a * scale_b` を
  `module._hswq_nvfp4_alpha` にキャッシュして初回固定している。`scale_a = 1.0` に
  固定しても既存キャッシュがあればそちらが使われるため、**block-only 専用パスでは
  alpha キャッシュを使わず `alpha = scale_b` を毎回直接使用する**
- 対象は明示的に `_hswq_nvfp4_convrot` 層のみ（convrot 済みの分布は平滑なので
  block-only で足りる可能性）。非 convrot 層は現行のまま（分離）
- 評価: moodyRealMix 20 シード cosine。**nv 探索（nv100〜nv84）が不要になれば
  変換ワークフロー自体が簡略化される**

**期待効果**: 精度は現状維持のまま「calib 工数ゼロ」化。もしくは改良 A と組み合わせて
精度プラス。変更コストが最小（Python 数行）なので**最初に実施**。

**実測結果（2026-09-10）— 実装失敗、全 revert**

- 実装したが、既定パス（env なし）で既存 calib NVFP4 のスコアを **0.974 → 0.924 相当に悪化**させた。
- 全変更を revert（`8c04a1b` で 2015feb の状態に完全復元）。
- **教訓**: 「env なしなら従来パス」という設計だけでは不十分。実装直後に**既存スコアを再計測**し、
  非劣化を確認する（リグレッションテスト必須）。これを怠ると既存の正常動作を壊したまま気づかない。
- 再挑戦する場合の条件: 専用関数への完全分離（既存関数を書き換えない）＋ リグレッションテスト合格。

### 改良 D: SA3 attention の併用導入（足し算の加速）

- HSWQ NVFP4 は Linear 対象、attention は未介入 → **SA3 attention と積で効く**
- 現状 whl: `dist/sageattn3-1.0.0+cu132torch2.14.0-cp313/cp314-win_amd64.whl`
- **前提確認（実施前チェックリスト）**:
  - [ ] ComfyUI 環境の Python (3.13/3.14?) と torch バージョン (2.14?) 一致
  - [ ] CUDA **12.8 以上**のランタイム（setup.py の要件。13.2 限定ではない）
  - [ ] GPU が sm120/sm121（RTX 50 系）
  - [ ] Z-Image attention の head_dim が **64 または 128**
    （SA3 は head_dim 64/128 のみ実装。256 以上は `api.py` で SDPA fallback、
    `DISPATCH_HEAD_DIM` / `launch.h static_assert` に分岐なし）
- 統合形態: 既存 patch 系（`patches/` または nodes の attention 差し替え）に
  **専用ノード/フラグ**で SA3 を選択可能に。SDPA フォールバック条件
  （head_dim ≥ 256、非対応形状、GQA）も SA3 側制約に合わせる
- 精度: README により image gen はほぼロスレス。ただし Z-Image での実測 parity は必須

**実測結果（2026-09-10）— 不採用（品質不成立）**

チェックリストは全項目合格（sageattn3 whl 導入済み、sm121、head_dim 128、CUDA 13.2）。
実装はベンチ側に限定（既定パス不変）: `zi_convrot_nvfp4_traj_compare.py --attention sage3`、
`zi_int8_bench.py --attention sage3`（いずれも quant モデルのみ attention を override、
FP16 ベースラインは stock attention のまま）。

**SA3 単体テスト**（ランダム q/k/v、fp16、HND、head_dim 128、SDPA 比較）:

| 条件 | cos vs SDPA |
|---|---|
| seq=4096, per_block_mean=True | **0.98192** |
| seq=4096, per_block_mean=False | 0.98185 |
| seq=4128 / 1024 | 同程度（0.9817〜0.9821） |

→ **SA3（FP4 attention）は SDPA に対して常時 ~1.8% の固有誤差**（per_block_mean 無関係）

**実モデル検証**（FP16 ベースライン基準、ローカル 5060 Ti）:

| 組合せ | 指標 | 結果 |
|---|---|---|
| NVFP4 Linear（既存 nv100）+ SA3 | final-cos（12step, seed42） | **0.0556（破綻）** per_block_mean True/False 同様 |
| INT8 Linear（sci_1off）+ SA3 | latent-cos（25step, seed42） | **0.0909（破綻）** 推論 43.3s（INT8 単体 62.5s より 30% 速い） |
| INT8 Linear 単体（SA3 なし） | latent-cos | 0.9862（正常） |
| NVFP4 Linear 単体（SA3 なし） | final-cos | 0.987 / 0.984（正常） |

**原因**: SA3 の固有誤差（~1.8%/attention 層）が、量子化 Linear の誤差（NVFP4 ~2.6%、INT8 ~1.4%）
と加算され、多層 × 多ステップの蓄積で軌道が分岐する。SA3 の単体誤差自体が README の
「ほぼロスレス」と乖離している（この whl バージョン・5060 Ti 環境での実測）。

- **判断: SA3 併用は品質面で不成立。改良 D は不採用**
- 速度面は有効（INT8+SA3 で 30% 高速化）だが、品質が使い物にならないため採用不可
- `--attention` 実装は既定 sdpa（不使用）のまま両ベンチに残置。NVFP4 側（traj_compare）の
  実装は Owner 指示で撤去済み

### 改良 E（参考・非対象）: P の FP4 保持

softmax 直後の P を FP4 保持するのは attention 固有。Z-Image の MLP/Linear は
層間に非線形 (SiLU 等) が挟まり適用不可。**本計画では実装しない**（将来の
attention 自前実装時の素材）。

### 改良 F（保留判断）: cuBLAS → 自前 MMA

- SA3 の命令自体（mxf4nvf4 MMA）は cuBLAS blockwise FP4 も内部で同系を使う。
  SA3 が速いのは fusion 設計のため
- 小 M 問題は改良 B の前処理削減で改善見込み → **まず改良 B を測り、
  それでも GEMM 本体がボトルネックだった場合のみ再評価**

---

## 4. 実装フェーズ（順序は依存関係とコスト順）

### Phase 0: ベースライン計測（半日）

- [ ] 現行 eager path の kernel breakdown を取得（torch profiler / nsys）
  - 対象: rotate matmul / quantize_nvfp4 / cublas_gemm / その他、wall time 比率
  - **計測ワークフロー**: moodyRealMix TC W4A4、1024×1024、12 ステップ、1 シード
  - **nsys コマンド例**: `nsys profile --stats=true -t cuda,nvtx -o baseline -- python ...`
  - **FP16 baseline**: 同一ワークフローを FP16 で実行し、同一条件の nsys trace と
    wall time を取得。各カーネルの比率を表で並べる
- [ ] 精度ベースライン: moodyRealMix 現行最良 (nv90〜nv84 の成績) を再確認
- [ ] moodyProMix (nv100) をリグレッション基準として記録
- **完了条件**: 「改良前の数値」が表 1 枚にまとまること
  （精度: 20 シード cosine summary、速度: FP16 / NVFP4 各カーネル wall time 比率）

### Phase 1: 改良 C block-only 実験（半日、コード最小）

- [ ] `HSWQ_NVFP4_BLOCKONLY` env + convrot 層専用分岐（専用関数）
- [ ] **alpha キャッシュ無効化の実装確認**: 専用パスで `module._hswq_nvfp4_alpha`
  を使わず `alpha = scale_b` を直接使用していることを検証
- [ ] alpha キャッシュ有無の 2 条件で moodyRealMix 20 シード cosine 計測（全文出力）
  → step-0 amax 問題（Krea2 で発見済み）が Z-Image にも存在するか同時に調査
- [ ] 判定: 現行 calib ありの成績と同等（±0.005）なら calib 不要化オプションとして採択
- **完了条件**: 20 シード全文 + summary 出力、採択/棄却の明確な判定

### Phase 2: 改良 A mean shift プロトタイプ（1 日、Python レベル）

- [ ] rotate 後 128-token mean shift + `delta_y = W·μ^T` 加算を Python で実装
  （まず正しさ優先。融合はしない）
- [ ] parity: mean shift OFF と ON で FP16 は一致、NVFP4 は cosine 向上を確認
- [ ] moodyRealMix 20 シード計測 → **0.95 到達可否をここで判定**
- [ ] input_scale 干渉の確認（改良 C 併用の 2 条件で計測）
- **完了条件**: 20 シード全文 + summary。未達なら原因分析（分布ヒストグラム）まで
- ※ ここで効果が薄い場合、シフト位置（rotate 前/後）の 2 パターンのみ追加検証して
  打ち切り判断。無限に設定を増やさない

### Phase 3: 改良 B 融合カーネル（2〜3 日、CUDA）

- [ ] 拡張 `hswq_fused_rotquant` 雛形（setup.py は SA3 流用、MSVC 回避込み）
- [ ] カーネル 1: convrot 用 butterfly+quantize (gs=256)
- [ ] カーネル 2: quantize のみ（非 convrot 用・SA3 流用）
- [ ] unit test（2 段階検証）:
  1. SF レイアウト: 融合カーネル vs 現行 2 段 → **ビット完全一致必須**
  2. FP4 qdata: butterfly vs dense GEMM の rotate 出力 cos を先行測定後に閾値決定
  （shape: M ∈ {1, 16, 128, 1024, 4096} × K/N 実モデル主要 shape）
- [ ] 接続: `HSWQ_FUSED_ROTQUANT=1` で `_tc_forward_pooled` の前処理を差し替え
- [ ] ベンチ: Phase 0 と同一条件で wall time 比較（3 回平均）
- **完了条件**: unit test 全緑 + ベンチ表（FP16 / 現行 NVFP4 / 融合 NVFP4）
- ※ 鉄則: カーネル修正は実測（デバッガ/dump 比較）で原因確定後のみ。
  デバッグ用 dump は tools/ 配下スクリプトで実施し本番コードに埋め込まない
- ※ **再キャリブ注意**: calib 済み `input_scale` は dense GEMM path の精度特性前提。
  butterfly に切り替えると float 演算順序差で精度特性が変わるため、
  融合カーネル導入後は再キャリブの要否を精度計測で判定する

### Phase 4: 改良 D SA3 attention 統合（1 日）

- [ ] 前提チェックリスト（環境バージョン照合）を全て通す
- [ ] 専用ノード/フラグで Z-Image attention → `sageattn3_blackwell` 差し替え
- [ ] parity 画像生成 + wall time 計測（改良 A/B と独立に ON/OFF 計測）
- **完了条件**: 画像 parity 目視 + 数値（PSNR/cosine）記録 + ベンチ表追加

### Phase 5: 統合検証・確定（1 日）

- [ ] 採択した改良の組合せで最終 20 シード × 12 ステップ（全文 + summary + GEMM MODE）
- [ ] moodyProMix nv100 リグレッション（劣化なし確認）
- [ ] 実画像 workflow での最終 wall time
- [ ] 結果を MEMORY.md / memory/YYYY-MM-DD.md に記録、必要なら repo docs 更新
  （docs は一般論のみ、個別数字は参照セクション）

**合計所要目安: 6〜8 日**

---

## 5. 検証プロトコル（既存規約を遵守）

| 項目 | 手順 |
|---|---|
| 精度 | 20 シード × 12 ステップ TC(W4A4) cosine。**結果は全行 + summary + GEMM MODE を全文出力** |
| 速度 | 同一シード・同一設定で 3 回実行の中央値。FP16 baseline との比率で報告 |
| parity | `nvfp4_comfy_parity` 既存フロー。改良ごとに ON/OFF で実施 |
| **既存非劣化（リグレッション）** | **変更後、既存 calib NVFP4 の基準スコアを同一条件で再計測し、非劣化を確認してから新機能を評価する（2026-09-10 の失敗の直接原因への対策）** |
| シード | **正規シード（10桁 20個: 42,137,5517,92048,...）を使用**。独自シードでは mean が最大 0.05 変動する |
| 環境差 | ローカル（5060 Ti）とクラウド（5090 等）で mean が ±0.005 程度変動する。完全一致は求めず、0.97 台で同等と判定 |
| 評価対象の照合 | 計測前に artifact（= 正規 nv100 であるか）とシード（= 正規 10桁か）を照合する |
| 失敗時 | 未達の生成物は即削除（既存ルール）。FAIL の原因を 1 行でも記録してから削除 |

---

## 6. リスクと回避策

| リスク | 影響 | 回避策 |
|---|---|---|
| **既定パス（env なし）の破壊** ★最重要★ | 既存の正常動作が壊れ、スコアが 0.974 → 0.924 に低下（2026-09-10 実例） | 変更は追加のみ（既存コードを書き換えない）。実装直後に既存スコアの再計測で非劣化を確認 |
| mean shift が実モデルで破綻（DC 極大層） | 改良 A の不採用（2026-09-10 実例） | 実装前に層ごとの mu/x 比を実測。DC が極大の層があるモデルでは適用しない |
| 誤った artifact / シードでの評価 | 無効な比較で誤結論（2026-09-10 実例） | 計測前に artifact とシードを正規のものと照合（§5） |
| SF レイアウト実装ミス（融合カーネル） | silent corruption（画質劣化が parity を通る） | unit test でビット一致比較。FAIL 時は dump 比較ツール（tools/ 配下）で特定 |
| mean shift の group 構造が画像統計と不整合 | 改良 A 効果なし | Phase 2 で早期打ち切り判断。分布ヒストグラムを保存 |
| torch/CUDA バージョン不一致（SA3 whl） | 改良 D 不発 | Phase 4 冒頭のチェックリストで全遮断。不一致ならソース再ビルド（build 実績スクリプトあり） |
| comfy_kitchen `_C` private API 変更 | 融合カーネル接続点が壊れる | 接続は 1 関数 (`_tc_forward_pooled`) に限定。バージョンを setup で pin |
| 融合カーネルが環境依存で動かない | Phase 3 全滅 | フラグ OFF の既定パスは完全現状維持（フォールバックで隠さず例外で止める） |
| Windows MSVC + CUDA ビルド問題 | ビルド不通 | SA3 の `setup.py` / `build_*.cmd` 実績（SAL マクロ回避は `.cu` ソース内で実施）を踏襲 |
| step-0 amax 問題の Z-Image 横展開 | alpha キャッシュ経由で Krea2 と同じ mis-scale が発生し精度劣化 | Phase 1 で alpha キャッシュ有無の 2 条件比較を追加。Z-Image でも同様の劣化が出るか調査 |
| butterfly 切替後の再キャリブ必要性 | calib 済み `input_scale` が dense GEMM 前提で算出されており、butterfly path での精度特性と不一致 | 改良 B 導入後に精度計測で再キャリブ要否を判定。必要なら butterfly path 用の再キャリブを Phase 3 に追加 |
| CUDA 拡張 `hswq_fused_rotquant` の配布方法 | MSVC + CUDA ビルド環境を持たないユーザーが利用不可 | Phase 3 に pre-built whl のビルド・配布手順を追加（build-windows-whl / build-linux-whl スキル活用） |

---

## 7. 対象ファイルマップ

### 読む（参照実装）
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention3_blackwell\sageattn3\api.py` — 前処理・量子化呼び出し
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention3_blackwell\csrc\quantization\fp4_quantization_4d.cu` — 融合量子化カーネル（改良 B のベース）
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention3_blackwell\csrc\blackwell\mainloop_tma_ws.h` — delta_s / P 量子化
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention3_blackwell\setup.py` — Windows ビルド実績

### 変更する（HSWQ 側）
- `nodes/zimage_nvfp4/zi_nvfp4_forward.py` — 改良 A/C の専用パス、改良 B 接続点
- `nodes/nvfp4/nvfp4_runtime.py` — 改良 C の scale 固定（convrot 専用分岐）
- `nodes/zimage_nvfp4/zi_nvfp4_hadamard.py` — butterfly 実装（カーネル側へ移植時の参照）
- 新規: `hswq_fused_rotquant/`（CUDA 拡張）または `nodes/zimage_nvfp4/csrc/`
- 改良 D: attention 差し替え用の専用 patch/nodes（既存 patch 構成に倣う）

### 変更しない
- `nodes/nvfp4/nvfp4_forward.py`（SDXL 側）— Z-Image と分離運用（混ぜない）
- 既存 parity / bench スクリプト（テスト対象であり変更しない）

---

## 8. 判断の履歴メモ（この計画の根拠）

- 改良 B を「本命」、改良 A を「精度の本命」と位置付ける根拠:
  SA3 の速度優位は fusion と smoothing の設計に由来し、命令自体は cuBLAS と同系
  （→ 改良 F を保留とする理由も同じ）
- per-block mean は回転と直交するため、ConvRot 実装済みの HSWQ に素直に足せる
  という構造的判断
- 改良 C を最初にやる理由: 変更コスト最小かつ、改良 A の scale 干渉評価に必要な
  基礎データになるため
- Z-Image の非線形層があるため SA3 の P-FP4 は転用不可（改良 E として明示的に範囲外）
