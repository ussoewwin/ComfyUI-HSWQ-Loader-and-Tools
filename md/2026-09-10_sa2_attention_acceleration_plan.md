# SA2 (SageAttention 2) 転用計画書 — HSWQ 量子化 Linear との併用による推論加速

- 作成日: 2026-09-10（v2: **Phase 0 / Phase 1 実測結果を反映**）
- 状態: **Phase 0・Phase 1 完了 → 品質・速度とも基準クリア（Phase 2 の 20 シード統計待ち）**
- 対象: HSWQ NVFP4 / INT8 Linear + SA2 attention の併用加速
- 情報源: `D:\USERFILES\fp8e4m3\SageAttention\sageattention\` ソース実読 + インストール済みパッケージ実測 + 実モデル計測
- 前身: `2026-09-09_sa3_nvfp4_hswq_acceleration_plan.md`（SA3 は改良 A/C/D 全滅、B 未着手）
- 参考: `implementation_plan.md`（SA2 計画ドラフト。本計画はこれを土台に実測値で再構成したもの）

### 改訂履歴

| 版 | 内容 |
|---|---|
| v1 | 計画立案（SA2 コード実読・Phase 0 の一部のみ実測） |
| v2 | Phase 0 完了（実モデル per-call 誤差）、Phase 1 完了（12 ステップ軌道・速度）、ハーネスのレイアウト必須要件を追記 |

---

## 0. 背景 — SA3 全滅と SA2 の位置づけ

### 0.1 前計画（SA3）の実測結果

| 改良 | 結果 | 原因 |
|---|---|---|
| A (mean shift, Linear) | **不採用** | 実モデルの DC 極大層で delta_y が出力 DC を 42〜64% ずらす（seed42: 0.615 vs OFF 0.984） |
| C (block-only) | **実装失敗** | 既定パスを破壊（既存 0.974 → 0.924 相当）→ 全 revert |
| D (SA3 attention) | **不採用** | SA3 FP4 attention の固有誤差が多層蓄積（NVFP4+SA3 = 0.0556、INT8+SA3 = 0.0909） |
| B (rotate+quantize 融合 / butterfly) | **未着手** | 独立して有効。本計画と並行可能 |

**Phase 0 プロファイル実測**（1024×1024、6step、5060 Ti）: softmax/attention 26.4% / rotate 16.9% / cuBLAS FP4 GEMM 19.6% / quantize 1.8%。**attention が最大の単一ボトルネック**であり、SA2 が効けば最大の速度改善になる。

### 0.2 SA3 vs SA2 の構造差

| 項目 | SA3 (FP4 attention) | SA2 (INT8/FP8 attention) |
|---|---|---|
| Q/K 量子化 | FP4 e2m1（仮数 1bit = 8 値） | **INT8**（256 値、per-warp/per-block/per-thread） |
| V 量子化 | FP4 e2m1 | **FP8 E4M3**（仮数 3bit = 16 値） |
| P (softmax 出力) | FP4 e2m1 | FP16 → FP8 E4M3（`S_FP8_OFFSET=8.807` で精度維持） |
| GQA/MQA | **非対応** | **対応**（`nqheads // nkheads` 自動展開） |
| head_dim | 64/128 のみ | **64/128/256**（パディング対応、>256 は ValueError） |
| Blackwell | sm120/121 のみ | **sm100/120/121 対応**（SM89 カーネルで動作） |

### 0.3 実測結果（核心・すべて実測値）

**Z-Image (moodyRealMix_xhsEdition) / 1024×1024 / euler+simple / cfg 2.5 / RTX 5060 Ti 16GB**

| # | 計測 | 結果 |
|---|---|---|
| 1 | SA2 単体（ランダム q/k/v, seq 4096, head_dim 128, fp16） | SDPA 比較 cos **0.999258**（SA3 は 0.98192） |
| 2 | **実モデル per-call 誤差**（NVFP4 モデル, 34 calls, bf16） | cos **0.99970〜0.99999** / max-abs 相対誤差 **1.1〜5.5%** |
| 3 | **FP16 モデル 12 ステップ**（stock vs SA2、量子化なし） | final x0 cos **0.998427**（per-step 最小 0.999046、bifurcation なし） |
| 4 | **NVFP4 モデル 12 ステップ**（同一モデル内 stock vs SA2） | final x0 cos **0.985622**（per-step 1.000 → 0.988 で滑らかに推移） |
| 5 | **NVFP4 + SA2 vs FP16 基準**（12 step, seed42, TC W4A4） | final-cos **0.98576**（NVFP4 単体 = 0.98700 → 差 **-0.0012**、same-image 判定） |
| 6 | **速度（NVFP4 モデル 12 ステップ、同一プロセス）** | stock **13.68 s** → SA2 **11.57 s** = **-15.4%**（2.11 s 短縮、0.176 s/step） |
| 7 | 速度（FP16 モデル 12 ステップ） | 34.13 s → 29.34 s = **-14.0%** |
| 8 | attention 単体速度（1 forward = 34 calls） | SDPA 189.7 ms → SA2 82.7 ms = **2.29x** |
| 9 | SA2 適用率（12 ステップ通算） | **816 / 816 calls（100%）**、フォールバック 0、エラー 0 |
| 10 | 量子化 GEMM への影響 | **GEMM MODE = TC (W4A4) 維持**、dequant_fallbacks = 0 |

**結論**: SA2 は **attention を 2.29x 高速化し、12 ステップ全体で -15.4%**。品質劣化は **cos -0.0012**（ローカル/クラウドの環境差 ±0.005 の範囲内）で、軌道は bifurcate しない（same-image）。**採用可**。

### 0.4 ハーネス実装の必須要件と、対照実験の教訓（重要）

v1 のハーネス（SA3 版テンプレートの流用）には **レイアウト変換のバグ**があり、12 ステップ軌道が崩壊した（final-cos 0.086、step10 で bifurcate）。**SA2 の品質問題と誤認しかけた。**

| 事実 | 内容 |
|---|---|
| stock (`attention_pytorch`) の入出力 | 入力 `[B,H,N,D]`（skip_reshape=True）→ 出力 **`transpose(1,2)` して `[B,N,H*D]`** |
| バグ版ハーネス | `[B,H,N,D]` を **transpose せず** `reshape(B,-1,H*D)` → head/token の順序が崩壊 |
| 発見の決め手 | **対照実験**: override 内で sageattn の代わりに `F.sdpa` を呼ぶ（数式は stock と同等）→ **同じく 0.085 に崩壊** = 原因は SA2 ではなくハーネス |
| 修正後 | 同対照実験で **per-step cos ≈ 1.000（final 0.999968）= HARNESS CLEAN** |

**教訓（今後の必須手順）**:
1. attention 差し替えを実装したら、**必ず「同一数式を返す対照実験」でハーネス自体を検証**してから性能・品質を評価する。
2. `transpose` を伴うレイアウト変換は、**stock 実装（`attention_pytorch`）の出力整形をそのまま写す**。

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

実測: `sageattention.core._cuda_archs = ['sm120']`、`torch.version.cuda = 13.2` → **SA2++ パス（fp32+fp16, per_warp）が選択される**。

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
- GQA では `repeat_interleave` で Q head 数に展開。`quant_per_block_int8_fuse_sub_mean_cuda` で mean 減算と量子化を 1 カーネルに融合
- **SA3 の mean shift（改良 A）と同概念だが attention 内部に閉じており、Linear 前処理には適用しない**（= DC 極大層の問題を構造的に回避）
- 実測では `smooth_k=True/False` の per-call cos 差はほぼ無し（0.999993 vs 0.999993、本モデルでは効かない可能性）

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

### 1.7 実モデルの呼び出し条件（実測）

| 項目 | 値 |
|---|---|
| attention 呼び出し元 | `comfy.ldm.lumina.model.JointAttention` → `optimized_attention_masked(..., skip_reshape=True, transformer_options=...)` |
| active backend | `attention_pytorch`（sage/flash/xformers 無効、pytorch 有効） |
| mask | **全 34 calls で None**（12 ステップ通算 816 calls すべて None） |
| 形状 / dtype | 30 calls `(1,30,4128,128)`、2 calls `(1,30,4096,128)`、2 calls `(1,30,32,128)` / fp16（FP16 モデル）・bf16（NVFP4 モデル） |
| q/k/v amax（代表） | 5.4〜10.9 / 6.6〜10.9 / 68〜478 |

### 1.8 ビルド資産

- ビルド済み whl: `sageattention-2.2.0+cu132torch2.12.0`（cp312/cp313）
- MSVC SAL 回避: `fused.cu` 冒頭で `#undef __in/__out/__inout`（SA3 と同方式）
- CUDA 要件: SM89 ≥ 12.4 / SM90 ≥ 12.3 / **SM120 ≥ 12.8**

---

## 2. 検証計画

### Phase 0: SA2 単体精度・ディスパッチ確認 — **完了**

- [x] パッケージ確認: `sageattention 2.2.0.post6+cu132torch2.14.0` / `_qattn_sm89` 存在
- [x] arch ディスパッチ確認: `sm120` → `sageattn_qk_int8_pv_fp8_cuda(per_warp, fp32+fp16)`
- [x] 単体精度: ランダム q/k/v → cos 0.999258（seq 4096） / 0.999254（4128）
- [x] 実モデル per-call 誤差: cos 0.99970〜0.99999、max-abs 相対誤差 1.1〜5.5%
- [ ] 追加条件（bf16 単体 / head_dim 64・256 / `pv_accum_dtype="fp32+fp32"` 比較）— 未実施（採用判断には不要）

**判定基準**: cos ≥ 0.999 → **クリア**

### Phase 1: 実モデル検証 — 量子化 Linear + SA2 attention — **完了**

- [x] **NVFP4 Linear (nv100) + SA2**: 12 step × seed42 TC(W4A4) → final-cos **0.98576**（基準 0.98700、差 -0.0012、same-image）
- [x] **FP16 + SA2**（量子化なし）: 12 step → final x0 cos **0.998427**（SA2 固有の軌道影響は 0.16%）
- [x] **速度計測**: NVFP4 12 step 13.68 s → 11.57 s（**-15.4%**） / FP16 12 step（**-14.0%**）
- [x] 適用率: 816/816 calls（100%）、フォールバック 0、GEMM MODE = TC (W4A4) 維持
- [ ] INT8 Linear + SA2（`zi_int8_bench.py --attention sage2`）— 未実施（INT8 側の追補）

**判定基準**: final-cos ≥ 0.96（NVFP4 単体から劣化 ≤ 0.015）→ **クリア（実劣化 0.0012）**

### Phase 2: 20 シード統計検証（採用確定のため、1 日）

- [ ] 正規 10 桁シード 20 個 × 12 ステップで cosine mean/std/95%CI
- [ ] NVFP4 単体（SA2 OFF）との同時計測によるリグレッション
- [ ] 最終ベンチ表（FP16 / NVFP4 / NVFP4+SA2 + 速度）
- [ ] INT8 + SA2 の追補計測
- [ ] SDPA フォールバック条件の整理（head_dim > 256 / mask あり / import 失敗）

**完了条件**: 20 シード summary + 最終ベンチ表

---

## 3. 実装設計

**設計原則（SA3 の失敗 + 本 Phase 1 のハーネスバグから確立した鉄則）**
1. **既定パス（env／フラグなし）の動作を一切変えない**
2. 実装直後に既存スコア（既定パス）を再計測し、非劣化を確認してから新機能を評価
3. 変更は追加のみ（既存コードの書き換え禁止）。専用フラグで明示的に有効化
4. **同一数式を返す対照実験でハーネス自体を先に検証する**（SA2 の品質とハーネスのバグを混同しない）

### 3.1 統合方式（実装済み）

ベンチスクリプト `benchmark/zi_convrot_nvfp4_traj_compare.py` に `--attention {sdpa,sage2}` を追加。**既定は sdpa（従来動作そのまま）**。実装は関数 `apply_sage2_attention()` / `print_sage2_attn_stats()` として**SA3 とは完全に分離**（混在なし）。FP16 ベースラインは stock attention のままで、量子化モデルのみ差し替える。

### 3.2 レイアウト変換（必須仕様・実測で確定）

```python
# 入力正規化（attention_pytorch と同一の意味論）
if skip_reshape:                      # q,k,v = [B,H,N,D]
    b, _, _, dim_head = q.shape
else:                                 # q,k,v = [B,N,H,D] -> transpose
    b, n, _ = q.shape; dim_head = q.shape[-1] // heads
    qh = q.view(b, n, heads, dim_head).transpose(1, 2)   # k,v も同様

out = sageattn(qh, kh, vh, tensor_layout="HND", is_causal=False)   # [B,H,N,D]

# 出力整形（★ここを誤ると軌道が崩壊する）
if skip_output_reshape:
    return out                                   # [B,H,N,D]
return out.transpose(1, 2).reshape(b, -1, heads * dim_head)   # [B,N,H*D]
```

### 3.3 SDPA フォールバック条件（実装済み）

| 条件 | 動作 |
|---|---|
| `mask is not None` | SDPA（本モデルでは発生しない：816/816 で None） |
| `head_dim > 256` | SDPA（SA2 が ValueError） |
| `import sageattention` 失敗 | SDPA |
| 実行時例外 | SDPA（reason をログ出力） |

---

## 4. 検証プロトコル（SA3 計画書から継承 + 教訓）

| 項目 | 手順 |
|---|---|
| 精度 | 20 シード × 12 ステップ TC(W4A4) cosine。**全行 + summary + GEMM MODE を全文出力** |
| 速度 | 同一シード・同一設定で 3 回実行の中央値。FP16 baseline との比率 |
| parity | `nvfp4_comfy_parity` 既存フロー。SA2 ON/OFF で実施 |
| **既存非劣化（リグレッション）** | **変更後、既定パス（sdpa）のスコアを再計測し非劣化を確認**（実施済み: 0.98700 で従来値と一致） |
| **ハーネス検証** | **override 内で SDPA を呼ぶ対照実験を先に実施**（per-step cos ≈ 1 を確認） |
| シード | **正規シード（10 桁 20 個）を使用** |
| 環境差 | ローカル（5060 Ti）とクラウドで mean ±0.005 変動。**0.97 台で同等と判定** |
| 評価対象の照合 | 計測前に artifact（正規品か）とシード（正規か）を照合 |
| 失敗時 | 未達の生成物は即削除。FAIL 原因を 1 行記録してから削除 |

---

## 5. リスクと回避策

| リスク | 影響 | 回避策 |
|---|---|---|
| **レイアウト変換ミス**（v1 で実際に発生） | 軌道崩壊を「SA2 の品質問題」と誤認 | 対照実験（override 内 SDPA）で per-step cos ≈ 1 を確認してから評価 |
| 多層蓄積での劣化 | SA3 と同じ破綻 | 実測済み: FP16+SA2 で 0.9984、NVFP4+SA2 で 0.98576（bifurcation なし） |
| Blackwell でのカーネル動作不良 | 未知のバグ | 実測済み: 816/816 正常、エラー 0 |
| fp16 per_warp カーネルが sm120 に無い | fp16 経路が使えない | fp8（SA2++）パスを使用。実測で問題なし |
| 20 シードで mean が基準未達 | 採用不可 | Phase 2 で判定（単体 -0.0012 なので可能性は低い） |
| INT8 Linear + SA2 の未計測 | INT8 経路の可否不明 | Phase 2 で追補計測 |
| 既定パスの破壊 | スコア劣化（SA3 で実例） | `--attention` フラグ制御・既定 sdpa・実装後の既存スコア再計測（実施済み） |
| 速度改善が僅少 | 導入意義が薄い | 実測 -15.4%（NVFP4 12 step）。基準（≤10% なら不採用）をクリア |

---

## 6. 対象ファイルマップ

### 読む（参照実装）
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention\core.py` — ディスパッチ・attention 実装（L171-178 が Blackwell 分岐）
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention\quant.py` — INT8/FP8 量子化・fused 操作
- `D:\USERFILES\fp8e4m3\SageAttention\sageattention\sm89_compile.py` — Blackwell バインディング
- `D:\USERFILES\fp8e4m3\SageAttention\csrc\qattn\` — SM89/Blackwell カーネル
- `D:\USERFILES\GitHub\hswq\ComfyUI-master\comfy\ldm\modules\attention.py` — `attention_pytorch`（出力整形の基準）
- `D:\USERFILES\GitHub\hswq\ComfyUI-master\comfy\ldm\lumina\model.py` — Z-Image の attention 呼び出し元

### 変更（追加のみ・実装済み）
- `benchmark/zi_convrot_nvfp4_traj_compare.py` — `--attention {sdpa,sage2}` / `apply_sage2_attention()` / `print_sage2_attn_stats()`（+98 行、削除 0）

### 未実施（必要時に追加）
- `benchmark/zi_int8_bench.py` — INT8 経路への `--attention sage2` 追加（Phase 2 で追補）

### 変更しない
- HSWQ Linear forward（NVFP4 / INT8 パス）— 一切触らない
- 既存 parity / bench スクリプトの既定動作
- `ComfyUI-master` 側（モンキーパッチはベンチスクリプト内で完結）

---

## 7. SA3 計画書との関係

本計画は SA3 計画書（`2026-09-09_sa3_nvfp4_hswq_acceleration_plan.md`）の**改良 D の後継**。

| SA3 計画の改良 | 状態 | 本計画との関係 |
|---|---|---|
| A (mean shift) | 不採用 | SA2 では attention 内の `smooth_k` として同等機能が内蔵（Linear には適用しない） |
| B (融合カーネル / butterfly) | 未着手 | **独立して有効**。SA2 と併用可能（Linear 側の速度改善） |
| C (block-only) | 実装失敗 | SA2 とは無関係 |
| D (SA3 attention) | 不採用 | **本計画で SA2 attention に置換。Phase 1 で採用基準クリア** |
