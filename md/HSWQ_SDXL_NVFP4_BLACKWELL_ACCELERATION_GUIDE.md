# HSWQ SDXL ConvRot NVFP4 Blackwell Tensor Core 高速化技術ガイド

## 概要

NVIDIA Blackwell アーキテクチャ (SM >= 100: B200 / GB200, RTX 5090 / SM120) における HSWQ SDXL ConvRot NVFP4 の推論パフォーマンスを最大化するため、**Per-Weight CUDA Graph 自動ディスパッチ機構 (Tensor Boost)** を導入した。

本変更は **`nodes/nvfp4/`（SDXL Product Tensor Core パス）内のみ** で閉じており、Z Image ConvRot NVFP4（`nodes/zimage_nvfp4/` comfy-parity パス）、SDXL ConvRot INT8、FP8、標準 FP16/BF16 パスには一切影響を与えない完全分離設計となっている。

---

## ログ出力・状態確認 (Tensor Boost ログ)

Tensor Boost (Blackwell Per-Weight CUDA Graph) の動作状況は、コンソールおよびログ出力で確認可能。

### 1. チェックポイントロード時
モデルロード時に Blackwell GPU (SM >= 100) を検出すると、以下のログが出力される:
```text
[HSWQ NVFP4 Tensor Boost] Blackwell GPU (SM >= 100) DETECTED: Per-Weight CUDA Graph Tensor Boost ACTIVE
```
（※ 非 Blackwell 環境の場合は `Standard SDXL NVFP4 Product path ACTIVE` が出力される）

### 2. キャプチャ実行時 (初回の順伝播時)
各 Linear レイヤーの重み用 CUDA Graph がキャプチャされる際に、レイヤー形状とともにインフォメーションログが出力される:
```text
[HSWQ NVFP4 Tensor Boost] Captured Blackwell per-weight CUDA Graph #1 (shape M=512 K=2048 N=2048, w_ptr=0x..., device=cuda:0)
```

### 3. ベンチマーク・統計情報 (`nvfp4_forward_stats()`)
`nvfp4_forward_stats()` の戻り値辞書に以下のフィールドが追加され、Blackwell のヒット数およびアクティブ状態を確認可能:
- `"blackwell_graph_hits"`: Blackwell Per-Weight CUDA Graph リプレイヒット回数
- `"blackwell_tensor_boost_active"`: Blackwell Tensor Boost の有効判定フラグ (`True` / `False`)

---

## 課題とアーキテクチャ背景

### 従来の CUDA Graph (Shape-Shared) の課題
従来の SDXL NVFP4 スタックにおける CUDA Graph は形状共有型 (`_GRAPH_CACHE`) であり、毎呼び出し時に重みテンソル全体を `static_w.copy_(w_qdata)` でコピーする実装であった。
この重みコピーオーバーヘッドにより、CUDA Graph 使用時（13.05秒）が eager pooled パス（~11.8秒）よりも低速化するため、`HSWQ_NVFP4_CUDAGRAPH=1` の明示的 opt-in 設定を必須としていた。

### Blackwell (SM100 / SM120) における最適化戦略
Blackwell では FP4 (E2M1) Tensor Core の演算能力（RTX 5090 で約 3,352 TOPS）が著しく向上したため、ホスト側・PyTorch オーバーヘッドおよび重み転送コピーがボトルネックの大部分を占める。
サンプリング中のモデル重みのアドレス (`data_ptr`) は VRAM 上で一定であるため、重みテンソルをコピーせず**直接グラフ内にキャプチャ**する `nvfp4_quant_mm_cudagraph_perweight` を新設した。

---

## 実装仕様

### 1. Blackwell GPU 自動検出 (`nodes/nvfp4/nvfp4_conf.py`)

```python
_GPU_CC: tuple | None = None

def _get_gpu_cc() -> tuple:
    global _GPU_CC
    if _GPU_CC is None:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            _GPU_CC = torch.cuda.get_device_capability()
        else:
            _GPU_CC = (0, 0)
    return _GPU_CC

def is_blackwell_gpu() -> bool:
    """True if GPU is Blackwell class (SM >= 100): B200, RTX 5090, etc."""
    major, _ = _get_gpu_cc()
    return major >= 10
```

### 2. Per-Weight CUDA Graph 機構 (`nodes/nvfp4/nvfp4_runtime.py`)

- **キャッシュキー**: `(w_ptr, m, k, n, str(out_dtype), bool(pad_16x), has_bias, int(orig_n))`
  - 重みの `data_ptr()` をキーに含めることで、重みごとのグラフエントリを個別保持 (`_PER_WEIGHT_GRAPH_CACHE_MAX = 160`)。
- **キャプチャ動作**:
  - `w_qdata` をコピーせず、キャプチャ内から直接参照。
- **リプレイ動作**:
  - 毎リプレイ時にはアクティベーション `x` およびスケール (`scale_a`, `alpha`, `bias`) のみコピー。重みのコピー処理を 0 に削減。

```python
def nvfp4_quant_mm_cudagraph_perweight(
    x, *, w_qdata, weight_scale, block_scale_w, scale_a, bias, out_dtype, alpha, pad_16x: bool, orig_n: int,
):
    ...
```

### 3. 自動ディスパッチガード (`nodes/nvfp4/nvfp4_forward.py`)

`_tc_forward_pooled` 内において、Blackwell 環境検出時に自動で Per-Weight CUDA Graph パスへ分岐する。

```python
_bw = is_blackwell_gpu()
if (
    _bw
    and orig_m <= _GRAPH_MAX_M
    and not getattr(module, "_hswq_nvfp4_no_cudagraph", False)
):
    try:
        result = nvfp4_quant_mm_cudagraph_perweight(...)
        _TC_HITS += 1
        _BLACKWELL_GRAPH_HITS += 1
        return result
    except torch.cuda.OutOfMemoryError:
        clear_nvfp4_cudagraphs()
        torch.cuda.empty_cache()
    ...
```

---

## パス分離と安全性の担保

| パス | モジュールフラグ | Blackwell 最適化の適用 |
|:---|:---|:---|
| **SDXL ConvRot NVFP4** | `module._hswq_nvfp4 = True` | ✅ 自動適用 |
| **Z Image ConvRot NVFP4** | `module._hswq_nvfp4_parity = True` (`_hswq_nvfp4 = False`) | ❌ 完全非適用 (Comfy Parity パス) |
| **SDXL ConvRot INT8** | ComfyUI MixedPrecision / INT8 Ops | ❌ 完全非適用 |
| **FP8 / Native FP16** | Stock ComfyUI Ops | ❌ 完全非適用 |

Z Image NVFP4 や INT8 のロード・推論コードには一切変更を加えておらず、SDXL ConvRot NVFP4 のテンソルコアプロダクトパス内でのみ動作するため、相互干渉や品質低下のリスクは存在しない。
