---
license: apache-2.0
tags:
- controlnet
- text-to-image
- image-to-image
- inpainting
- qwen-image
- qwen-image-2512
- quantized
- int8
- convrot
- comfyui
pipeline_tag: image-to-image
---

# ControlNet Models (ConvRot INT8)

High-fidelity **ConvRot INT8** quantized weights for ControlNet models.

---

## 🌟 Model Overview

This repository hosts the **ConvRot INT8** quantized weights of **`Qwen-Image-2512-Fun-Controlnet-Union-2602`**, based on [alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union).

- **Base Model:** [alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union)
- **Supported Conditions:** Canny, HED, Depth, Pose, MLSD, Scribble, Gray, Inpainting
- **Quantization:** ConvRot INT8
- **License:** Apache-2.0

---

## 📦 Available Models

| Filename | Base Model | Supported Conditions | Precision | File Size | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen-Image-2512-Fun-Controlnet-Union-2602_convrot_int8.safetensors` | Qwen-Image-2512-Fun-Controlnet-Union-2602 | Canny, HED, Depth, Pose, MLSD, Scribble, Gray, Inpaint | ConvRot INT8 | ~1.64 GB | Apache-2.0 |

---

## 🛠️ Model Features

- Multi-condition ControlNet Union model (added on 5 layer blocks of Qwen-Image-2512).
- Supported control conditions: Canny, HED, Depth, Pose, MLSD, Scribble, Gray, and Inpainting.
- Quantized to **ConvRot INT8** to reduce VRAM and disk footprint while maintaining structural control fidelity.

---

## 🚀 Usage in ComfyUI

ComfyUI does not natively support the **ConvRot INT8** format for ControlNet models. To load and execute these weights in ComfyUI, the dedicated loader extension is required:

- **Dedicated Loader Extension:** [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)

### Installation

Clone the repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools.git
```

Place `Qwen-Image-2512-Fun-Controlnet-Union-2602_convrot_int8.safetensors` into your ComfyUI `models/controlnet/` directory and load it using the dedicated loader node.

---

## 📜 Credits & License

- **Original Base Model:** [alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/Qwen-Image-2512-Fun-Controlnet-Union)
- **Upstream Repository:** [aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)
- **License:** Apache-2.0
