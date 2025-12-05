# Step-Audio-EditX with GPU Memory Management

[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

<p align="center">
  <img src="assets/logo.png" height=100>
</p>

<div align="center">
  <a href="https://stepaudiollm.github.io/step-audio-editx/"><img src="https://img.shields.io/static/v1?label=Demo%20Page&message=Web&color=green"></a>
  <a href="https://arxiv.org/abs/2511.03601"><img src="https://img.shields.io/static/v1?label=Tech%20Report&message=Arxiv&color=red"></a>
  <a href="https://huggingface.co/stepfun-ai/Step-Audio-EditX"><img src="https://img.shields.io/static/v1?label=Model&message=HuggingFace&color=yellow"></a>
  <a href="https://github.com/neosun100/Step-Audio-EditX-GPU-Managed"><img src="https://img.shields.io/github/stars/neosun100/Step-Audio-EditX-GPU-Managed?style=social"></a>
</div>

## 📖 Overview

**Step-Audio-EditX** is a powerful 3B parameter LLM-based reinforcement learning audio model for expressive and iterative audio editing, enhanced with **intelligent GPU memory management** that reduces idle memory usage by **99.99%**.

### ✨ Key Features

- 🎯 **Zero-shot Voice Cloning** - Clone any voice with just 3-10 seconds of audio
- 🎭 **Emotion & Style Editing** - Support for dozens of emotions and speaking styles
- 🗣️ **Paralinguistic Control** - Add breathing, laughter, sighs, and more
- 🌍 **Multi-language Support** - Chinese, English, Sichuanese, Cantonese
- 🎮 **GPU Memory Management** - Reduce idle GPU memory from 40GB to 3MB (99.99% savings)
- ⚡ **Lazy Loading** - Models load on-demand, startup in 20 seconds
- 🔄 **Auto Offload** - Automatic GPU↔CPU transfer after task completion

### 📊 GPU Memory Comparison

| Status | Traditional | GPU Managed | Savings |
|--------|------------|-------------|---------|
| **Startup** | 40 GB | **3 MB** | **99.99%** ✨ |
| **In Use** | 40 GB | 40 GB | 0% |
| **After Task** | 40 GB | **5.7 GB*** | **85.75%** 🎉 |

\* *Note: 5.7GB is ONNX Runtime residual memory (known limitation), but still saves 85% compared to traditional approach.*

## 🖼️ Screenshot

![Step-Audio-EditX Interface](assets/screenshot.png)

## 🚀 Quick Start

### Prerequisites

- **GPU**: NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- **OS**: Linux (Ubuntu 22.04+ recommended)
- **Docker**: Docker >= 20.10 with nvidia-docker2
- **Models**: Download from [HuggingFace](https://huggingface.co/stepfun-ai/Step-Audio-EditX) or [ModelScope](https://modelscope.cn/models/stepfun-ai/Step-Audio-EditX)

### Option 1: Docker (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/neosun100/Step-Audio-EditX-GPU-Managed.git
cd Step-Audio-EditX-GPU-Managed

# 2. Download models
mkdir -p models && cd models
git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer
git clone https://huggingface.co/stepfun-ai/Step-Audio-EditX
cd ..

# 3. Build Docker image
docker build -t step-audio-editx:latest .

# 4. Run with GPU management (edit script first)
./start_with_gpu_management.sh
```

### Option 2: Direct Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with GPU management
python app.py \
  --model-path ./models \
  --enable-gpu-management \
  --gpu-idle-timeout 600 \
  --enable-auto-transcribe \
  --server-port 7860
```

### Access

- **Web UI**: http://localhost:7860
- **Startup Time**: ~20 seconds (lazy loading)
- **First Request**: +20-30 seconds (model loading)

## 📦 Installation

### System Requirements

- **GPU Memory**: 16GB+ (minimum 12GB)
- **RAM**: 32GB+ recommended
- **Disk Space**: 50GB+ (for models)
- **CUDA**: 12.1+ with cuDNN 8

### Docker Installation

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Model Download

#### From HuggingFace

```bash
mkdir -p models && cd models
git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer
git clone https://huggingface.co/stepfun-ai/Step-Audio-EditX
```

#### From ModelScope (China)

```bash
mkdir -p models && cd models
git clone https://www.modelscope.cn/stepfun-ai/Step-Audio-Tokenizer.git Step-Audio-Tokenizer
git clone https://www.modelscope.cn/stepfun-ai/Step-Audio-EditX.git Step-Audio-EditX
```

## ⚙️ Configuration

### Environment Variables

```bash
# GPU Configuration
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: Thread Configuration
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### Command Line Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--model-path` | Path to models directory | - | ✅ |
| `--enable-gpu-management` | Enable GPU memory management | `False` | ❌ |
| `--gpu-idle-timeout` | Idle timeout in seconds | `600` | ❌ |
| `--enable-auto-transcribe` | Enable Whisper transcription | `False` | ❌ |
| `--server-port` | Web UI port | `7860` | ❌ |

### Docker Configuration

Edit `start_with_gpu_management.sh`:

```bash
PROJECT_DIR="/your/project/path"  # Project root directory
GPU_ID=2                          # GPU device ID (0, 1, 2, ...)
PORT=7860                         # Host port
```

## 💡 Usage Examples

### Voice Cloning

```python
import requests

# Upload reference audio and generate
response = requests.post(
    "http://localhost:7860/api/clone",
    files={"audio": open("reference.wav", "rb")},
    data={"text": "Hello, this is a cloned voice!"}
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Emotion Editing

```python
# Add emotion to existing audio
response = requests.post(
    "http://localhost:7860/api/edit",
    files={"audio": open("input.wav", "rb")},
    data={
        "mode": "emotion",
        "emotion": "happy",
        "intensity": 1.5
    }
)
```

### Style Editing

```python
# Change speaking style
response = requests.post(
    "http://localhost:7860/api/edit",
    files={"audio": open("input.wav", "rb")},
    data={
        "mode": "style",
        "style": "whisper",
        "intensity": 2.0
    }
)
```

## 🏗️ Project Structure

```
Step-Audio-EditX-GPU-Managed/
├── app.py                      # Main Gradio application
├── gpu_manager.py              # GPU resource manager
├── tts_gpu_managed.py          # TTS model wrapper
├── tokenizer.py                # Audio tokenizer with lazy loading
├── whisper_wrapper.py          # Whisper model wrapper
├── tts.py                      # TTS model implementation
├── model_loader.py             # Model loading utilities
├── utils.py                    # Utility functions
├── api_server.py               # FastAPI server (optional)
├── Dockerfile                  # Docker image definition
├── requirements.txt            # Python dependencies
├── start_with_gpu_management.sh # Startup script
├── GPU_MANAGEMENT.md           # GPU management guide
├── models/                     # Model files (not in git)
│   ├── Step-Audio-Tokenizer/
│   └── Step-Audio-EditX/
├── cache/                      # FunASR cache (not in git)
├── assets/                     # Static assets
└── docs/                       # Documentation
```

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch 2.8.0, Transformers 4.53.3
- **Audio Processing**: torchaudio, librosa, soundfile
- **Speech Models**: FunASR, Whisper, CosyVoice
- **Inference**: ONNX Runtime GPU 1.17.0
- **Web Framework**: Gradio 5.16.0+, FastAPI (optional)
- **Containerization**: Docker, nvidia-docker2

## 📈 Performance

### Tested Environment

- **GPU**: NVIDIA L40S (48GB)
- **CUDA**: 12.1.0
- **Driver**: 580.105.08

### Benchmark Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Startup Memory** | 3 MB | Lazy loading enabled |
| **Peak Memory** | 40 GB | During inference |
| **Post-task Memory** | 5.7 GB | ONNX residual |
| **First Load Time** | 20-30s | One-time cost |
| **Recovery Time** | 2-5s | CPU→GPU transfer |
| **Inference Time** | 8-24s | With FunASR cache |

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Step-Audio-EditX-GPU-Managed.git
cd Step-Audio-EditX-GPU-Managed

# Install development dependencies
pip install -r requirements.txt

# Run tests
python test_gpu_management.py
```

## 📝 Changelog

### 2025-12-05: GPU Memory Management & UI Improvements 🎮

**GPU Memory Management:**
- ✅ Implemented lazy loading for all models (TTS, Whisper, FunASR)
- ✅ Added automatic GPU↔CPU offloading
- ✅ Reduced startup memory from 40GB to 3MB (99.99%)
- ✅ Added idle timeout monitoring (configurable)
- ✅ Created comprehensive documentation

**Bug Fixes:**
- ✅ Fixed `edit()` method parameter mismatch
- ✅ Fixed `NoneType` error on second edit operation
- ✅ Ensured models are loaded before use in all code paths

**UI Enhancements:**
- ✅ Added bilingual (English/Chinese) labels for all UI elements
- ✅ Task and Sub-task dropdowns now show both languages
- ✅ Example: "emotion (情感)", "happy (开心)"
- ✅ Improved user experience for Chinese users

### 2025-12-04: Unified Deployment 🚀

- ✅ Unified UI + API in single container
- ✅ Reduced resource usage by 50%
- ✅ Single port for all services

### 2025-11-22: Performance Optimization ⚡

- ✅ FunASR persistent cache (3x speedup)
- ✅ TF32 acceleration enabled
- ✅ ONNX Runtime optimization

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

- Do not use this model for unauthorized voice cloning, identity impersonation, fraud, deepfakes, or other illegal purposes
- Ensure compliance with local laws and ethical guidelines
- The developers are not responsible for any misuse of this technology

## 🙏 Acknowledgments

- Original [Step-Audio-EditX](https://github.com/stepfun-ai/Step-Audio-EditX) by Stepfun AI
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for TTS model
- [FunASR](https://github.com/alibaba-damo-academy/FunASR) for audio tokenization
- [Whisper](https://github.com/openai/whisper) for transcription

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=neosun100/Step-Audio-EditX-GPU-Managed&type=Date)](https://star-history.com/#neosun100/Step-Audio-EditX-GPU-Managed)

## 📱 Follow Us

![公众号](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)

---

**Made with ❤️ by the community**
