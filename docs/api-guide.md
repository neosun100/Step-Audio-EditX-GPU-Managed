# Step-Audio-EditX OpenAI 兼容 API 手册

> 统一基地址：`http://localhost:8003`（本地部署）或 `https://your-api-domain.com`（生产环境）  
> Swagger & ReDoc：`http://localhost:8003/docs` / `http://localhost:8003/redoc`

本指南覆盖所有可用端点、请求字段以及来自官方 Demo 的实战示例，帮助你快速完成自动化调用或集成第三方系统（Open WebUI、AnythingLLM、Vercel AI SDK 等）。

> **⚠️ 部署注意事项**：
> - 每个服务实例（UI 或 API）加载完整模型约需 **23-30GB GPU 显存**
> - 启动前请运行 `nvidia-smi` 检查可用显存，使用 `--gpus '"device=N"'` 指定空闲 GPU
> - 若遇到 `CUDA out of memory` 错误，请停止容器并选择其他空闲 GPU 重新启动
> - 详细 GPU 选择指南请参考主 `README.md` 的 "GPU 选择重要提示" 章节

> **🚀 三种模型性能对比**（UI 实际测试，完整流程）：
> 
> | 模型变体 | 实际耗时 | 相对性能 | 磁盘占用 | 使用建议 |
> |---------|---------|---------|----------|----------|
> | `base` | 24s | 1.00x | 16 GB | 最稳定 |
> | `bnb` | 24s | 1.00x ⚡ | 7.1 GB | **强烈推荐**（默认）|
> | `awq` | 34s | 0.71x | 7.1 GB | 不推荐（慢 42%）|
> 
> **💡 关键发现**：BnB 和 Base 速度**完全相同**（24s）！  
> 原因：LLM 只占 8% 时间，音频编码（83%）不受量化影响。
> 
> **选择指南**：
> - **默认推荐** → `bnb`（速度 = base + 省 56% 磁盘）⭐
> - 追求稳定 → `base`（无量化）
> - 避免使用 → `awq`（慢 42%）
> 
> 详细分析：[`ui-performance-test-result.md`](ui-performance-test-result.md)

---

## 1. 认证与通用 headers

| 项           | 说明                                                                 |
|--------------|----------------------------------------------------------------------|
| Base URL     | `http://localhost:8003/v1`（本地部署）或 `https://your-api-domain.com/v1`（生产环境）       |
| API Key      | 兼容 OpenAI，`Authorization: Bearer <任意字符串>` 即可                |
| Content-Type | `/v1/audio/speech` 使用 `application/json`；`/v1/audio/speech/upload` 使用 `multipart/form-data` |

> **提示**：如需上传大文件，建议先放到对象存储/HTTP，可通过 `step_audio.prompt_audio_url` 或 `input_audio_url` 引用。

---

## 2. 可用端点

| 方法 | 路径              | 说明                                                                                 |
|------|-------------------|--------------------------------------------------------------------------------------|
| GET  | `/healthz`        | 健康检查，返回 `{"status":"ok"}`                                                     |
| GET  | `/v1/models`      | OpenAI 格式模型列表（目前只有 `step-audio-editx`）                                   |
| GET  | `/v1/voices`      | 预置声线（fear_female / happy_en / whisper_cn / story_teller 等）                   |
| GET  | `/v1/tags`        | 项目已有的音频编辑标签（emotion/style/speed/denoise/vad/paralinguistic 等）          |
| POST | `/v1/audio/speech`| **核心接口：TTS、克隆、情绪/风格/副语言/降噪/去静音/调速均在此完成**（支持 `model_variant` / `intensity`） |
| POST | `/v1/audio/speech/upload` | `multipart/form-data` 版本，可直接上传 `input_audio_file` / `prompt_audio_file` |

Swagger 页面展示完整 Schema；也可下载 `openapi.json` 供 SDK 使用。

---

## 3. 请求结构速览

```jsonc
{
  "model": "step-audio-editx",
  "voice": "happy_en",                  // 可选，使用内置样例语音
  "response_format": "wav",             // wav | mp3 | flac
  "input": "目标文本/提示",
  "metadata": { "trace_id": "demo" },   // 可选，原样返回在响应 headers
  "step_audio": {
    "mode": "clone",                    // clone | emotion | style | paralinguistic | speed | denoise | vad
    "model_variant": "bnb",             // bnb (推荐，默认) | base (稳定) | awq (慢42%不推荐)
    "prompt_text": "参考音频的文本",      // clone 模式建议提供
    "prompt_audio_base64": "...",       // clone 自定义音色
    "prompt_audio_url": "https://...",  // 上述任一即可；缺少则使用 voice preset
    "input_audio_base64": "...",        // 各类 edit 模式需要已有音频
    "input_audio_url": "https://...",
    "audio_text": "原音频文本",          // 可缺省，系统会走 Whisper 自动转写
    "edit_info": "happy / remove / ...",// emotion/style/speed 等模式的附加参数
    "n_edit_iter": 1                    // 1~4（保留为未来扩展次数）
  }
}
```

- **音频输入方式**  
  1. `prompt/input_audio_base64`: 直接在 JSON 中携带 Base64（适合脚本、SDK）。  
  2. `prompt/input_audio_url`: 指向可访问的 HTTP/HTTPS 资源，服务端自动下载。  
  3. `/v1/audio/speech/upload`: 通过 `multipart/form-data` 表单字段 `input_audio_file`、`prompt_audio_file` 上传文件，其余参数仍放在 `payload` JSON 字段中，内部会自动转为 Base64。
- **模型选择**  
  - `model_variant: "base"`（默认）：全精度版本，音质最佳；适合 GPU 资源充足的场景。  
  - `model_variant: "awq"`：加载 `Step-Audio-EditX-AWQ-4bit` 量化模型，显存占用更低、推理更快。
- **语气强度 (`step_audio.intensity`)**  
  - 范围 `0.5 ~ 3.0`，默认 `1.0`。  
  - 数值越大，编辑/情绪的效果越明显。  
  - 系统会自动将数值映射为 `Slightly/Gently/Noticeably/Strongly/Vigorously/Dramatically` 等提示词插入到指令中。

---

## 4. 快速自检

```bash
# 健康检查
curl -i http://localhost:8003/healthz

# 查看可用声线
curl http://localhost:8003/v1/voices | jq

# 查看编辑标签
curl http://localhost:8003/v1/tags | jq
```

---

## 5. 常用示例

### 5.1 预置声线克隆（无需上传音频）

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Authorization: Bearer demo" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "voice": "happy_en",
        "response_format": "wav",
        "input": "Hi, this is the Happy English voice from Step-Audio-EditX.",
        "step_audio": {
          "mode": "clone",
          "prompt_text": "You know, I just finished that big project and feel so relieved."
        }
      }' --output happy_en.wav
```

### 5.2 自定义参考音色（上传本地音频）

```bash
# 将本地 wav/mp3 转 Base64
python3 - <<'PY'
from pathlib import Path
import base64, json, sys
data = base64.b64encode(Path("examples/fear_zh_female_prompt.wav").read_bytes()).decode()
print(json.dumps({"b64": data}))
PY
```

拷贝 `b64` 字符串并发起请求：

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "response_format": "mp3",
        "input": "请用同样的语气说：今晚我们必须尽快撤离。",
        "step_audio": {
          "mode": "clone",
          "prompt_text": "我总觉得，有人在跟着我，我能听到奇怪的脚步声。",
          "prompt_audio_base64": "<上一步的Base64>"
        }
      }' --output custom_clone.mp3
```

### 5.3 加深语气 / 强化效果（intensity + model_variant）

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "response_format": "wav",
        "input": "把这段录音调整成更开心的语气。",
        "step_audio": {
          "mode": "emotion",
          "edit_info": "happy",
          "intensity": 2.3,
          "model_variant": "awq",
          "input_audio_url": "https://example.com/your-audio-file.wav",
          "audio_text": "这是一段示例音频的文本内容"
        }
      }' --output emotion_happy_intense.wav
```

> 将 `intensity` 提升到 2.0 以上会生成 “Strongly/ Vigorously / Dramatically” 的提示，从而在 LLM 指令层面强调“更强烈的开心语气”；`model_variant: "awq"` 可切换到 4-bit 量化模型以降低显存和延迟。反之可传 `0.7` 获得更轻微的变化并保持 `model_variant: "base"` 获取最高音质。

### 5.4 直接上传音频文件（multipart/form-data）

```bash
curl -X POST http://localhost:8003/v1/audio/speech/upload \
  -F 'payload={
        "model": "step-audio-editx",
        "response_format": "wav",
        "input": "把这段录音调整成更开心的语气。",
        "step_audio": {
          "mode": "emotion",
          "edit_info": "happy",
          "audio_text": "这是一段示例音频的文本内容"
        }
      }' \
  -F "input_audio_file=@/tmp/user_demo.wav;type=audio/wav" \
  --output emotion_happy_upload.wav
```

> 说明：`payload` 为 JSON 字符串；如需自定义克隆，可同时添加 `-F "prompt_audio_file=@reference.wav"`。其余字段与 `/v1/audio/speech` 完全一致。

### 5.5 情绪编辑（自动转写 + Emotion）

```bash
# 读取示例 vad_prompt.wav 并发送
python3 - <<'PY'
import base64, json, requests, pathlib
audio = base64.b64encode(pathlib.Path("examples/vad_prompt.wav").read_bytes()).decode()
payload = {
    "model": "step-audio-editx",
    "response_format": "wav",
    "input": "把这段话读得更加愉快一些。",
    "step_audio": {
        "mode": "emotion",
        "edit_info": "happy",
        "input_audio_base64": audio
    }
}
r = requests.post("http://localhost:8003/v1/audio/speech",
                  headers={"Authorization": "Bearer demo"},
                  json=payload)
open("emotion_edit.wav", "wb").write(r.content)
print("Status:", r.status_code)
PY
```

> `audio_text` 未提供时，会自动调用 Whisper（容器启动参数中已开启 `--enable-auto-transcribe`）。

### 5.6 风格（Style）+ 新文本

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "response_format": "wav",
        "input": "我想要用更正式、演讲的语气。",
        "step_audio": {
          "mode": "style",
          "edit_info": "formal",
          "input_audio_url": "https://example-bucket/meeting-origin.wav",
          "audio_text": "原始音频文案"
        }
      }' --output style_formal.wav
```

### 5.7 副语言（Paralinguistic）+ 新文本

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "response_format": "wav",
        "input": "请在故事开头加入笑声，结尾加入叹息。",
        "step_audio": {
          "mode": "paralinguistic",
          "input_audio_base64": "<Base64>",
          "audio_text": "故事正文",
          "edit_info": "storytelling"
        }
      }' --output paralinguistic.wav
```

### 5.8 去静音（VAD）与降噪（Denoise）

```bash
# VAD
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "input": "去除所有空白段落",
        "step_audio": {
          "mode": "vad",
          "input_audio_url": "https://example.com/raw_call.wav"
        }
      }' --output vad_clean.wav

# Denoise
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "input": "请保留语音内容，但最大程度消除噪声",
        "step_audio": {
          "mode": "denoise",
          "input_audio_url": "https://example.com/noisy.wav"
        }
      }' --output denoise.wav
```

### 5.9 语速调节（Speed）

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "input": "保持原意但整体放慢到0.8x",
        "step_audio": {
          "mode": "speed",
          "edit_info": "0.8x",
          "input_audio_base64": "<Base64>"
        }
      }' --output slow.wav
```

---

## 6. Python SDK 示例（requests）

```python
import base64
import requests

BASE = "http://localhost:8003/v1"
headers = {"Authorization": "Bearer demo"}

def clone_with_preset(text: str, voice="whisper_cn"):
    payload = {
        "model": "step-audio-editx",
        "voice": voice,
        "response_format": "mp3",
        "input": text,
        "step_audio": {"mode": "clone"}
    }
    resp = requests.post(f"{BASE}/audio/speech", headers=headers, json=payload)
    resp.raise_for_status()
    return resp.content

audio_bytes = clone_with_preset("欢迎使用 Step-Audio-EditX 的 OpenAI API。")
open("python_clone.mp3", "wb").write(audio_bytes)
```

---

## 7. 快速排障

| 现象                                 | 排查步骤                                                                                             |
|--------------------------------------|------------------------------------------------------------------------------------------------------|
| 403/404                              | 确认反向代理是否把 `/v1/*` 转发到 `http://<host>:8800`，TLS/Host 头是否被篡改                         |
| 415 / Unsupported Media Type         | 必须使用 `application/json`；音频数据通过 Base64/URL 传递                                            |
| 400 / 缺少必填字段                   | clone 模式需要 `voice` 或 `prompt_audio_*`；edit 模式需要 `input_audio_*`                            |
| 500 / CUDA 内存不足                  | 同时多路大模型推理可能溢出，可减少并发或指定不同 GPU（当前 UI=GPU2，API=GPU3）                       |
| 返回空字符串                         | 自动转写失败时会写日志 `Audio transcription failed`；建议传 `audio_text` 或提供更清晰的音频          |

---

## 8. 更多想法

- 可在客户端实现 **SRT/字幕**：先调用 `/v1/audio/speech` 获得处理后的音频，再使用 Whisper 本地或 `/v1/audio/speech` 以 `mode=vad` + `response_format=wav` 输出清晰音频，随后离线转写。
- 结合 `Gradio UI` 与 API：UI 仍由 `http://localhost:7860` 提供可视化，而后端系统则直接走 `http://localhost:8003/v1`.
- 未来可添加 SSE/流式或 `text->token` 接口来兼容 OpenAI `responses` API；当前已预留 `stream` 字段，响应中 `X-StepAudio-Model` 便于追踪。

如需扩展进一步的 preset、角色模板或集成示例（例如 Vercel AI / LangChain / AnythingLLM 配置截图），请告知。***
