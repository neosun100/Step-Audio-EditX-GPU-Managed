# Step-Audio-EditX API cURL Cookbook

> **Base URL**：`http://localhost:8003`（本地部署）  
> **认证**：API 支持可选的认证头，格式为 `Authorization: Bearer <token>`  
> **说明**：本地部署通常不需要认证，生产环境请配置适当的认证机制。

---

## 0. 准备工作

### 0.1 健康检查

```bash
curl http://localhost:8003/healthz
```

### 0.2 查看可用声线与标签

```bash
# 声线列表（用于 voice 字段）
curl http://localhost:8003/v1/voices | jq

# 编辑标签
curl http://localhost:8003/v1/tags | jq
```

### 0.3 将本地音频转为 Base64（复用官方示例或你的音频）

```bash
python3 - <<'PY'
from pathlib import Path
import base64, json
data = base64.b64encode(Path("examples/fear_zh_female_prompt.wav").read_bytes()).decode()
print(json.dumps({"base64": data}, ensure_ascii=False))
PY
```

输出中的 `base64` 字段可直接填入 `step_audio.prompt_audio_base64` 或 `input_audio_base64`。

---

## 1. 语音克隆（Clone）

### 1.1 使用预置声线

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "voice": "story_teller",
        "response_format": "mp3",
        "input": "故事要从风很大的那天说起。",
        "step_audio": { "mode": "clone" }
      }' --output clone_preset.mp3
```

### 1.2 自定义参考音色（Base64）

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "response_format": "wav",
        "input": "请保持同样的情绪说：这是一个示例文本。",
        "step_audio": {
          "mode": "clone",
          "prompt_text": "这是一段示例音频的文本内容",
          "prompt_audio_base64": "<BASE64_AUDIO>"
        }
      }' --output clone_custom.wav
```

---

## 2. 情绪 / 风格 / 副语言

> 以下例子均复用了同一段输入音频 Base64（将 `<BASE64_AUDIO>` 替换成真实值）。若省略 `audio_text`，系统会自动调用 Whisper 做转写。

### 2.1 情绪增强（Emotion）

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
          "input_audio_base64": "<BASE64_AUDIO>",
          "audio_text": "这是一段示例音频的文本内容"
        }
      }' --output emotion_happy.wav
```

### 2.2 加深语气（intensity）

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
          "intensity": 2.2,
          "input_audio_url": "https://example.com/your-audio-file.wav",
          "audio_text": "这是一段示例音频的文本内容"
        }
      }' --output emotion_happy_intense.wav
```

> `intensity` 值越大效果越明显，可在 0.5~3.0 之间调节；也可以继续使用 Base64 方式，只需保留 `input_audio_base64` 即可。

### 2.3 切换 AWQ 量化模型

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
          "model_variant": "awq",
          "input_audio_url": "https://img.aws.xin/uPic/%E5%AF%B9%E5%95%8A%E4%BB%8A%E5%A4%A9%E5%A4%9A%E5%A5%BD%E6%9C%BA%E4%BC%9A%EF%BC%8C%E6%88%91%E6%9C%AC%E6%83%B3%E5%A5%BD%E5%A5%BD%E8%B7%9F%E5%A4%A7%E9%9B%B7%E5%93%A5%E5%94%A0%E5%94%A0%EF%BC%8C%E6%93%8D.wav",
          "audio_text": "这是一段示例音频的文本内容"
        }
      }' --output emotion_awq.wav
```

> 当本地显存紧张或需要更快推理时，可将 `model_variant` 设为 `awq` 使用 4-bit 量化模型；如需最高音质，则继续使用 `base`。其余参数（Base64 / URL / intensity）完全兼容。

### 2.4 风格转换（Style）

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "response_format": "wav",
        "input": "我想让这段话变得更正式稳重。",
        "step_audio": {
          "mode": "style",
          "edit_info": "formal",
          "input_audio_base64": "<BASE64_AUDIO>",
          "audio_text": "这是一段示例音频的文本内容"
        }
      }' --output style_formal.wav
```

> **直接引用在线音频**：若你的素材已经放在 HTTP/HTTPS 地址，可像上文的 intensity 示例那样改用 `input_audio_url`，无需再进行 Base64。

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
          "input_audio_url": "https://example.com/your-audio-file.wav",
          "audio_text": "这是一段示例音频的文本内容"
        }
      }' --output emotion_happy.wav
```

### 2.5 副语言（Paralinguistic）

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "response_format": "wav",
        "input": "在开头加点感叹，结尾来个叹气。",
        "step_audio": {
          "mode": "paralinguistic",
          "edit_info": "storytelling",
          "input_audio_base64": "<BASE64_AUDIO>",
          "audio_text": "这是一段示例音频的文本内容"
        }
      }' --output paralinguistic_story.wav
```

### 2.6 直接上传音频（/v1/audio/speech/upload）

```bash
curl -X POST http://localhost:8003/v1/audio/speech/upload \
  -F 'payload={
        "model": "step-audio-editx",
        "response_format": "wav",
        "input": "请把开头加上欢呼声，整体更开心。",
        "step_audio": {
          "mode": "emotion",
          "edit_info": "happy",
          "audio_text": "这是一段示例音频的文本内容"
        }
      }' \
  -F "input_audio_file=@examples/fear_zh_female_prompt.wav;type=audio/wav" \
  --output emotion_upload.wav
```

> `payload` 为 JSON 字符串；若要同时上传参考音频，可再加 `-F "prompt_audio_file=@/path/to/ref.wav"。服务器收到文件后会自动转为 Base64 并复用同一处理逻辑。`

---

## 3. Speed / Denoise / VAD

### 3.1 语速调节

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "response_format": "wav",
        "input": "整体节奏放慢到0.85x。",
        "step_audio": {
          "mode": "speed",
          "edit_info": "0.85x",
          "input_audio_base64": "<BASE64_AUDIO>",
          "audio_text": "这是一段示例音频的文本内容"
        }
      }' --output speed_slow.wav
```

### 3.2 降噪（Denoise）

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "response_format": "wav",
        "input": "去掉所有背景噪声。",
        "step_audio": {
          "mode": "denoise",
          "input_audio_base64": "<BASE64_AUDIO>"
        }
      }' --output denoise.wav
```

### 3.3 去静音（VAD）

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "response_format": "wav",
        "input": "裁剪静音，只留下语音内容。",
        "step_audio": {
          "mode": "vad",
          "input_audio_base64": "<BASE64_AUDIO>"
        }
      }' --output vad_trim.wav
```

---

## 4. 组合提示（metadata & response_format）

```bash
curl -X POST http://localhost:8003/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
        "model": "step-audio-editx",
        "voice": "happy_en",
        "response_format": "flac",
        "input": "欢迎使用 Step-Audio-EditX 的 OpenAI 兼容 API。",
        "metadata": { "request_id": "demo-001" },
        "step_audio": { "mode": "clone" }
      }' --output welcome.flac
```

响应头将包含 `X-StepAudio-Model` 以及 `x-metadata-request_id`，便于调试或日志追踪。

---

## 5. 常见问题速查

| 现象 | 解决办法 |
|------|----------|
| 返回 401 | Base URL 前面需要 Basic 认证头（或你自定义的 Header）；确认反向代理是否透传。 |
| 返回 400 | 检查 `step_audio` 必填字段：clone 模式至少选择 `voice` 或 `prompt_audio_*`；编辑模式需 `input_audio_*`。 |
| 返回 500 + “Preset audio not found” | 检查声线名称是否在 `/v1/voices` 列表；也可把示例音频 copy 到 `/model/examples`。 |
| 音频为空或不完整 | 确保 `input_audio_base64` 为 WAV/MP3 等标准格式；必要时先 `ffmpeg -i sample.wav sample16k.wav`。 |
| Whisper 自动转写失败 | 在 Body 中显式提供 `audio_text`，或确保音频语速清晰；日志可通过 `docker logs step-audio-api` 查看。 |

---

将本文档与 `docs/api-guide.md` 配套使用，可实现“看文档即复制、立即调用”的效果。若需要针对特定业务（如客服质检、配音批处理）编写更复杂的脚本示例，请告知。***
