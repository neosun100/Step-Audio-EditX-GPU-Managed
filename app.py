import gradio as gr
import os
import argparse
import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from datetime import datetime
import time
import torchaudio
import librosa
import soundfile as sf

# Project imports
from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from tts_gpu_managed import GPUManagedTTS
from model_loader import ModelSource
from config.edit_config import get_supported_edit_types, get_edit_type_key, get_edit_info_key
from whisper_wrapper import WhisperWrapper
from gpu_manager import get_gpu_manager

# Configure logging
logger = logging.getLogger(__name__)

# Save audio to temporary directory
def save_audio(audio_type, audio_data, sr, tmp_dir):
    """Save audio data to a temporary file with timestamp"""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(tmp_dir, audio_type, f"{current_time}.wav")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        if isinstance(audio_data, torch.Tensor):
            torchaudio.save(save_path, audio_data, sr)
        else:
            sf.write(save_path, audio_data, sr)
        logger.debug(f"Audio saved to: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        raise


class EditxTab:
    """Audio editing and voice cloning interface tab"""

    def __init__(self, args, encoder=None, tts_engine=None):
        self.args = args
        self.encoder = encoder  # Store encoder for cache stats
        self.tts_engine = tts_engine  # Store TTS engine for GPU management
        self.edit_type_list = list(get_supported_edit_types().keys())
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.enable_auto_transcribe = getattr(args, 'enable_auto_transcribe', False)
        self.live_logs = []  # Store live execution logs
        self.max_logs = 100  # Maximum number of logs to keep

    def history_messages_to_show(self, messages):
        """Convert message history to gradio chatbot format"""
        show_msgs = []
        for message in messages:
            edit_type = message['edit_type']
            edit_info = message['edit_info']
            source_text = message['source_text']
            target_text = message['target_text']
            raw_audio_part = message['raw_wave']
            edit_audio_part = message['edit_wave']
            type_str = f"{edit_type}-{edit_info}" if edit_info is not None else f"{edit_type}"
            show_msgs.extend([
                {"role": "user", "content": f"任务类型：{type_str}\n文本：{source_text}"},
                {"role": "user", "content": gr.Audio(value=raw_audio_part, interactive=False)},
                {"role": "assistant", "content": f"输出音频：\n文本：{target_text}"},
                {"role": "assistant", "content": gr.Audio(value=edit_audio_part, interactive=False)}
            ])
        return show_msgs

    def generate_clone(self, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, model_variant, intensity, state):
        """Generate cloned audio"""
        self.add_log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.add_log("🎤 开始 CLONE 操作")
        self.add_log(f"   模型: {model_variant} | 强度: {intensity}")
        self.logger.info("Starting voice cloning process")
        self.logger.info(f"   Model: {model_variant}, Intensity: {intensity}")
        state['history_audio'] = []
        state['history_messages'] = []

        # Input validation
        if not prompt_text_input or prompt_text_input.strip() == "":
            error_msg = "[Error] Uploaded text cannot be empty."
            self.logger.error(error_msg)
            self.add_log(f"❌ {error_msg}")
            return [{"role": "user", "content": error_msg}], state, "", self.get_live_logs()
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            self.logger.error(error_msg)
            self.add_log(f"❌ {error_msg}")
            return [{"role": "user", "content": error_msg}], state, "", self.get_live_logs()
        if not generated_text or generated_text.strip() == "":
            error_msg = "[Error] Clone content cannot be empty."
            self.logger.error(error_msg)
            self.add_log(f"❌ {error_msg}")
            return [{"role": "user", "content": error_msg}], state, "", self.get_live_logs()
        if get_edit_type_key(edit_type) != "clone":
            error_msg = "[Error] CLONE button must use clone task."
            self.logger.error(error_msg)
            self.add_log(f"❌ {error_msg}")
            return [{"role": "user", "content": error_msg}], state, "", self.get_live_logs()

        try:
            # Use common_tts_engine for cloning
            self.add_log("📥 输入验证通过，开始克隆...")
            clone_start = time.time()
            output_audio, output_sr = common_tts_engine.clone(
                prompt_audio_input, prompt_text_input, generated_text
            )
            clone_time = time.time() - clone_start
            self.add_log(f"✅ 克隆完成，耗时: {clone_time:.2f}s")

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": prompt_text_input,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                state["history_audio"].append((output_sr, audio_numpy, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                
                # 自动更新缓存统计
                cache_stats_text = self.format_cache_stats()
                self.logger.info("Voice cloning completed successfully")
                self.add_log("🎉 操作成功完成！")
                return show_msgs, state, cache_stats_text, self.get_live_logs()
            else:
                error_msg = "[Error] Clone failed"
                self.logger.error(error_msg)
                self.add_log(f"❌ {error_msg}")
                return [{"role": "user", "content": error_msg}], state, "", self.get_live_logs()

        except Exception as e:
            error_msg = f"[Error] Clone failed: {str(e)}"
            self.logger.error(error_msg)
            self.add_log(f"❌ 异常: {str(e)}")
            cache_stats_text = self.format_cache_stats()
            return [{"role": "user", "content": error_msg}], state, cache_stats_text, self.get_live_logs()
        
    def generate_edit(self, prompt_text_input, prompt_audio_input, generated_text, edit_type, edit_info, model_variant, intensity, state):
        """Generate edited audio"""
        self.logger.info(f"   Model: {model_variant}, Intensity: {intensity}")
        self.logger.info("Starting audio editing process")

        # Input validation
        if not prompt_audio_input:
            error_msg = "[Error] Uploaded audio cannot be empty."
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

        try:
            # Determine which audio to use
            if len(state["history_audio"]) == 0:
                # First edit - use uploaded audio
                audio_to_edit = prompt_audio_input
                text_to_use = prompt_text_input
                self.logger.debug("Using prompt audio, no history found")
            else:
                # Use previous edited audio - save it to temp file first
                sample_rate, audio_numpy, previous_text = state["history_audio"][-1]
                temp_path = save_audio("temp", audio_numpy, sample_rate, self.args.tmp_dir)
                audio_to_edit = temp_path
                text_to_use = previous_text
                self.logger.debug(f"Using previous audio from history, count: {len(state['history_audio'])}")

            # 提取实际的编辑类型和信息键
            actual_edit_type = get_edit_type_key(edit_type)
            actual_edit_info = get_edit_info_key(edit_info) if edit_info else None
            
            # For para-linguistic, use generated_text; otherwise use source text
            if actual_edit_type not in {"paralinguistic"}:
                generated_text = text_to_use

            # Use common_tts_engine for editing
            output_audio, output_sr = common_tts_engine.edit(
                audio_to_edit, text_to_use, actual_edit_type, actual_edit_info, generated_text
            )

            if output_audio is not None and output_sr is not None:
                # Convert tensor to numpy if needed
                if isinstance(output_audio, torch.Tensor):
                    audio_numpy = output_audio.cpu().numpy().squeeze()
                else:
                    audio_numpy = output_audio

                # Load original audio for comparison
                if len(state["history_audio"]) == 0:
                    input_audio_data_numpy, input_sample_rate = librosa.load(prompt_audio_input)
                else:
                    input_sample_rate, input_audio_data_numpy, _ = state["history_audio"][-1]

                # Create message for history
                cur_assistant_msg = {
                    "edit_type": edit_type,
                    "edit_info": edit_info,
                    "source_text": text_to_use,
                    "target_text": generated_text,
                    "raw_wave": (input_sample_rate, input_audio_data_numpy),
                    "edit_wave": (output_sr, audio_numpy),
                }
                state["history_audio"].append((output_sr, audio_numpy, generated_text))
                state["history_messages"].append(cur_assistant_msg)

                show_msgs = self.history_messages_to_show(state["history_messages"])
                self.logger.info("Audio editing completed successfully")
                return show_msgs, state
            else:
                error_msg = "[Error] Edit failed"
                self.logger.error(error_msg)
                return [{"role": "user", "content": error_msg}], state

        except Exception as e:
            error_msg = f"[Error] Edit failed: {str(e)}"
            self.logger.error(error_msg)
            return [{"role": "user", "content": error_msg}], state

    def clear_history(self, state):
        """Clear conversation history"""
        state["history_messages"] = []
        state["history_audio"] = []
        return [], state

    def init_state(self):
        """Initialize conversation state"""
        return {
            "history_messages": [],
            "history_audio": []
        }

    def register_components(self):
        """Register gradio components - maintaining exact layout from original"""
        with gr.Tab("Editx"):
            with gr.Row():
                with gr.Column():
                    self.model_input = gr.Textbox(label="Model Name", value="Step-Audio-EditX", scale=1)
                    self.prompt_text_input = gr.Textbox(label="Prompt Text", value="", scale=1)
                    self.prompt_audio_input = gr.Audio(
                        sources=["upload", "microphone"],
                        format="wav",
                        type="filepath",
                        label="Input Audio",
                    )
                    self.generated_text = gr.Textbox(label="Target Text", lines=1, max_lines=200, max_length=1000)
                    
                    # Model Variant Selection
                    self.model_variant = gr.Radio(
                        label="🎯 Model Variant",
                        choices=["base", "awq", "bnb"],
                        value="base",
                        info="base: 原始模型 | awq: AWQ 4-bit | bnb: BnB 4-bit"
                    )
                    
                    # Intensity Slider
                    self.intensity = gr.Slider(
                        label="🎚️ Effect Intensity (强度)",
                        minimum=0.1,
                        maximum=3.0,
                        value=1.0,
                        step=0.1,
                        info="调整效果强度 (0.1=最弱, 1.0=标准, 3.0=最强)"
                    )
                    
                    # FunASR Cache Stats
                    with gr.Accordion("📊 FunASR 缓存统计", open=True):
                        self.cache_stats_display = gr.Textbox(
                            label="缓存性能",
                            value="等待数据...\n点击 CLONE 按钮后自动更新",
                            lines=8,
                            max_lines=10,
                            interactive=False,
                            show_copy_button=True
                        )
                        with gr.Row():
                            self.refresh_cache_btn = gr.Button("🔄 刷新统计", size="sm")
                            self.clear_cache_btn = gr.Button("🗑️ 清空缓存", size="sm")
                    
                    # GPU Management (if enabled)
                    if self.args.enable_gpu_management:
                        with gr.Accordion("🎮 GPU 显存管理", open=True):
                            self.gpu_status_display = gr.Textbox(
                                label="GPU 状态",
                                value="等待查询...",
                                lines=6,
                                max_lines=10,
                                interactive=False,
                                show_copy_button=True
                            )
                            with gr.Row():
                                self.refresh_gpu_btn = gr.Button("🔄 刷新状态", size="sm")
                                self.offload_gpu_btn = gr.Button("💾 卸载到CPU", size="sm")
                                self.release_gpu_btn = gr.Button("🗑️ 完全释放", size="sm")
                    
                with gr.Column():
                    with gr.Row():
                        self.edit_type = gr.Dropdown(label="Task (任务)", choices=self.edit_type_list, value="clone (克隆)")
                        self.edit_info = gr.Dropdown(label="Sub-task (子任务)", choices=[], value=None)
                    self.chat_box = gr.Chatbot(label="History (历史记录)", type="messages", height=480*1)
                    
                    # 🔥 实时日志显示区域
                    with gr.Accordion("📋 实时运行日志", open=True):
                        self.live_log_display = gr.Textbox(
                            label="执行日志 (带时间戳)",
                            value="等待执行...\n日志将在 CLONE/EDIT 操作时自动更新",
                            lines=12,
                            max_lines=20,
                            interactive=False,
                            show_copy_button=True,
                            autoscroll=True
                        )
                        with gr.Row():
                            self.refresh_log_btn = gr.Button("🔄 刷新日志", size="sm")
                            self.clear_log_btn = gr.Button("🗑️ 清空日志", size="sm")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        self.button_tts = gr.Button("CLONE", variant="primary")
                        self.button_edit = gr.Button("EDIT", variant="primary")
                with gr.Column():
                    self.clean_history_submit = gr.Button("Clear History", variant="primary")

            gr.Markdown("---")
            gr.Markdown("""
                **Button Description:**
                - CLONE: Synthesizes audio based on uploaded audio and text, only used for clone mode, will clear history information when used.
                - EDIT: Edits based on uploaded audio, or continues to stack edit effects based on the previous round of generated audio.
                """)
            gr.Markdown("""
                **Operation Workflow:**
                - Upload the audio to be edited on the left side and fill in the corresponding text content of the audio;
                - If the task requires modifying text content (such as clone, para-linguistic), fill in the text to be synthesized in the "clone text" field. For all other tasks, keep the uploaded audio text content unchanged;
                - Select tasks and subtasks on the right side (some tasks have no subtasks, such as vad, etc.);
                - Click the "CLONE" or "EDIT" button on the left side, and audio will be generated in the dialog box on the right side.
                """)
            gr.Markdown("""
                **Para-linguistic Description:**
                - Supported tags include: [Breathing] [Laughter] [Surprise-oh] [Confirmation-en] [Uhm] [Surprise-ah] [Surprise-wa] [Sigh] [Question-ei] [Dissatisfaction-hnn]
                - Example:
                    - Fill in "clone text" field: "Great, the weather is so nice today." Click the "CLONE" button to get audio.
                    - Change "clone text" field to: "Great[Laughter], the weather is so nice today[Surprise-ah]." Click the "EDIT" button to get para-linguistic audio.
                """)

    def register_events(self):
        """Register event handlers"""
        # Create independent state for each session
        state = gr.State(self.init_state())

        self.button_tts.click(self.generate_clone,
            inputs=[self.prompt_text_input, self.prompt_audio_input, self.generated_text, self.edit_type, self.edit_info, self.model_variant, self.intensity, state],
            outputs=[self.chat_box, state, self.cache_stats_display, self.live_log_display])
        self.button_edit.click(self.generate_edit,
            inputs=[self.prompt_text_input, self.prompt_audio_input, self.generated_text, self.edit_type, self.edit_info, self.model_variant, self.intensity, state],
            outputs=[self.chat_box, state])
        
        # Cache control events
        self.refresh_cache_btn.click(
            fn=self.get_cache_stats,
            inputs=[],
            outputs=self.cache_stats_display
        )
        self.clear_cache_btn.click(
            fn=self.clear_cache,
            inputs=[],
            outputs=self.cache_stats_display
        )
        
        # Log control events
        self.refresh_log_btn.click(
            fn=self.get_live_logs,
            inputs=[],
            outputs=self.live_log_display
        )
        self.clear_log_btn.click(
            fn=self.clear_live_logs,
            inputs=[],
            outputs=self.live_log_display
        )
        
        # GPU management events (if enabled)
        if self.args.enable_gpu_management:
            self.refresh_gpu_btn.click(
                fn=self.get_gpu_status,
                inputs=[],
                outputs=self.gpu_status_display
            )
            self.offload_gpu_btn.click(
                fn=self.offload_gpu,
                inputs=[],
                outputs=[gr.Textbox(visible=False), self.gpu_status_display]
            )
            self.release_gpu_btn.click(
                fn=self.release_gpu,
                inputs=[],
                outputs=[gr.Textbox(visible=False), self.gpu_status_display]
            )

        self.clean_history_submit.click(self.clear_history, inputs=[state], outputs=[self.chat_box, state])
        self.edit_type.change(
            fn=self.update_edit_info,
            inputs=self.edit_type,
            outputs=self.edit_info,
        )

        # Add audio transcription event only if enabled
        if self.enable_auto_transcribe:
            self.prompt_audio_input.change(
                fn=self.transcribe_audio,
                inputs=[self.prompt_audio_input, self.prompt_text_input],
                outputs=self.prompt_text_input,
            )

    def update_edit_info(self, category):
        """Update sub-task dropdown based on main task selection"""
        category_items = get_supported_edit_types()
        choices = category_items.get(category, [])
        value = None if len(choices) == 0 else choices[0]
        return gr.Dropdown(label="Sub-task", choices=choices, value=value)
    
    def get_cache_stats(self):
        """获取 FunASR 缓存统计（返回格式化文本）"""
        return self.format_cache_stats()
    
    def format_cache_stats(self):
        """格式化缓存统计为易读文本"""
        if not hasattr(self, 'encoder'):
            return "❌ 错误：Encoder 未初始化"
        
        if not hasattr(self.encoder, 'get_cache_stats'):
            return "❌ 错误：Encoder 没有 get_cache_stats 方法"
        
        try:
            stats = self.encoder.get_cache_stats()
            self.logger.info(f"✅ Retrieved cache stats: {stats}")
            
            # 格式化为易读文本
            text = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            text += "📊 FunASR 缓存性能统计\n"
            text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            if "error" in stats:
                text += f"❌ {stats.get('error')}\n"
                text += f"   {stats.get('info', '')}\n"
            else:
                text += f"✅ 缓存状态：{'启用' if stats.get('enabled') else '禁用'}\n\n"
                text += f"📈 统计数据：\n"
                text += f"   • 命中次数：{stats.get('hits', 0)} 次\n"
                text += f"   • 未命中次数：{stats.get('misses', 0)} 次\n"
                text += f"   • 总请求数：{stats.get('total_requests', 0)} 次\n"
                text += f"   • 命中率：{stats.get('hit_rate', '0.0%')}\n\n"
                text += f"💾 缓存使用：\n"
                text += f"   • 当前大小：{stats.get('cache_size', 0)} 项\n"
                text += f"   • 最大容量：{stats.get('max_size', 0)} 项\n\n"
                text += f"⏱️ 性能提升：\n"
                text += f"   • 预估节省时间：{stats.get('time_saved_estimate', '0s')}\n"
                text += f"   • 每次命中节省：~1.65s\n\n"
                
                # 添加性能建议
                hit_rate_num = float(stats.get('hit_rate', '0%').rstrip('%'))
                if hit_rate_num > 50:
                    text += "🎉 缓存效果很好！\n"
                elif hit_rate_num > 0:
                    text += "💡 提示：使用相同音频可提高命中率\n"
                else:
                    text += "💡 提示：执行几次 clone 后查看效果\n"
            
            text += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            text += f"🕐 更新时间：{time.strftime('%H:%M:%S')}\n"
            
            return text
            
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return f"❌ 获取统计失败：{str(e)}"
    
    def clear_cache(self):
        """清空 FunASR 缓存"""
        if hasattr(self, 'encoder') and hasattr(self.encoder, 'clear_cache'):
            self.encoder.clear_cache()
            self.logger.info("🗑️ Cache cleared")
            return self.format_cache_stats()
        return "❌ 错误：Cache not available"
    
    def add_log(self, message):
        """添加日志条目（带时间戳）"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.live_logs.append(log_entry)
        # Keep only the last max_logs entries
        if len(self.live_logs) > self.max_logs:
            self.live_logs = self.live_logs[-self.max_logs:]
    
    def get_live_logs(self):
        """获取格式化的实时日志"""
        if not self.live_logs:
            return "暂无日志记录\n执行 CLONE/EDIT 操作后将显示日志"
        
        # Return last 50 logs (most recent)
        recent_logs = self.live_logs[-50:]
        return "\n".join(recent_logs)
    
    def clear_live_logs(self):
        """清空实时日志"""
        self.live_logs.clear()
        self.add_log("📋 日志已清空")
        return self.get_live_logs()
    
    def get_gpu_status(self):
        """获取 GPU 状态"""
        if not self.args.enable_gpu_management or not self.tts_engine:
            return "GPU 管理未启用"
        
        try:
            status = self.tts_engine.get_gpu_status()
            
            if not status.get('enabled', True):
                return "GPU 管理未启用"
            
            # 格式化显示
            lines = []
            lines.append(f"🎮 GPU 显存管理状态")
            lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"GPU 显存占用: {status.get('gpu_memory_mb', 0):.1f} MB")
            lines.append(f"空闲超时: {status.get('idle_timeout', 0)} 秒")
            lines.append("")
            
            models = status.get('models', {})
            if models:
                lines.append("📦 模型状态:")
                for model_name, model_info in models.items():
                    location = model_info.get('location', 'unknown')
                    idle_sec = model_info.get('idle_seconds', 0)
                    
                    location_icon = {
                        'gpu': '🟢 GPU',
                        'cpu': '🟡 CPU',
                        'unloaded': '⚪ 未加载'
                    }.get(location, '❓ 未知')
                    
                    lines.append(f"  • {model_name}: {location_icon}")
                    lines.append(f"    空闲时间: {idle_sec} 秒")
            else:
                lines.append("📦 暂无模型加载")
            
            return "\n".join(lines)
        except Exception as e:
            return f"❌ 获取状态失败: {str(e)}"
    
    def offload_gpu(self):
        """手动卸载 GPU 到 CPU"""
        if not self.args.enable_gpu_management or not self.tts_engine:
            return "GPU 管理未启用", self.get_gpu_status()
        
        try:
            self.tts_engine.force_offload()
            return "✅ 模型已卸载到 CPU", self.get_gpu_status()
        except Exception as e:
            return f"❌ 卸载失败: {str(e)}", self.get_gpu_status()
    
    def release_gpu(self):
        """完全释放 GPU 显存"""
        if not self.args.enable_gpu_management or not self.tts_engine:
            return "GPU 管理未启用", self.get_gpu_status()
        
        try:
            self.tts_engine.force_release()
            return "✅ 模型已完全释放", self.get_gpu_status()
        except Exception as e:
            return f"❌ 释放失败: {str(e)}", self.get_gpu_status()

    def transcribe_audio(self, audio_input, current_text):
        """Transcribe audio using Whisper ASR when prompt text is empty"""
        # Only transcribe if current text is empty
        if current_text and current_text.strip():
            return current_text  # Keep existing text
        if not audio_input:
            return ""  # No audio to transcribe
        if whisper_asr is None:
            self.logger.error("Whisper ASR not initialized.")
            return ""

        try:
            # Transcribe audio
            transcribed_text = whisper_asr(audio_input)
            self.logger.info(f"Audio transcribed: {transcribed_text}")
            return transcribed_text

        except Exception as e:
            self.logger.error(f"Failed to transcribe audio: {e}")
            return ""


def launch_demo(args, editx_tab, encoder, tts_engines, whisper_asr_instance):
    """Launch the gradio demo with optional API support"""
    with gr.Blocks(
            theme=gr.themes.Soft(), 
            title="🎙️ Step-Audio-EditX",
            css="""
    :root {
        --font: "Helvetica Neue", Helvetica, Arial, sans-serif;
        --font-mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }
    """) as demo:
        gr.Markdown("## 🎙️ Step-Audio-EditX")
        gr.Markdown("Audio Editing and Zero-Shot Cloning using Step-Audio-EditX")

        # Register components
        editx_tab.register_components()

        # Register events
        editx_tab.register_events()

    # Check if API should be enabled
    enable_api = getattr(args, 'enable_api', False)
    
    if enable_api:
        # Import API components
        from pathlib import Path
        from api_server import build_fastapi_app
        
        logger.info("🔌 启用 API 支持，共享模型实例")
        
        # Build FastAPI app with shared models
        model_path = Path(args.model_path)
        asset_roots = [model_path.parent / "examples"] if (model_path.parent / "examples").exists() else []
        
        api_app = build_fastapi_app(
            model_engines=tts_engines,
            model_root=model_path,
            asset_roots=asset_roots,
            whisper_asr=whisper_asr_instance
        )
        
        # Mount Gradio to FastAPI
        app = gr.mount_gradio_app(api_app, demo, path="/")
        
        logger.info("=" * 80)
        logger.info(f"✓ 统一服务器启动成功")
        logger.info(f"UI 界面: http://{args.server_name}:{args.server_port}")
        logger.info(f"API 文档: http://{args.server_name}:{args.server_port}/docs")
        logger.info(f"健康检查: http://{args.server_name}:{args.server_port}/healthz")
        logger.info(f"共享模型: UI 和 API 使用同一个模型实例")
        logger.info("=" * 80)
        
        # Use uvicorn to run the combined app
        import uvicorn
        uvicorn.run(
            app,
            host=args.server_name,
            port=args.server_port,
            log_level="info"
        )
    else:
        # Launch demo only (original behavior)
        demo.queue().launch(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share if hasattr(args, 'share') else False
        )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Step-Audio Edit Demo")
    parser.add_argument("--model-path", type=str, required=True, help="Model path.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Demo server name.")
    parser.add_argument("--server-port", type=int, default=7860, help="Demo server port.")
    parser.add_argument("--tmp-dir", type=str, default="/tmp/gradio", help="Save path.")
    parser.add_argument("--share", action="store_true", help="Share gradio app.")

    # Multi-source loading support parameters
    parser.add_argument(
        "--model-source",
        type=str,
        default="auto",
        choices=["auto", "local", "modelscope", "huggingface"],
        help="Model source: auto (detect automatically), local, modelscope, or huggingface"
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
        help="Tokenizer model ID for online loading"
    )
    parser.add_argument(
        "--tts-model-id",
        type=str,
        default=None,
        help="TTS model ID for online loading (if different from model-path)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["int4", "int8", "awq-4bit"],
        help="Enable quantization for the TTS model to reduce memory usage."
             "Choices: int4 (online), int8 (online), awq-4bit (AWQ 4-bit quantization)."
             "When quantization is enabled, data types are handled automatically by the quantization library."
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="PyTorch data type for model operations. This setting only applies when quantization is disabled. "
             "When quantization is enabled, data types are managed automatically."
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="cuda",
        help="Device mapping for model loading (default: cuda)"
    )
    parser.add_argument(
        "--enable-auto-transcribe",
        action="store_true",
        help="Enable automatic audio transcription when uploading audio files (default: disabled)"
    )
    parser.add_argument(
        "--enable-api",
        action="store_true",
        help="Enable FastAPI endpoints (UI and API will share the same model instance)"
    )
    parser.add_argument(
        "--enable-gpu-management",
        action="store_true",
        default=True,
        help="Enable GPU memory management (lazy loading + auto offload). "
             "Models will be loaded on first use and offloaded to CPU after each task. (Default: enabled)"
    )
    parser.add_argument(
        "--gpu-idle-timeout",
        type=int,
        default=600,
        help="GPU idle timeout in seconds before auto-offloading to CPU (default: 600 = 10 minutes)"
    )

    args = parser.parse_args()

    # Map string arguments to actual types
    source_mapping = {
        "auto": ModelSource.AUTO,
        "local": ModelSource.LOCAL,
        "modelscope": ModelSource.MODELSCOPE,
        "huggingface": ModelSource.HUGGINGFACE
    }
    model_source = source_mapping[args.model_source]

    # Map torch dtype string to actual torch dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_mapping[args.torch_dtype]

    logger.info(f"Loading models with source: {args.model_source}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Tokenizer model ID: {args.tokenizer_model_id}")
    logger.info(f"Torch dtype: {args.torch_dtype}")
    logger.info(f"Device map: {args.device_map}")
    if args.tts_model_id:
        logger.info(f"TTS model ID: {args.tts_model_id}")
    if args.quantization:
        logger.info(f"🔧 {args.quantization.upper()} quantization enabled")

    # Initialize models
    whisper_asr = None
    try:
        # Load StepAudioTokenizer
        encoder = StepAudioTokenizer(
            os.path.join(args.model_path, "Step-Audio-Tokenizer"),
            model_source=model_source,
            funasr_model_id=args.tokenizer_model_id,
            enable_gpu_management=args.enable_gpu_management
        )
        logger.info("✓ StepAudioTokenizer loaded successfully")
        
        # Initialize TTS engine with optional GPU management
        tts_model_path = os.path.join(
            args.model_path, 
            "Step-Audio-EditX-AWQ-4bit" if args.quantization == "awq-4bit" else "Step-Audio-EditX"
        )
        
        if args.enable_gpu_management:
            logger.info(f"🚀 GPU 管理已启用 (超时: {args.gpu_idle_timeout}秒)")
            common_tts_engine = GPUManagedTTS(
                model_path=tts_model_path,
                audio_tokenizer=encoder,
                model_source=model_source,
                tts_model_id=args.tts_model_id,
                quantization_config=args.quantization,
                torch_dtype=torch_dtype,
                device_map=args.device_map,
                gpu_idle_timeout=args.gpu_idle_timeout,
                enable_gpu_management=True
            )
        else:
            logger.info("ℹ️  使用传统加载方式（GPU 管理已禁用）")
            common_tts_engine = StepAudioTTS(
                tts_model_path,
                encoder,
                model_source=model_source,
                tts_model_id=args.tts_model_id,
                quantization_config=args.quantization,
                torch_dtype=torch_dtype,
                device_map=args.device_map
            )
        logger.info("✓ StepCommonAudioTTS loaded successfully")
        
        # Prepare tts_engines dict for API (if enabled)
        tts_engines = {"base": common_tts_engine}
        
        if args.enable_auto_transcribe:
            whisper_asr = WhisperWrapper(enable_gpu_management=args.enable_gpu_management)
            logger.info("✓ Automatic audio transcription enabled")
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        logger.error("Please check your model paths and source configuration.")
        exit(1)

    # Create EditxTab instance (pass encoder for cache stats and tts_engine for GPU management)
    editx_tab = EditxTab(args, encoder=encoder, tts_engine=common_tts_engine)

    # Launch demo with shared models
    launch_demo(args, editx_tab, encoder, tts_engines, whisper_asr)
