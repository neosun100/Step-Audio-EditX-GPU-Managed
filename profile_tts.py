#!/usr/bin/env python3
"""
详细的 TTS 流程性能剖析
"""
import time
import torch
import argparse
from pathlib import Path
from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS
from model_loader import ModelSource

def profile_clone(model_variant: str, model_engines: dict):
    """详细剖析 clone 流程"""
    print(f"\n{'='*70}")
    print(f"🔍 剖析 {model_variant.upper()} 模型 - Clone 任务")
    print(f"{'='*70}")
    
    engine = model_engines[model_variant]
    
    # 测试参数
    prompt_wav_path = "/app/examples/en_happy_prompt.wav"
    prompt_text = "You know, I just finished that big project and feel so relieved."
    target_text = "Hi! I am your Step-Audio-EditX clone. This is a test of voice cloning."
    
    # 总计时开始
    t_total_start = time.time()
    
    # 1. 音频加载
    t1 = time.time()
    import torchaudio
    prompt_wav, _ = torchaudio.load(prompt_wav_path)
    t_load_audio = time.time() - t1
    print(f"  1️⃣  音频加载:            {t_load_audio*1000:.1f} ms")
    
    # 2. 音频编码 (Tokenizer)
    t2 = time.time()
    vq0206_codes, vq02_codes_ori, vq06_codes_ori, speech_feat, _, speech_embedding = (
        engine.encoder.encode(prompt_wav, prompt_text)
    )
    t_encode = time.time() - t2
    print(f"  2️⃣  音频编码 (Tokenizer): {t_encode*1000:.1f} ms")
    
    # 3. LLM 生成 (核心量化部分)
    t3 = time.time()
    vq02_codes = engine._build_and_generate(
        prompt_text, target_text, "clone", None, 
        vq02_codes_ori, vq06_codes_ori, speech_feat, 
        speech_embedding, intensity=1.0
    )
    t_llm = time.time() - t3
    print(f"  3️⃣  LLM 生成:             {t_llm*1000:.1f} ms  ⚡ [量化影响主要在这里]")
    
    # 4. 音频解码 (CosyVoice)
    t4 = time.time()
    audio_data = engine.cosyvoice_model.decode(
        vq02_codes, 
        vq06_codes_ori, 
        prompt_wav, 
        speech_embedding
    )
    t_decode = time.time() - t4
    print(f"  4️⃣  音频解码 (CosyVoice): {t_decode*1000:.1f} ms")
    
    t_total = time.time() - t_total_start
    
    print(f"\n  📊 时间分布:")
    print(f"     音频加载:   {t_load_audio*1000:>7.1f} ms ({t_load_audio/t_total*100:>5.1f}%)")
    print(f"     音频编码:   {t_encode*1000:>7.1f} ms ({t_encode/t_total*100:>5.1f}%)")
    print(f"     LLM 生成:   {t_llm*1000:>7.1f} ms ({t_llm/t_total*100:>5.1f}%) ⚡")
    print(f"     音频解码:   {t_decode*1000:>7.1f} ms ({t_decode/t_total*100:>5.1f}%)")
    print(f"     ─────────────────────────────────")
    print(f"     总耗时:     {t_total*1000:>7.1f} ms (100.0%)")
    
    return {
        'load': t_load_audio,
        'encode': t_encode,
        'llm': t_llm,
        'decode': t_decode,
        'total': t_total
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='/model')
    args = parser.parse_args()
    
    print("\n🚀 Step-Audio-EditX 详细性能剖析")
    print("=" * 70)
    
    # 加载模型
    print("正在加载模型...")
    base_dir = Path(args.model_path)
    
    encoder = StepAudioTokenizer(
        str(base_dir / "Step-Audio-Tokenizer"),
        model_source=ModelSource.LOCAL,
        funasr_model_id="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"
    )
    
    model_engines = {}
    
    # Base 模型
    model_engines["base"] = StepAudioTTS(
        str(base_dir / "Step-Audio-EditX"),
        encoder,
        model_source=ModelSource.LOCAL,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    print("✓ Base 模型加载完成")
    
    # BnB 模型
    bnb_path = base_dir / "Step-Audio-EditX-bnb-4bit"
    if bnb_path.exists():
        model_engines["bnb"] = StepAudioTTS(
            str(bnb_path),
            encoder,
            model_source=ModelSource.LOCAL,
            quantization_config="int4",
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        print("✓ BnB 模型加载完成")
    
    # AWQ 模型
    awq_path = base_dir / "Step-Audio-EditX-AWQ-4bit"
    if awq_path.exists():
        model_engines["awq"] = StepAudioTTS(
            str(awq_path),
            encoder,
            model_source=ModelSource.LOCAL,
            quantization_config="awq-4bit",
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        print("✓ AWQ 模型加载完成")
    
    print("\n开始性能剖析...")
    
    # 剖析每个模型
    results = {}
    for variant in ["base", "bnb", "awq"]:
        if variant in model_engines:
            results[variant] = profile_clone(variant, model_engines)
            time.sleep(2)  # 清理缓存
    
    # 对比总结
    print(f"\n{'='*70}")
    print("📈 性能对比总结")
    print(f"{'='*70}")
    
    print(f"\n{'阶段':<15} {'Base':<12} {'BnB':<12} {'AWQ':<12} {'关键发现'}")
    print("-" * 70)
    
    for stage, name in [('load', '音频加载'), ('encode', '音频编码'), 
                         ('llm', 'LLM 生成'), ('decode', '音频解码')]:
        base_t = results.get('base', {}).get(stage, 0) * 1000
        bnb_t = results.get('bnb', {}).get(stage, 0) * 1000
        awq_t = results.get('awq', {}).get(stage, 0) * 1000
        
        note = ""
        if stage == 'llm':
            note = "⚡ 量化影响"
        elif stage in ['encode', 'decode']:
            note = "🔧 无量化差异"
        
        print(f"{name:<15} {base_t:>8.0f} ms  {bnb_t:>8.0f} ms  {awq_t:>8.0f} ms  {note}")
    
    print("-" * 70)
    base_total = results.get('base', {}).get('total', 0) * 1000
    bnb_total = results.get('bnb', {}).get('total', 0) * 1000
    awq_total = results.get('awq', {}).get('total', 0) * 1000
    print(f"{'总耗时':<15} {base_total:>8.0f} ms  {bnb_total:>8.0f} ms  {awq_total:>8.0f} ms")
    
    # 关键洞察
    print(f"\n💡 关键洞察:")
    if 'base' in results and 'bnb' in results:
        llm_ratio = results['base']['llm'] / results['base']['total'] * 100
        print(f"   • LLM 生成只占总时间的 {llm_ratio:.1f}%")
        print(f"   • 音频编码+解码占 {100-llm_ratio:.1f}%，不受量化影响")
        print(f"   • 这就是为什么 BnB 和 Base 实际使用速度差不多！")

if __name__ == '__main__':
    main()
