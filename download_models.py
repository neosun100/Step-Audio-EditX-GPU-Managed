from modelscope import snapshot_download
import os
import shutil

os.makedirs("models", exist_ok=True)

# Tokenizer
print("Downloading Step-Audio-Tokenizer...")
try:
    tokenizer_path = snapshot_download('stepfun-ai/Step-Audio-Tokenizer')
    target_tokenizer_path = os.path.join("models", "Step-Audio-Tokenizer")
    if os.path.exists(target_tokenizer_path):
        shutil.rmtree(target_tokenizer_path)
    shutil.copytree(tokenizer_path, target_tokenizer_path)
    print(f"Moved Tokenizer to {target_tokenizer_path}")
except Exception as e:
    print(f"Failed to download Tokenizer: {e}")

# EditX
print("Downloading Step-Audio-EditX...")
try:
    editx_path = snapshot_download('stepfun-ai/Step-Audio-EditX')
    target_editx_path = os.path.join("models", "Step-Audio-EditX")
    if os.path.exists(target_editx_path):
        shutil.rmtree(target_editx_path)
    shutil.copytree(editx_path, target_editx_path)
    print(f"Moved EditX to {target_editx_path}")
except Exception as e:
    print(f"Failed to download EditX: {e}")
