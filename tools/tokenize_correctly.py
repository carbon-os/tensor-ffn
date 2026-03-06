import os
import struct
import array
from transformers import AutoTokenizer

# Configuration
MODEL_DIR = "../trainer/qwen2.5-0.5b-instruct/"
DATA_DIR = "/root/markdownv2"  # Your input folder
OUT_FILE = "../trainer/corpus.bin"

# 1. Load the official tokenizer (guaranteed to be correct)
print(f"Loading tokenizer from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

all_ids = []
file_count = 0

# 2. Walk directories and tokenize
print(f"Scanning {DATA_DIR}...")
for root, _, files in os.walk(DATA_DIR):
    for filename in files:
        # Filter for text/code files
        if filename.endswith(('.go', '.md', '.txt', '.c', '.h', '.cpp', '.py', '.sh')):
            path = os.path.join(root, filename)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    if not text.strip(): continue
                    
                    # Encode and append EOS
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    all_ids.extend(ids)
                    all_ids.append(tokenizer.eos_token_id)
                    file_count += 1
            except Exception as e:
                print(f"Skipping {path}: {e}")

print(f"Tokenized {file_count} files.")
print(f"Total tokens: {len(all_ids)}")

# 3. Write binary format for tensor-ffn
# Format: [Magic: 4b] [Version: 4b] [Count: 8b] [Tokens: 4b * N]
CORPUS_MAGIC = 0x54464E4E
CORPUS_VERSION = 1

print(f"Writing to {OUT_FILE}...")
with open(OUT_FILE, 'wb') as f:
    # Header
    f.write(struct.pack('<I', CORPUS_MAGIC))
    f.write(struct.pack('<I', CORPUS_VERSION))
    f.write(struct.pack('<Q', len(all_ids)))
    
    # Body (Fast write using array)
    # 'I' = unsigned int (32-bit), matches uint32_t in C++
    token_array = array.array('I', all_ids)
    token_array.tofile(f)

print("Done. You can now train.")