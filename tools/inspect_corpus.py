import struct
import os
from transformers import AutoTokenizer

# Config
MODEL_DIR = "../trainer/qwen2.5-0.5b-instruct/"
CORPUS_FILE = "../trainer/corpus.bin"

def inspect():
    if not os.path.exists(CORPUS_FILE):
        print("Error: corpus.bin not found.")
        return

    print(f"Loading tokenizer from {MODEL_DIR}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    print(f"Reading {CORPUS_FILE}...")
    with open(CORPUS_FILE, "rb") as f:
        # Read Header
        # Magic (4), Version (4), Count (8)
        header_data = f.read(16)
        if len(header_data) < 16:
            print("Error: File too short for header.")
            return
            
        magic, version, count = struct.unpack('<IIQ', header_data)
        
        print("--- Header Info ---")
        print(f"Magic:   0x{magic:08X} (Expected: 0x54464E4E)")
        print(f"Version: {version}")
        print(f"Count:   {count} tokens")
        
        # Read First 50 Tokens
        # Each token is a uint32 (4 bytes)
        token_data = f.read(50 * 4)
        tokens = struct.unpack(f'<{len(token_data)//4}I', token_data)
        
        print("\n--- First 50 Tokens (Raw IDs) ---")
        print(tokens)
        
        print("\n--- Decoded Text ---")
        decoded = tokenizer.decode(tokens)
        print(f"'{decoded}'")

if __name__ == "__main__":
    inspect()