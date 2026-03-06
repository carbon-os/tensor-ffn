import json
import os

# Adjust path if needed
MODEL_DIR = "../trainer/qwen2.5-0.5b-instruct/"
SAFE_FILE = os.path.join(MODEL_DIR, "model.safetensors")

def check():
    if not os.path.exists(SAFE_FILE):
        print(f"Error: {SAFE_FILE} not found.")
        return

    with open(SAFE_FILE, "rb") as f:
        # Read header length (first 8 bytes, little-endian uint64)
        header_len_bytes = f.read(8)
        header_len = int.from_bytes(header_len_bytes, 'little')
        
        # Read header JSON
        header_json = f.read(header_len).decode('utf-8')
        header = json.loads(header_json)
        
        # Check the dtype of the embedding layer
        if "model.embed_tokens.weight" in header:
            dtype = header["model.embed_tokens.weight"]["dtype"]
            print(f"TENSOR DTYPE: {dtype}")
            
            if dtype == "F16":
                print(">>> PROBLEM DETECTED: Weights are F16, but Trainer expects BF16.")
                print(">>> The bits are being misinterpreted, causing high loss.")
            elif dtype == "BF16":
                print(">>> DType is BF16 (Correct). The issue is elsewhere.")
            else:
                print(f">>> Unknown dtype: {dtype}")
        else:
            print("Could not find embedding weights in header.")

if __name__ == "__main__":
    check()