from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding works

# Expose functions
def encode(text, max_length=64):
    return tokenizer.encode(text, truncation=True, max_length=max_length, return_tensors="pt")[0]

def decode(token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True)

def get_vocab_size():
    return tokenizer.vocab_size

def get_eos_token_id():
    return tokenizer.eos_token_id  # <- ✅ This line is key
