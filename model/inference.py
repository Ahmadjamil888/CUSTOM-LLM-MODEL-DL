import torch
from model import GPTModel, GPTConfig
from tokenizer import encode, decode, get_vocab_size

# Config
BLOCK_SIZE = 64
N_LAYERS = 4
N_HEADS = 4
EMBED_DIM = 128
MODEL_PATH = "trained_model/gpt_custom.pth"

# Load model
config = GPTConfig(
    vocab_size=get_vocab_size(),
    block_size=BLOCK_SIZE,
    n_layer=N_LAYERS,
    n_head=N_HEADS,
    n_embd=EMBED_DIM
)

model = GPTModel(config)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(prompt, max_new_tokens=50):
    """
    Generate text using the trained GPT model from a given prompt.
    """
    model.eval()
    
    input_ids = encode(prompt, max_length=BLOCK_SIZE).unsqueeze(0).to(device)  # [1, T]

    for _ in range(max_new_tokens):
        input_crop = input_ids[:, -BLOCK_SIZE:]  # crop to block size
        with torch.no_grad():
            logits = model(input_crop)
            next_token_logits = logits[:, -1, :]  # last token's logits
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

        input_ids = torch.cat((input_ids, next_token), dim=1)

    output = input_ids[0]
    text = decode(output)
    
    # Return only the new response (exclude original prompt)
    return text[len(prompt):].strip()
