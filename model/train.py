import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from model import GPTModel, GPTConfig
from tokenizer import get_vocab_size

# ✅ Configs
BLOCK_SIZE = 64
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 3e-4
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
MODEL_PATH = "trained_model/gpt_custom.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Padding for batch

# ✅ Load super small dataset
dataset = load_dataset("daily_dialog", split="train[:1000]")  # ~45MB total dataset

# ✅ Tokenize dialogue
def tokenize_function(example):
    joined_text = " ".join(example["dialog"])
    tokens = tokenizer(
        joined_text,
        truncation=True,
        padding="max_length",
        max_length=BLOCK_SIZE,
        return_tensors="pt"
    )
    return {"input_ids": tokens["input_ids"].squeeze()}

tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)
tokenized_dataset.set_format(type="torch")

# ✅ Dataloader
dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ✅ Define model
config = GPTConfig(
    vocab_size=get_vocab_size(),
    block_size=BLOCK_SIZE,
    n_layer=NUM_LAYERS,
    n_head=NUM_HEADS,
    n_embd=EMBED_DIM
)

model = GPTModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ✅ Training loop
model.train()
print("🚀 Training started...")
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        inputs = batch["input_ids"].to(device)
        targets = inputs.clone()

        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# ✅ Save model
torch.save(model.state_dict(), MODEL_PATH)
print(f"📦 Model saved to {MODEL_PATH}")
