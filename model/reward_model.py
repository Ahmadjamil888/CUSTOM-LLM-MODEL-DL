import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class RewardModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output is a scalar reward
        )

    def forward(self, prompt, response):
        """
        prompt: str
        response: str
        """
        input_text = f"{prompt} [SEP] {response}"
        tokens = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        input_ids = tokens["input_ids"].to(self.encoder.device)
        attention_mask = tokens["attention_mask"].to(self.encoder.device)

        with torch.no_grad():  # freeze encoder for simplicity (optional)
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        reward = self.classifier(cls_embedding)

        return reward.squeeze()  # scalar

    def compute_reward(self, prompt, response):
        self.eval()
        with torch.no_grad():
            return self.forward(prompt, response).item()
