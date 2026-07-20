import torch
import torch.nn as nn

CONFIG = {
    "embed_dim":128,
    "hidden_dim":128,
    "dropout":0.3
}

class BiLSTMAttention(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()

    self.embedding = nn.Embedding(
        vocab_size,
        CONFIG["embed_dim"],
        padding_idx = 0
    )

    self.lstm = nn.LSTM(
        CONFIG["embed_dim"],
        CONFIG["hidden_dim"],
        batch_first = True,
        bidirectional = True
    )

    self.attention = nn.Linear(CONFIG["hidden_dim"] * 2 ,1)
    self.dropout   = nn.Dropout(CONFIG["dropout"])
    self.fc        = nn.Linear(CONFIG["hidden_dim"] * 2 ,5)

  def forward(self, input_ids, attention_mask):
    x = self.embedding(input_ids)
    x, _ = self.lstm(x)

    scores = self.attention(x).squeeze(-1)

    scores = scores.masked_fill(attention_mask == 0, -1e9)

    weights = torch.softmax(scores, dim=1)

    # Apply weights to LSTM outputs
    context = torch.sum(
        x * weights.unsqueeze(-1),
        dim = 1
    )

    context = self.dropout(context)
    return self.fc(context)