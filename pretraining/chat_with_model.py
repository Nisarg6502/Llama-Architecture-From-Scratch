import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
from dataclasses import dataclass

# ====================================================================
# 1. ARCHITECTURE DEFINITIONS (Needed to rebuild the brain locally)
# ====================================================================
@dataclass
class TokenizerConfig:
    name: str = "gpt2"
    vocab_size: int = 50257

class SimpleTokenizer:
    def __init__(self, config=None):
        self.config = config or TokenizerConfig()
        self.enc = tiktoken.get_encoding(self.config.name)
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = self.enc.encode(self.eos_token, allowed_special={self.eos_token})[0]

    def encode(self, text):
        return self.enc.encode(text, allowed_special={self.eos_token})

    def decode(self, ids):
        return self.enc.decode(ids)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048, theta=10000.0):
        super().__init__()
        assert d_model % 2 == 0
        dim_indices = torch.arange(0, d_model, 2).float()
        inv_freq = 1.0 / (theta ** (dim_indices / d_model))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = freqs.repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    @staticmethod
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x, seq_len):
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return (x * cos) + (self.rotate_half(x) * sin)

def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rotary(q, seq_len)
        k = self.rotary(k, seq_len)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        return self.resid_dropout(output)

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model, expansion_factor=4):
        super().__init__()
        hidden_dim = expansion_factor * d_model
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    max_seq_len: int = 512
    dropout: float = 0.1
    embd_dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 10000
    batch_size: int = 8
    grad_accum_steps: int = 8
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embd_dropout = nn.Dropout(config.embd_dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        x = self.token_embedding(input_ids)
        x = self.embd_dropout(x)
        mask = create_causal_mask(seq_len, input_ids.device)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, None

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

# ====================================================================
# 2. RUNNING THE MODEL INTERACTIVELY
# ====================================================================
def main():
    print("Loading Model...")
    # Use CPU by default for local laptop inference (or "mps" for Mac M1/M2/M3)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the checkpoint
    checkpoint = torch.load("model.pt", map_location=device, weights_only=False)
    
    # Rebuild model with the saved config
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    tokenizer = SimpleTokenizer()
    print("\nModel loaded successfully! Type 'quit' to exit.")
    print("-" * 50)

    # Interactive chat loop
    while True:
        prompt = input("\nStart a sentence: ")
        if prompt.lower() in ['quit', 'exit']:
            break
            
        print("\nThinking...")
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        
        # Generate 100 new tokens
        output_ids = model.generate(input_ids, max_new_tokens=100, temperature=0.7, top_k=40)
        generated_text = tokenizer.decode(output_ids[0].tolist())
        
        print("\n--- Output ---")
        print(generated_text)
        print("--------------")

if __name__ == "__main__":
    main()
