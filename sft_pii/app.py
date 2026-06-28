import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
from dataclasses import dataclass
from huggingface_hub import hf_hub_download

# ==========================================
# 1. ARCHITECTURE (Required to load pii_model.pt)
# ==========================================
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
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)

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
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rotary(q, seq_len), self.rotary(k, seq_len)
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = self.attn_dropout(F.softmax(attn_scores, dim=-1))
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

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

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embd_dropout = nn.Dropout(config.embd_dropout)
        self.layers = nn.ModuleList([TransformerBlock(config.d_model, config.num_heads, config.dropout) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        x = self.embd_dropout(self.token_embedding(input_ids))
        mask = create_causal_mask(seq_len, input_ids.device)
        for layer in self.layers: x = layer(x, mask)
        return self.lm_head(self.final_norm(x))

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=0.2, stop_token_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
            
            logits = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if stop_token_id is not None and next_token.item() == stop_token_id:
                break
                
        return input_ids

# ==========================================
# 2. GRADIO INTERFACE SETUP (The Pro Way)
# ==========================================
print("Starting up PII Firewall...")
device = torch.device("cpu")
tokenizer = SimpleTokenizer()

try:
    print("Downloading weights from Hugging Face Hub...")
    # Replace 'your-username/pii-redactor-150m' with your actual HF repo ID
    # Replace the filename with your exact uploaded .pt file
    model_path = hf_hub_download(
        repo_id="nisarg6502/Llama3-150M-PII-Redactor", 
        filename="pii_model_epoch_3.pt"
    )
    
    print("Loading weights into memory...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    model_loaded = True
    print("Model loaded successfully!")
    
except Exception as e:
    model_loaded = False
    error_msg = str(e)
    print(f"Failed to load model: {error_msg}")

# ... (The rest of your scrub_text function and Gradio UI code stays exactly the same!) ...

def scrub_text(user_input):
    if not model_loaded:
        return f"Error: Could not load pii_model.pt."
    
    if not user_input.strip():
        return "Please enter text to redact."

    # INVISIBLE FORMATTING: The user just types normal text, but we wrap it in the triggers!
    prompt = f"[RAW] {user_input} [REDACTED] "
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    # Generate text with low temperature for strict factual output
    output_ids = model.generate(input_ids, max_new_tokens=100, temperature=0.2, stop_token_id=tokenizer.eos_token_id)
    full_output = tokenizer.decode(output_ids[0].tolist())
    
    # Extract only the redacted portion to show the user
    if "[REDACTED]" in full_output:
        final_result = full_output.split("[REDACTED]")[-1].replace("<|endoftext|>", "").strip()
    else:
        final_result = full_output

    return final_result

# Build the Web UI
with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ Local Privacy Firewall (150M Parameters)")
    gr.Markdown("This model was fine-tuned from scratch to detect and redact Personally Identifiable Information (PII) before it ever leaves the local network.")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(lines=4, label="Raw Text (Contains PII)", placeholder="Please send the receipt to michael.scott@dundermifflin.com...")
            submit_btn = gr.Button("Scrub Data", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(lines=4, label="Safe Text (Redacted)")

    submit_btn.click(fn=scrub_text, inputs=[prompt_input], outputs=output_text)

demo.launch(share=True, theme=gr.themes.Monochrome())
