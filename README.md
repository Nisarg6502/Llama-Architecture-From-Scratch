# 🧠 Custom 150M Parameter LLM: Pre-Trained Base to Privacy Firewall

🎯 **Live Demo:** [Click here to test the PII Firewall live on HuggingFace!](https://huggingface.co/spaces/nisarg6502/Custom-Llama-150M-Model)  
📦 **Model Weights:** [Download from HuggingFace](https://huggingface.co/nisarg6502/Llama3-150M-PII-Redactor)

This repository demonstrates the complete, end-to-end lifecycle of a Large Language Model. It contains the code to build a custom 150M-parameter architecture from scratch in PyTorch, pre-train it on general text, and strictly fine-tune it to act as a local Personally Identifiable Information (PII) redaction microservice.

Sending raw user data to external APIs (like OpenAI or Anthropic) poses a massive security risk. This project solves that by providing a lightweight, locally hosted model that intercepts and scrubs PII *before* it leaves your server.

---

## 🏗️ Stage 1: Architecture & Upgrades
Rather than relying on the standard 2019 GPT-2 architecture, this model's internals were modernized to mimic Meta's **Llama 3** architecture, significantly improving training throughput, context understanding, and stability.

The model features 12 Transformer blocks, 12 attention heads, a hidden dimension of 768, and a context window of 512 tokens.

* **Rotary Positional Embeddings (RoPE):** Replaced standard absolute positional embeddings to provide better relative distance understanding between tokens.
* **RMSNorm:** Replaced standard `LayerNorm` to strip unnecessary mean calculations, accelerating the forward/backward pass.
* **SwiGLU Activations:** Replaced standard ReLU with Swish-Gated Linear Units in the Feed-Forward Networks for higher expressivity.

## 📊 Stage 2: Pre-Training Infrastructure
The base model was pre-trained from scratch on the **WikiText-103** dataset (1.8 million documents, ~103M tokens).

* **Hardware:** NVIDIA RTX 5090 (32GB VRAM) provisioned via RunPod.
* **Optimization:** Overcame CUDA Out-of-Memory (OOM) constraints by engineering a gradient accumulation pipeline (`batch_size=8`, `grad_accum_steps=8`), maintaining an effective batch size of 64 for stable convergence.

## 🛡️ Stage 3: Fine-Tuning for Data Privacy
Once the base model understood general English grammar, it was fine-tuned on the `ai4privacy/pii-masking-200k` dataset to recognize and scrub sensitive entities (names, emails, phone numbers, addresses). 

Instead of traditional next-token prediction, the training loop utilizes **Loss Masking**. The model is only penalized for missing the actual PII redaction tags, forcing it to learn strict data scrubbing without degrading its underlying language capabilities.

**Deployment Profile:**
* **Memory Footprint:** ~600MB (Calculated footprint for 150M parameters in 32-bit float).
* **Hardware Requirement:** Designed specifically for fast, headless inference on standard consumer CPU infrastructure without requiring dedicated GPU accelerators.

---

## 💻 How to Run Locally

Because this is a ~150M parameter model, it is designed for fast, cheap inference on standard CPUs. You can run it as an interactive web UI or a headless REST API.
