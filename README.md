# 🧠 Custom 150M Parameter Language Model

A 150-million parameter autoregressive language model built entirely from scratch in PyTorch.

Rather than relying on the standard 2019 GPT-2 architecture, this model's internals were modernized to mimic Meta's **Llama 3** architecture, significantly improving training throughput, context understanding, and stability.

🔗 [**Click here to chat with the model live on HuggingFace!**](#) *(Add your HuggingFace Space link here later)*

## 🏗️ Architecture & Upgrades

The model features 12 Transformer blocks, 12 attention heads, and a hidden dimension of 768. To modernize the architecture, the following upgrades were implemented:

* **Rotary Positional Embeddings (RoPE):** Replaced standard absolute positional embeddings with RoPE to provide better relative distance understanding between tokens.

* **RMSNorm:** Replaced standard `LayerNorm` with Root Mean Square Normalization to strip unnecessary mean calculations, accelerating the forward/backward pass.

* **SwiGLU Activations:** Replaced standard ReLU with Swish-Gated Linear Units in the Feed-Forward Networks for higher expressivity.

## 📊 Training & Infrastructure

The model was pre-trained on the **WikiText-103** dataset (1.8 million documents, ~103M tokens).

* **Hardware:** NVIDIA RTX 5090 (32GB VRAM) provisioned via RunPod.

* **Context Window:** 512 tokens.

* **Optimization:** Overcame CUDA Out-of-Memory (OOM) constraints by engineering a gradient accumulation pipeline (`batch_size=8`, `grad_accum_steps=8`), maintaining an effective batch size of 64 for stable convergence.

## 💻 How to Run Locally

Because this is a ~150M parameter model, it is highly optimized for fast inference on standard consumer CPUs.

1. Clone this repository.

2. Install requirements: `pip install torch tiktoken`

3. Download the `model.pt` weights (Link provided in the HuggingFace space).

4. Run the interactive chat script:

    ```bash
    python chat_with_model.py
    ```
