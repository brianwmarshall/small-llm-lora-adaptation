# Local Small LLM Adaptation Pipeline (Qwen2.5-1.5B via LoRA)

A lightweight pipeline demonstrating end-to-end Low-Rank Adaptation (LoRA) fine-tuning of an open-weights LLM using Unsloth, quantized export to GGUF, and offline deployment via Ollama.

## Overview
- **Base Model:** Qwen/Qwen2.5-1.5B-Instruct
- **Fine-Tuning Method:** LoRA (Low-Rank Adaptation) target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Dataset:** Instruction-following conversation subset (`mlabonne/FineTome-100k`)
- **Quantization:** 4-bit (`q4_k_m` GGUF)
- **Inference Runtime:** Local execution via Ollama / Apple Silicon

---

## Technical Workflow

1. **Parameter-Efficient Fine-Tuning (PEFT):**
   - Frozen base model weights in 4-bit precision to minimize memory overhead during backpropagation.
   - Rank $r = 16$ adapter matrices injected into attention and feed-forward MLP layers.

2. **Quantization & Export:**
   - Compiled via `llama.cpp` to `Q4_K_M` GGUF format (~1.1 GB).

3. **Local Deployment:**
   - Configured custom `Modelfile` and registered model into local Ollama instance.

---

## Quickstart & Local Setup

### 1. Build & Register Model in Ollama
```bash
# Clone repository
git clone [https://github.com/brianwmarshall/small-llm-lora-adaptation.git](https://github.com/brianwmarshall/small-llm-lora-adaptation.git)
cd small-llm-lora-adaptation

# Create Modelfile pointing to local GGUF
echo "FROM ./qwen2.5-1.5b-instruct.Q4_K_M.gguf" > Modelfile

# Create and run
ollama create my-custom-qwen -f Modelfile
ollama run my-custom-qwen
