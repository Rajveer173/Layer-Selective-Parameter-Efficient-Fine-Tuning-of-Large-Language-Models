# Layer-Selective Parameter-Efficient Fine-Tuning of Large Language Models

## Overview
Fine-tuning large language models (LLMs) is computationally expensive and often unnecessary across all layers.  
This project investigates whether **fine-tuning only a subset of transformer layers** using parameter-efficient methods can significantly reduce training cost while maintaining stable performance.

We propose a **layer-selective LoRA fine-tuning strategy**, where Low-Rank Adaptation (LoRA) is applied only to the final transformer layers instead of all layers.

---

## Key Idea
Instead of applying LoRA adapters to every transformer layer, we apply them **only to the last 8 layers** of the model and compare against a standard QLoRA baseline.

**Research Question:**
> Do all layers of an LLM need fine-tuning, or can selective adaptation achieve similar results with lower cost?

---

## Method
- **Base Model:** Phi-2 (2.7B parameters)
- **Fine-Tuning Method:** QLoRA (4-bit quantization + LoRA)
- **Baseline:** LoRA applied to all 24 transformer layers
- **Proposed Method:** LoRA applied only to the last 8 layers
- **Dataset:** Alpaca (5,000 instruction samples)
- **Hardware:** Google Colab (Tesla T4 GPU)

---

## Results
| Method | Layers Tuned | Trainable Parameters | Trainable % | Training Stability |
|------|-------------|----------------------|-------------|-------------------|
| Baseline QLoRA | 24 | Higher | ~0.4% | Stable |
| **Layer-Selective LoRA** | **8 (last)** | **3.93M** | **0.14%** | **Stable** |

Despite adapting only a small subset of layers, the model:
- Converges stably
- Produces coherent outputs
- Uses significantly fewer trainable parameters

---

## Qualitative Example
**Prompt:**  
Explain the bias-variance tradeoff in machine learning.

**Output:**  
The layer-selective fine-tuned model produces a coherent explanation discussing overfitting, underfitting, and the tradeoff between bias and variance.

---

## Repository Structure
