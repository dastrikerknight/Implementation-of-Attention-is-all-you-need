# Implementation-of-Attention-is-all-you-need

# Transformer-Based Neural Machine Translation (NMT) with PyTorch

## 📌 Introduction
This repository implements a **Transformer** model for Neural Machine Translation (NMT) based on the seminal paper:

> **"Attention Is All You Need"** – Vaswani et al., 2017  
> https://arxiv.org/abs/1706.03762

The model translates between **English** and **German** using the **Multi30k** dataset.  
The Transformer architecture replaces recurrent and convolutional structures with pure **self-attention**, enabling faster training and better handling of long-range dependencies.

<p align="center">
  <img src="https://raw.githubusercontent.com/asyml/texar-pytorch/master/docs/_static/img/transformer.png" width="700"/>
</p>

*Figure: The Transformer architecture, showing the encoder-decoder structure and attention mechanisms.*

---

## 🧠 How the Transformer Works (Paper Summary)
The Transformer is composed of:
1. **Encoder** – Reads the source sentence and produces context-aware vector representations.  
   - Each encoder layer consists of:
     - **Multi-Head Self-Attention** (source attends to itself)
     - **Feed Forward Network**
     - **Residual Connections + LayerNorm**
2. **Decoder** – Generates the target sentence token-by-token.  
   - Each decoder layer consists of:
     - **Masked Multi-Head Self-Attention** (target attends to itself, future tokens masked)
     - **Encoder-Decoder Attention** (decoder attends to encoder outputs)
     - **Feed Forward Network**
     - **Residual Connections + LayerNorm**
3. **Positional Encoding** – Adds sequence order information to embeddings since attention is position-agnostic.
4. **Attention Mechanism** – Allows the model to dynamically focus on relevant words regardless of their distance in the sequence.

---

## ⚙️ Hyperparameters
| Parameter         | Value     | Description |
|-------------------|-----------|-------------|
| `d_model`         | 512       | Embedding size / model hidden size |
| `n_head`          | 8         | Number of attention heads |
| `ffn_hidden`      | 2048      | Feed-forward hidden layer size |
| `n_layers`        | 6         | Encoder and decoder layers |
| `dropout`         | 0.1       | Dropout rate |
| `max_len`         | 100       | Maximum sequence length |
| `batch_size`      | 32        | Training batch size |
| `learning_rate`   | 1e-4      | Adam optimizer learning rate |
| `epochs`          | 30        | Training epochs (planned) |

---

## 📦 Packages Used
| Package         | Usage |
|-----------------|-------|
| **torch**       | Core deep learning framework (model, training loop, GPU acceleration) |
| **torchtext**   | Dataset loading (Multi30k), token processing |
| **sentencepiece** | Tokenization (subword units) |
| **tqdm**        | Progress bars for training and evaluation |
| **numpy**       | Array operations |
| **math**        | Positional encoding calculations |

---

## 🚫 Computational Limitation
Training a Transformer from scratch on **Multi30k** with a **T4 GPU** (as provided by Google Colab free tier) is computationally demanding:
- The **T4 GPU** in free Colab sessions offers **~12 GB VRAM** and **~12 hours max runtime**.
- A full Transformer (6-layer encoder/decoder, `d_model=512`) requires **several days** of training for convergence even on small datasets like Multi30k.
- Free Colab sessions reset after 12 hours, leading to training interruptions and loss of progress unless checkpointing is implemented.
- Training at reasonable speeds would require **multi-GPU** or **high-memory GPUs** like V100/A100 for practical convergence.

Due to these limitations, **full training was not performed** here.  
Instead, a **dummy dataset test** was used to validate the full pipeline.

---

## 🧪 Dummy Test Results
A small **synthetic dataset** was used to ensure that:
- Data loading, tokenization, batching, and masks work correctly
- Model forward/backward pass runs without shape errors
- Loss decreases as expected

### Results:
| Metric       | Value |
|--------------|-------|
| Train Loss   | **2.53** |
| Validation Loss | **2.76** |

These results confirm the implementation is **functionally correct**.

---

## 🚀 How to Run
1. **Install dependencies**:
```bash
pip install torch torchtext sentencepiece tqdm
