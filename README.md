# Generative Adversarial Network (GAN) — CIFAR-10

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A faithful, production-quality PyTorch implementation of the foundational GAN paper.**

*Based on: [Generative Adversarial Nets — Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661)*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [The Core Idea — The Minimax Game](#-the-core-idea--the-minimax-game)
- [Project Structure](#-project-structure)
- [Tech Stack](#️-tech-stack)
- [Architecture](#️-architecture)
  - [Generator](#generator-gz-θg)
  - [Discriminator](#discriminator-dx-θd)
  - [Weight Initialization](#weight-initialization)
- [Training Configuration](#️-training-configuration)
  - [Hyperparameters](#hyperparameters)
  - [The Training Loop](#the-training-loop)
  - [The Saturation Fix](#the-generator-saturation-fix)
  - [Checkpointing](#checkpointing)
- [Evaluation — Memorization Check](#-evaluation--memorization-check)
- [Results & Visualizations](#-results--visualizations)
- [Getting Started](#-getting-started)
- [Author](#-author)

---

## 🧩 Overview

This project implements the original GAN framework applied to the **CIFAR-10** dataset (50,000 training images across 10 classes: planes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks).

The key goal is to train a **Generator** network that produces synthetic 32×32 RGB images so convincing that a simultaneously trained **Discriminator** network cannot distinguish them from real CIFAR-10 images. The project follows the paper closely while incorporating modern DCGAN-style architectural improvements (convolutional layers, batch normalization) for training stability on image data.

---

## 🧠 The Core Idea — The Minimax Game

The GAN framework is a two-player zero-sum game between two neural networks:

- **Generator ($G$):** Takes random noise $z \sim p_z(z)$ as input and outputs a synthetic image. Its goal is to **fool** the Discriminator.
- **Discriminator ($D$):** Takes an image (real or fake) as input and outputs a scalar probability of authenticity. Its goal is to **correctly classify** real vs. fake.

Both networks are trained simultaneously to optimize the following value function:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

At the **Nash Equilibrium**, the Generator perfectly replicates the real data distribution $p_g = p_{\text{data}}$, and the Discriminator is maximally confused: $D(x) = 0.5$ for all inputs.

---

## 📁 Project Structure

```
gan-cifar10/
│
├── gan.ipynb                   # Main notebook — full implementation
│
├── data/                       # CIFAR-10 dataset (auto-downloaded)
│   └── cifar-10-batches-py/
│
├── results/                    # Saved output visualizations
│   ├── final_generated.png     # 8×8 grid of final generated images
│   ├── generation_progress.png # Generator evolution across epochs
│   ├── training_curves.png     # Loss & discriminator confidence plots
│   └── nearest_neighbor.png   # Generated vs. nearest real comparison
│
└── checkpoints/                # Saved model weights
    ├── gan_epoch_100.pth
    ├── gan_epoch_200.pth
    ├── gan_epoch_300.pth
    └── gan_final.pth           # Final model (G + D state dicts + losses)
```

---

## 🛠️ Tech Stack

| Category | Library |
|---|---|
| **Core Framework** | `torch`, `torch.nn`, `torch.optim` |
| **Computer Vision** | `torchvision`, `torchvision.transforms`, `torchvision.utils` |
| **Evaluation** | `scipy.linalg`, `torchvision.models` (Inception-v3) |
| **Data & Math** | `numpy` |
| **Visualization** | `matplotlib.pyplot` |
| **Utilities** | `os`, `time` |

---

## 🏗️ Architecture

Both networks follow the **DCGAN** convention: fully convolutional, no fully connected layers in the main pipeline, and weights initialized from a normal distribution.

### Weight Initialization

All convolutional and transposed-convolutional weights are initialized from $\mathcal{N}(0, 0.02)$. BatchNorm weights are initialized from $\mathcal{N}(1.0, 0.02)$ with biases set to $0$. This follows the standard DCGAN recipe for stable early-stage training.

```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
```

---

### Generator $G(z;\ \theta_g)$

The Generator maps a **100-dimensional latent noise vector** $z \sim \mathcal{U}(-1, 1)$ to a full **32×32 RGB image** through a series of fractional-strided (transposed) convolutions.

| Layer | Operation | Output Shape |
|---|---|---|
| **Input** | $z \sim \text{Uniform}(-1, 1)$ | $(B,\ 100,\ 1,\ 1)$ |
| **Block 1** | `ConvTranspose2d(100 → 512, k=4, s=1, p=0)` → `BatchNorm2d` → `ReLU` | $(B,\ 512,\ 4,\ 4)$ |
| **Block 2** | `ConvTranspose2d(512 → 256, k=4, s=2, p=1)` → `BatchNorm2d` → `ReLU` | $(B,\ 256,\ 8,\ 8)$ |
| **Block 3** | `ConvTranspose2d(256 → 128, k=4, s=2, p=1)` → `BatchNorm2d` → `ReLU` | $(B,\ 128,\ 16,\ 16)$ |
| **Output** | `ConvTranspose2d(128 → 3, k=4, s=2, p=1)` → `Sigmoid` | $(B,\ 3,\ 32,\ 32)$ |

**Design choices:**
- `ReLU` activations in hidden layers ensure strong, non-sparse gradient flow through the upsampling path.
- `Sigmoid` at the output maps pixel values to $[0, 1]$, consistent with the normalized real images.
- `BatchNorm2d` on every hidden block accelerates convergence and reduces sensitivity to learning rate.
- `bias=False` on all `ConvTranspose2d` layers since BatchNorm already handles the bias offset.

---

### Discriminator $D(x;\ \theta_d)$

The Discriminator takes a **32×32 RGB image** (real or generated) and outputs a **scalar probability** that the image is real.

| Layer | Operation | Output Shape |
|---|---|---|
| **Input** | Real or Fake image | $(B,\ 3,\ 32,\ 32)$ |
| **Block 1** | `Conv2d(3 → 128, k=4, s=2, p=1)` → `LeakyReLU(0.2)` → `Dropout(0.3)` | $(B,\ 128,\ 16,\ 16)$ |
| **Block 2** | `Conv2d(128 → 256, k=4, s=2, p=1)` → `BatchNorm2d` → `LeakyReLU(0.2)` → `Dropout(0.3)` | $(B,\ 256,\ 8,\ 8)$ |
| **Block 3** | `Conv2d(256 → 512, k=4, s=2, p=1)` → `BatchNorm2d` → `LeakyReLU(0.2)` → `Dropout(0.3)` | $(B,\ 512,\ 4,\ 4)$ |
| **Output** | `Conv2d(512 → 1, k=4, s=1, p=0)` → `Sigmoid` | $(B,\ 1,\ 1,\ 1)$ |

**Design choices:**
- `LeakyReLU(0.2)` replaces the paper's Maxout activations. It preserves gradient flow for negative inputs, which is critical to avoid the Discriminator dying early.
- `Dropout(0.3)` on every hidden block acts as strong regularization to prevent the Discriminator from memorizing the training set rather than learning meaningful features.
- No `BatchNorm` on the first block (common DCGAN practice), but applied to all subsequent hidden blocks.
- `Sigmoid` at output delivers a probability score in $(0, 1)$.

---

## ⚙️ Training Configuration

### Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `LATENT_DIM` | `100` | Size of the input noise vector $z$ |
| `IMG_SIZE` | `32` | CIFAR-10 native resolution |
| `IMG_CHANNELS` | `3` | RGB |
| `FEATURE_G` | `128` | Base feature map multiplier for Generator |
| `FEATURE_D` | `128` | Base feature map multiplier for Discriminator |
| `BATCH_SIZE` | `128` | Mini-batch size |
| `LR` | `2e-4` | Learning rate for both Adam optimizers |
| `BETA1` | `0.5` | Adam momentum ($\beta_1$); lower than default for GAN stability |
| `NUM_EPOCHS` | `300` | Total training epochs |
| `K_STEPS` | `1` | Discriminator update steps per Generator step |
| `SEED` | `42` | For reproducibility |

### Dataset Preprocessing

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Maps to [-1, 1]
])
```

The normalization maps pixel values from $[0, 1]$ to $[-1, 1]$. The DataLoader uses `pin_memory=True` for faster GPU transfers and `drop_last=True` to ensure consistent batch sizes.

---

### The Training Loop

The algorithm strictly alternates between one Discriminator update and one Generator update ($k = 1$) per batch.

**Step 1 — Train the Discriminator ($D$):**

The Discriminator is updated to maximize its ability to distinguish real from fake images. It is trained on both real data (label = 1) and freshly generated fakes (label = 0) in each step.

$$\mathcal{L}_D = -\left[\log D(x) + \log(1 - D(G(z)))\right]$$

**Step 2 — Train the Generator ($G$):**

The Generator is updated to fool the Discriminator. See the saturation fix below for the exact objective used.

The loop also tracks four metrics per batch:
- `Loss_D` — Discriminator total loss
- `Loss_G` — Generator loss
- `D(x)` — Average Discriminator confidence on real images (should start near 1, converge toward 0.5)
- `D(G(z))` — Average Discriminator confidence on fake images (should start near 0, converge toward 0.5)

---

### The Generator Saturation Fix

Early in training, when the Discriminator is much stronger than the Generator, the original objective $\log(1 - D(G(z)))$ **saturates** — it provides near-zero gradients that stall Generator learning.

**Solution:** Instead of minimizing $\log(1 - D(G(z)))$, the Generator is trained to **maximize** $\log D(G(z))$:

$$\mathcal{L}_G = -\log D(G(z))$$

This is mathematically equivalent to training the Generator with **real labels** (label = 1) on the fake images. It produces much stronger gradients in the early stages and is the standard practical fix recommended in the original paper.

---

### Checkpointing

Model checkpoints (Generator, Discriminator, both optimizers, and loss histories) are saved every 100 epochs and at the end of training:

```
checkpoints/
├── gan_epoch_100.pth
├── gan_epoch_200.pth
├── gan_epoch_300.pth
└── gan_final.pth
```

To resume training or load for inference:

```python
checkpoint = torch.load("checkpoints/gan_final.pth")
netG.load_state_dict(checkpoint["netG_state_dict"])
netD.load_state_dict(checkpoint["netD_state_dict"])
```

---

## 📊 Evaluation — Memorization Check

> *"Adversarial nets do not explicitly represent the probability distribution $p_g(x)$."* — Goodfellow et al., 2014

Since GANs have no explicit likelihood function, evaluation is performed visually and via metric-based methods.

### 1. Training Curves (`results/training_curves.png`)
Two side-by-side plots are generated after training:
- **Loss curves** for Generator and Discriminator over all epochs.
- **Discriminator confidence** tracking $D(x)$ (should decrease from ~1 to ~0.5) and $D(G(z))$ (should increase from ~0 to ~0.5). Convergence of both toward the 0.5 equilibrium line indicates a healthy training dynamic.

### 2. Generation Progress (`results/generation_progress.png`)
Snapshots of a **fixed noise vector** fed through the Generator are captured at epoch 1 and every 25 epochs thereafter. This directly visualizes how the Generator's output evolves from random noise into structured images over the 300-epoch run.

### 3. Final Generated Grid (`results/final_generated.png`)
An 8×8 grid of 64 images generated from fresh random noise at the end of training.

### 4. Nearest Neighbor Search (`results/nearest_neighbor.png`)
The core memorization check. For each of 8 generated images, the closest real CIFAR-10 training image is found using **L2 distance in pixel space** (batched for efficiency):

```python
dists = torch.norm(real_batch_flat - fake_flat, dim=1)
```

If the nearest real neighbor looks significantly different from the generated image (high L2 distance, different content), this is strong evidence that the Generator has learned to synthesize **novel features** rather than memorize and reproduce training samples.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib scipy
```

A CUDA-capable GPU is strongly recommended for the 300-epoch training run. The code automatically detects and uses CUDA if available:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Run

1. Clone the repository:
   ```bash
   git clone https://github.com/nadeem-ahmad3/gan-cifar10.git
   cd gan-cifar10
   ```

2. Launch the notebook:
   ```bash
   jupyter notebook gan.ipynb
   ```

3. Run all cells sequentially. CIFAR-10 will be downloaded automatically on the first run to `./data/`.

### Generate Images from a Saved Checkpoint

```python
import torch

# Load checkpoint
checkpoint = torch.load("checkpoints/gan_final.pth", map_location=DEVICE)
netG.load_state_dict(checkpoint["netG_state_dict"])
netG.eval()

# Generate
with torch.no_grad():
    noise = torch.randn(64, 100, 1, 1, device=DEVICE)
    fake_images = netG(noise).cpu()
```

---

## 👨‍💻 Author

<div align="center">

**Nadeem Ahmad**

*Software Engineer & AI Developer*
*Final Year Software Engineering Student — FAST NUCES*

[![Email](https://img.shields.io/badge/Email-engrnadeem.26%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:engrnadeem.26@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Nadeem%20Ahmad-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nadeem-ahmad3/)

</div>

---

<div align="center">

**Reference:** Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). *Generative Adversarial Nets*. NeurIPS 2014. [arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)

</div>
