# 👗 AI-Powered Fashion Designer using GANs

A two-stage generative AI pipeline that creates realistic fashion designs from scratch using Generative Adversarial Networks (GANs).This project showcases how AI can assist designers in generating sketch outlines and rendering them into vibrant, fashion-ready images.

---

## 📌 Project Overview

This project simulates the traditional fashion design workflow using AI, from idea generation (via sketches) to full-color rendering:

- **Sketch GAN**: Generates grayscale fashion sketches from random noise vectors.
- **Render GAN**: Transforms these sketches into realistic, high-resolution, color garment images.
- **Classifier**: Validates the realism of synthetic images using CNN-based evaluation.

---

## 📂 Dataset & Preprocessing

We used the [DeepFashion2 Dataset](https://www.kaggle.com/datasets/thusharanair/deepfashion2-original-with-dataframes) for training.

- Resized all images to `128x128`
- Applied **Canny edge detection** for sketch creation
- Normalized pixel intensity to [-1, 1]
- Trained on 5,000 image-sketch pairs

---

## 🧠 Architecture

### 1️⃣ Sketch GAN

- **Input**: 100-dim noise vector
- **Generator**: DCGAN-style upsampling with Conv2DTranspose, InstanceNorm, LeakyReLU
- **Discriminator**: PatchGAN with dropout and label smoothing
- **Training Tricks**: Label smoothing, noisy labels, mixed precision, gradient clipping

### 2️⃣ Render GAN

- **Input**: Grayscale sketch
- **Generator**: U-Net architecture with residual blocks and skip connections
- **Discriminator**: Conditional GAN with sketch-color pair as input
- **Loss Function**:
  - `L_total = L_GAN + λ1 * L1 + λ2 * SSIM`
  - Balances realism, pixel accuracy, and structural similarity

---

## 📊 Results

| Model       | Accuracy | F1 Score | SSIM  | Description                             |
|-------------|----------|----------|-------|-----------------------------------------|
| Sketch GAN  | 60%      | 0.68     | 0.31  | Generated sketches from random noise    |
| Render GAN  | 82%      | 0.82     | -     | Rendered sketches into color images     |

- **Visual Quality**: Sketches captured garment structure; renders had textures and shading.
- **Classifier Test**: Render GAN outputs were highly realistic and hard to distinguish from real ones.

---

## 📷 Output Samples

1. Real image from DeepFashion2  
2. Canny sketch version  
3. AI-generated sketch via Sketch GAN  
4. Final colored render via Render GAN

> *All outputs are generated at 128×128 resolution.*

---

## 🛠️ Tech Stack

- Python, TensorFlow (Keras)
- GANs (DCGAN, cGAN)
- Canny edge detection
- Mixed Precision Training
- CNN Classifier for evaluation

---

## ✅ Key Takeaways

- Fully automated AI pipeline for fashion visualization
- Sketch GAN helps with ideation by generating outlines
- Render GAN brings sketches to life with realistic colorization
- Classifier ensures quality through real/fake differentiation

---

## 🔗 References

1. [Image-to-Image Translation with cGANs](https://arxiv.org/abs/1611.07004)  
2. [DeepFashion Dataset](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)  
3. [CycleGAN](https://arxiv.org/abs/1703.10593)  
4. [High-Res Synthesis with cGANs](https://arxiv.org/abs/1711.11585)

---
## ⚙️ Installation

```bash
pip install -r requirements(1).txt

