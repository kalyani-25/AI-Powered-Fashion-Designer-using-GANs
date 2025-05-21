#!/usr/bin/env python3
import os, glob, math, tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, Model
from tensorflow.keras.mixed_precision import set_global_policy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio.v2 as imageio

# 1. Setup
set_global_policy('mixed_float16')

IMG_SIZE, Z_DIM = 128, 100
BATCH_SIZE, EPOCHS = 64, 20
LR_G, LR_D, BETA_1 = 2e-4, 1e-5, 0.5
DATA_DIR = "/Users/kalyanivaidya/AI Fashion Designer/data/synthetic_sketches"
OUT_DIR = "/Users/kalyanivaidya/AI Fashion Designer/outputs/sketch_gan_final_report_optimized"
os.makedirs(OUT_DIR, exist_ok=True)

# 2. Dataset
paths = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
if not paths:
    raise ValueError("No .jpg images found.")

def load_image(p):
    img = tf.io.read_file(p)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img

ds = tf.data.Dataset.from_tensor_slices(paths)
ds = ds.cache()
ds = ds.shuffle(buffer_size=len(paths))
ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(BATCH_SIZE, drop_remainder=True)
ds = ds.prefetch(tf.data.AUTOTUNE)
steps_per_epoch = len(paths) // BATCH_SIZE

# 3. Generator
def build_generator():
    z = layers.Input(shape=(Z_DIM,))
    x = layers.Dense((IMG_SIZE//16)*(IMG_SIZE//16)*512)(z)
    x = layers.Reshape((IMG_SIZE//16, IMG_SIZE//16, 512))(x)
    for f in [256, 128, 64, 32]:
        x = layers.Conv2DTranspose(f, 4, strides=2, padding='same')(x)
        x = layers.InstanceNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
    out = layers.Conv2D(1, 3, padding='same', activation='tanh', dtype='float32')(x)
    return Model(z, out, name="Generator")

# 4. Discriminator
def build_discriminator():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = inp
    for f in [64, 128, 256, 512]:
        x = layers.Conv2D(f, 4, strides=2, padding='same')(x)
        x = layers.InstanceNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
    out = layers.Conv2D(1, 4, padding='same', dtype='float32')(x)
    return Model(inp, out, name="Discriminator")

G, D = build_generator(), build_discriminator()

# 5. Loss and Optimizers
bce = losses.BinaryCrossentropy(from_logits=True)
optG = optimizers.Adam(LR_G, beta_1=BETA_1)
optD = optimizers.Adam(LR_D, beta_1=BETA_1)

# 6. Label Smoothing and Noise
def smooth_labels(y, smoothing=0.9):
    noise = tf.random.uniform(tf.shape(y), 0, 0.1)
    return y * smoothing + noise

def noisy_labels(y, flip_rate=0.05):
    flip = tf.random.uniform(tf.shape(y)) < flip_rate
    return tf.where(flip, 1-y, y)

# 7. Training Step
@tf.function
def train_step(real_images):
    bs = tf.shape(real_images)[0]
    noise = tf.random.normal((bs, Z_DIM))

    with tf.GradientTape() as td:
        fake_images = G(noise, training=True)
        real_logits = D(real_images, training=True)
        fake_logits = D(fake_images, training=True)

        real_labels = smooth_labels(tf.ones_like(real_logits))
        fake_labels = noisy_labels(tf.zeros_like(fake_logits))

        d_loss_real = bce(real_labels, real_logits)
        d_loss_fake = bce(fake_labels, fake_logits)
        d_loss = d_loss_real + d_loss_fake

    grads_D = td.gradient(d_loss, D.trainable_variables)
    grads_D = [tf.clip_by_value(g, -1.0, 1.0) for g in grads_D]
    optD.apply_gradients(zip(grads_D, D.trainable_variables))

    # Train Generator twice
    g_losses = []
    for _ in range(2):
        noise = tf.random.normal((bs, Z_DIM))
        with tf.GradientTape() as tg:
            fake_images = G(noise, training=True)
            fake_logits = D(fake_images, training=True)
            g_loss = bce(tf.ones_like(fake_logits), fake_logits)
        grads_G = tg.gradient(g_loss, G.trainable_variables)
        grads_G = [tf.clip_by_value(g, -1.0, 1.0) for g in grads_G]
        optG.apply_gradients(zip(grads_G, G.trainable_variables))
        g_losses.append(g_loss)

    return d_loss, tf.reduce_mean(g_losses)

# 8. Training Loop
all_d, all_g, epoch_d, epoch_g = [], [], [], []

for epoch in range(1, EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
    d_losses, g_losses = [], []

    for step, real_batch in enumerate(ds, start=1):
        d_l, g_l = train_step(real_batch)
        d_losses.append(float(d_l))
        g_losses.append(float(g_l))
        all_d.append(float(d_l))
        all_g.append(float(g_l))

        if step == 1 or step % 100 == 0:
            print(f" Step {step}/{steps_per_epoch}  D={d_l:.4f}  G={g_l:.4f}")

        if step >= steps_per_epoch:
            break

    mean_d = np.mean(d_losses)
    mean_g = np.mean(g_losses)
    epoch_d.append(mean_d)
    epoch_g.append(mean_g)
    print(f" → Mean D loss: {mean_d:.4f}   Mean G loss: {mean_g:.4f}")

    # Save generated sample
    sample = G(tf.random.normal((1, Z_DIM)), training=False)[0]
    sample = ((sample + 1) * 127.5).numpy().astype('uint8')
    tf.keras.preprocessing.image.save_img(os.path.join(OUT_DIR, f"epoch{epoch:02d}.png"), sample)

print("\n✅ Training complete! Outputs saved in:", OUT_DIR)

# 9. Save batch-wise loss curve
plt.figure(figsize=(10, 5))
plt.plot(all_d, label="Discriminator Loss (batch)")
plt.plot(all_g, label="Generator Loss (batch)")
plt.title("Batch-wise Loss Curve")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "batch_loss_curve.png"))

# 10. Save epoch-wise loss curve
plt.figure(figsize=(10, 5))
plt.plot(epoch_d, label="Discriminator Loss (epoch)")
plt.plot(epoch_g, label="Generator Loss (epoch)")
plt.title("Epoch-wise Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "epoch_loss_curve.png"))

# 11. Create training animation GIF
frames_path = os.path.join(OUT_DIR, "epoch*.png")
frame_files = sorted(glob.glob(frames_path))

gif_path = os.path.join(OUT_DIR, "training_progress.gif")
with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
    for frame_file in frame_files:
        image = imageio.imread(frame_file)
        writer.append_data(image)

print(f"✅ Training animation GIF saved at: {gif_path}")

# 12. Create Final Report Page
loss_curve_batch = os.path.join(OUT_DIR, "batch_loss_curve.png")
loss_curve_epoch = os.path.join(OUT_DIR, "epoch_loss_curve.png")

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

batch_img = mpimg.imread(loss_curve_batch)
axs[0, 0].imshow(batch_img)
axs[0, 0].axis('off')
axs[0, 0].set_title("Batch-wise Loss Curve")

epoch_img = mpimg.imread(loss_curve_epoch)
axs[0, 1].imshow(epoch_img)
axs[0, 1].axis('off')
axs[0, 1].set_title("Epoch-wise Loss Curve")

gif_img = mpimg.imread(gif_path)
axs[1, 0].imshow(gif_img)
axs[1, 0].axis('off')
axs[1, 0].set_title("Sketch Evolution (GIF Preview)")

axs[1, 1].axis('off')

report_path = os.path.join(OUT_DIR, "final_report_page.png")
plt.tight_layout()
plt.savefig(report_path)

print(f"✅ Final report page saved at: {report_path}")
