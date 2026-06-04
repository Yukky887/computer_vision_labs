import os
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, initializers

# Параметры
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAME = "watch"
IMG_SIZE = 64
CHANNELS = 3
NOISE_DIM = 128
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0002
BETA_1 = 0.5

ARTIFACTS_DIR = "artifacts_gan"
RESULTS_DIR = "results_gan"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Загрузка датасета
print("Загрузка датасета Caltech-101...")
info = tfds.builder('caltech101').info
class_names = info.features['label'].names
class_idx = class_names.index(CLASS_NAME)

ds = tfds.load('caltech101', split='all', shuffle_files=True, as_supervised=True, with_info=False)

def filter_class(image, label):
    return tf.equal(tf.cast(label, tf.int32), tf.cast(class_idx, tf.int32))

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5
    return image

dataset = (ds
           .filter(filter_class)
           .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
           .shuffle(500, seed=SEED)
           .batch(BATCH_SIZE, drop_remainder=True)
           .prefetch(tf.data.AUTOTUNE))

# Подсчет изображений
count = 0
for batch in dataset:
    count += batch.shape[0]
print(f"Загружено изображений класса '{CLASS_NAME}': {count}")

# Генератор
def build_generator():
    init = initializers.RandomNormal(mean=0.0, stddev=0.02)
    return models.Sequential([
        layers.Input(shape=(NOISE_DIM,)),
        layers.Dense(4 * 4 * 512, use_bias=False, kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Reshape((4, 4, 512)),
        layers.Conv2DTranspose(256, 5, strides=2, padding="same", use_bias=False, kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2DTranspose(128, 5, strides=2, padding="same", use_bias=False, kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False, kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2DTranspose(32, 5, strides=2, padding="same", use_bias=False, kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2D(CHANNELS, 5, padding="same", activation="tanh", kernel_initializer=init)
    ], name="generator")

# Дискриминатор
def build_discriminator():
    init = initializers.RandomNormal(mean=0.0, stddev=0.02)
    return models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        layers.Conv2D(64, 5, strides=2, padding="same", kernel_initializer=init),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, 5, strides=2, padding="same", kernel_initializer=init),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(256, 5, strides=2, padding="same", kernel_initializer=init),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(512, 5, strides=2, padding="same", kernel_initializer=init),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid")
    ], name="discriminator")

generator = build_generator()
discriminator = build_discriminator()

loss_fn = tf.keras.losses.BinaryCrossentropy()
gen_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
disc_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)

@tf.function
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)
        real_out = discriminator(real_images, training=True)
        fake_out = discriminator(fake_images, training=True)
        d_loss_real = loss_fn(tf.ones_like(real_out) * 0.95, real_out)
        d_loss_fake = loss_fn(tf.zeros_like(fake_out), fake_out)
        d_loss = d_loss_real + d_loss_fake
        g_loss = loss_fn(tf.ones_like(fake_out), fake_out)
        real_acc = tf.reduce_mean(tf.cast(real_out >= 0.5, tf.float32))
        fake_acc = tf.reduce_mean(tf.cast(fake_out < 0.5, tf.float32))
        d_acc = (real_acc + fake_acc) / 2
        g_fool_rate = tf.reduce_mean(tf.cast(fake_out >= 0.5, tf.float32))
    gen_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    return g_loss, d_loss, d_acc, real_acc, fake_acc, g_fool_rate

# Сохранение изображений
def save_images(generator, epoch, noise):
    images = generator(noise, training=False).numpy()
    images = (images + 1) / 2
    plt.figure(figsize=(6, 6))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.suptitle(f"Эпоха {epoch}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"epoch_{epoch:03d}.png"), dpi=150)
    plt.close()

# Цикл обучения
fixed_noise = tf.random.normal([16, NOISE_DIM], seed=SEED)

g_losses, d_losses = [], []
d_accs, real_accs, fake_accs, g_fool_rates = [], [], [], []

print("\nОбучение GAN...")
for epoch in range(1, EPOCHS + 1):
    epoch_g, epoch_d = [], []
    epoch_d_acc, epoch_real_acc, epoch_fake_acc, epoch_fool = [], [], [], []

    for real_batch in dataset:
        g_loss, d_loss, d_acc, real_acc, fake_acc, g_fool_rate = train_step(real_batch)
        epoch_g.append(float(g_loss))
        epoch_d.append(float(d_loss))
        epoch_d_acc.append(float(d_acc))
        epoch_real_acc.append(float(real_acc))
        epoch_fake_acc.append(float(fake_acc))
        epoch_fool.append(float(g_fool_rate))

    g_losses.append(np.mean(epoch_g))
    d_losses.append(np.mean(epoch_d))
    d_accs.append(np.mean(epoch_d_acc))
    real_accs.append(np.mean(epoch_real_acc))
    fake_accs.append(np.mean(epoch_fake_acc))
    g_fool_rates.append(np.mean(epoch_fool))

    print(
        f"Эпоха {epoch}/{EPOCHS} | G loss: {g_losses[-1]:.4f} | D loss: {d_losses[-1]:.4f} | D acc: {d_accs[-1]*100:.1f}% | Fool: {g_fool_rates[-1]*100:.1f}%"
    )

    if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
        save_images(generator, epoch, fixed_noise)

# График потерь
plt.figure(figsize=(8, 4))
plt.plot(g_losses, label="Генератор")
plt.plot(d_losses, label="Дискриминатор")
plt.title("Потери GAN")
plt.xlabel("Эпоха")
plt.ylabel("Потери")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "losses.png"), dpi=150)
plt.close()

# График статистики
plt.figure(figsize=(8, 4))
plt.plot(d_accs, label="Точность дискриминатора")
plt.plot(real_accs, label="Реальные")
plt.plot(fake_accs, label="Сгенерированные")
plt.plot(g_fool_rates, label="Обман генератора")
plt.title("Статистика обучения")
plt.xlabel("Эпоха")
plt.ylabel("Доля")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "stats.png"), dpi=150)
plt.close()

# Сохранение моделей
generator.save(os.path.join(ARTIFACTS_DIR, "generator.keras"))
discriminator.save(os.path.join(ARTIFACTS_DIR, "discriminator.keras"))
np.save(os.path.join(ARTIFACTS_DIR, "fixed_noise.npy"), fixed_noise.numpy())

print(f"\nМодели сохранены в: {ARTIFACTS_DIR}/")
print("Готово!")

# Проверка загрузки модели
generator_loaded = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, "generator.keras"))
noise = np.load(os.path.join(ARTIFACTS_DIR, "fixed_noise.npy"))[:16]
images = generator_loaded(noise, training=False).numpy()
images = np.clip((images + 1) / 2, 0, 1)

plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(images[i])
    plt.axis("off")
plt.suptitle("Генерация загруженной моделью")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "loaded_model.png"), dpi=150)
plt.show()