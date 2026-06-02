import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models, callbacks
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Фиксируем seed для воспроизводимости
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Параметры
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_HEAD = 5
EPOCHS_FINE = 3
LEARNING_RATE = 1e-3
DROPOUT = 0.4
AUTOTUNE = tf.data.AUTOTUNE

SELECTED_CLASSES = ["airplanes", "motorbikes", "faces", "watch", "lotus"]
NUM_CLASSES = len(SELECTED_CLASSES)

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("Загрузка датасета Caltech-101...")

# Получаем индексы нужных классов
info = tfds.builder('caltech101').info
class_names_full = info.features['label'].names

SELECTED_INDICES = []
CLASS_MAPPING = {}
for new_idx, name in enumerate(SELECTED_CLASSES):
    for old_idx, full_name in enumerate(class_names_full):
        if full_name == name:
            SELECTED_INDICES.append(old_idx)
            CLASS_MAPPING[old_idx] = new_idx
            break

# Загружаем датасет
ds_train, ds_test = tfds.load(
    'caltech101', 
    split=['train', 'test'], 
    shuffle_files=True, 
    as_supervised=True, 
    with_info=False
)

# Фильтрация и переназначение меток
def filter_by_class(image, label):
    label = tf.cast(label, tf.int32)
    return tf.reduce_any(tf.equal(label, tf.constant(SELECTED_INDICES, dtype=tf.int32)))

def remap_labels(image, label):
    label = tf.cast(label, tf.int32)
    result = tf.constant(-1, dtype=tf.int64)
    for old_idx in SELECTED_INDICES:
        new_idx = CLASS_MAPPING[old_idx]
        result = tf.where(
            tf.equal(label, tf.cast(old_idx, tf.int32)),
            tf.cast(new_idx, tf.int64),
            result
        )
    return image, result

ds_train = ds_train.filter(filter_by_class).map(remap_labels)
ds_test = ds_test.filter(filter_by_class).map(remap_labels)

# Предобработка и аугментация
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    return image, label

# Создаем пайплайны данных
train_ds = (ds_train
            .shuffle(1000, seed=SEED)
            .map(augment, num_parallel_calls=AUTOTUNE)
            .map(preprocess, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

test_ds = (ds_test
           .map(preprocess, num_parallel_calls=AUTOTUNE)
           .batch(BATCH_SIZE)
           .prefetch(AUTOTUNE))

print(f"Выбрано классов: {NUM_CLASSES}")

# Построение модели
def build_model():
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False
    
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    
    model = models.Model(inputs, outputs)
    return model, base

model, base = build_model()
model.summary()

# Коллбеки
cbs = [
    callbacks.ModelCheckpoint(
        os.path.join(ARTIFACTS_DIR, "best_model.keras"),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max"
    ),
    callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        verbose=1
    )
]

# Этап 1: обучаем классификатор
print("\n--- Этап 1: Обучение классификатора ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history_head = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS_HEAD,
    callbacks=cbs
)

# Этап 2: дообучение с разморозкой
print("\n--- Этап 2: Дообучение ---")
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
history_fine = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS_FINE,
    callbacks=cbs
)

# Графики обучения
history = {}
for k in history_head.history:
    history[k] = history_head.history[k] + history_fine.history[k]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="train")
plt.plot(history["val_accuracy"], label="val")
plt.title("Точность (Accuracy)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="train")
plt.plot(history["val_loss"], label="val")
plt.title("Функция потерь (Loss)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, "training_history.png"), dpi=150)
plt.show()

# Оценка на тестовой выборке
print("\n--- Оценка модели ---")
y_true = []
y_pred = []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

print(classification_report(
    y_true, y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=SELECTED_CLASSES,
    digits=4,
    zero_division=0
))

# Матрица ошибок
cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues", interpolation='nearest')
plt.title("Матрица ошибок")
plt.colorbar()
ticks = np.arange(len(SELECTED_CLASSES))
plt.xticks(ticks, SELECTED_CLASSES, rotation=45, ha='right')
plt.yticks(ticks, SELECTED_CLASSES)
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')

for i in range(len(SELECTED_CLASSES)):
    for j in range(len(SELECTED_CLASSES)):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, "confusion_matrix.png"), dpi=150)
plt.show()

# Сохранение модели
model.save(os.path.join(ARTIFACTS_DIR, "caltech101_mobilenetv2.keras"))
with open(os.path.join(ARTIFACTS_DIR, "class_names.json"), "w", encoding="utf-8") as f:
    json.dump(SELECTED_CLASSES, f, ensure_ascii=False, indent=2)

print(f"\nМодель сохранена в: {ARTIFACTS_DIR}/")
print("Готово!")