import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

def load_dataset(root, img_size=(128, 128)):
    images, masks = [], []
    for tile in sorted(os.listdir(root)):
        img_dir = os.path.join(root, tile, "images")
        mask_dir = os.path.join(root, tile, "masks")

        if os.path.isdir(img_dir):
            for f in os.listdir(img_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(img_dir, f)
                    mask_path = os.path.join(mask_dir, os.path.splitext(f)[0] + ".png")
                    if not os.path.exists(mask_path):
                        continue

                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size) / 255.0

                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue
                    mask = cv2.resize(mask, img_size)
                    mask = np.expand_dims(mask, axis=-1) / 255.0

                    images.append(img.astype("float32"))
                    masks.append(mask.astype("float32"))

    return np.array(images, dtype="float32"), np.array(masks, dtype="float32")


# dataset yükle
X, y = load_dataset("aerial_dataset", img_size=(128, 128))
print(f"Toplam örnek: {len(X)}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Toplam train örnek: {len(X_train)}")
print(f"Toplam val örnek: {len(X_val)}")


def build_unet(input_size=(128, 128, 3)):
    inputs = keras.Input(input_size)

    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(c3)  # aynı kanal sayısı tutuldu
    p3 = layers.MaxPooling2D()(c3)

    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(c4)
    p4 = layers.MaxPooling2D()(c4)

    c5 = layers.Conv2D(256, 3, activation="relu", padding="same")(p4)
    c5 = layers.Conv2D(256, 3, activation="relu", padding="same")(c5)

    u6 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding="same")(c5)
    u6 = layers.concatenate([u6, c4])  # <-- düzeltme
    c6 = layers.Conv2D(128, 3, activation="relu", padding="same")(u6)
    c6 = layers.Conv2D(128, 3, activation="relu", padding="same")(c6)

    u7 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, 3, activation="relu", padding="same")(u7)
    c7 = layers.Conv2D(64, 3, activation="relu", padding="same")(c7)

    u8 = layers.Conv2DTranspose(32, 2, strides=(2, 2), padding="same")(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, 3, activation="relu", padding="same")(u8)
    c8 = layers.Conv2D(32, 3, activation="relu", padding="same")(c8)

    u9 = layers.Conv2DTranspose(16, 2, strides=(2, 2), padding="same")(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(16, 3, activation="relu", padding="same")(u9)
    c9 = layers.Conv2D(16, 3, activation="relu", padding="same")(c9)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c9)

    return keras.Model(inputs, outputs)


model = build_unet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint("model_best.h5", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
]


history = model.fit(
    X_train, y_train,
    validation_data = (X_val,y_val),
    epochs = 15,
    batch_size = 16,
    callbacks = callbacks
)

import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label = "train_loss")
plt.plot(history.history["val_loss"], label = "val_loss")
plt.legend()
plt.show()

def show_prediction(idx=0):
    img = X_val[idx]
    mask_true = y_val[idx].squeeze()
    pred_raw = model.predict(img[None, ...])[0].squeeze()
    mask_pred = (pred_raw > 0.5).astype("float32")

    plt.figure(figsize=(10,5))

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(mask_true, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(pred_raw, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

show_prediction(1)
