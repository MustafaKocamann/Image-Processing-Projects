# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError


df = pd.read_csv("boneage-training-dataset.csv")

image_folder = "boneage-training-dataset"
available_files = set(os.listdir(image_folder))
available_ids = set(f.replace(".png", "")for f in available_files if f.endswith(".png"))
df = df[df["id"].astype(str).isin(available_ids)].reset_index(drop=True)

df["boneage"] = df["boneage"] / 240.0
df["path"] = df["id"].apply(lambda x: os.path.join(image_folder, f"{x}.png"))

print(df.head())

plt.hist(df["boneage"]*240, bins= 50)
plt.xlabel("Kemik Yaşı (Ay)")
plt.ylabel("Frekans")
plt.title("Kemik Yaşı Dağılımı")
plt.tight_layout()
plt.show()

def load_images(df, img_size=128):
    images = []
    valid_indices = []
    
    for i, path in enumerate(df["path"]):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Görsel okunamadı: {path}")
            continue
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        images.append(img)
        valid_indices.append(i)
    
    new_df = df.iloc[valid_indices].reset_index(drop=True)
    X = np.array(images).reshape(-1, img_size, img_size, 1)
    y = new_df["boneage"].values
    
    return X, y

X, y = load_images(df)
print("X shape:", X.shape)
print("y shape:", y.shape)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

datagen = ImageDataGenerator(
    horizontal_flip = True,
    zoom_range = 0.1,
    width_shift_range = 0.1, 
    height_shift_range = 0.1
)

datagen.fit(X_train)

model = Sequential()
model.add(Conv2D(32,(3,3), activation = "relu", input_shape = (128,128,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3), activation = "relu"))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation = "linear"))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mae",
    metrics=[MeanAbsoluteError()]
)

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
    ModelCheckpoint("bone_age_model.keras", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss")
]

history = model.fit(
    datagen.flow(X_train, y_train, batch_size = 32),
    validation_data = (X_val, y_val),
    epochs = 10,
    callbacks = callbacks
)

plt.plot(history.history["loss"], label = "training_mae")
plt.plot(history.history["val_loss"], label = "validation_mae")
plt.xlabel("epochs")
plt.ylabel("MAE")
plt.title("Training performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

predictions = model.predict(X_val) * 240
actuals = y_val * 240

pred_arr = np.array(predictions).reshape(-1)   
y_arr = np.array(y_val).reshape(-1)          

if y_arr.max() <= 1.1:
    y_display = y_arr * 240.0
else:
    y_display = y_arr.copy()


pred_display = pred_arr.copy()

print("DEBUG shapes -> X_val:", np.array(X_val).shape,
      "pred_display:", pred_display.shape, "y_display:", y_display.shape)


fig, axes = plt.subplots(2, 5, figsize=(14, 6))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i >= len(X_val):
        ax.axis("off")
        continue
    try:
        img = X_val[i]
        
        if img.ndim == 4:
            img = img.reshape(img.shape[1], img.shape[2])
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = img.reshape(img.shape[0], img.shape[1])
        ax.imshow(img, cmap="gray")
        
        t = pred_display[i]
        g = y_display[i]
        ax.set_title(f"Tahmin: {t:.0f}\nGerçek: {g:.0f}", fontsize=9)
        ax.axis("off")
    except Exception as e:
        ax.set_title("Hata")
        ax.text(0.5, 0.5, str(e), fontsize=7, ha="center")
        ax.axis("off")

plt.suptitle("Kemik Yaşı Tahmin Sonuçları", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


