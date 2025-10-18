import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import DenseNet121

train_datagen = ImageDataGenerator(
    rescale=1./255.0,
    horizontal_flip=True,
    rotation_range=10,
    brightness_range=[0.8, 1.2],
    validation_split=0.1 
)

test_datagen = ImageDataGenerator(rescale=1./255.0)

DATA_DIR = "chest_xray"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_MODE = "binary"
SEED = 42

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    subset="training",
    shuffle=True,
    seed=SEED
)

val_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=CLASS_MODE,
    subset="validation",
    shuffle=False,
    seed=SEED
)

test_dir = os.path.join(DATA_DIR, "test")
if os.path.isdir(test_dir):
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        shuffle=False
    )
else:
    test_gen = None

class_names = list(train_gen.class_indices.keys())


images, labels = next(train_gen)
plt.figure(figsize=(10, 5))
for i in range(4):
    ax = plt.subplot(1, 4, i + 1)
    label_idx = int(np.round(labels[i])) if labels.ndim > 1 else int(labels[i])
    ax.imshow(images[i])
    ax.set_title(class_names[label_idx])
    ax.axis("off")
plt.tight_layout()
plt.show()
plt.close('all')

base_model = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)


base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=pred)


model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])


callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6),
    ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True)
]

print("Model Summary:")
model.summary()


history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=2,
    callbacks=callbacks,
    verbose=1
)


if test_gen is not None:

    pred_probs = model.predict(test_gen, verbose=1)
   
    pred_labels = (pred_probs.flatten() > 0.5).astype(int)
    true_labels = test_gen.classes

    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap="Blues", colorbar=False, ax=plt.gca())
    plt.title("Test Set Confusion Matrix")
    plt.show()
    plt.close('all')
else:
    print("test_gen bulunamadı; confusion matrix atlanıyor.")
