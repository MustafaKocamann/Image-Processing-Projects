"""
MNIST Dataset:
 * Rakamlama için yaygın olarak kullanılan bir veri setidir.
 * 0-9 arasındaki el yazısı rakamları içerir.
 * 10 sınıfı vardır (her rakam bir sınıf).
 * 28x28 piksel boyutunda gri tonlamalı görüntülerden oluşur.
 * gray_scale: Görüntülerin siyah-beyaz olduğunu belirtir.
 * 60000 eğitim örneği ve 10000 test örneği içerir.


 Amacımız:ANN ile rakamları sınıflandırmak.

 1-) Image Processing (Görüntü İşleme):
 * Histogram Eşitleme (Histogram Equalization): Görüntü kontrastını artırmak için kullanılır.
 * Gaussian Blur: Görüntüdeki gürültüyü azaltmak için kullanılır.
 * Canny Edge Detection: Görüntüdeki kenarları tespit etmek için kullanılır.

"""
# import necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train.shape, y_train.shape), (x_test.shape, y_test.shape)

# image processing
img = x_train[0]

stages = {"original": img}

# Histogram Equalization
eg = cv2.equalizeHist(img)
stages["Histogram Equalization"] = eg

# Gaussian Blur
blur = cv2.GaussianBlur(eg, (5, 5), 0)
stages["Gaussian Blur"] = blur

# Canny Edge Detection
edges = cv2.Canny(blur, 50, 150)
stages["Canny Edge Detection"] = edges

#Visualization
fig,axes = plt.subplots(2, 2, figsize=(6, 6))
axes = axes.flat
for ax, (title,im) in zip(axes, stages.items()):
    ax.imshow(im, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.suptitle("MNIST Image Processing Stages")
plt.tight_layout()
plt.show()

# preprocessing function
def preprocess_image(img):
    """
    - histogram equalization, 
    - gaussian blur, canny edge detection
    - flattering: converting 28x28 image to 784 vector
    - normalization: 0-255 -> 0-1
    """
    img_eg = cv2.equalizeHist(img)
    img_blur = cv2.GaussianBlur(img_eg, (5, 5), 0)
    img_edges = cv2.Canny(img_blur, 50, 150)
    features = img_edges.flatten() / 255.0
    return features

num_train = 10000
num_test = 2000

x_train = np.array([preprocess_image(img) for img in x_train[:num_train]])
y_train_sub = y_train[:num_train]

X_test = np.array([preprocess_image(img) for img in x_test[:num_test]])
y_test_sub = y_test[:num_test]

# model creation
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']

)

model.summary()

# early stopping

early_stop = EarlyStopping(
    monitor='val_loss',   
    patience=4,           
    restore_best_weights=True,  
    mode='min'            
)

# model training

history = model.fit(
    x_train, y_train_sub,
    validation_data=(X_test, y_test_sub),
    epochs=50,
    batch_size=32,
    verbose=2,
    callbacks=[early_stop]
)

test_loss, test_acc = model.evaluate(X_test, y_test_sub)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()