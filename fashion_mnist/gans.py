import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers
import os


BUFFER_SIZE = 60000
BATCH_SIZE = 128
NOISE_DIM = 100
IMG_SHAPE = (28, 28, 1)
EPOCHS = 3


(train_images, _), (_, _) = fashion_mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32")
train_images = (train_images - 127.5) / 127.5
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def make_generator_model():
    model = tf.keras.Sequential([
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((7, 7, 256)),

        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh")
    ])
    return model


def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=IMG_SHAPE),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model



generator = make_generator_model()
discriminator = make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)



def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)



generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

seed = tf.random.normal([16, NOISE_DIM])



def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    out_dir = "generated_images"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, 0] + 1) / 2, cmap="gray")
        plt.axis("off")

    plt.savefig(os.path.join(out_dir, f"image_at_epoch{epoch:03d}.png"))
    plt.close(fig)


def train(dataset, epochs):
    for epoch in range(1, epochs + 1):
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        batch_count = 0

        for image_batch in dataset:
            batch_size = tf.shape(image_batch)[0]
            noise = tf.random.normal([batch_size, NOISE_DIM])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)

                real_output = discriminator(image_batch, training=True)
                fake_output = discriminator(generated_images, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

            gen_loss_total += float(gen_loss)
            disc_loss_total += float(disc_loss)
            batch_count += 1

        print(f"Epoch:{epoch}/{epochs} -- Generator Loss: {gen_loss_total / batch_count:.3f} -- Discriminator Loss: {disc_loss_total / batch_count:.3f}")
        generate_and_save_images(generator, epoch, seed)

train(train_dataset, EPOCHS)

