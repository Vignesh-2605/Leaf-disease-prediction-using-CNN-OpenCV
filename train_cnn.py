# train_cnn.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model

from preprocess import load_data

import matplotlib.pyplot as plt
import os
import pickle


# Build Functional CNN Model (Grad-CAM Compatible)
def build_functional_cnn(num_classes):

    inputs = Input(shape=(224, 224, 3))

    x = Conv2D(32, (3,3), activation="relu", name="conv1")(inputs)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3,3), activation="relu", name="conv2")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3,3), activation="relu", name="conv3")(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)

    return model


# Train CNN Model
def train():

    print("📌 Loading Dataset...")
    train_data, val_data = load_data()

    # Save class names
    class_names = list(train_data.class_indices.keys())

    os.makedirs("models", exist_ok=True)

    with open("models/class_names.pkl", "wb") as f:
        pickle.dump(class_names, f)

    print("✅ Class names saved: models/class_names.pkl")

    print("📌 Building Functional CNN Model...")
    model = build_functional_cnn(train_data.num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("📌 Training Started...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )

    # Save model
    model.save("models/cnn_model.h5")
    print("✅ Model Saved: models/cnn_model.h5")

    # Accuracy Curve
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("CNN Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Loss Curve
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("CNN Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()