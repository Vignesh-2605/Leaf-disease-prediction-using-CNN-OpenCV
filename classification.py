import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

MODEL_PATH = "crop_disease_model.h5"

def classify(img_path):

    model = tf.keras.models.load_model(MODEL_PATH)

    img = image.load_img(img_path, target_size=(224,224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0

    prediction = model.predict(arr)[0]

    return prediction


if __name__ == "__main__":

    img_path = "sample_leaf.jpg"

    probs = classify(img_path)

    # 📊 Probability Plot
    plt.figure(figsize=(10,5))
    plt.bar(range(len(probs)), probs)
    plt.title("Disease Classification Probabilities")
    plt.xlabel("Class Index")
    plt.ylabel("Probability")
    plt.show()

    print("Predicted Class:", np.argmax(probs))