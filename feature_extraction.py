import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image

def extract_features(img_path):

    model = ResNet50(weights="imagenet",
                     include_top=False,
                     pooling="avg")

    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    features = model.predict(img_array)

    return features[0]


if __name__ == "__main__":

    img_path = "sample_leaf.jpg"

    feat_vector = extract_features(img_path)

    # 📊 Feature Plot
    plt.figure(figsize=(12,4))
    plt.plot(feat_vector)
    plt.title("Deep Feature Vector (ResNet50 Output)")
    plt.xlabel("Feature Index")
    plt.ylabel("Activation Value")
    plt.show()

    print("Feature Shape:", feat_vector.shape)