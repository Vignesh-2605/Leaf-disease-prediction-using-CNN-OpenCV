import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

from severity import estimate_severity
from recommend import recommend_solution

MODEL_PATH = "models/cnn_model.h5"

def predict_disease(img_path):

    model = tf.keras.models.load_model(MODEL_PATH)

    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    disease_classes = sorted(list(model.class_names)) if hasattr(model, "class_names") else None

    severity, mask = estimate_severity(img_path)

    print("\n🌿 Disease Predicted Index:", class_index)
    print("🔥 Severity: {:.2f}%".format(severity))

    return class_index, severity