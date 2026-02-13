# recommend.py

def recommend_solution(disease):

    solutions = {

        # Tomato Diseases
        "Tomato___Late_blight":
            "Spray fungicide immediately and remove infected leaves.",

        "Tomato___Early_blight":
            "Use neem oil or copper fungicide. Avoid overhead irrigation.",

        "Tomato___Leaf_Mold":
            "Improve air circulation and apply sulfur-based fungicide.",

        "Tomato___healthy":
            "Plant is healthy. Continue regular monitoring.",


        # Potato Diseases
        "Potato___Early_blight":
            "Apply copper-based fungicide and remove affected leaves.",

        "Potato___Late_blight":
            "Use certified fungicide spray and isolate infected plants.",

        "Potato___healthy":
            "Plant is healthy. Continue monitoring.",


        # Pepper Diseases
        "Pepper,_bell___Bacterial_spot":
            "Use disease-free seeds and apply bactericide spray.",

        "Pepper,_bell___healthy":
            "Plant is healthy. No action needed."
    }

    return solutions.get(
        disease,
        "Disease detected. Please consult an agricultural expert for treatment."
    )