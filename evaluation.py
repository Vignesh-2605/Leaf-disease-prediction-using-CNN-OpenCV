# 7_evaluation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

import seaborn as sns

from preprocess import load_data   # Uses Module 1 preprocessing loader


MODEL_PATH = "C:\\Users\\Vignesh S\\OneDrive\\Documents\\Vignesh_S\\College SIMATS\\ComputerVision\\Capstone\\models\\cnn_model.h5"


# 1. Load Model + Dataset
def load_model_and_data():

    print("📌 Loading Trained Model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("📌 Loading Validation Dataset...")
    train_data, val_data = load_data()

    return model, val_data


# 2. Accuracy & Loss Curve Plot
def plot_training_curves(history):

    plt.figure(figsize=(8,5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# 3. Confusion Matrix Heatmap
def plot_confusion_matrix(y_true, y_pred, class_names):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10,7))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


# 4. Classification Report Table
def generate_classification_report(y_true, y_pred, class_names):

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    df_report = pd.DataFrame(report).transpose()

    print("\n✅ Classification Report Table:\n")
    print(df_report)

    return df_report



# 5. ROC Curve Plot (Multi-Class)
def plot_multiclass_roc(y_true, y_probs, num_classes):

    print("\n📌 Plotting ROC Curve...")

    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    plt.figure(figsize=(8,6))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"Class {i} AUC={roc_auc:.2f}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Multi-Class ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


# 6. Final Evaluation Runner
def evaluate_model():

    model, val_data = load_model_and_data()

    print("\n📌 Running Predictions on Validation Set...")

    y_true = val_data.classes
    y_probs = model.predict(val_data)
    y_pred = np.argmax(y_probs, axis=1)

    class_names = list(val_data.class_indices.keys())
    num_classes = len(class_names)

    # Classification Report Table
    report_df = generate_classification_report(
        y_true, y_pred, class_names
    )

    # Confusion Matrix Heatmap
    plot_confusion_matrix(y_true, y_pred, class_names)

    # ROC Curve
    plot_multiclass_roc(y_true, y_probs, num_classes)

    # Summary Table Output
    summary = {
        "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"],
        "Value": [
            report_df.loc["accuracy", "precision"],
            report_df.loc["macro avg", "precision"],
            report_df.loc["macro avg", "recall"],
            report_df.loc["macro avg", "f1-score"]
        ]
    }

    summary_df = pd.DataFrame(summary)

    print("\n✅ Final Evaluation Summary Table:\n")
    print(summary_df)

    # Save summary as CSV
    summary_df.to_csv("evaluation_summary.csv", index=False)
    print("\n📌 Saved: evaluation_summary.csv")


if __name__ == "__main__":
    evaluate_model()