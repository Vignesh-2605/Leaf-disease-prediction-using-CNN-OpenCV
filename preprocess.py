import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 32

def load_data(dataset_path="C:\\Users\\Vignesh S\\OneDrive\\Documents\\Vignesh_S\\College SIMATS\\ComputerVision\\Capstone\\Dataset\\PlantVillage"):

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=25,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        subset="training",
        class_mode="categorical"
    )

    val_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        subset="validation",
        class_mode="categorical"
    )

    return train_data, val_data