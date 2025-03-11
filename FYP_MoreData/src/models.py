import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Dropout, 
    GlobalAveragePooling2D, Dense
)
from tensorflow.keras.models import Model

def build_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Builds and returns a CNN model for diabetes detection using thermal images.

    Args:
        input_shape (tuple): Shape of the input image.
        num_classes (int): Number of output classes (1 for binary classification).

    Returns:
        model (tf.keras.Model): Compiled CNN model.
    """
    
    inputs = Input(shape=input_shape, name="input_layer")

    # Convolutional Block 1
    x = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Convolutional Block 2
    x = Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Convolutional Block 3
    x = Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Convolutional Block 4
    x = Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Fully Connected Layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.4)(x)
    
    # Output Layer (Sigmoid for Binary Classification)
    outputs = Dense(num_classes, activation="sigmoid", name="output_layer")(x)

    # Model Definition
    model = Model(inputs, outputs, name="Improved_Diabetes_CNN")
    
    # Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# Debugging: Run the model independently to check if it executes correctly
if __name__ == "__main__":
    model = build_model()
    model.summary()
