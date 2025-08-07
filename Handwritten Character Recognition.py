import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Suppress verbose TensorFlow logging
tf.get_logger().setLevel('ERROR')

print("TensorFlow Version:", tf.__version__)

## 1. LOAD AND PREPROCESS THE DATA
# Load the MNIST dataset, which is conveniently included in Keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the images
# Normalize pixel values from the range [0, 255] to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape images to include a single channel dimension (for grayscale)
# The CNN expects input shape: (batch_size, height, width, channels)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convert labels to one-hot encoded vectors
# e.g., 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Training labels shape: {y_train.shape}")


## 2. BUILD THE CONVOLUTIONAL NEURAL NETWORK (CNN) MODEL
model = tf.keras.Sequential([
    # Input layer
    tf.keras.layers.Input(shape=(28, 28, 1)),

    # First convolutional block
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Second convolutional block
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten the feature maps to feed into dense layers
    tf.keras.layers.Flatten(),

    # Add a dropout layer to prevent overfitting
    tf.keras.layers.Dropout(0.5),

    # Output layer with 'softmax' activation for classification
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

# Display the model's architecture
model.summary()


## 3. COMPILE THE MODEL
# We use 'categorical_crossentropy' because our labels are one-hot encoded
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


## 4. TRAIN THE MODEL
batch_size = 128
epochs = 15

print("\nStarting model training...")
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1 # Use 10% of training data for validation
)
print("Model training finished.")


## 5. EVALUATE THE MODEL
print("\nEvaluating model on test data...")
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")


## 6. MAKE A PREDICTION ON A SAMPLE IMAGE
print("\nMaking a prediction on a sample test image...")
# Select a random image from the test set
image_index = np.random.randint(0, len(x_test))
sample_image = x_test[image_index]
true_label = np.argmax(y_test[image_index])

# Predict the digit
# The model expects a batch of images, so we add a batch dimension
prediction = model.predict(np.expand_dims(sample_image, axis=0))
predicted_label = np.argmax(prediction)

# Display the result
plt.figure()
plt.imshow(sample_image, cmap="gray")
plt.title(f"True Label: {true_label}\nPredicted Label: {predicted_label}")
plt.axis("off")
plt.show()