# ============================================================
# NEURAL NETWORK FOR IMAGE CLASSIFICATION
# Minor Project - 6th Semester
# ============================================================

# -----------------------------------------------------------
# PART 1: IMPORTING LIBRARIES
# Libraries are pre-built tools that simplify development
# -----------------------------------------------------------
import numpy as np                          # For numerical computations
import matplotlib.pyplot as plt             # For plotting graphs and images
import tensorflow as tf                     # For building neural networks
from tensorflow import keras        # type: ignore        # High-level neural network API
from tensorflow.keras import layers     # type: ignore    # For defining network layers
from tensorflow.keras.datasets import mnist    # type: ignore  # Handwritten digits dataset
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns                       # For enhanced visualizations
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("   NEURAL NETWORK IMAGE CLASSIFICATION")
print("=" * 50)
print("\nStep 1: Libraries loaded successfully!")

# -----------------------------------------------------------
# PART 2: LOADING THE DATASET
# MNIST = 70,000 handwritten digit images (digits 0 through 9)
# -----------------------------------------------------------
print("\nStep 2: Loading dataset...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"   Training images: {x_train.shape[0]} images")
print(f"   Testing images:  {x_test.shape[0]} images")
print(f"   Image size:      {x_train.shape[1]}x{x_train.shape[2]} pixels")
print(f"   Classes (labels): {np.unique(y_train)}")
print("   Dataset loaded successfully!")

# -----------------------------------------------------------
# PART 3: VISUALIZING SAMPLE IMAGES
# Displays sample images to understand the dataset
# -----------------------------------------------------------
print("\nStep 3: Displaying sample images...")

plt.figure(figsize=(12, 5))
plt.suptitle("MNIST Dataset - Sample Handwritten Digits", fontsize=14, fontweight='bold')
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Digit: {y_train[i]}", fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Sample images saved: sample_images.png")

# -----------------------------------------------------------
# PART 4: PREPROCESSING THE DATA
# Prepares the data for model input
# Pixel values range from 0-255; normalizing them to 0-1
# -----------------------------------------------------------
print("\nStep 4: Preprocessing data...")

# Normalize pixel values (0-255 → 0-1)
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# Flatten each 28x28 image into a 1D array of 784 values
x_train_flat = x_train.reshape(-1, 784)
x_test_flat  = x_test.reshape(-1, 784)

print(f"   Training data shape: {x_train_flat.shape}")
print(f"   Testing data shape:  {x_test_flat.shape}")
print(f"   Pixel range: {x_train_flat.min():.1f} to {x_train_flat.max():.1f}")
print("   Preprocessing complete!")

# -----------------------------------------------------------
# PART 5: BUILDING THE NEURAL NETWORK MODEL
# Core component of the project
# Each layer progressively learns more abstract features
# -----------------------------------------------------------
print("\nStep 5: Building Neural Network model...")

model = keras.Sequential([
    # Input Layer: accepts 784 pixel values
    layers.Dense(256, activation='relu', input_shape=(784,),
                 name='Hidden_Layer_1'),

    # Dropout: prevents overfitting by randomly disabling neurons
    layers.Dropout(0.3, name='Dropout_1'),

    # Hidden Layer 2
    layers.Dense(128, activation='relu', name='Hidden_Layer_2'),

    layers.Dropout(0.2, name='Dropout_2'),

    # Hidden Layer 3
    layers.Dense(64, activation='relu', name='Hidden_Layer_3'),

    # Output Layer: 10 neurons for digits 0-9
    layers.Dense(10, activation='softmax', name='Output_Layer')
])

print("\n   --- Model Architecture ---")
model.summary()

# -----------------------------------------------------------
# PART 6: COMPILING THE MODEL
# Defines the optimizer, loss function, and evaluation metric
# -----------------------------------------------------------
print("\nStep 6: Compiling model...")

model.compile(
    optimizer='adam',                          # Optimization algorithm
    loss='sparse_categorical_crossentropy',    # Loss function for classification
    metrics=['accuracy']                       # Evaluation metric
)
print("   Model compiled successfully!")

# -----------------------------------------------------------
# PART 7: TRAINING THE MODEL
# Feeds training data to the model so it can learn
# Epochs = number of full passes over the training dataset
# -----------------------------------------------------------
print("\nStep 7: Training model...")
print("   (Please wait - this may take 3-5 minutes)\n")

history = model.fit(
    x_train_flat, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,    # Reserve 10% of data for validation
    verbose=1
)

print("\n   Training complete!")

# -----------------------------------------------------------
# PART 8: EVALUATING THE MODEL
# Tests the model on unseen data to measure real-world accuracy
# -----------------------------------------------------------
print("\nStep 8: Evaluating model...")

test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=0)
print(f"\n   ✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"   📉 Test Loss:     {test_loss:.4f}")

# -----------------------------------------------------------
# PART 9: PLOTTING TRAINING GRAPHS
# Visualizes how accuracy and loss changed during training
# -----------------------------------------------------------
print("\nStep 9: Generating training graphs...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Neural Network Training Results", fontsize=14, fontweight='bold')

# Accuracy Graph
ax1.plot(history.history['accuracy'],     label='Training Accuracy',   color='blue',   linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch Number')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss Graph
ax2.plot(history.history['loss'],     label='Training Loss',   color='red',   linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', color='green', linewidth=2)
ax2.set_title('Loss over Epochs')
ax2.set_xlabel('Epoch Number')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_graphs.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Graphs saved: training_graphs.png")

# -----------------------------------------------------------
# PART 10: DISPLAYING PREDICTIONS
# Shows what the model predicts on new test images
# -----------------------------------------------------------
print("\nStep 10: Generating predictions...")

predictions = model.predict(x_test_flat, verbose=0)

plt.figure(figsize=(14, 7))
plt.suptitle("Model Predictions  |  Green = Correct  |  Red = Wrong",
             fontsize=12, fontweight='bold')

for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    predicted = np.argmax(predictions[i])
    actual    = y_test[i]
    confidence = np.max(predictions[i]) * 100
    color = 'green' if predicted == actual else 'red'
    plt.title(f"P:{predicted} A:{actual}\n{confidence:.0f}%",
              color=color, fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Predictions saved: predictions.png")

# -----------------------------------------------------------
# PART 11: CONFUSION MATRIX
# Shows which digits the model confused with each other
# -----------------------------------------------------------
print("\nStep 11: Generating Confusion Matrix...")

y_pred = np.argmax(model.predict(x_test_flat, verbose=0), axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10),
            yticklabels=range(10))
plt.title('Confusion Matrix\n(Rows = Actual, Columns = Predicted)',
          fontsize=13, fontweight='bold')
plt.xlabel('Predicted Digit', fontsize=11)
plt.ylabel('Actual Digit',    fontsize=11)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Confusion Matrix saved: confusion_matrix.png")

# -----------------------------------------------------------
# PART 12: FINAL RESULTS SUMMARY
# -----------------------------------------------------------
print("\n" + "=" * 50)
print("         FINAL PROJECT RESULTS")
print("=" * 50)
print(f"  Model Accuracy : {test_accuracy * 100:.2f}%")
print(f"  Model Loss     : {test_loss:.4f}")
print(f"  Total Params   : {model.count_params():,}")
print("\n  Files generated:")
print("  📸 sample_images.png")
print("  📈 training_graphs.png")
print("  🎯 predictions.png")
print("  🔲 confusion_matrix.png")
print("=" * 50)
print("\n✅ PROJECT COMPLETE!")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=[f'Digit {i}' for i in range(10)]))
