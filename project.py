# ============================================================
# NEURAL NETWORK FOR IMAGE CLASSIFICATION
# Minor Project - 6th Semester
# ============================================================

# -----------------------------------------------------------
# PART 1: LIBRARIES IMPORT KARNA
# Libraries = ready-made tools jo humara kaam asaan karte hain
# -----------------------------------------------------------
import numpy as np                          # Numbers ke saath kaam karne ke liye
import matplotlib.pyplot as plt             # Graphs aur images dikhane ke liye
import tensorflow as tf                     # Neural Network banane ke liye
from tensorflow import keras        # type: ignore        # Neural Network ka shortcut tool
from tensorflow.keras import layers     # type: ignore    # Network ki layers banane ke liye
from tensorflow.keras.datasets import mnist    # type: ignore  # Handwritten digit dataset
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns                       # Sundar graphs ke liye
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("   NEURAL NETWORK IMAGE CLASSIFICATION")
print("=" * 50)
print("\nStep 1: Libraries successfully load ho gayi!")

# -----------------------------------------------------------
# PART 2: DATA LOAD KARNA
# MNIST = 70,000 handwritten digit images (0 se 9 tak)
# -----------------------------------------------------------
print("\nStep 2: Dataset load ho raha hai...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"   Training images: {x_train.shape[0]} images")
print(f"   Testing images:  {x_test.shape[0]} images")
print(f"   Image size:      {x_train.shape[1]}x{x_train.shape[2]} pixels")
print(f"   Digits (labels): {np.unique(y_train)}")
print("   Data successfully load ho gaya!")

# -----------------------------------------------------------
# PART 3: SAMPLE IMAGES DIKHANA
# Ye dekhenge ki dataset mein kaisa data hai
# -----------------------------------------------------------
print("\nStep 3: Sample images dikhata hun...")

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
print("   Sample images save ho gayi: sample_images.png")

# -----------------------------------------------------------
# PART 4: DATA PREPROCESS KARNA
# Preprocess = data ko model ke liye ready karna
# Pixel values 0-255 hoti hain, inhe 0-1 mein convert karenge
# -----------------------------------------------------------
print("\nStep 4: Data preprocess ho raha hai...")

# Normalize karna (0-255 → 0-1)
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# 28x28 image ko flat 784 numbers ki list mein convert karna
x_train_flat = x_train.reshape(-1, 784)
x_test_flat  = x_test.reshape(-1, 784)

print(f"   Training data shape: {x_train_flat.shape}")
print(f"   Testing data shape:  {x_test_flat.shape}")
print(f"   Pixel range: {x_train_flat.min():.1f} to {x_train_flat.max():.1f}")
print("   Preprocessing complete!")

# -----------------------------------------------------------
# PART 5: NEURAL NETWORK MODEL BANANA
# Ye project ka main part hai
# Layers = network ki manjile, har layer kuch seekhti hai
# -----------------------------------------------------------
print("\nStep 5: Neural Network model ban raha hai...")

model = keras.Sequential([
    # Input Layer: 784 pixels lega
    layers.Dense(256, activation='relu', input_shape=(784,),
                 name='Hidden_Layer_1'),

    # Dropout: Overfitting rokta hai (ratta maarna band karta hai)
    layers.Dropout(0.3, name='Dropout_1'),

    # Hidden Layer 2
    layers.Dense(128, activation='relu', name='Hidden_Layer_2'),

    layers.Dropout(0.2, name='Dropout_2'),

    # Hidden Layer 3
    layers.Dense(64, activation='relu', name='Hidden_Layer_3'),

    # Output Layer: 10 outputs (digits 0-9)
    layers.Dense(10, activation='softmax', name='Output_Layer')
])

print("\n   --- Model Architecture ---")
model.summary()

# -----------------------------------------------------------
# PART 6: MODEL COMPILE KARNA
# Compile = model ko batana ki kaise seekhna hai
# -----------------------------------------------------------
print("\nStep 6: Model compile ho raha hai...")

model.compile(
    optimizer='adam',                          # Seekhne ka tarika
    loss='sparse_categorical_crossentropy',    # Galtiyon ko measure karna
    metrics=['accuracy']                       # Hume accuracy chahiye
)
print("   Model compile ho gaya!")

# -----------------------------------------------------------
# PART 7: MODEL TRAIN KARNA
# Train = model ko data dikhana taaki woh seekhe
# Epochs = kitni baar poora data dekhega
# -----------------------------------------------------------
print("\nStep 7: Model train ho raha hai...")
print("   (Thoda wait kar - 3-5 min lag sakte hain)\n")

history = model.fit(
    x_train_flat, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,    # 10% data validation ke liye
    verbose=1
)

print("\n   Training complete!")

# -----------------------------------------------------------
# PART 8: MODEL TEST KARNA
# Test data pe check karo ki model kitna accurate hai
# -----------------------------------------------------------
print("\nStep 8: Model test ho raha hai...")

test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=0)
print(f"\n   ✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"   📉 Test Loss:     {test_loss:.4f}")

# -----------------------------------------------------------
# PART 9: TRAINING GRAPHS BANANA
# Graphs se dikhega ki model kaise improve hua
# -----------------------------------------------------------
print("\nStep 9: Training graphs ban rahe hain...")

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
print("   Graphs save ho gaye: training_graphs.png")

# -----------------------------------------------------------
# PART 10: PREDICTIONS DIKHANA
# Model naye images pe kya predict karta hai
# -----------------------------------------------------------
print("\nStep 10: Predictions ban rahi hain...")

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
print("   Predictions save ho gayi: predictions.png")

# -----------------------------------------------------------
# PART 11: CONFUSION MATRIX
# Ye dikhata hai ki kaun sa digit kaun se digit se confuse hua
# -----------------------------------------------------------
print("\nStep 11: Confusion Matrix ban raha hai...")

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
print("   Confusion Matrix save ho gaya: confusion_matrix.png")

# -----------------------------------------------------------
# PART 12: FINAL REPORT PRINT KARNA
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
print("\n✅ PROJECT COMPLETE! BEST OF LUCK BHAI!")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=[f'Digit {i}' for i in range(10)]))