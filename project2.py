# ============================================================
# NEURAL NETWORK FOR IMAGE CLASSIFICATION - ANIMALS & VEHICLES
# CIFAR-10 Dataset - CNN Version (Higher Accuracy)
# Minor Project - 6th Semester
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras   # type: ignore
from tensorflow.keras import layers   # type: ignore
from tensorflow.keras.datasets import cifar10   # type: ignore
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

CLASS_NAMES = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

print("=" * 55)
print("   CNN - ANIMALS & VEHICLES CLASSIFICATION")
print("=" * 55)
print("\nStep 1: Libraries load ho gayi!")

# -----------------------------------------------------------
# PART 2: DATA LOAD KARNA
# -----------------------------------------------------------
print("\nStep 2: Dataset load ho raha hai...")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test  = y_test.flatten()

print(f"   Training images : {x_train.shape[0]}")
print(f"   Testing images  : {x_test.shape[0]}")
print("   Data load ho gaya!")

# -----------------------------------------------------------
# PART 3: SAMPLE IMAGES
# -----------------------------------------------------------
print("\nStep 3: Sample images dikhata hun...")

plt.figure(figsize=(14, 6))
plt.suptitle("CIFAR-10 Dataset - Sample Images", fontsize=14, fontweight='bold')
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(f"{CLASS_NAMES[y_train[i]]}", fontsize=9)
    plt.axis('off')
plt.tight_layout()
plt.savefig('cifar_sample_images.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: cifar_sample_images.png")

# -----------------------------------------------------------
# PART 4: PREPROCESS
# -----------------------------------------------------------
print("\nStep 4: Data preprocess ho raha hai...")

# Normalize - CNN ko flat nahi karna, shape same rakhna
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# Data Augmentation - training images ko thoda rotate/flip karke
# zyada data banate hain taaki model better seekhe
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10
)
datagen.fit(x_train)

print(f"   Shape: {x_train.shape}")
print("   Preprocessing complete!")

# -----------------------------------------------------------
# PART 5: CNN MODEL BANANA
# Conv2D = image ke features dhundhta hai (edges, shapes, colors)
# MaxPooling = image ko compress karta hai important parts rakhke
# -----------------------------------------------------------
print("\nStep 5: CNN Model ban raha hai...")

model = keras.Sequential([

    # Block 1
    layers.Conv2D(32, (3,3), padding='same', activation='relu',
                  input_shape=(32,32,3), name='Conv_1'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), padding='same', activation='relu', name='Conv_2'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.2),

    # Block 2
    layers.Conv2D(64, (3,3), padding='same', activation='relu', name='Conv_3'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), padding='same', activation='relu', name='Conv_4'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.3),

    # Block 3
    layers.Conv2D(128, (3,3), padding='same', activation='relu', name='Conv_5'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.4),

    # Flatten + Dense Layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax', name='Output')
])

print("\n   --- CNN Architecture ---")
model.summary()

# -----------------------------------------------------------
# PART 6: COMPILE
# -----------------------------------------------------------
print("\nStep 6: Model compile ho raha hai...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("   Compile ho gaya!")

# -----------------------------------------------------------
# PART 7: TRAIN
# -----------------------------------------------------------
print("\nStep 7: Model train ho raha hai...")
print("   (10-15 min lag sakte hain - song sun le beech mein 😄)\n")

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(x_test, y_test),
    verbose=1
)

print("\n   Training complete!")

# -----------------------------------------------------------
# PART 8: EVALUATE
# -----------------------------------------------------------
print("\nStep 8: Model test ho raha hai...")

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n   ✅ Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"   📉 Test Loss     : {test_loss:.4f}")

# -----------------------------------------------------------
# PART 9: TRAINING GRAPHS
# -----------------------------------------------------------
print("\nStep 9: Graphs ban rahe hain...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("CNN Training Results - CIFAR-10", fontsize=14, fontweight='bold')

ax1.plot(history.history['accuracy'],     label='Training Accuracy',   color='blue',   linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch Number')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'],     label='Training Loss',   color='red',   linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', color='green', linewidth=2)
ax2.set_title('Loss over Epochs')
ax2.set_xlabel('Epoch Number')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cifar_training_graphs.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: cifar_training_graphs.png")

# -----------------------------------------------------------
# PART 10: PREDICTIONS
# -----------------------------------------------------------
print("\nStep 10: Predictions...")

predictions = model.predict(x_test, verbose=0)

plt.figure(figsize=(15, 8))
plt.suptitle("CNN Predictions  |  Green = Correct  |  Red = Wrong",
             fontsize=12, fontweight='bold')

for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_test[i])
    predicted  = np.argmax(predictions[i])
    actual     = y_test[i]
    confidence = np.max(predictions[i]) * 100
    color = 'green' if predicted == actual else 'red'
    plt.title(f"P:{CLASS_NAMES[predicted][:4]}\nA:{CLASS_NAMES[actual][:4]}\n{confidence:.0f}%",
              color=color, fontsize=7)
    plt.axis('off')

plt.tight_layout()
plt.savefig('cifar_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: cifar_predictions.png")

# -----------------------------------------------------------
# PART 11: CONFUSION MATRIX
# -----------------------------------------------------------
print("\nStep 11: Confusion Matrix...")

y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix - CIFAR-10 CNN',
          fontsize=13, fontweight='bold')
plt.xlabel('Predicted Category', fontsize=11)
plt.ylabel('Actual Category',    fontsize=11)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('cifar_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: cifar_confusion_matrix.png")

# -----------------------------------------------------------
# PART 12: FINAL REPORT
# -----------------------------------------------------------
print("\n" + "=" * 55)
print("           FINAL PROJECT RESULTS - CNN")
print("=" * 55)
print(f"  Dataset        : CIFAR-10")
print(f"  Model Type     : CNN (Convolutional Neural Network)")
print(f"  Categories     : {len(CLASS_NAMES)}")
print(f"  Model Accuracy : {test_accuracy * 100:.2f}%")
print(f"  Model Loss     : {test_loss:.4f}")
print(f"  Total Params   : {model.count_params():,}")
print("\n  Files generated:")
print("  📸 cifar_sample_images.png")
print("  📈 cifar_training_graphs.png")
print("  🎯 cifar_predictions.png")
print("  🔲 cifar_confusion_matrix.png")
print("=" * 55)
print("\n✅ CNN PROJECT COMPLETE! BHAI TU CHAMPION HAI! 🏆")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
