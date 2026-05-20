# ============================================================
# NEURAL NETWORK FOR IMAGE CLASSIFICATION - ANIMALS & VEHICLES
# CIFAR-10 Dataset - CNN Version (Higher Accuracy)
# Minor Project - 6th Semester
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras import layers # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.datasets import cifar10 # pyright: ignore[reportMissingImports]
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------
# MODEL FILE NAME
# -----------------------------------------------------------

MODEL_PATH = "cifar10_cnn_model.keras"

# -----------------------------------------------------------
# CLASS NAMES
# -----------------------------------------------------------

CLASS_NAMES = [
    'Airplane', 'Car', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

print("=" * 55)
print("   CNN - ANIMALS & VEHICLES CLASSIFICATION")
print("=" * 55)

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------

print("\nStep 1: Dataset load ho raha hai...")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train.flatten()
y_test  = y_test.flatten()

print(f"   Training images : {x_train.shape[0]}")
print(f"   Testing images  : {x_test.shape[0]}")

# -----------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------

print("\nStep 2: Preprocessing...")

x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# -----------------------------------------------------------
# CHECK IF MODEL EXISTS
# -----------------------------------------------------------

if os.path.exists(MODEL_PATH):

    print("\n✅ Saved model found!")
    print("Loading existing model...")

    model = keras.models.load_model(MODEL_PATH)

    print("✅ Model loaded successfully!")

else:

    print("\n❌ No saved model found.")
    print("Training new model...")

    # -------------------------------------------------------
    # DATA AUGMENTATION
    # -------------------------------------------------------

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10
    )

    datagen.fit(x_train)

    # -------------------------------------------------------
    # CNN MODEL
    # -------------------------------------------------------

    model = keras.Sequential([

        # Block 1
        layers.Conv2D(
            32,
            (3,3),
            padding='same',
            activation='relu',
            input_shape=(32,32,3),
            name='Conv_1'
        ),

        layers.BatchNormalization(),

        layers.Conv2D(
            32,
            (3,3),
            padding='same',
            activation='relu',
            name='Conv_2'
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(2,2),

        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(
            64,
            (3,3),
            padding='same',
            activation='relu',
            name='Conv_3'
        ),

        layers.BatchNormalization(),

        layers.Conv2D(
            64,
            (3,3),
            padding='same',
            activation='relu',
            name='Conv_4'
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(2,2),

        layers.Dropout(0.3),

        # Block 3
        layers.Conv2D(
            128,
            (3,3),
            padding='same',
            activation='relu',
            name='Conv_5'
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(2,2),

        layers.Dropout(0.4),

        # Dense Layers
        layers.Flatten(),

        layers.Dense(
            256,
            activation='relu'
        ),

        layers.BatchNormalization(),

        layers.Dropout(0.5),

        layers.Dense(
            10,
            activation='softmax',
            name='Output'
        )
    ])

    # -------------------------------------------------------
    # MODEL SUMMARY
    # -------------------------------------------------------

    print("\n--- CNN Architecture ---")

    model.summary()

    # -------------------------------------------------------
    # COMPILE MODEL
    # -------------------------------------------------------

    print("\nStep 3: Compiling model...")

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss='sparse_categorical_crossentropy',

        metrics=['accuracy']
    )

    print("✅ Compile complete!")

    # -------------------------------------------------------
    # TRAIN MODEL
    # -------------------------------------------------------

    print("\nStep 4: Training model...")
    print("   (First time training mein 10-15 min lag sakte hain)\n")

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=30,
        validation_data=(x_test, y_test),
        verbose=1
    )

    print("\n✅ Training complete!")

    # -------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------

    print("\nSaving model...")

    model.save(MODEL_PATH)

    print(f"✅ Model saved as {MODEL_PATH}")

    # -------------------------------------------------------
    # TRAINING GRAPHS
    # -------------------------------------------------------

    print("\nGenerating training graphs...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')

    ax1.set_title('Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')

    ax2.set_title('Loss')
    ax2.legend()

    plt.tight_layout()

    plt.savefig(
        'cifar_training_graphs.png',
        dpi=150,
        bbox_inches='tight'
    )

    plt.show()

# -----------------------------------------------------------
# EVALUATE MODEL
# -----------------------------------------------------------

print("\nStep 5: Evaluating model...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"\n✅ Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"📉 Test Loss     : {test_loss:.4f}")

# -----------------------------------------------------------
# PREDICTIONS
# -----------------------------------------------------------

print("\nStep 6: Making predictions...")

predictions = model.predict(
    x_test,
    verbose=0
)

y_pred = np.argmax(predictions, axis=1)

# -----------------------------------------------------------
# CONFUSION MATRIX
# -----------------------------------------------------------

print("\nStep 7: Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.tight_layout()

plt.savefig(
    'cifar_confusion_matrix.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

# -----------------------------------------------------------
# FINAL REPORT
# -----------------------------------------------------------

print("\n" + "=" * 55)
print("           FINAL PROJECT RESULTS")
print("=" * 55)

print(f"Dataset        : CIFAR-10")
print(f"Model Type     : CNN")
print(f"Accuracy       : {test_accuracy * 100:.2f}%")
print(f"Loss           : {test_loss:.4f}")
print(f"Parameters     : {model.count_params():,}")

print("\nGenerated Files:")
print("📈 cifar_training_graphs.png")
print("🔲 cifar_confusion_matrix.png")
print(f"🧠 {MODEL_PATH}")

print("=" * 55)

print("\nDetailed Classification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES
    )
)