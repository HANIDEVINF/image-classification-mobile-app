from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "public" / "model"
REPORT_DIR = ROOT / "reports"
SEED = 42
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def downsample(images: np.ndarray) -> np.ndarray:
    images = images.astype("float32") / 255.0
    images = images.reshape((-1, 16, 2, 16, 2, 3)).mean(axis=(2, 4))
    return images.reshape((images.shape[0], -1))


def main() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    x_train = downsample(x_train)
    x_test = downsample(x_test)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(768,), name="downsampled_rgb"),
            tf.keras.layers.Dense(256, activation="relu", name="visual_features"),
            tf.keras.layers.Dropout(0.25, name="dropout"),
            tf.keras.layers.Dense(128, activation="relu", name="semantic_projection"),
            tf.keras.layers.Dense(10, activation="softmax", name="class_probabilities"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.12,
        epochs=18,
        batch_size=256,
        verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)],
    )

    probabilities = model.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "dataset": "CIFAR-10",
        "dataset_source": "tf.keras.datasets.cifar10",
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "input_features": 768,
        "classes": CLASSES,
        "architecture": ["Dense(256, relu)", "Dropout(0.25)", "Dense(128, relu)", "Dense(10, softmax)"],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "epochs_ran": len(history.history["loss"]),
    }

    weights = model.get_weights()
    export = {
        "model_type": "keras_dense_cifar10_classifier",
        "classes": CLASSES,
        "input": {"width": 16, "height": 16, "channels": 3, "normalization": "rgb_0_1"},
        "weights": {
            "dense1_kernel": weights[0].tolist(),
            "dense1_bias": weights[1].tolist(),
            "dense2_kernel": weights[2].tolist(),
            "dense2_bias": weights[3].tolist(),
            "out_kernel": weights[4].tolist(),
            "out_bias": weights[5].tolist(),
        },
        "metrics": metrics,
    }

    model.save(MODEL_DIR / "keras_cifar10_classifier.keras")
    (MODEL_DIR / "cifar10_classifier.json").write_text(json.dumps(export), encoding="utf-8")
    (REPORT_DIR / "keras_cifar10_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, predictions, target_names=CLASSES),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
