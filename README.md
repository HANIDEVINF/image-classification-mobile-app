# Keras Vision Classifier

Portfolio project using a real Keras model trained on CIFAR-10.

## Train

```bash
python scripts/train_keras_cifar_classifier.py
```

The script downloads CIFAR-10 through `tf.keras.datasets`, trains a neural network, saves the `.keras` model, exports browser-readable weights to `public/model/cifar10_classifier.json`, and writes metrics into `reports/`.

## Run

```bash
npm install
npm run dev
```
