const fs = require("fs")
const path = require("path")

const root = path.resolve(__dirname, "..")
const modelPath = path.join(root, "public", "model", "cifar10_classifier.json")
const indexPath = path.join(root, "public", "index.html")

if (!fs.existsSync(modelPath)) {
  throw new Error("Missing public/model/cifar10_classifier.json. Run python scripts/train_keras_cifar_classifier.py first.")
}

if (!fs.existsSync(indexPath)) {
  throw new Error("Missing public/index.html.")
}

console.log("Static Keras vision app is ready.")
