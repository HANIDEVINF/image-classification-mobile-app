"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { BarChart3, Brain, Camera, ImageUp, Loader2, RotateCcw, Sparkles, UploadCloud } from "lucide-react"

type ModelExport = {
  classes: string[]
  weights: Record<string, number[] | number[][]>
  metrics: {
    dataset: string
    accuracy: number
    macro_f1: number
    train_size: number
    test_size: number
    epochs_ran: number
  }
}

type Prediction = {
  label: string
  score: number
}

function relu(values: number[]) {
  return values.map((value) => Math.max(0, value))
}

function softmax(values: number[]) {
  const max = Math.max(...values)
  const exp = values.map((value) => Math.exp(value - max))
  const total = exp.reduce((sum, value) => sum + value, 0)
  return exp.map((value) => value / total)
}

function dense(input: number[], kernel: number[][], bias: number[]) {
  const output = new Array(bias.length).fill(0)
  for (let out = 0; out < bias.length; out += 1) {
    let sum = bias[out]
    for (let i = 0; i < input.length; i += 1) sum += input[i] * kernel[i][out]
    output[out] = sum
  }
  return output
}

async function imageToFeatures(file: File) {
  const bitmap = await createImageBitmap(file)
  const canvas = document.createElement("canvas")
  canvas.width = 16
  canvas.height = 16
  const context = canvas.getContext("2d")
  if (!context) throw new Error("Canvas is not available")
  context.drawImage(bitmap, 0, 0, 16, 16)
  const pixels = context.getImageData(0, 0, 16, 16).data
  const features: number[] = []
  for (let i = 0; i < pixels.length; i += 4) {
    features.push(pixels[i] / 255, pixels[i + 1] / 255, pixels[i + 2] / 255)
  }
  return { features, preview: URL.createObjectURL(file) }
}

function runModel(model: ModelExport, features: number[]) {
  const w = model.weights
  const h1 = relu(dense(features, w.dense1_kernel as number[][], w.dense1_bias as number[]))
  const h2 = relu(dense(h1, w.dense2_kernel as number[][], w.dense2_bias as number[]))
  const probabilities = softmax(dense(h2, w.out_kernel as number[][], w.out_bias as number[]))
  return model.classes
    .map((label, index) => ({ label, score: probabilities[index] }))
    .sort((a, b) => b.score - a.score)
}

export default function Home() {
  const [model, setModel] = useState<ModelExport | null>(null)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [preview, setPreview] = useState("")
  const [status, setStatus] = useState("Loading trained Keras model")
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetch("/model/cifar10_classifier.json")
      .then((response) => response.json())
      .then((loaded: ModelExport) => {
        setModel(loaded)
        setStatus("Upload a photo to test the model")
      })
      .catch(() => setStatus("Model artifact is missing. Run the training script first."))
  }, [])

  const top = predictions[0]
  const metrics = useMemo(() => model?.metrics, [model])

  async function classify(file: File) {
    if (!model) return
    setStatus("Reading image and running browser inference")
    const { features, preview: nextPreview } = await imageToFeatures(file)
    setPreview(nextPreview)
    setPredictions(runModel(model, features))
    setStatus("Prediction complete")
  }

  return (
    <main className="min-h-screen bg-[#071013] text-slate-50">
      <section className="mx-auto grid min-h-screen max-w-7xl gap-8 px-5 py-8 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="flex flex-col justify-center">
          <div className="mb-5 inline-flex w-fit items-center gap-2 rounded-full border border-sky-300/30 bg-sky-300/10 px-4 py-2 text-sm text-sky-100">
            <Brain className="h-4 w-4" />
            Real Keras model trained on CIFAR-10
          </div>
          <h1 className="max-w-3xl text-5xl font-black leading-none md:text-7xl">Mobile Vision Classifier</h1>
          <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-300">
            Upload any image and the app resizes it to the same 16x16 RGB representation used during training, then runs
            the exported Keras dense network directly in the browser.
          </p>
          <div className="mt-8 grid gap-3 sm:grid-cols-3">
            {metrics &&
              [
                ["Dataset", metrics.dataset],
                ["Accuracy", `${(metrics.accuracy * 100).toFixed(1)}%`],
                ["Macro F1", `${(metrics.macro_f1 * 100).toFixed(1)}%`],
              ].map(([label, value]) => (
                <div key={label} className="rounded-lg border border-white/10 bg-white/[0.04] p-4">
                  <div className="text-sm text-slate-400">{label}</div>
                  <div className="mt-2 text-2xl font-black text-sky-200">{value}</div>
                </div>
              ))}
          </div>
          <div className="mt-8 rounded-lg border border-white/10 bg-white/[0.04] p-5">
            <div className="mb-4 flex items-center gap-2 font-bold text-sky-100">
              <UploadCloud className="h-5 w-5" />
              Test with your own image
            </div>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) classify(file)
              }}
            />
            <div className="flex flex-col gap-3 sm:flex-row">
              <button
                onClick={() => inputRef.current?.click()}
                className="flex items-center justify-center gap-2 rounded-md bg-sky-300 px-5 py-3 font-black text-slate-950"
              >
                <ImageUp className="h-4 w-4" />
                Upload Image
              </button>
              <button
                onClick={() => {
                  setPreview("")
                  setPredictions([])
                  setStatus("Upload a photo to test the model")
                }}
                className="flex items-center justify-center gap-2 rounded-md border border-white/10 px-5 py-3 font-bold text-slate-200"
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </button>
            </div>
            <div className="mt-4 flex items-center gap-2 text-sm text-slate-400">
              {!model && <Loader2 className="h-4 w-4 animate-spin" />}
              {status}
            </div>
          </div>
        </div>

        <div className="grid content-center gap-5">
          <div className="overflow-hidden rounded-lg border border-white/10 bg-white/[0.04]">
            {preview ? (
              <img src={preview} alt="Uploaded preview" className="h-[440px] w-full object-cover" />
            ) : (
              <div className="flex h-[440px] items-center justify-center bg-[#102027]">
                <div className="text-center text-slate-400">
                  <Camera className="mx-auto mb-3 h-12 w-12 text-sky-200" />
                  Your uploaded image appears here
                </div>
              </div>
            )}
          </div>
          <div className="rounded-lg border border-white/10 bg-white/[0.04] p-5">
            <div className="mb-5 flex items-center justify-between">
              <div className="flex items-center gap-2 font-bold">
                <BarChart3 className="h-5 w-5 text-emerald-300" />
                Prediction probabilities
              </div>
              {top && (
                <div className="rounded-full bg-emerald-300 px-3 py-1 text-sm font-black text-slate-950">
                  {top.label} {(top.score * 100).toFixed(0)}%
                </div>
              )}
            </div>
            <div className="space-y-3">
              {(predictions.length ? predictions : model?.classes.map((label) => ({ label, score: 0 })) || []).map((item) => (
                <div key={item.label}>
                  <div className="mb-1 flex justify-between text-sm">
                    <span className="capitalize text-slate-300">{item.label}</span>
                    <span>{(item.score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-white/10">
                    <div className="h-2 rounded-full bg-sky-300" style={{ width: `${Math.max(2, item.score * 100)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="rounded-lg border border-emerald-300/20 bg-emerald-300/10 p-4 text-sm leading-6 text-emerald-50">
            <Sparkles className="mb-2 h-5 w-5" />
            This is a real exported neural network. The app does not call an API or use fixed suggestions.
          </div>
        </div>
      </section>
    </main>
  )
}
