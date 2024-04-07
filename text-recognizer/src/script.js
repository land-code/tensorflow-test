import {MnistData} from './data.js';
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

async function showExamples(data) {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

function getModel () {
  const model = tf.sequential()

  const IMAGE_WIDTH = 28
  const IMAGE_HEIGHT = 28
  const IMAGE_CHANNELS = 1

  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }))

  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
  }))

  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }))

  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
  }))

  model.add(tf.layers.flatten())

  const NUM_OUTPUT_CLASSES = 10
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }))

  const optimizer = tf.train.adam()
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  return model
}

/**
 * A function to train the model
 * @param {tf.Sequential} model 
 * @param {} data 
 * @returns 
 */
async function train (model, data) {
  const metris = ['loss', 'val_loss', 'acc', 'val_acc']
  const container = {
    name: 'Model Training',
    tab: 'Model',
    styles: { height: '1000px' }
  }

  const fitCallbacks = tfvis.show.fitCallbacks(container, metris)

  const BATCH_SIZE = 512
  const TRAIN_DATA_SIZE = 55000
  const TEST_DATA_SIZE = 10000

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE)
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      d.labels
    ]
  })

  const [textXs, testYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TEST_DATA_SIZE)
    return [
      d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      d.labels
    ]
  })

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    model,
    epochs: 10,
    validationData: [textXs, testYs],
    shuffle: true,
    callbacks: fitCallbacks
  })
}

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}


async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});

  labels.dispose();
}

function initFileUpload (model) {
  const filePicker = document.getElementById('fileUpload')
  filePicker.onchange = async () => {
    const file = filePicker.files[0]
    const reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onload = async () => {
      const image = new Image()
      image.src = reader.result
      image.width = 28
      image.height = 28
      image.onload = async () => {
        const tensor = tf.browser.fromPixels(image)
          .resizeNearestNeighbor([28, 28])
          .mean(2)
          .toFloat()
          .reshape([1, 28, 28, 1])
        const prediction = model.predict(tensor)
        const pIndex = tf.argMax(prediction, 1).dataSync()
        alert(`The digit is ${pIndex[0]}`)
      }
    }
  }
}

function initCanvas (model) {
  const canvas = document.getElementById('canvas')
  canvas.addEventListener('mousemove', (e) => {
    if (e.buttons === 1) {
      const ctx = canvas.getContext('2d')
      ctx.fillStyle = 'rgb(255, 255, 255)'
      ctx.fillRect(e.offsetX, e.offsetY, 25, 25)
    }
  })

  canvas.addEventListener('mouseup', () => {
    const img = new Image()
    img.src = canvas.toDataURL('image/png')
    img.width = 28
    img.height = 28
    img.onload = async () => {
      const tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([28, 28])
        .mean(2)
        .toFloat()
        .reshape([1, 28, 28, 1])
      const prediction = model.predict(tensor)
      const pIndex = tf.argMax(prediction, 1).dataSync()
      alert(`The digit is ${pIndex[0]}`)

      canvas.getContext('2d').clearRect(0, 0, 400, 400)
    }
  })
}

async function run() {  
  const data = new MnistData();
  await data.load();
  await showExamples(data);
  
  const model = getModel();
  tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);
  
  initFileUpload(model)
  initCanvas(model)

  await train(model, data);
  await showAccuracy(model, data);
  await showConfusion(model, data);

  
}

document.addEventListener('DOMContentLoaded', run);
