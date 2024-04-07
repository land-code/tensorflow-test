import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

const form = document.querySelector('form') as HTMLFormElement
const result = document.querySelector('#result') as HTMLDivElement

const resultWeight = document.querySelector('#result-weight') as HTMLSpanElement
const resultBias = document.querySelector('#result-bias') as HTMLSpanElement

const filterInputs = (inputs: FormDataEntryValue[]): number[] => {
  return inputs
    .filter(input => {
      if (input === '') {
        return false
      }
      if (typeof input !== 'string') {
        return false
      }
      if (Number.isNaN(parseInt(input))) {
        return false
      }

      return true
    })
    .map(input => parseInt(input as string))
}

form.addEventListener('submit', async e => {
  e.preventDefault()

  const formData = new FormData(form)
  const inputs = formData.getAll('input')
  const outputs = formData.getAll('output')

  const filteredInputs = filterInputs(inputs)
  const filteredOutputs = filterInputs(outputs)

  console.log({ filteredInputs, filteredOutputs })
  // show data in tfvis sidebar
  tfvis.render.scatterplot(
    { name: 'Input vs Output' },
    {
      values: filteredInputs.map((input, index) => ({
        x: input,
        y: filteredOutputs[index],
      })),
    },
    {
      xAxisDomain: [0, Math.max(...filteredInputs) * 2],
      yAxisDomain: [0, Math.max(...filteredOutputs) * 2],
    },
  )

  // create a model
  const model = tf.sequential()
  model.add(tf.layers.dense({ inputShape: [1], units: 1 }))
  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' })

  // convert data to tensors
  const xs = tf.tensor2d(filteredInputs, [filteredInputs.length, 1])
  const ys = tf.tensor2d(filteredOutputs, [filteredOutputs.length, 1])

  // train the model
  await model.fit(xs, ys, { epochs: 500 })

  // get weight and bias
  const weight = await model.getWeights()[0].data()
  const bias = await model.getWeights()[1].data()

  // show result
  resultWeight.innerText = Math.round(weight[0]).toString()
  resultBias.innerText = Math.round(bias[0]).toString()

  // show linear regression line with weight and bias and show the previous points on the same graph. The points and the line should be visible in the same graph.

  function calculateLinePoints(weight: number, bias: number, max: number) {
    const points = []
    for (let i = 0; i < max * 100; i += 0.01) {
      points.push({
        x: i,
        y: weight * i + bias,
      })
    }

    return points
  }

  const linePoints = calculateLinePoints(
    weight[0],
    bias[0],
    Math.max(...filteredInputs),
  )

  tfvis.render.scatterplot(
    { name: 'Input vs Output' },
    {
      values: [
        filteredInputs.map((input, index) => ({
          x: input,
          y: filteredOutputs[index],
        })),
        linePoints.map(point => ({
          x: point.x,
          y: point.y,
        })),
      ],
      series: ['actual', 'line'],
    },
    {
      xAxisDomain: [0, Math.max(...filteredInputs) * 2],
      yAxisDomain: [0, Math.max(...filteredOutputs) * 2],
      seriesColors: ['red', 'blue'],

    },
  )

  result.hidden = false
})
