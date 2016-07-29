using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkBackPropegation
{
    public partial class NeuralNetwork
    {
        #region Train Network

        public double[] TrainNetwork(double[][] trainingData, int maxEpochs,
          double learningRate, double momentum)                                 // return error vector associated with each training data set
        {
            #region Local Variables

            double[][] hiddenOutputGrads = InitializeMatrix(numOutput, numHidden); // hidden-to-output weight gradients
            double[] outputBiasGrads = new double[numOutput];                   // output bias gradients

            double[][] inputHiddenGrads = InitializeMatrix(numHidden, numInput);  // input-to-hidden weight gradients
            double[] hiddenBiasGrads = new double[numHidden];                   // hidden bias gradients

            double[] outputSignalGrads = new double[numOutput];                  // local gradient output signals - gradients w/o associated input terms
            double[] hiddenSignalGrads = new double[numHidden];                  // local gradient hidden node signals

            // back-prop momentum specific arrays 
            double[][] inputHiddenPrevWeightsDelta = InitializeMatrix(numHidden, numInput);
            double[] hiddenPrevBiasesDelta = new double[numHidden];
            double[][] hiddenOutputPrevWeightsDelta = InitializeMatrix(numOutput, numHidden);
            double[] outputPrevBiasesDelta = new double[numOutput];

            double[] inputs = new double[numInput]; // inputs
            double[] desiredOutput = new double[numOutput]; // target values
            double[] actualOutput = new double[numOutput];

            double[] trainingError = new double[maxEpochs];

            double derivative = 0.0;
            double errorSignal = 0.0;

            #endregion

            for (int iteration = 0; iteration < maxEpochs; iteration++)
            {
                trainingError[iteration] = MeanSquaredError(trainingData);

                for (int k = 0; k < trainingData.Length; k++)  // iterate through each data set 
                {
                    Array.Copy(trainingData[k], inputs, numInput);    // extract input vector
                    Array.Copy(trainingData[k], numInput, desiredOutput, 0, numOutput); // extract desired output vector

                    actualOutput = ComputeOutput(inputs);   // compute output vector from current neural network

                    // i = inputs, h = hidden, p = output

                    #region Calculate Weight Gradients

                    // output node signals (softmax)
                    for (int p = 0; p < numOutput; p++)
                    {
                        errorSignal = desiredOutput[p] - actualOutput[p];
                        derivative = DerivativeSigmoid(actualOutput[p]);    // for softmax
                        outputSignalGrads[p] = errorSignal * derivative;
                    }

                    //compute hidden-to-output weight gradients using output signals
                    for (int h = 0; h < numHidden; h++)
                    {
                        for (int p = 0; p < numOutput; p++)
                        {
                            hiddenOutputGrads[p][h] = outputSignalGrads[p] * hiddenOutputs[h];
                        }
                    }

                    //compute output bias gradients using output signals
                    //outputBiasGrads[p] = outputSignalGrads[p] * 1.0
                    Array.Copy(outputSignalGrads, outputBiasGrads, numOutput);

                    //compute hidden node signals
                    double sum = 0;
                    for (int h = 0; h < numHidden; h++)
                    {
                        derivative = DerivativeSigmoid(hiddenOutputs[h]);   // for sigmoid
                        sum = 0;    // need sums of output signals times hidden-to-output weights
                        for (int p = 0; p < numOutput; p++)
                        {
                            sum += outputSignalGrads[p] * hiddenOutputWeights[p][h];    // represents error signal
                        }

                        hiddenSignalGrads[h] = derivative * sum;
                    }

                    //compute input-hidden weight gradients
                    for (int i = 0; i < numInput; i++)
                    {
                        for (int h = 0; h < numHidden; h++)
                        {
                            inputHiddenGrads[h][i] = hiddenSignalGrads[h] * inputs[i];
                        }
                    }

                    //compute hidden node bias gradients
                    //hiddenBiasGrads[h] = hiddenSignalGrads[h] * 1.0
                    Array.Copy(hiddenSignalGrads, hiddenBiasGrads, numHidden);

                    #endregion

                    //----update weights and bias in network----//
                    #region Update Network weight Matrix

                    double delta = 0;
                    // update input-to-hidden weights
                    for (int i = 0; i < numInput; i++)
                    {
                        for (int h = 0; h < numHidden; h++)
                        {
                            delta = inputHiddenGrads[h][i] * learningRate;
                            inputHiddenWeights[h][i] += delta;
                            inputHiddenWeights[h][i] += inputHiddenPrevWeightsDelta[h][i] * momentum;
                            inputHiddenPrevWeightsDelta[h][i] = delta;
                        }
                    }

                    // update hidden biases
                    for (int h = 0; h < numHidden; h++)
                    {
                        delta = hiddenBiasGrads[h] * learningRate;
                        hiddenBiases[h] += delta;
                        hiddenBiases[h] += hiddenPrevBiasesDelta[h] * momentum;
                        hiddenPrevBiasesDelta[h] = delta;
                    }

                    // update hidden-to-output weights
                    for (int h = 0; h < numHidden; h++)
                    {
                        for (int p = 0; p < numOutput; p++)
                        {
                            delta = hiddenOutputGrads[p][h] * learningRate;
                            hiddenOutputWeights[p][h] += delta;
                            hiddenOutputWeights[p][h] += hiddenOutputPrevWeightsDelta[p][h] * momentum;
                            hiddenOutputPrevWeightsDelta[p][h] = delta;
                        }
                    }

                    // update output node biases
                    for (int p = 0; p < numOutput; p++)
                    {
                        delta = outputBiasGrads[p] * learningRate;
                        outputBiases[p] += delta;
                        outputBiases[p] += outputPrevBiasesDelta[p] * momentum;
                        outputPrevBiasesDelta[p] = delta;
                    }

                    #endregion
                }
            }

            return trainingError;
        }

        #endregion

        #region Initialize Matrix

        private double[][] InitializeMatrix(int rows, int column)
        {
            double[][] matrix = new double[rows][];

            for (int i = 0; i < matrix.Length; i++)
            {
                matrix[i] = new double[column];
                for (int j = 0; j < column; j++)
                {
                    matrix[i][j] = 0.0;
                }
            }

            return matrix;
        } 

        #endregion
    }
}
