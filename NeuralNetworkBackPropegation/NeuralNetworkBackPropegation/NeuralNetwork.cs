using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkBackPropegation
{
    public partial class NeuralNetwork
    {
        #region Private Variables

        private int numInput; // number input nodes
        private int numHidden;
        private int numOutput;

        private double[][] inputHiddenWeights; // input-hidden weight matrix
        private double[] hiddenBiases;  // bias vector of the hidden layer
        private double[] hiddenOutputs; //output vector of the hidden layer

        private double[][] hiddenOutputWeights; // hidden-output weight matrix
        private double[] outputBiases;  // bias vector of the output layer

        private Random rnd;

        #endregion

        #region Constructor

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            hiddenOutputs = new double[numHidden];

            hiddenBiases = new double[numHidden];
            outputBiases = new double[numOutput];

            inputHiddenWeights = InitializeMatrix(numHidden,numInput);
 
            hiddenOutputWeights = InitializeMatrix(numOutput, numHidden);

            rnd = new Random(0);
            InitializeWeights(); // all weights and biases

        }

        #endregion

        #region Initialize Weights

        private void InitializeWeights()
        {
            // initialize weights and biases to small random values
            // in this input-hidden and hidden-output weightmatrises
            // the 0th column seperated to bias weight vector for hidden and output layer respectivly 

            double[][] inputHiddenWeightMat = InitializeMatrix(numHidden,numInput + 1);// numinput + bias

            double[][] hiddenOutputWeightMat = InitializeMatrix(numOutput, numHidden + 1);// numhidden + bias  

            // randomly generate weights between 0 and 1
            for (int i = 0; i < numHidden; i++)
            {
                for (int j = 0; j < numInput + 1; j++)
                {
                    inputHiddenWeightMat[i][j] = RandomDouble(6);
                }
            }

            for (int i = 0; i < numOutput; i++)
            {
                for (int j = 0; j < numHidden + 1; j++)
                {
                    hiddenOutputWeightMat[i][j] = RandomDouble(6);
                }
            }

            this.SetWeights(inputHiddenWeightMat, hiddenOutputWeightMat);   // set weights on network
        }

        #endregion

        #region Random Number

        private double RandomDouble(int precision)
        {
            double random = rnd.NextDouble();
            return Math.Round(random, precision);
        } 

        #endregion

        #region Set Weights

        public void SetWeights(double[][] inputHiddenWeightMat, double[][] hiddenOutputWeightMat)
        {
            // check for the dimension errors of the matrises
            if (inputHiddenWeightMat[0].Length != (numInput + 1)
                || inputHiddenWeightMat.Length != numHidden
                || hiddenOutputWeightMat[0].Length != (numHidden + 1)
                || hiddenOutputWeightMat.Length != (numOutput))
                throw new Exception("Weight matrix dimension error");

            for (int i = 0; i < numHidden; i++)
            {
                hiddenBiases[i] = inputHiddenWeightMat[i][0]; // seperate hidden bias vector

                for (int j = 1; j < numInput + 1; j++)
                {
                    inputHiddenWeights[i][j - 1] = inputHiddenWeightMat[i][j];  // seperate input-hidden weight matrix
                }
            }

            for (int i = 0; i < numOutput; i++)
            {
                outputBiases[i] = hiddenOutputWeightMat[i][0];  // seperate output bias vector

                for (int j = 1; j < numHidden + 1; j++)
                {
                    hiddenOutputWeights[i][j - 1] = hiddenOutputWeightMat[i][j];    // seperate hidden-output weight matrix
                }
            }
        }

        #endregion

        #region Get Current Wegiht Matrix

        public double[][] GetWeightMatrix(WeightMatrix type)
        {
            if (type == WeightMatrix.InputHidden)
            {
                double[][] inputHiddenWeightMat = InitializeMatrix(numHidden, numInput + 1);
 
                for (int i = 0; i < numHidden; i++)
                {
                    inputHiddenWeightMat[i][0] = hiddenBiases[i]; // seperate hidden bias vector

                    for (int j = 1; j < numInput; j++)
                    {
                        inputHiddenWeightMat[i][j] = inputHiddenWeights[i][j - 1];  // seperate input-hidden weight matrix
                    }
                }

                return inputHiddenWeightMat;
            }
            else if (type == WeightMatrix.HiddenOutput)
            {
                double[][] hiddenOutputWeightMat = InitializeMatrix(numOutput, numHidden + 1);

                for (int i = 0; i < numOutput; i++)
                {
                    hiddenOutputWeightMat[i][0] = outputBiases[i];  // seperate output bias vector

                    for (int j = 1; j < numHidden; j++)
                    {
                        hiddenOutputWeightMat[i][j] = hiddenOutputWeights[i][j - 1];    // seperate hidden-output weight matrix
                    }
                }

                return hiddenOutputWeightMat;
            }
            else
            {
                return InitializeMatrix(1, 1);
            }
        }

        #endregion

        #region Sigmoid Function

        private static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        #endregion

        #region Derivative Sigmoid

        private static double DerivativeSigmoid(double output)
        {
            return (1 - output) * output;
        }

        #endregion

        #region SoftMax Function

        private static double[] Softmax(double[] output)
        {
            double sum = 0.0;
            for (int i = 0; i < output.Length; ++i)
                sum += Math.Exp(output[i]);

            double[] result = new double[output.Length];
            for (int i = 0; i < output.Length; ++i)
                result[i] = Math.Exp(output[i]) / sum;

            return result; // now scaled so that xi sum to 1.0
        }

        #endregion

        #region Compute Output Vector

        public double[] ComputeOutput(double[] input)
        {
            double[] hiddenSums = new double[numHidden]; // hidden nodes sums scratch array
            double[] outputSums = new double[numOutput]; // output nodes sums

            for (int i = 0; i < numHidden; ++i)
            {
                hiddenSums[i] += this.hiddenBiases[i];  // add biases to hidden sums

                for (int j = 0; j < numInput; ++j)
                {
                    hiddenSums[i] += input[j] * this.inputHiddenWeights[i][j];  // compute input-hidden sum of weights * inputs
                }
            }

            for (int i = 0; i < numHidden; ++i) // apply activation
            {
                this.hiddenOutputs[i] = Sigmoid(hiddenSums[i]);
            }

            for (int i = 0; i < numOutput; ++i)
            {
                outputSums[i] += outputBiases[i];   // add biases to output sums

                for (int j = 0; j < numHidden; ++j)
                {
                    outputSums[i] += hiddenOutputs[j] * hiddenOutputWeights[i][j];  // compute hidden-output sum of weights * hOutputs
                }
            }

            return Softmax(outputSums); // softmax all outputs to support classification
        }

        #endregion

        #region Mean Squared Error

        public double MeanSquaredError(double[][] trainingData, double[][] desiredOutputMat)
        {
            // average squared error per training item
            double sumSquaredError = 0.0;
            double[] actualOutput;

            for (int i = 0; i < trainingData.Length; i++)
            {
                actualOutput = ComputeOutput(trainingData[i]);   // get output for current weight matrix

                for (int j = 0; j < numOutput; j++)
                {
                    double error = desiredOutputMat[i][j] - actualOutput[j];
                    sumSquaredError += error * error;
                }
            }

            return sumSquaredError / (trainingData.Length * 2); // devide by 2 to cancel out derivative coefficient
        }

        #endregion

        #region Get Accuracy

        public double Accuracy(double[][] testData, double[][] desiredOutputMat)
        {
            int correctTestCases = 0;
            int incorrectTestCases = 0;
            int desiredIndex = 0;
            int actualIndex = 0;

            double[] actualOutput;

            for (int i = 0; i < testData.Length; i++)
            {
                actualOutput = ComputeOutput(testData[i]);   // get output for current weight matrix

                desiredIndex = MaxIndex(desiredOutputMat[i]);
                actualIndex = MaxIndex(actualOutput);

                if (desiredIndex == actualIndex)
                {
                    correctTestCases++;
                }
                else
                {
                    incorrectTestCases++;
                }

            }

            if ((correctTestCases + incorrectTestCases) != 0)
            {
                return correctTestCases / (correctTestCases + incorrectTestCases);
            }
            else
            {
                throw new Exception("Test cases calculation failed");
            }
        }

        #endregion

        #region Max Index

        private static int MaxIndex(double[] vector)
        {
            int maxIndex = 0;
            double maxValue = vector[0];

            for (int i = 1; i < vector.Length; i++)
            {
                if (maxValue < vector[i])
                {
                    maxValue = vector[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }

        #endregion
    }
}
