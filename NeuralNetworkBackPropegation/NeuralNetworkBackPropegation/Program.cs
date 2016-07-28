using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkBackPropegation
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork(2, 3, 2);
            double[][] result1 = nn.GetWeightMatrix(WeightMatrix.HiddenOutput);
            double[][] result2 = nn.GetWeightMatrix(WeightMatrix.InputHidden);

            for (int i=0; i< result1.Length; i++)
            {
                for (int j=0; j < result1[i].Length; j++)
                {
                    Console.Write(result1[i][j]+" ");
                }
                Console.Write("\n");
            }

            for (int i = 0; i < result2.Length; i++)
            {
                for (int j = 0; j < result2[i].Length; j++)
                {
                    Console.Write(result2[i][j] + " ");
                }
                Console.Write("\n");
            }

            Console.ReadLine();
        }
    }
}
