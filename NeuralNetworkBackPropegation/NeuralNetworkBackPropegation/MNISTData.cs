using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualBasic;
using Microsoft.VisualBasic.FileIO;
using System.IO;
namespace NeuralNetworkBackPropegation
{
    public class MNISTData
    {
        public double[][] ReadData(string path)
        {
            List<List<double>> matrix = new List<List<double>>();
            double[][] dataMatrix;
            double[] dataVector;
            int row = 0, column = 0;
            bool isInitalized = false;

            using (TextFieldParser parser = new TextFieldParser(path))
            {
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(",");
                
                while (!parser.EndOfData)
                {
                    string[] fields = parser.ReadFields();
                    dataVector = Array.ConvertAll<string, double>(fields, double.Parse);
                    matrix.Add(dataVector.ToList());
                    row++;
                    if (!isInitalized)
                    {
                        column = fields.Length;
                        isInitalized = true;
                    }
                }       
            }

            dataMatrix = new double[row][];

            for (int i=0; i< row; i++)
            {
                dataMatrix[i] = new double[column];
                dataMatrix[i] = matrix[i].ToArray();
            }

            return dataMatrix;
        }
    }
}
