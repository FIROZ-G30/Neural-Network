﻿using System;
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
        #region Properties

        private int numRows = 0;

        public int NumRows
        {
            get { return numRows; }
        }

        private int numColumn = 0;

        public int NumColumn
        {
            get { return numColumn; }
        }

        #endregion

        #region Data Read Helper

        public double[][] ReadData(string path)
        {
            List<List<double>> matrix = new List<List<double>>();
            double[][] dataMatrix;
            double[] dataVector;
            bool isInitalized = false;

            using (TextFieldParser parser = new TextFieldParser(path))
            {
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(",");

                while (!parser.EndOfData)
                {
                    string[] fields = parser.ReadFields();
                    dataVector = Array.ConvertAll(fields, double.Parse);
                    matrix.Add(dataVector.ToList());
                    numRows++;
                    if (!isInitalized)
                    {
                        numColumn = fields.Length;
                        isInitalized = true;
                    }
                }
            }

            dataMatrix = new double[numRows][];

            for (int i = 0; i < numRows; i++)
            {
                dataMatrix[i] = new double[numColumn];
                dataMatrix[i] = matrix[i].ToArray();
            }

            return dataMatrix;
        } 

        #endregion
    }
}
