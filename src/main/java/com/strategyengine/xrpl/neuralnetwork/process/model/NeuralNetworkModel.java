package com.strategyengine.xrpl.neuralnetwork.process.model;

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NeuralNetworkModel {
	private MultiLayerNetwork model;

	public NeuralNetworkModel(int numInputs, int numOutputs, int numHiddenNodes, double learningRate,
			LossFunctions.LossFunction lossFunction) {
		// Define the neural network architecture
		model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
				.gradientNormalizationThreshold(1.0).seed(1234).updater(new Adam(learningRate)).list()
				.layer(0,
						new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).activation(Activation.RELU)
								.build())//Dense = every neuron connected to every other neuron
				
				.layer(1, new OutputLayer.Builder() // Use OutputLayer instead of DenseLayer
						.nIn(numHiddenNodes).nOut(numOutputs).activation(Activation.IDENTITY) // Or other appropriate
																								// activation function
						.lossFunction(lossFunction) // Choose an appropriate loss function
						.build())
				.build());
		model.init();
	}

	public void train(DataSet dataSet, int numEpochs) {
		// Train the model
		for (int epoch = 0; epoch < numEpochs; epoch++) {
			model.fit(dataSet);
		}
	}

	//returns a predicted price for each of the inputs
	public double[] predict(INDArray inputFeatures) {
		int numPredictions = Long.valueOf(inputFeatures.size(0)).intValue(); // Number of input data points

		// Make predictions for the given input features
		INDArray predictions = model.output(inputFeatures, false);

		// You may want to adjust the predictions for the desired number of days
		double[] predictedPrices = new double[numPredictions];

		for (int i = 0; i < numPredictions; i++) {
			INDArray prediction = predictions.getRow(i); // Get the prediction for the i-th data point
			double predictedPrice = prediction.getDouble(0); // Assuming the prediction is in the first column

			predictedPrices[i] = predictedPrice;
		}

		return predictedPrices;
	}
}