package com.strategyengine.xrpl.neuralnetwork.model;

import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.strategyengine.xrpl.neuralnetwork.process.model.NeuralNetworkModel;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Builder
@EqualsAndHashCode
@ToString
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Setter
public class PredictionConfig {

	private int numHiddenNodes;

	private int numEpochs;

	private LossFunctions.LossFunction lossFunction;
	private double learningRate;

	private Prediction prediction;

	@ToString.Exclude
	private NeuralNetworkModel model;

}
