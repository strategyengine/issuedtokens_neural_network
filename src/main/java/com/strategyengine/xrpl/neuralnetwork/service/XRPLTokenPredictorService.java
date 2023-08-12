package com.strategyengine.xrpl.neuralnetwork.service;

import com.strategyengine.xrpl.neuralnetwork.model.PredictionConfig;

public interface XRPLTokenPredictorService {


	PredictionConfig trainAndPredict(int tokenId);

	PredictionConfig retrainModel();

}
