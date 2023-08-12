package com.strategyengine.xrpl.neuralnetwork.service;

import com.strategyengine.xrpl.neuralnetwork.model.Prediction;
import com.strategyengine.xrpl.neuralnetwork.model.PredictionConfig;

public interface XRPLTokenPredictorService {

	PredictionConfig retrainModel();

	Prediction predict(int tokenId);

}
