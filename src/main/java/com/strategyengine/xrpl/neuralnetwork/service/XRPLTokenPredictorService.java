package com.strategyengine.xrpl.neuralnetwork.service;

import com.strategyengine.xrpl.neuralnetwork.model.Prediction;

public interface XRPLTokenPredictorService {

	void retrainModel();

	Prediction predict(int tokenId);

}
