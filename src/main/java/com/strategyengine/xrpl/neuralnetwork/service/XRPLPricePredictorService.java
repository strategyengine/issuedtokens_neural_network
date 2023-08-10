package com.strategyengine.xrpl.neuralnetwork.service;

import com.strategyengine.xrpl.neuralnetwork.model.PredictionConfig;

public interface XRPLPricePredictorService {

	PredictionConfig trainAndPredict();

}
