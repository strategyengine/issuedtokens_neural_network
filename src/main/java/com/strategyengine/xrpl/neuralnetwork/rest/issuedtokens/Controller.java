package com.strategyengine.xrpl.neuralnetwork.rest.issuedtokens;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import com.google.common.annotations.VisibleForTesting;
import com.strategyengine.xrpl.neuralnetwork.model.PredictionConfig;
import com.strategyengine.xrpl.neuralnetwork.service.XRPLPricePredictorService;

import io.swagger.annotations.Api;
import lombok.extern.log4j.Log4j2;

@Log4j2
@Api(tags = "Neural Network Endpoints")
@RestController
public class Controller {

	@VisibleForTesting
	@Autowired
	protected XRPLPricePredictorService xRPLPricePredictorService;

	
	@GetMapping(value = "/learning/predict/issuedtoken/price")
	public PredictionConfig getHolderofferbyvalueById() {
		
		return xRPLPricePredictorService.trainAndPredict();
		
	}
	
}