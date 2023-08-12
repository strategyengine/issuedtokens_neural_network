package com.strategyengine.xrpl.neuralnetwork.rest.issuedtokens;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

import com.google.common.annotations.VisibleForTesting;
import com.strategyengine.xrpl.neuralnetwork.model.Prediction;
import com.strategyengine.xrpl.neuralnetwork.service.XRPLTokenPredictorService;

import io.swagger.annotations.Api;
import lombok.extern.log4j.Log4j2;

@Log4j2
@Api(tags = "Neural Network Endpoints")
@RestController
public class Controller {


	@VisibleForTesting
	@Autowired
	protected XRPLTokenPredictorService xRPLTokenPredictorService;
	
	//model retrains itself weekly
	@GetMapping(value = "/learning/predict/model/train}")
	public void trainModel() {
		
		long start = System.currentTimeMillis();
	    xRPLTokenPredictorService.retrainModel();
		
		double minutes = (System.currentTimeMillis()-start)/1000/60;
		log.info("Model Retrained in minutes : " + minutes);
		
	}

	
	@GetMapping(value = "/learning/predict/issuedtoken/price/{tokenId}")
	public Prediction getHolderofferbyvalueById(@PathVariable int tokenId) {
		
		return xRPLTokenPredictorService.predict(tokenId);
		
	}
}