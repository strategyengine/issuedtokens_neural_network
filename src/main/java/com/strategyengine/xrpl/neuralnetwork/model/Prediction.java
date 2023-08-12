package com.strategyengine.xrpl.neuralnetwork.model;

import java.util.List;

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
public class Prediction {

	private String token;
	
	private double sumErrors;
	
	private List<PredictionDate> predictionDates;
	
	private double mostRecentPrice;

	
	
}
