package com.strategyengine.xrpl.neuralnetwork.model;

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
public class BollingerBand {

	private double[] sma;
	
	private double[] standardDeviation;
	
	private double[] upperBand;
	
	private double[] lowerBand;


}
