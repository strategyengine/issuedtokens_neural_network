package com.strategyengine.xrpl.neuralnetwork.metrics;

import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;

import com.strategyengine.xrpl.neuralnetwork.entity.IssuedTokenStatEnt;
import com.strategyengine.xrpl.neuralnetwork.model.BollingerBand;

@Service
public class BollingerBandsCalculator {

	private int period = 14;

	private double stddevMultiplier = 2.0;

	public Optional<BollingerBand> calculateBollingerBands(IssuedTokenStatEnt stat,
			List<IssuedTokenStatEnt> statsForToken) {

		if (statsForToken.size() < period + 1) {
			return Optional.empty();
		}

		double[] closingPrices = new double[statsForToken.size()];
		for (int i = 0; i < statsForToken.size(); i++) {
			closingPrices[i] = statsForToken.get(i).getPrice().doubleValue();
		}

		double[] sma = calculateSMA(closingPrices, period);
		double[] standardDeviations = calculateStandardDeviations(closingPrices, period, sma);

		double[] upperBollingerBands = new double[sma.length];
		double[] lowerBollingerBands = new double[sma.length];

		for (int i = 0; i < sma.length; i++) {
			upperBollingerBands[i] = sma[i] + (stddevMultiplier * standardDeviations[i]);
			lowerBollingerBands[i] = sma[i] - (stddevMultiplier * standardDeviations[i]);
		}

		if (upperBollingerBands.length == 0) {
			return Optional.empty();
		}

		return Optional.of(BollingerBand.builder().sma(sma).upperBand(upperBollingerBands)
				.lowerBand(lowerBollingerBands).standardDeviation(standardDeviations).build());

	}

	private double[] calculateSMA(double[] closingPrices, int period) {
		double[] sma = new double[closingPrices.length - period + 1];

		for (int i = 0; i <= closingPrices.length - period; i++) {
			double sum = 0;
			for (int j = i; j < i + period; j++) {
				sum += closingPrices[j];
			}
			sma[i] = sum / period;
		}

		return sma;
	}

	private double[] calculateStandardDeviations(double[] closingPrices, int period, double[] sma) {
		double[] standardDeviations = new double[closingPrices.length - period + 1];

		for (int i = 0; i <= closingPrices.length - period; i++) {
			double sumSquaredDeviations = 0;
			for (int j = i; j < i + period; j++) {
				double deviation = closingPrices[j] - sma[i];
				sumSquaredDeviations += deviation * deviation;
			}
			double meanSquaredDeviations = sumSquaredDeviations / period;
			double standardDeviation = Math.sqrt(meanSquaredDeviations);
			standardDeviations[i] = standardDeviation;
		}

		return standardDeviations;
	}
}