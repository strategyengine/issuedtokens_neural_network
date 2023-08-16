package com.strategyengine.xrpl.neuralnetwork.metrics;

import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;

import com.strategyengine.xrpl.neuralnetwork.entity.IssuedTokenStatEnt;

@Service
public class RsiCalculator {

	//14 day RSI
	private int period  = 14;
	
    public Optional<Double> calculateRSI(IssuedTokenStatEnt stat, List<IssuedTokenStatEnt> stats) {
    	
    	//calculate RSI 
    	List<IssuedTokenStatEnt> statsForToken = stats.subList(0, stats.indexOf(stat)+1);
    	

		if (statsForToken.size() < period + 1) {
			return Optional.empty();
		}
		
		if (statsForToken.size() > period) {
			statsForToken = statsForToken.subList(statsForToken.size() - (period+1), statsForToken.size());
		}
                
        double[] priceChanges = new double[statsForToken.size() - 1];
        for (int i = 1; i < statsForToken.size(); i++) {
            priceChanges[i - 1] = statsForToken.get(i).getPrice().doubleValue() - statsForToken.get(i - 1).getPrice().doubleValue();
        }

        double averageGain = 0;
        double averageLoss = 0;

        for (int i = 0; i < period; i++) {
            if (priceChanges[i] > 0) {
                averageGain += priceChanges[i];
            } else {
                averageLoss -= priceChanges[i];
            }
        }

        averageGain /= period;
        averageLoss /= period;
        
        if(averageLoss==0) {
        	return Optional.empty();
        }

        double rs = averageGain / averageLoss;
        double rsi = 100 - (100 / (1 + rs));

        if(Double.isNaN(rsi)) {
        	return Optional.empty();
        }
        return Optional.of(rsi);
    }
}





