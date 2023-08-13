package com.strategyengine.xrpl.neuralnetwork.metrics;

import java.util.List;

import org.springframework.stereotype.Service;

import com.strategyengine.xrpl.neuralnetwork.entity.IssuedTokenStatEnt;

@Service
public class MovingAverageCalculator {

  
	public double averagePrice(IssuedTokenStatEnt stat, List<IssuedTokenStatEnt> statsForToken, int averageDays) {
		int endIdxOfStat = statsForToken.indexOf(stat);
		int idxOfStat = endIdxOfStat - averageDays;
		if (idxOfStat < 0) {
			idxOfStat = 0;
		}
		double cnt = 0;
		double sum = 0;
		while (idxOfStat <= endIdxOfStat) {
			sum += statsForToken.get(idxOfStat).getPrice().doubleValue();
			cnt++;
			idxOfStat++;
		}

		return sum / cnt;

	}
	
}





