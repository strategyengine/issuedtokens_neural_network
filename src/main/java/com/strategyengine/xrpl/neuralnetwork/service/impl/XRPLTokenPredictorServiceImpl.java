package com.strategyengine.xrpl.neuralnetwork.service.impl;

import java.math.BigDecimal;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.shade.protobuf.common.collect.ImmutableList;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Example;
import org.springframework.data.domain.Sort;
import org.springframework.data.domain.Sort.Direction;
import org.springframework.stereotype.Service;

import com.strategyengine.xrpl.neuralnetwork.entity.IssuedTokenEnt;
import com.strategyengine.xrpl.neuralnetwork.entity.IssuedTokenStatEnt;
import com.strategyengine.xrpl.neuralnetwork.model.Prediction;
import com.strategyengine.xrpl.neuralnetwork.model.PredictionConfig;
import com.strategyengine.xrpl.neuralnetwork.model.PredictionDate;
import com.strategyengine.xrpl.neuralnetwork.process.model.NeuralNetworkModel;
import com.strategyengine.xrpl.neuralnetwork.repo.IssuedTokenRepo;
import com.strategyengine.xrpl.neuralnetwork.repo.IssuedTokenStatRepo;
import com.strategyengine.xrpl.neuralnetwork.service.XRPLTokenPredictorService;

import lombok.extern.log4j.Log4j2;

@Log4j2
@Service
public class XRPLTokenPredictorServiceImpl implements XRPLTokenPredictorService {

	@Autowired
	private IssuedTokenRepo issuedTokenRepo;

	@Autowired
	private IssuedTokenStatRepo issuedTokenStatRepo;

	int FIELD_CREATE_DATE = 0;
	int FIELD_ISSUED_AMOUNT = 1;
	int FIELD_TRUSTLINES = 2;
	int FIELD_HOLDERS = 3;
	int FIELD_OFFERS = 4;
	int FIELD_ID = 5;
	int FIELD_PRICE = 6;
//	int FIELD_EXCHANGES7 = 7;
//	int FIELD_VOL1 = 8;
//	int FIELD_VOL7 = 9;

	private int numInputs = 7;

	private int numOutputs = 1;

	private int predictDaysInFuture = 5;// number of days in the future to predict a price

	private NeuralNetworkModel bestModel = null;
	
	private NeuralNetworkModel model = null;

	@Override
	public PredictionConfig trainAndPredict(int tokenId) {

		int numHiddenNodes = 1;// 1;//5;//10;20;

		int numEpochs = 2;

		double learningRate = Adam.DEFAULT_ADAM_LEARNING_RATE;

		PredictionConfig best = null;

		for (int i = 1; i < 50; i++) {

			numHiddenNodes = numHiddenNodes++;
			numEpochs = numEpochs++;

			for (int epochRun = numEpochs; epochRun > 0; epochRun--) {

				for (double learningRateRun = learningRate; learningRateRun < (Adam.DEFAULT_ADAM_LEARNING_RATE
						* 5); learningRateRun += .001) {

					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR, best, tokenId);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.COSINE_PROXIMITY, best, tokenId);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.HINGE, best, tokenId);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.KL_DIVERGENCE, best, tokenId);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.L1, best, tokenId);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.L2, best, tokenId);
					// best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
					// LossFunctions.LossFunction.MCXENT, best, tokenId);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.MEAN_ABSOLUTE_PERCENTAGE_ERROR, best, tokenId);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR, best, tokenId);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.MSE, best, tokenId);
					// best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
					// LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD, best, tokenId);

					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.POISSON, best, tokenId);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY, best, tokenId);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.SQUARED_HINGE, best, tokenId);
					// best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
					// LossFunctions.LossFunction.SPARSE_MCXENT, best, tokenId);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.SQUARED_LOSS, best, tokenId);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.WASSERSTEIN, best, tokenId);
					// best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
					// LossFunctions.LossFunction.XENT, best, tokenId);

				}
			}
		}

		log.info("Best config " + best);

		return best;

	}

	int runCount = 0;

	private PredictionConfig runScenario(int epochRun, double learningRateRun, int numHiddenNodes, int numEpochs,
			LossFunction lossFunction, PredictionConfig best, int tokenId) {
		runCount++;
		log.info("Runs " + runCount);
		try {
			Map<Integer, Prediction> predictions = trainAndPredict(epochRun, numHiddenNodes, learningRateRun,
					lossFunction, tokenId);

			double sumErrors = predictions.get(tokenId).getSumErrors();

			if (best == null || sumErrors < best.getSumErrors()) {

				best = PredictionConfig.builder().numEpochs(numEpochs).learningRate(learningRateRun)
						.lossFunction(lossFunction).numHiddenNodes(numHiddenNodes).prediction(predictions)
						.sumErrors(sumErrors).build();

				log.info("Better config found \n" + best);
				
				bestModel = model;
			}

		} catch (Exception e) {
			log.error(e);
		}
		return best;

	}

	Map<Integer, List<IssuedTokenStatEnt>> statsMap = new HashMap<>();

	public Map<Integer, Prediction> trainAndPredict(int numEpochs, int numHiddenNodes, double learningRate,
			LossFunctions.LossFunction lossFunction, int tokenId) {

		// Build the neural network model
		 model = buildModel(numHiddenNodes, learningRate, lossFunction);

		IssuedTokenEnt token = issuedTokenRepo.findById(tokenId).get();
				
		boolean trainUsingAllTokens = true;
		if (!trainUsingAllTokens) {;
			trainModel(token, model, numEpochs);
		} else {
			issuedTokenRepo.findAll(Example.of(IssuedTokenEnt.builder().blackholed(true).build())).stream()
					.forEach(y -> trainModel(y, model, numEpochs));
		}

		Map<Integer, Prediction> preds = new HashMap<>();

		Prediction pred = predict(token, model);
		if (pred != null) {
			preds.put(token.getId(), pred);
		}

		return preds;
	}

	private void trainModel(IssuedTokenEnt token, NeuralNetworkModel model, int numEpochs) {

		List<IssuedTokenStatEnt> statsForToken = statsMap.get(token.getId());
		if (statsForToken == null) {
			statsForToken = issuedTokenStatRepo
					.findAll(Example.of(IssuedTokenStatEnt.builder().issuedTokenId(token.getId()).build()),
							Sort.by(Direction.ASC, "createDate"))
					.stream().filter(t -> t.getPrice() != null).collect(Collectors.toList());

			statsMap.put(token.getId(), statsForToken);
		}
		//start training with at least 15 data points
		for (int maxStat = 15; maxStat + predictDaysInFuture < statsForToken.size(); maxStat++) {

			try {
				AtomicInteger i = new AtomicInteger();

				List<IssuedTokenStatEnt> statsSubList = statsForToken.subList(0, maxStat);

				// create and train each sublist. Subsequent input data has an additional day
				// added, so each run has more valid data than the previous
				double[][] inputData = new double[statsSubList.size()][numInputs];
				double[] outputData = new double[statsSubList.size()]; // Adjust the size based on the number of target
																		// prices

				statsSubList.stream().forEach(s -> convertDataToInput(s, inputData, i.getAndIncrement()));

				i.set(0);

				// The predicted price is 5 days in the future for each input data row
				List<IssuedTokenStatEnt> statsPredictedPricesSubList = statsForToken.subList(predictDaysInFuture,
						maxStat + predictDaysInFuture);
				statsPredictedPricesSubList.stream().forEach(s -> {
					try {
						int idx = i.getAndIncrement();
						if (s.getPrice() != null) {

							outputData[idx] = s.getPrice().doubleValue();
						}
					} catch (Exception e) {
						log.error("Parse price " + s, e);
					}
				});

				if (inputData.length <= 1) {
					return;
				}

				INDArray inputArray = Nd4j.create(inputData);
				INDArray outputArray = Nd4j.create(outputData).reshape(-1, 1);

				DataSet dataSet = new DataSet(inputArray, outputArray);

				//PredictionConfig(numHiddenNodes=1, numEpochs=2, lossFunction=MEAN_ABSOLUTE_ERROR, learningRate=0.001, sumErrors=1074.6160367064178) - no normalize
				 NormalizerStandardize normalizer = new NormalizerStandardize();
				 normalizer.fit(dataSet); // Fit the normalizer on your training data
				 normalizer.transform(dataSet); // Transform both training and test data

				SplitTestAndTrain testAndTrainSplit = dataSet.splitTestAndTrain(0.8); // 80% for training, 20% for
																						// testing

				DataSet testAndTrain = testAndTrainSplit.getTrain();
				DataSet validationData = testAndTrainSplit.getTest();

				// Train the model
				model.train(testAndTrain, numEpochs);
			} catch (Exception e) {
				log.error(e);
			}
		}
	}

	private Prediction predict(IssuedTokenEnt token, NeuralNetworkModel model) {

		List<IssuedTokenStatEnt> statsForToken = statsMap.get(token.getId());

		if (statsForToken == null) {
			statsForToken = issuedTokenStatRepo
					.findAll(Example.of(IssuedTokenStatEnt.builder().issuedTokenId(token.getId()).build()));
		}
		double[][] tokenInputData = new double[statsForToken.size()][numInputs];
		AtomicInteger i = new AtomicInteger();

		statsForToken.stream().forEach(s -> convertDataToInput(s, tokenInputData, i.getAndIncrement()));

		if (tokenInputData.length <= 1) {
			return Prediction.builder().token(token.getCurrency()).build();
		}

		INDArray tokenFeatures = Nd4j.create(tokenInputData);

		Calendar c = Calendar.getInstance();
		c.add(Calendar.DATE, predictDaysInFuture);

		// Make predictions for 1 day, 5 days, and 30 days intervals
		double[] predictedPrices = model.predict(tokenFeatures);

		double sumErrors = sumErrors(predictedPrices, statsForToken);

		double mostRecentPrice = 0;
		for (int p = statsForToken.size() - 1; p > 0; p--) {

			BigDecimal price = statsForToken.get(p).getPrice();

			if (BigDecimal.ZERO.equals(price)) {
				continue;
			}

			mostRecentPrice = price.doubleValue();
			break;
		}
		if (mostRecentPrice == 0) {
			return null;
		}
		return Prediction.builder().token(token.getCurrency()).sumErrors(sumErrors).mostRecentPrice(mostRecentPrice)
				.predictionDates(ImmutableList.of(PredictionDate.builder().date(c.getTime())
						.price(predictedPrices[predictedPrices.length - 1]).build()))
				.build();

	}

	private double sumErrors(double[] predictedPrices, List<IssuedTokenStatEnt> statsForToken) {

		double totalErrors = 0;

		for (int i = 0; i < predictedPrices.length; i++) {

			if (statsForToken.size() <= i + predictDaysInFuture) {
				break;
			}

			double expectedPrice = statsForToken.get(i + predictDaysInFuture).getPrice().doubleValue();

			if (predictedPrices[i] > expectedPrice) {
				totalErrors += predictedPrices[i] - expectedPrice;
			} else {
				totalErrors += expectedPrice - predictedPrices[i];
			}

		}

		return totalErrors;
	}

	private NeuralNetworkModel buildModel(int numHiddenNodes, double learningRate,
			LossFunctions.LossFunction lossFunction) {
		return new NeuralNetworkModel(numInputs, numOutputs, numHiddenNodes, learningRate, lossFunction);
	}

	private void convertDataToInput(IssuedTokenStatEnt stat, double[][] inputData, final int idx) {

		inputData[idx][FIELD_CREATE_DATE] = stat.getCreateDate().getTime();
		inputData[idx][FIELD_ISSUED_AMOUNT] = stat.getIssuedAmount().longValue();
		inputData[idx][FIELD_TRUSTLINES] = stat.getTrustlines();
		inputData[idx][FIELD_HOLDERS] = stat.getHolders();
		inputData[idx][FIELD_OFFERS] = stat.getOffers();
		inputData[idx][FIELD_ID] = stat.getIssuedTokenId();
		inputData[idx][FIELD_PRICE] = stat.getPrice().doubleValue();

	}

	@Override
	public PredictionConfig retrainModel() {
		// TODO Auto-generated method stub
		return null;
	}

}
