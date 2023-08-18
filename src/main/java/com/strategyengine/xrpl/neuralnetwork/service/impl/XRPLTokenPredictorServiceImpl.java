package com.strategyengine.xrpl.neuralnetwork.service.impl;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
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
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import com.strategyengine.xrpl.neuralnetwork.entity.IssuedTokenEnt;
import com.strategyengine.xrpl.neuralnetwork.entity.IssuedTokenStatEnt;
import com.strategyengine.xrpl.neuralnetwork.metrics.BollingerBandsCalculator;
import com.strategyengine.xrpl.neuralnetwork.metrics.MovingAverageCalculator;
import com.strategyengine.xrpl.neuralnetwork.metrics.RsiCalculator;
import com.strategyengine.xrpl.neuralnetwork.model.BollingerBand;
import com.strategyengine.xrpl.neuralnetwork.model.Prediction;
import com.strategyengine.xrpl.neuralnetwork.model.PredictionConfig;
import com.strategyengine.xrpl.neuralnetwork.model.PredictionDate;
import com.strategyengine.xrpl.neuralnetwork.process.model.NeuralNetworkModel;
import com.strategyengine.xrpl.neuralnetwork.repo.IssuedTokenRepo;
import com.strategyengine.xrpl.neuralnetwork.repo.IssuedTokenStatRepo;
import com.strategyengine.xrpl.neuralnetwork.rest.exception.BadRequestException;
import com.strategyengine.xrpl.neuralnetwork.service.XRPLTokenPredictorService;

import lombok.extern.log4j.Log4j2;

@Log4j2
@Service
public class XRPLTokenPredictorServiceImpl implements XRPLTokenPredictorService {

	@Autowired
	private IssuedTokenRepo issuedTokenRepo;

	@Autowired
	private IssuedTokenStatRepo issuedTokenStatRepo;

	@Autowired
	private RsiCalculator rsiCalculator;

	@Autowired
	private MovingAverageCalculator movingAverageCalculator;

	@Autowired
	private BollingerBandsCalculator bollingerBandsCalculator;

	// TODO
	// add XRP price for each row
	// add Relative Strength Index (RSI), Moving Average Convergence Divergence
	// (MACD), and Bollinger Bands,
	int FIELD_CREATE_DATE = 0;
	int FIELD_ISSUED_AMOUNT = 1;
	int FIELD_TRUSTLINES = 2;
	int FIELD_HOLDERS = 3;
	int FIELD_OFFERS = 4;
	int FIELD_ID = 5;
	int FIELD_PRICE = 6;
	int FIELD_MA_10d = 7;
	int FIELD_MA_30d = 8;
	int FIELD_RSI = 9;
	int FIELD_BOLLINGER_UPPER = 10;
	int FIELD_BOLLINGER_LOWER = 11;
	int FIELD_BOLLINGER_SMA = 12;
	int FIELD_BOLLINGER_STANDARD_DEV = 13;

//	int FIELD_VOL7 = 9;

	private int numInputs = 14;

	private int numOutputs = 1;

	private int predictDaysInFuture = 5;// number of days in the future to predict a price

	private Map<Integer, NeuralNetworkModel> models = new HashMap<>();

	@Override
	public Prediction predict(int tokenId) {

		if (models.get(tokenId) == null) {
			throw new BadRequestException("No model trained yet, please try back later");
		}

		Optional<IssuedTokenEnt> issuedToken = issuedTokenRepo.findById(tokenId);

		if (issuedToken.isEmpty()) {
			throw new BadRequestException("No issued token found for " + tokenId);
		}

		return this.predict(issuedToken.get(), models.get(tokenId));
	}

	// every 7 days
	@Scheduled(fixedRate = 1000 * 60 * 60 * 24 * 7)
	@Override
	public void retrainModel() {

		issuedTokenRepo.findAll(Example.of(IssuedTokenEnt.builder().blackholed(true).build()), Sort.by("id")).stream()
				.forEach(t -> trainAndPredict(t));

	}

	public PredictionConfig trainAndPredict(IssuedTokenEnt token) {

	//	if (!token.getCurrency().equals("FSE")) {// TODO remove
	//		return null;
	//	}
		// 5:3 - 147

		// 50:30 - 123 after 10
		// PredictionConfig(numHiddenNodes=51, numEpochs=28,
		// lossFunction=MEAN_ABSOLUTE_ERROR, learningRate=0.001,
		// prediction=Prediction(token=FSE, sumErrors=123.54745043076362,
		// mostRecentPrice=2.131),
		// model=com.strategyengine.xrpl.neuralnetwork.process.model.NeuralNetworkModel@15c66e43)
		// 2 mins per 5 iterations
		// 100:60 - 127 after 5
		// 50: 200 - 140
		// 40:40 -121
		// 30:30 - 140
		// 40:30 - 128
		// 40:20 - 127
		// 45:40 -127

		int numHiddenNodes = 40;// 1;//5;//10;20;

		int numEpochs = 40;

		double learningRate = Adam.DEFAULT_ADAM_LEARNING_RATE;

		PredictionConfig best = null;

		for (int i = 1; i < 2; i++) {

			numHiddenNodes++;

				best = runScenario(learningRate, numHiddenNodes, numEpochs,
						LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR, best, token);

		}

		log.info("Best config " + best);

		return best;

	}

	int runCount = 0;

	private PredictionConfig runScenario(double learningRateRun, int numHiddenNodes, int numEpochs,
			LossFunction lossFunction, PredictionConfig best, IssuedTokenEnt token) {
		runCount++;

		try {
			PredictionConfig predictionConfig = trainAndPredict(numEpochs, numHiddenNodes, learningRateRun,
					lossFunction, token);

			double averagePercentError = predictionConfig.getPrediction().getAveragePercentError();

			if (best == null || averagePercentError < best.getPrediction().getAveragePercentError()) {

				best = predictionConfig;

				log.info("Better config found \n" + best);

				if(predictionConfig.getPrediction().getAveragePercentError() < 40) {
					models.put(token.getId(), predictionConfig.getModel());
				}
			}

			if (runCount % 5 == 0) {
				log.info("Run {}", predictionConfig);
			}

		} catch (Exception e) {
			log.error(e);
		}
		return best;

	}

	Map<Integer, List<IssuedTokenStatEnt>> statsMap = new HashMap<>();

	public PredictionConfig trainAndPredict(int numEpochs, int numHiddenNodes, double learningRate,
			LossFunctions.LossFunction lossFunction, IssuedTokenEnt token) {

		// Build the neural network model
		NeuralNetworkModel model = new NeuralNetworkModel(numInputs, numOutputs, numHiddenNodes, learningRate,
				lossFunction);

		trainModel(token, model, numEpochs);

		Prediction pred = predict(token, model);

		return PredictionConfig.builder().numEpochs(numEpochs).learningRate(learningRate).lossFunction(lossFunction)
				.numHiddenNodes(numHiddenNodes).prediction(pred).model(model).build();

	}

	private void trainModel(IssuedTokenEnt token, NeuralNetworkModel model, int numEpochs) {

		List<IssuedTokenStatEnt> statsForToken = statsMap.get(token.getId());
		if (statsForToken == null) {
			statsForToken = issuedTokenStatRepo
					.findAll(Example.of(IssuedTokenStatEnt.builder().issuedTokenId(token.getId()).build()),
							Sort.by(Direction.ASC, "createDate"))
					.stream().filter(t -> t.getPrice() != null && t.getPrice().compareTo(BigDecimal.ZERO) > 0)
					.collect(Collectors.toList());

			statsMap.put(token.getId(), statsForToken);
		}
		// start training with at least 15 data points
		for (int maxStat = 15; maxStat + predictDaysInFuture < statsForToken.size(); maxStat++) {

			try {
				AtomicInteger i = new AtomicInteger();

				List<IssuedTokenStatEnt> statsSubList = statsForToken.subList(0, maxStat);

				// create and train each sublist. Subsequent input data has an additional day
				// added, so each run has more valid data than the previous
				double[][] inputData = new double[statsSubList.size()][numInputs];
				double[] outputData = new double[statsSubList.size()]; // Adjust the size based on the number of target
																		// prices

				final List<IssuedTokenStatEnt> statsFnl = new ArrayList<>(statsForToken);

				statsSubList.stream().forEach(s -> convertDataToInput(s, inputData, i.getAndIncrement(), statsFnl));

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

				// sum errors after 20 iterations
				// 156 Norm and not denormalized
				// 1055 not norm or denorm
				// 42800000000000000 norm and denorm
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
			log.error("Could not find stats for " + token);
			return null;
		}
		double[][] tokenInputData = new double[statsForToken.size()][numInputs];
		AtomicInteger i = new AtomicInteger();

		final List<IssuedTokenStatEnt> statsFnl = new ArrayList<>(statsForToken);
		statsForToken.stream().forEach(s -> convertDataToInput(s, tokenInputData, i.getAndIncrement(), statsFnl));

		if (tokenInputData.length <= 1) {
			return Prediction.builder().token(token.getCurrency()).build();
		}

		double[] outputData = new double[tokenInputData.length];
		INDArray inputArray = Nd4j.create(tokenInputData);
		INDArray outputArray = Nd4j.create(outputData).reshape(-1, 1);

		DataSet dataSet = new DataSet(inputArray, outputArray);

		// PredictionConfig(numHiddenNodes=1, numEpochs=2,
		// lossFunction=MEAN_ABSOLUTE_ERROR, learningRate=0.001,
		// sumErrors=1074.6160367064178) - no normalize
		NormalizerStandardize normalizer = new NormalizerStandardize();
		normalizer.fit(dataSet); // Fit the normalizer on your training data
		normalizer.transform(dataSet); // Transform both training and test data

		// Make a normalized prediction
		double[] predictedPrices = model.predict(dataSet.getFeatures());
		/**
		 * double[] normalizedPredictedValue = model.predict(dataSet.getFeatures());
		 * 
		 * INDArray tempFeatures = Nd4j.create(tokenInputData); // Create an INDArray
		 * with the same shape as input data // INDArray normalizedPredictionArray =
		 * Nd4j.create(normalizedPredictedValue); // Your normalized prediction array
		 * 
		 * 
		 * // Copy the normalized prediction values into the appropriate positions in
		 * the features array for (int j = 0; j < normalizedPredictedValue.length; j++)
		 * { tempFeatures.putScalar(j, numInputs - 1, normalizedPredictedValue[j]); //
		 * Assuming the last column corresponds to the prediction column }
		 * 
		 * DataSet tempDataSet = new DataSet(tempFeatures, tempFeatures); // Use the
		 * modified features array for both features and labels
		 * 
		 * normalizer.revert(tempDataSet);
		 * 
		 * Object o = tempDataSet.getFeatures();
		 * 
		 * double[] predictedPrices = tempDataSet.getFeatures().dup().data().asDouble();
		 */

		double averagePercentError = averagePercentError(predictedPrices, statsForToken);

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

		return Prediction.builder().token(token.getCurrency()).averagePercentError(averagePercentError).mostRecentPrice(mostRecentPrice)
				.predictionDates(ImmutableList.of(
						PredictionDate.builder().date(predictedDate(-4))
								.price(predictedPrices[predictedPrices.length - 5]).build(),
						PredictionDate.builder().date(predictedDate(-3))
								.price(predictedPrices[predictedPrices.length - 4]).build(),
						PredictionDate.builder().date(predictedDate(-2))
								.price(predictedPrices[predictedPrices.length - 3]).build(),
						PredictionDate.builder().date(predictedDate(-1))
								.price(predictedPrices[predictedPrices.length - 2]).build(),
						PredictionDate.builder().date(predictedDate(0))
								.price(predictedPrices[predictedPrices.length - 1]).build()))
				.build();

	}

	private Date predictedDate(int i) {
		Calendar c = Calendar.getInstance();
		c.add(Calendar.DATE, predictDaysInFuture - i);
		return c.getTime();
	}

	private double averagePercentError(double[] predictedPrices, List<IssuedTokenStatEnt> statsForToken) {

		double percentErrorsSum = 0;

		for (int i = 0; i < predictedPrices.length; i++) {

			if (statsForToken.size() <= i + predictDaysInFuture) {
				break;
			}

			double expectedPrice = statsForToken.get(i + predictDaysInFuture).getPrice().doubleValue();
			double predictedPrice = predictedPrices[i];
			double percentErr = (predictedPrice - expectedPrice)/expectedPrice;
			percentErrorsSum += Math.abs(percentErr);

		}

		return (percentErrorsSum / statsForToken.size())*100;
	}

	private void convertDataToInput(IssuedTokenStatEnt stat, double[][] inputData, final int idx,
			List<IssuedTokenStatEnt> statsForToken) {

		inputData[idx][FIELD_CREATE_DATE] = stat.getCreateDate().getTime();
		inputData[idx][FIELD_ISSUED_AMOUNT] = stat.getIssuedAmount().longValue();
		inputData[idx][FIELD_TRUSTLINES] = stat.getTrustlines();
		inputData[idx][FIELD_HOLDERS] = stat.getHolders();
		inputData[idx][FIELD_OFFERS] = stat.getOffers();
		inputData[idx][FIELD_ID] = stat.getIssuedTokenId();
		inputData[idx][FIELD_PRICE] = stat.getPrice().doubleValue();
		inputData[idx][FIELD_MA_10d] = movingAverageCalculator.averagePrice(stat, statsForToken, 10);
		inputData[idx][FIELD_MA_30d] = movingAverageCalculator.averagePrice(stat, statsForToken, 10);

		Optional<Double> rsi = rsiCalculator.calculateRSI(stat, statsForToken);
		if (rsi.isPresent()) {
			inputData[idx][FIELD_RSI] = rsi.get();
		}

		Optional<BollingerBand> bollingerBand = bollingerBandsCalculator.calculateBollingerBands(stat, statsForToken);

		if (bollingerBand.isPresent()) {
			BollingerBand bb = bollingerBand.get();
			inputData[idx][FIELD_BOLLINGER_UPPER] = bb.getUpperBand()[bb.getUpperBand().length - 1];
			inputData[idx][FIELD_BOLLINGER_LOWER] = bb.getLowerBand()[bb.getLowerBand().length - 1];
			inputData[idx][FIELD_BOLLINGER_SMA] = bb.getSma()[bb.getSma().length - 1];
			inputData[idx][FIELD_BOLLINGER_STANDARD_DEV] = bb.getStandardDeviation()[bb.getStandardDeviation().length
					- 1];
		}

	}

}
