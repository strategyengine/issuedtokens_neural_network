package com.strategyengine.xrpl.neuralnetwork.service.impl;

import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
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
import com.strategyengine.xrpl.neuralnetwork.repo.IssuedTokenRepo;
import com.strategyengine.xrpl.neuralnetwork.repo.IssuedTokenStatRepo;
import com.strategyengine.xrpl.neuralnetwork.service.XRPLPricePredictorService;

import lombok.extern.log4j.Log4j2;

@Log4j2
@Service
public class XRPLPricePredictorServiceImpl implements XRPLPricePredictorService {

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
	int FIELD_EXCHANGES1 = 6;
	int FIELD_EXCHANGES7 = 7;
	int FIELD_VOL1 = 8;
	int FIELD_VOL7 = 9;

	private int numInputs = 10;

	private int numOutputs = 1;

	@Override
	public PredictionConfig trainAndPredict() {

		int numHiddenNodes = 5;// 1;//5;//10;20;

		int numEpochs = 10;

		double learningRate = Adam.DEFAULT_ADAM_LEARNING_RATE;

		PredictionConfig best = null;

		for (int i = 1; i < 50; i++) {

			numHiddenNodes = numHiddenNodes++;
			numEpochs = numEpochs++;

			for (int epochRun = numEpochs; epochRun > 0; epochRun--) {

				for (double learningRateRun = learningRate; learningRateRun < (Adam.DEFAULT_ADAM_LEARNING_RATE
						* 5); learningRateRun += .001) {

					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR, best);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.COSINE_PROXIMITY, best);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.HINGE, best);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.KL_DIVERGENCE, best);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.L1, best);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.L2, best);
					// best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
					// LossFunctions.LossFunction.MCXENT, best);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.MEAN_ABSOLUTE_PERCENTAGE_ERROR, best);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR, best);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.MSE, best);
					// best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
					// LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD, best);

					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.POISSON, best);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY, best);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.SQUARED_HINGE, best);
					// best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
					// LossFunctions.LossFunction.SPARSE_MCXENT, best);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.SQUARED_LOSS, best);
					best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
							LossFunctions.LossFunction.WASSERSTEIN, best);
					// best = runScenario(epochRun, learningRateRun, numHiddenNodes, numEpochs,
					// LossFunctions.LossFunction.XENT, best);

				}
			}
		}

		log.info("Best config " + best);

		return best;

	}

	int runCount = 0;

	private PredictionConfig runScenario(int epochRun, double learningRateRun, int numHiddenNodes, int numEpochs,
			LossFunction lossFunction, PredictionConfig best) {
		runCount++;
		log.info("Runs " + runCount);
		try {
			Map<IssuedTokenEnt, Prediction> predictions = trainAndPredict(epochRun, numHiddenNodes, learningRateRun,
					lossFunction);

			double priceDistance = 0;
			for (IssuedTokenEnt token : predictions.keySet()) {

				Prediction prediction = predictions.get(token);

				double predictedPrice = prediction.getPredictionDates() != null
						? prediction.getPredictionDates().get(0).getPrice()[0]
						: 0;

				priceDistance += Math.abs(predictedPrice - prediction.getMostRecentPrice());
			}

			if (best == null || priceDistance < best.getPriceDistance()) {

				best = PredictionConfig.builder().numEpochs(numEpochs).learningRate(learningRateRun)
						.lossFunction(lossFunction).numHiddenNodes(numHiddenNodes).prediction(predictions)
						.priceDistance(priceDistance).build();

				log.info("Better config found \n" + best);
			}

		} catch (Exception e) {
			log.error(e.getMessage());
		}
		return best;

	}

	List<IssuedTokenEnt> tokens = null;

	Map<Integer, List<IssuedTokenStatEnt>> statsMap = new HashMap<>();

	public Map<IssuedTokenEnt, Prediction> trainAndPredict(int numEpochs, int numHiddenNodes, double learningRate,
			LossFunctions.LossFunction lossFunction) {

		// Build the neural network model
		NeuralNetworkModel model = buildModel(numHiddenNodes, learningRate, lossFunction);

		if (tokens == null) {
			tokens = issuedTokenRepo.findAll(Example.of(IssuedTokenEnt.builder().blackholed(true).build())).stream()
					// .filter(t -> "FSE".equals(t.getCurrency()))
					.collect(Collectors.toList());
		}
		// Prepare the data for training and testing
		tokens.stream().forEach(t -> trainModel(t, model, numEpochs));

		Map<IssuedTokenEnt, Prediction> preds = new HashMap<>();
		for (IssuedTokenEnt token : tokens) {
			preds.put(token, predict(token, model));
		}
		// Prepare the data for training and testing
		// return tokens.stream().map(t -> predict(t,
		// model)).collect(Collectors.toList());

		return preds;
	}

	private void trainModel(IssuedTokenEnt token, NeuralNetworkModel model, int numEpochs) {

		List<IssuedTokenStatEnt> stats = statsMap.get(token.getId());
		if (stats == null) {
			stats = issuedTokenStatRepo.findAll(
					Example.of(IssuedTokenStatEnt.builder().issuedTokenId(token.getId()).build()),
					Sort.by(Direction.ASC, "createDate"));

			statsMap.put(token.getId(), stats);
		}
		double[][] inputData = new double[stats.size()][numInputs];

		AtomicInteger i = new AtomicInteger();

		stats.stream().forEach(s -> convertDataToInput(s, inputData, i.getAndIncrement()));

		i.set(0);

		double[] outputData = new double[stats.size()]; // Adjust the size based on the number of target prices

		stats.stream().forEach(s -> {
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

		NormalizerStandardize normalizer = new NormalizerStandardize();
		normalizer.fit(dataSet); // Fit the normalizer on your training data
		normalizer.transform(dataSet); // Transform both training and test data

		SplitTestAndTrain testAndTrainSplit = dataSet.splitTestAndTrain(0.8); // 80% for training, 20% for testing

		DataSet testAndTrain = testAndTrainSplit.getTrain();
		DataSet validationData = testAndTrainSplit.getTest();

		// Train the model
		model.train(testAndTrain, numEpochs);

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
		c.add(Calendar.DATE, 1);
		Date futureOne = c.getTime();
		c.add(Calendar.DATE, 4);
		Date futureFive = c.getTime();

		c.add(Calendar.DATE, 25);
		Date futureThirty = c.getTime();

		// Make predictions for 1 day, 5 days, and 30 days intervals
		double[] oneDayPrediction = model.predict(tokenFeatures, 1);
		// double[] fiveDaysPrediction = model.predict(tokenFeatures, 5);
		// double[] thirtyDaysPrediction = model.predict(tokenFeatures, 30);

		// Print the predictions
//		log.info("Predictions for token: " + token.getCurrency());
//		log.info("1 day interval: " + Arrays.toString(oneDayPrediction));
//		log.info("5 days interval: " + Arrays.toString(fiveDaysPrediction));
//		log.info("30 days interval: " + Arrays.toString(thirtyDaysPrediction));

		return Prediction.builder().token(token.getCurrency())
				.mostRecentPrice(statsForToken.get(statsForToken.size() - 1).getPrice().doubleValue()).predictionDates(
						ImmutableList.of(PredictionDate.builder().date(futureOne).price(oneDayPrediction).build()
						// ,PredictionDate.builder().date(futureFive).price(fiveDaysPrediction).build(),
						// PredictionDate.builder().date(futureThirty).price(thirtyDaysPrediction).build()
						)).build();

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
		inputData[idx][FIELD_EXCHANGES1] = convertLong(stat.getExchanges24h());
		inputData[idx][FIELD_EXCHANGES7] = convertLong(stat.getExchanges7d());
		inputData[idx][FIELD_VOL1] = convertLong(stat.getVolume24h());
		inputData[idx][FIELD_VOL7] = convertLong(stat.getVolume7d());

	}

	private long convertLong(String v) {
		try {
			return Long.parseLong(v);
		} catch (Exception e) {
			return 0l;
		}
	}

}

class NeuralNetworkModel {
	private MultiLayerNetwork model;

	public NeuralNetworkModel(int numInputs, int numOutputs, int numHiddenNodes, double learningRate,
			LossFunctions.LossFunction lossFunction) {
		// Define the neural network architecture
		model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
				.gradientNormalizationThreshold(1.0).seed(1234).updater(new Adam(learningRate)).list()
				.layer(0,
						new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).activation(Activation.RELU)
								.build())

				.layer(1, new OutputLayer.Builder() // Use OutputLayer instead of DenseLayer
						.nIn(numHiddenNodes).nOut(numOutputs).activation(Activation.IDENTITY) // Or other appropriate
																								// activation function
						.lossFunction(lossFunction) // Choose an appropriate loss function
						.build())
//						new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numOutputs).activation(Activation.IDENTITY)
				// .build())
				.build());
		model.init();
	}

	public void train(DataSet dataSet, int numEpochs) {
		// Train the model
		for (int epoch = 0; epoch < numEpochs; epoch++) {
			model.fit(dataSet);
		}
	}

	public double[] predict(INDArray inputFeatures, int numDays) {
		int numPredictions = Long.valueOf(inputFeatures.size(0)).intValue(); // Number of input data points

		// Make predictions for the given input features
		INDArray predictions = model.output(inputFeatures);

		// You may want to adjust the predictions for the desired number of days
		double[] predictedPrices = new double[numPredictions];

		for (int i = 0; i < numPredictions; i++) {
			INDArray prediction = predictions.getRow(i); // Get the prediction for the i-th data point
			double predictedPrice = prediction.getDouble(0); // Assuming the prediction is in the first column

			// Here you can apply adjustments for the desired number of days
			// For example, if numDays is 5, you might want to apply a 5-day compound return
			// Adjust the logic here based on your specific requirements

			// Example of a simple adjustment for illustration (adjust this according to
			// your needs)
			double compoundFactor = Math.pow(predictedPrice, numDays); // Assuming numDays is an integer
			predictedPrice = predictedPrice * compoundFactor;

			predictedPrices[i] = predictedPrice;
		}

		return predictedPrices;
	}
}