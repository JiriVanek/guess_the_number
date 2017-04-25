package cz.zcu.kiv.eeg.gtn.application.classification;

import cz.zcu.kiv.eeg.gtn.application.featureextraction.IFeatureExtraction;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

// creates instance of Stacked Denoising Autoencoder @author Pumprdlici group
public class MixDeepLearning4jEarlyStop implements IERPClassifier {
    private final int NEURON_COUNT_DEFAULT = 30;    //default number of neurons
    private IFeatureExtraction fe;                //type of feature extraction (MatchingPursuit, FilterAndSubampling or WaveletTransform)
    //private MultiLayerNetwork model;            //multi layer neural network with a logistic output layer and multiple hidden neuralNets
    private MultiLayerNetwork bestModel;
    private int neuronCount;                    // Number of neurons
    private int iterations;                    //Iterations used to classify
    private String directory = "C:\\Temp\\";
    private int maxTime =5; //max time in minutes
    private int maxEpochs = 10000;
    private EarlyStoppingResult result;
    private int noImprovementEpochs = 20;
    private EarlyStoppingConfiguration esConf;
    private String pathname = "C:\\Temp\\SDAEStop"; //pathname+file name for saving model


    /*Default constructor*/
    public MixDeepLearning4jEarlyStop() {
        this.neuronCount = NEURON_COUNT_DEFAULT; // sets count of neurons in layer(0) to default number
    }

    /*Parametric constructor */
    public MixDeepLearning4jEarlyStop(int neuronCount) {
        this.neuronCount = neuronCount; // sets count of neurons in layer(0) to param
    }

    /*Classifying features*/
    @Override
    public double classify(double[][] epoch) {
        double[] featureVector = this.fe.extractFeatures(epoch); // Extracting features to vector
        INDArray features = Nd4j.create(featureVector); // Creating INDArray with extracted features
        return bestModel.output(features, Layer.TrainingMode.TEST).getDouble(0); // Result of classifying
    }

    @Override
    public void train(List<double[][]> epochs, List<Double> targets, int numberOfiter, IFeatureExtraction fe) {
        // Customizing params of classifier
        final int numRows = fe.getFeatureDimension();   // number of targets on a line
        final int numColumns = 2;   // number of labels needed for classifying
        this.iterations = numberOfiter; // number of iteration in the learning phase
        int listenerFreq = numberOfiter / 10; // frequency of output strings
        int seed = 123; //  seed - one of parameters. For more info check http://deeplearning4j.org/iris-flower-dataset-tutorial
        //Load Data - when target is 0, label[0] is 0 and label[1] is 1.
        double[][] labels = new double[targets.size()][numColumns]; // Matrix of labels for classifier
        double[][] features_matrix = new double[targets.size()][numRows]; // Matrix of features
        for (int i = 0; i < epochs.size(); i++) { // Iterating through epochs
            double[][] epoch = epochs.get(i); // Each epoch
            double[] features = fe.extractFeatures(epoch); // Feature of each epoch
            for (int j = 0; j < numColumns; j++) {   //setting labels for each column
                labels[i][0] = targets.get(i); // Setting label on position 0 as target
                labels[i][1] = Math.abs(1 - targets.get(i));  // Setting label on position 1 to be different from label[0]
            }
            features_matrix[i] = features; // Saving features to features matrix
        }

        // Creating INDArrays and DataSet
        INDArray output_data = Nd4j.create(labels); // Create INDArray with labels(targets)
        INDArray input_data = Nd4j.create(features_matrix); // Create INDArray with features(data)
        DataSet dataSet = new DataSet(input_data, output_data); // Create dataSet with features and labels
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true; // Setting to enforce numerical stability

        // Building a neural net
        MultiLayerConfiguration conf = this.createConfiguration(numRows, numColumns, seed, listenerFreq);
        //model = new MultiLayerNetwork(conf); // Passing built configuration to instance of multilayer network
       // model.init(); // Initialize model
        //model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq))); // Setting listeners
        //model.setListeners(new ScoreIterationListener(100));

        SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(0.35);
        //EarlyStoppingModelSaver saver = new LocalFileModelSaver(directory);
        InMemoryModelSaver <MultiLayerNetwork> saver = new InMemoryModelSaver();


        List<EpochTerminationCondition> list = new ArrayList<>(2);
        list.add(new MaxEpochsTerminationCondition(maxEpochs));
        list.add(new ScoreImprovementEpochTerminationCondition(noImprovementEpochs, 0.00001));
        //list.add(new ScoreImprovementEpochTerminationCondition(noImprovementEpochs));

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        //DataSetIterator testData= new TestDataSetIterator(dataSet);
        //DataSetIterator trainData= new TestDataSetIterator(dataSet);


                esConf = new EarlyStoppingConfiguration.Builder()
                //.epochTerminationConditions(new MaxEpochsTerminationCondition(maxEpochs))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(maxTime, TimeUnit.MINUTES))
                //.epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(noImprovementEpochs))
                //
                .scoreCalculator(new DataSetLossCalculator(new ListDataSetIterator(dataSet.asList()),true))
                //.scoreCalculator(new DataSetLossCalculator(new ListDataSetIterator(testAndTrain.getTest().asList(), 100), true))
                .evaluateEveryNEpochs(3)
                .modelSaver(saver)
                .epochTerminationConditions(list)
                .build();

        //create Estop trainer
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, net, new ListDataSetIterator(testAndTrain.getTrain().asList(), 100));
        //prepare UI
        UIServer uiServer = UIServer.getInstance();
        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains
        ArrayList listeners = new ArrayList();
        listeners.add(new ScoreIterationListener(100));
        listeners.add(new StatsListener(statsStorage));
        net.setListeners(listeners);
        result = trainer.fit();

        bestModel = (MultiLayerNetwork) result.getBestModel();

        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        Evaluation eval = new Evaluation(numColumns);
        eval.eval(dataSet.getLabels(), bestModel.output(dataSet.getFeatureMatrix(), Layer.TrainingMode.TEST));
        System.out.println(eval.stats());
    }

    //  initialization of neural net with params. For more info check http://deeplearning4j.org/iris-flower-dataset-tutorial where is more about params
    private MultiLayerConfiguration createConfiguration(int numRows, int outputNum, int seed, int listenerFreq) {
        System.out.print("Build model....SDA EStop");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder() // Starting builder pattern
                .seed(seed) // Locks in weight initialization for tuning
                //.weightInit(WeightInit.XAVIER)
                //.activation(Activation.LEAKYRELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.06)
                .iterations(1)
                //.momentum(0.5) // Momentum rate
                //.momentumAfter(Collections.singletonMap(3, 0.9)) //Map of the iteration to the momentum rate to apply at that iteration
                .list() // # NN layers (doesn't count input layer)
                .layer(0, new AutoEncoder.Builder()
                        .nIn(numRows)
                        .nOut(64)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.LEAKYRELU)
                        .corruptionLevel(0.2) // Set level of corruption
                        .lossFunction(LossFunctions.LossFunction.XENT)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(64).nOut(200)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.LEAKYRELU)
                        //.corruptionLevel(0.2) // Set level of corruption
                        .build())
                .layer(2, new AutoEncoder.Builder().nOut(24).nIn(200)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.RELU)
                        //.corruptionLevel(0.1) // Set level of corruption
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nOut(outputNum).nIn(24).build())
                .pretrain(false) // Do pre training
                .backprop(true)
                .build(); // Build on set configuration
        return conf;
    }

    // method for testing the classifier.
    @Override
    public ClassificationStatistics test(List<double[][]> epochs, List<Double> targets) {
        ClassificationStatistics resultsStats = new ClassificationStatistics(); // initialization of classifier statistics
        for (int i = 0; i < epochs.size(); i++) {   //iterating epochs
            double output = this.classify(epochs.get(i));   //   output means score of a classifier from method classify
            resultsStats.add(output, targets.get(i));   // calculating statistics
        }
        return resultsStats;    //  returns classifier statistics
    }

    // method not implemented. For loading use load(String file)
    @Override
    public void load(InputStream is) {

    }

    // method not implemented. For saving use method save(String file)
    @Override
    public void save(OutputStream dest) {

    }

    /**
     * save model in file
     * @param file path + filename without extension
     */
    @Override
    public void save(String file) {
        File locationToSave = new File(file + ".zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;   //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        try {
            ModelSerializer.writeModel(result.getBestModel(), locationToSave, saveUpdater);
            System.out.println("Saved network params " + result.getBestModel().params());
            System.out.println("Saved");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * load model in file
     * @param file path + filename without extension
     */
    @Override
    public void load(String file) {
        File locationToLoad = new File(file +".zip");
        try {
            result.setBestModel(ModelSerializer.restoreMultiLayerNetwork(locationToLoad));
            System.out.println("Loaded");
            System.out.println("Loaded network params " + result.getBestModel().params());
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Original network params " + result.getBestModel().params());
        System.out.println("Loaded");
    }

    @Override
    public IFeatureExtraction getFeatureExtraction() {
        return fe;
    }

    @Override
    public void setFeatureExtraction(IFeatureExtraction fe) {
        this.fe = fe;
    }
}
