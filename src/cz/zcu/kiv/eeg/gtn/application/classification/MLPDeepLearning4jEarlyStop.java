package cz.zcu.kiv.eeg.gtn.application.classification;

import cz.zcu.kiv.eeg.gtn.application.featureextraction.IFeatureExtraction;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.termination.EpochTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Created by lukasvareka on 4. 7. 2016.
 */
public class MLPDeepLearning4jEarlyStop implements IERPClassifier {
    private final int NEURON_COUNT_DEFAULT = 30;    //default number of neurons
    private IFeatureExtraction fe;                //type of feature extraction (MatchingPursuit, FilterAndSubampling or WaveletTransform)
    private MultiLayerNetwork model;            //multi layer neural network with a logistic output layer and multiple hidden neuralNets
    private int neuronCount;                    // Number of neurons
    private int iterations;                    //Iterations used to classify
    private Model model1;                       //model from new lbraries
    private int maxTime =5; //max time in minutes
    private int maxEpochs = 3000;
    private EarlyStoppingResult result;
    private int noImprovementEpochs = 10;
    private EarlyStoppingConfiguration esConf;
    private String pathname = "C:\\Temp\\MLPEStop"; //pathname+file name for saving model
    private String directory = "C:\\Temp\\";



    /*Default constructor*/
    public MLPDeepLearning4jEarlyStop() {
        this.neuronCount = NEURON_COUNT_DEFAULT; // sets count of neurons in layer(0) to default number
    }

    /*Parametric constructor */
    public MLPDeepLearning4jEarlyStop(int neuronCount) {
        this.neuronCount = neuronCount; // sets count of neurons in layer(0) to param
    }

    /*Classifying features*/
    @Override
    public double classify(double[][] epoch) {
        double[] featureVector = this.fe.extractFeatures(epoch); // Extracting features to vector
        INDArray features = Nd4j.create(featureVector); // Creating INDArray with extracted features
        return model.output(features, Layer.TrainingMode.TEST).getDouble(0); // Result of classifying

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
        //SplitTestAndTrain tat = dataSet.splitTestAndTrain(80);

        /*

        NOT WORKING WITH NEW LIBRARY

        DataSetIterator dataSetTrainIterator = new ListDataSetIterator(tat.getTrain().batchBy(8));
        DataSetIterator dataSetTestIterator = new ListDataSetIterator(tat.getTest().batchBy(8));

        */
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true; // Setting to enforce numerical stability

        // Building a neural net
        MultiLayerConfiguration conf = build(numRows, numColumns, seed, listenerFreq);
        //SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(80);

        //EarlyStoppingModelSaver saver = new LocalFileModelSaver(directory);
        EarlyStoppingModelSaver saver = new InMemoryModelSaver();

        List<EpochTerminationCondition> list = new ArrayList<EpochTerminationCondition>(2);
        list.add(new MaxEpochsTerminationCondition(maxEpochs));
        list.add(new ScoreImprovementEpochTerminationCondition(noImprovementEpochs, 0.00001));

        esConf = new EarlyStoppingConfiguration.Builder()
                //.epochTerminationConditions(new MaxEpochsTerminationCondition(maxEpochs))
                //.epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(noImprovementEpochs))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(maxTime, TimeUnit.MINUTES))
                //.scoreCalculator(new DataSetLossCalculator(new ListDataSetIterator(testAndTrain.getTest().asList(), 100), true))
                .scoreCalculator(new DataSetLossCalculator(new ListDataSetIterator(dataSet.asList(), 200), true))
                .evaluateEveryNEpochs(2)
                .modelSaver(saver)
                .epochTerminationConditions(list)
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,conf,new ListDataSetIterator(dataSet.asList(), 200));

//Conduct early stopping training:
        this.result = trainer.fit();

//Print out the results:
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

//Get the best model
        this.model = (MultiLayerNetwork) result.getBestModel();

        Evaluation eval = new Evaluation(numColumns);
        eval.eval(dataSet.getLabels(), model.output(dataSet.getFeatureMatrix(), Layer.TrainingMode.TEST));
        System.out.println(eval.stats());
    }

    //  initialization of neural net with params. For more info check http://deeplearning4j.org/iris-flower-dataset-tutorial where is more about params
    private MultiLayerConfiguration build(int numRows, int outputNum, int seed, int listenerFreq) {
        System.out.print("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                //.iterations(5)
                //.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.05)
               .updater(Updater.NESTEROVS).momentum(0.9)
                .weightInit(WeightInit.RELU)
                .activation(Activation.LEAKYRELU)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numRows).nOut(200)
                        //.corruptionLevel(0.2) // Set level of corruption
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(200).nOut(100)
                        //.corruptionLevel(0.2) // Set level of corruption
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(100).nOut(outputNum).build())
                .pretrain(false).backprop(true).build();


        //model.setListeners(new ScoreIterationListener(10));// Setting listeners
        //model.setListeners(new HistogramIterationListener(10));
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
        throw new NotImplementedException();
    }

    // method not implemented. For saving use method save(String file)
    @Override
    public void save(OutputStream dest) {
        throw new NotImplementedException();
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
        File locationToLoad = new File(file + ".zip");
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
