package pacman.entries.pacman;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.global.K2;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/**
 * Class that creates the bayes network.
 * 
 * @author Carlos Bailón Pérez and Daniel Castaño Estrella
 * @version 1.0
 */
public class BayesNetGenerator
{
	private Instances dataset;			//dataset to use creating bayes network
	private BayesNet my_bayes_net;		//bayes network
	private Instances dataset_orig;		//variable to store dataset without being discretized

	/**
	 * Constructor of the class.
	 * Builds and evaluate the bayes network.
	 * @throws Exception
	 */
	public BayesNetGenerator() throws Exception
	{
		//GET DATA
		//get dataset previously filtered by attribute selection in weka application
		DataSource source = new DataSource("arff/pacman_attr_selected.arff");
		dataset_orig = source.getDataSet();
		
		//set the class index (chosenDirection)
		dataset_orig.setClassIndex(dataset_orig.numAttributes()-1);
		
		//make a copy to avoid discretizing dataset_orig
		dataset = new Instances(dataset_orig);
		
		//DISCRETIZATION
		//set discretizing options
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = "first-last";
		//Apply discretization:
		Discretize discretize = new Discretize();
		try {
			discretize.setOptions(options);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		try {
			discretize.setInputFormat(dataset);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		try {
			dataset = Filter.useFilter(dataset, discretize);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		//BUILD BAYESNET
		my_bayes_net = new BayesNet();
		my_bayes_net.buildClassifier(dataset);
		
		//CLASSIFIER ALGORITHM
		//set K2 classifier with max2 parents option
		K2 search_alg = new K2();
		String[] options_k2 = new String[4];
		options_k2[0] = "-P"; options_k2[1] = "2";
		options_k2[2] = "-S"; options_k2[3] = "Cumulative-CV";
		search_alg.setOptions(options_k2);
		
		//ESTIMATOR
		//Creates a simple estimator
		SimpleEstimator estimator = new SimpleEstimator();
		String[] options_estimator = new String[2];
		options_estimator[0] = "-A"; options_estimator[1] = "0.5";
		estimator.setOptions(options_estimator);
		
		//SETTING CLASSIFIER AND ESTIMATOR
		my_bayes_net.setSearchAlgorithm(search_alg);
		my_bayes_net.setEstimator(estimator);
		
		//EVALUATION
		//validates the model with cross validation using leave one out cross validation method
		Evaluation eval = new Evaluation(dataset);
		eval.crossValidateModel(my_bayes_net,dataset,10,new Random(1));
		
		System.out.println(eval.toSummaryString("Evaluation results:\n", false));
		
		System.out.println("BAYES NET FINISHED!");
	}
	
	/**
	 * Getter method of the bayes network
	 * @return bayes network
	 */
	public BayesNet getBayesNet() {
		return my_bayes_net;
	}
	
	/**
	 * Getter method of the dataset
	 * @return dataset
	 */
	public Instances getDataset() {
		return dataset;
	}
	
	/**
	 * Getter method of the original dataset (without being discretized)
	 * @return original dataset
	 */
	public Instances getDatasetOrig() {
		return dataset_orig;
	}
}
