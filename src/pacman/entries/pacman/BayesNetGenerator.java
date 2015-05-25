package pacman.entries.pacman;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.global.K2;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class BayesNetGenerator
{
	private Instances dataset;
	private BayesNet my_bayes_net;

	public BayesNetGenerator() throws Exception
	{
		//GET DATASET
		DataSource source = new DataSource("arff/pacman_attr_selected.arff");
		dataset = source.getDataSet();
		
		//SET CLASS
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		//BUILD BAYESNET
		my_bayes_net = new BayesNet();
		my_bayes_net.buildClassifier(dataset);
		
		//SEARCH ALGORITHM
		K2 search_alg = new K2();
		String[] options_k2 = new String[4];
		options_k2[0] = "-P"; options_k2[1] = "2";
		options_k2[2] = "-S"; options_k2[3] = "LOO-CV";
		search_alg.setOptions(options_k2);
		
		//ESTIMATOR
		SimpleEstimator estimator = new SimpleEstimator();
		String[] options_estimator = new String[2];
		options_estimator[0] = "-A"; options_estimator[1] = "0.5";
		estimator.setOptions(options_estimator);
		
		//SETTING SEARCH ALG AND ESTIMATOR
		my_bayes_net.setSearchAlgorithm(search_alg);
		my_bayes_net.setEstimator(estimator);
		
		//EVALUATION
		Evaluation eval = new Evaluation(dataset);
		eval.crossValidateModel(my_bayes_net,dataset,10,new Random(1));
		
		System.out.println(eval.toSummaryString("Evaluation results:\n", false));
		
		System.out.println("BAYES NET FINISHED!");
	}
	
	public BayesNet getBayesNet() {
		return my_bayes_net;
	}
	
	public Instances getDataset() {
		return dataset;
	}
}
