package pacman.entries.pacman;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.BayesNet;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.attribute.AttributeSelection;


public class BayesController {
	private Instances originalDataset;
	private Instances trainingDataset;
	private BayesNet my_bayes_net;
	
	public BayesController() throws Exception
	{
		DataSource source = new DataSource("pacman_weka_v2.arff");
		originalDataset = source.getDataSet();
		originalDataset.setClassIndex(0);
		
		trainingDataset = new Instances(originalDataset);
		
		try {
			DiscretizeDataset();
		} catch (Exception e) {
			System.out.println("DISCRETIZE EXCEPTION: " + e);
		}
		
		try {
			BuildClassifier();
		} catch (Exception e) {
			System.out.println("BUILD CLASSIFIER EXCEPTION: " + e);
		}
		
	}
	
	private void BuildClassifier() throws Exception
	{
		my_bayes_net = new BayesNet();
		my_bayes_net.buildClassifier(trainingDataset);
	}

	private void DiscretizeDataset() throws Exception
	{
		//set options
		String[] options = new String[5];
		//choose the number of intervals, e.g. 2 :
		options[0] = "-B"; options[1] = "10";
		//choose the range of attributes on which to apply the filter:
		options[2] = "-R";
		options[3] = "1-1";
		options[4] = "-V";
		//Apply discretization:
		Discretize discretize = new Discretize();
		discretize.setOptions(options);
		discretize.setInputFormat(trainingDataset);
		trainingDataset = Filter.useFilter(trainingDataset, discretize);
	}
	/*
	private void FilterDataset()
	{
		//use a simple filter to remove a certain attribute	
		//set up options to remove 1st attribute	
		String[] opts = new String[]{ "-R", "1"};
		//create a Remove object (this is the filter class)
		AttributeSelection att_selection = new AttributeSelection();
		//Remove remove = new Remove();
		//set the filter options
		att_selection.setOptions(opts);
		//pass the dataset to the filter
		remove.setInputFormat(trainDataset);
		remove.setInputFormat(testingDataset);
		//apply the filter
		trainDataset = Filter.useFilter(trainDataset, remove);
		testingDataset = Filter.useFilter(testingDataset, remove);
	}
	*/
	public BayesNet getBayesNet()
	{
		return my_bayes_net;
	}
	
	public Instances getOriginalDataset()
	{
		return originalDataset;
	}
	
	public Instances getTrainingDataset()
	{
		return trainingDataset;
	}
}
