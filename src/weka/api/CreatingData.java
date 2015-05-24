package weka.api;

import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;

public class CreatingData {
	public static void main(String args[]) throws Exception{
		
		DataSource source = new DataSource("pacman_weka_v2.arff");
		Instances trainDataset = source.getDataSet();
		
		Instances testing;
		double[] vals;
		
//		for (int i = 0, atts_length = trainDataset.numAttributes(); i < atts_length; i++) {
//			atts.addElement(trainDataset.attribute(i));
//		}
//		
//		testing = new Instances("pacman", atts, 0);
		
		trainDataset.setClassIndex(0);
		testing = new Instances(trainDataset);
		
		vals = new double[testing.numAttributes()];
		vals[0] = testing.attribute("directionChosen").indexOfValue("NEUTRAL");
		vals[3] = 300;
		testing.add(new Instance(1.0, vals));
		
		
		
		//////////////////////////////////////////////////////////////////
		//DISCRETIZE
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
		discretize.setInputFormat(trainDataset);
		discretize.setInputFormat(testing);
		trainDataset = Filter.useFilter(trainDataset, discretize);
		testing = Filter.useFilter(testing, discretize);
		///////////////////////////////////////////////////////////////////
		
////////////////////////////////////////////////////////////////////
////FILTER
////use a simple filter to remove a certain attribute	
////set up options to remove 1st attribute	
//String[] opts = new String[]{ "-R", "1"};
////create a Remove object (this is the filter class)
//Remove remove = new Remove();
////set the filter options
//remove.setOptions(opts);
////pass the dataset to the filter
//remove.setInputFormat(trainDataset);
//remove.setInputFormat(testing);
////apply the filter
//trainDataset = Filter.useFilter(trainDataset, remove);
//testing = Filter.useFilter(testing, remove);
//////////////////////////////////////////////////////////////////////
		
		while(testing.firstInstance() != testing.lastInstance())
		{
			testing.delete(0);
		}
		
		
		
		//System.out.println(trainDataset);
		//System.out.println(testing);
		
		int numClasses = trainDataset.numClasses();
		//print out class values in the training dataset
		for(int i = 0; i < numClasses; i++){
			//get class string value using the class index
			String classValue = trainDataset.classAttribute().value(i);
			System.out.println("Class Value "+i+" is " + classValue);
		}
		//create and build the classifier
		J48 nb = new J48();
		//NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(trainDataset);
		//load new dataset
		Instances testDataset = testing;	
		//set class index to the last attribute
		testDataset.setClassIndex(0);
		//loop through the new dataset and make predictions
		System.out.println("===================");
		System.out.println("Actual Class, NB Predicted");
		for (int i = 0; i < testDataset.numInstances(); i++) {
			//get class double value for current instance
			double actualClass = testDataset.instance(i).classValue();
			//get class string value using the class index using the class's int value
			String actual = testDataset.classAttribute().value((int) actualClass);
			//get Instance object of current instance
			Instance newInst = testDataset.instance(i);
			//call classifyInstance, which returns a double value for the class
			double predNB = nb.classifyInstance(newInst);
			//use this value to get string value of the predicted class
			String predString = testDataset.classAttribute().value((int) predNB);
			System.out.println(actual+", "+predString);
		}
	}
}
