package weka.api;
//import required classes
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import java.io.File;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class DiscretizeAttribute{
	public static void main(String args[]) throws Exception{
		//load dataset
		DataSource source = new DataSource("pacman_weka_v2_3.arff");
		Instances dataset = source.getDataSet();		
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
		discretize.setInputFormat(dataset);
		Instances newData = Filter.useFilter(dataset, discretize);

		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(newData);
		saver.setFile(new File("discretized_filtered.arff"));
		saver.writeBatch();
	}
}