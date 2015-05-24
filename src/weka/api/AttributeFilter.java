package weka.api;

//import required classes
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import java.io.File;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class AttributeFilter{
	public static void main(String args[]) throws Exception{
		//load dataset
		DataSource source = new DataSource("pacman_weka_v2_3.arff");
		Instances dataset = source.getDataSet();
		
		//use a simple filter to remove a certain attribute	
		//set up options to remove 1st attribute	
		String[] opts = new String[]{ "-R", "1"};
		//create a Remove object (this is the filter class)
		Remove remove = new Remove();
		//set the filter options
		remove.setOptions(opts);
		//pass the dataset to the filter
		remove.setInputFormat(dataset);
		//apply the filter
		Instances newData = Filter.useFilter(dataset, remove);

		//now save the dataset to a new file as we learned before
		ArffSaver saver = new ArffSaver();
		saver.setInstances(newData);
		saver.setFile(new File("x.arff"));
		saver.writeBatch();
	}
}