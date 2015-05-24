package weka.api;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

//import java.io.BufferedReader;
import java.io.File;
//import java.io.FileReader;

import weka.core.converters.ConverterUtils.DataSource;
public class LoadSaveData{
	public static void main(String args[]) throws Exception{
		DataSource source = new DataSource("C:/Users/Dani/Documents/Projects/java_workspace/Ms_Pacman_weka_bayes/pacman_weka.arff");
		Instances dataset = source.getDataSet();
		//Instances dataset = new Instances(new BufferedReader(new FileReader("/home/likewise-open/ACADEMIC/csstnns/test/house.arff")));		
		
		System.out.println(dataset.toSummaryString());
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(dataset);
		saver.setFile(new File("C:/Users/Dani/Documents/Projects/java_workspace/Ms_Pacman_weka_bayes/pacman_weka2.arff"));
		saver.writeBatch();
	}
}