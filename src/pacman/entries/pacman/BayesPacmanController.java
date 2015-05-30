package pacman.entries.pacman;

import pacman.controllers.Controller;
import pacman.game.Constants.DM;
import pacman.game.Constants.GHOST;
import pacman.game.Game;
import pacman.game.Constants.MOVE;
import weka.classifiers.bayes.BayesNet;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.Filter;

/**
 * Implementation of the controller.
 * BayesPacmanController creates a bayes network using BayesNetGenerator class and uses it to get every move of the Pacman.
 * 
 * @author Carlos Bailón Pérez and Daniel Castaño Estrella
 * @version 1.0
 */
public class BayesPacmanController extends Controller<MOVE>
{
	private Instance current_instance;		//data set with data of current game time
	private Attribute[] attributes;			//array of attributes used in the Instance
	private BayesNet bayesnet = null;		//Bayes network
	private Instances dataset;				//dataset to use every time to discretize
	private Instances dataset_base;			//base dataset without being discretized
	private MOVE myMove=MOVE.NEUTRAL;		//next move
	
	/**
	 * Method that returnsthe next move
	 */
	public MOVE getMove(Game game, long timeDue) 
	{
		//first time, we create the bayes network
		if(bayesnet == null)
		{
			try {
				BayesNetGenerator gen = new BayesNetGenerator();
				bayesnet = gen.getBayesNet();
				dataset_base = gen.getDatasetOrig();
			} catch (Exception e) {
				e.printStackTrace();
			}
			//call the method that creates current instance and set the attributes
			CreateInstance();
		}
		//make copy ofthe original dataset of the bayes network
		dataset = new Instances(dataset_base);
		//call the method that populates the current instance with game data
		PopulateInstance(game, timeDue);
		
		//predNB is the next move nominal position (UP;RIGHT;DOWN;LEFT;NEUTRAL) we set it to NEUTRAL to makethatmove in case classifyInstance fails
		double predNB = 4;
		//setting of the class attribute (chosenDirection)
		dataset.setClassIndex(dataset.numAttributes()-1);
		try {
			//make the prediction
			predNB = bayesnet.classifyInstance(dataset.instance(dataset.numInstances()-1));
		} catch (Exception e) {
			System.out.println("EVALUATE EXCEPTION: " + e.getMessage());
			e.printStackTrace();
		}
		//set myMove and return it
		myMove = MOVE.valueOf(dataset.classAttribute().value((int) predNB));
		System.out.println(myMove.toString());
		return myMove;
	}

	/**
	 * Method that populates the array attributes with current game data and set it to the current instance
	 * @param game game data
	 * @param timeDue time data
	 */
	private void PopulateInstance(Game game, long timeDue)
	{
		for (int i = 0; i < attributes.length; i++)
		{
			switch (i)
			{
				case 0:
					//int time = game.getTotalTime();
					current_instance.setValue(i, game.getTotalTime()); 
					break;
				case 1:
					current_instance.setValue(i, game.getScore());
					break;
				case 2:
					int a = game.getNumberOfActivePills();
					current_instance.setValue(i, a);
					break;
				case 3:
					current_instance.setValue(i, game.getNextMoveTowardsTarget(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.BLINKY), DM.PATH).ordinal());
					break;
				case 4:
					current_instance.setValue(i, game.getNextMoveTowardsTarget(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.SUE), DM.PATH).ordinal());
					break;
				case 5:
					current_instance.setValue(i, game.getNextMoveTowardsTarget(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.PINKY), DM.PATH).ordinal());
					break;
				case 6:
					current_instance.setValue(i, game.getNextMoveTowardsTarget(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.INKY), DM.PATH).ordinal());
					break;
				case 7:
					current_instance.setValue(i, game.getCurrentLevel());
					break;
				case 8:
					current_instance.setValue(i, game.getPacmanCurrentNodeIndex());
					break;
				case 9:
					current_instance.setValue(i, game.getGhostCurrentNodeIndex(GHOST.INKY));
					break;
				case 10:
					current_instance.setValue(i, game.getGhostCurrentNodeIndex(GHOST.BLINKY));
					break;
				case 11:
					current_instance.setValue(i, game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.INKY)));
					break;
				case 12:
					current_instance.setValue(i, game.getGhostCurrentNodeIndex(GHOST.PINKY));
					break;
				case 13:
					current_instance.setValue(i, game.getGhostCurrentNodeIndex(GHOST.SUE));
					break;
				case 14:
					current_instance.setValue(i, game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.SUE)));
					break;
				case 15:
					current_instance.setValue(i, game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.PINKY)));
					break;
				case 16:
					current_instance.setValue(i, game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.BLINKY)));
					break;
				case 17:
					current_instance.setValue(i,4);
					break;
			}
		}
		
		//add current instance to dataset
		dataset.add(current_instance);
		
		//call discretization method
		Discretize();
	}
	
	/**
	 * Method that applies a discretization filterto the dataset
	 */
	private void Discretize()
	{
		//set options -> discretize all numerics
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
	}

	/**
	 * Method that creates the base of current instances. That means setting the attributes
	 */
	private void CreateInstance()
	{
		//construct an instance with 18 attributes
		current_instance = new Instance(18); 
		//construct array of attributes
		attributes = new Attribute[18];
		
		//construct and set direction nominal values
		FastVector direction_nominals = new FastVector(5); 
		direction_nominals.addElement("UP");
		direction_nominals.addElement("RIGHT");
		direction_nominals.addElement("DOWN");
		direction_nominals.addElement("LEFT");
		direction_nominals.addElement("NEUTRAL");
		
		//set the attributes
		attributes[0] = new Attribute("totalGameTime");
		attributes[1] = new Attribute("currentScore");
		attributes[2] = new Attribute("numOfPillsLeft");
		attributes[3] = new Attribute("blinkyDir", direction_nominals);
		attributes[4] = new Attribute("sueDir", direction_nominals);
		attributes[5] = new Attribute("pinkyDir", direction_nominals);
		attributes[6] = new Attribute("inkyDir", direction_nominals);
		attributes[7] = new Attribute("currentLevelTime");
		attributes[8] = new Attribute("pacmanPosition");
		attributes[9] = new Attribute("inkyPosition");
		attributes[10] = new Attribute("blinkyPosition");
		attributes[11] = new Attribute("inkyDist");
		attributes[12] = new Attribute("pinkyPosition");
		attributes[13] = new Attribute("suePosition");
		attributes[14] = new Attribute("sueDist");
		attributes[15] = new Attribute("pinkyDist");
		attributes[16] = new Attribute("blinkyDist");
		attributes[17] = new Attribute("directionChosen", direction_nominals);
	}
}
