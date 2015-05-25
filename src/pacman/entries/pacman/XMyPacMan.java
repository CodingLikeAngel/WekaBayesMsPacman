package pacman.entries.pacman;

import dataRecording.DataTuple;
import pacman.controllers.Controller;
import pacman.game.Constants.DM;
import pacman.game.Constants.GHOST;
import pacman.game.Constants.MOVE;
import pacman.game.Game;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

/*
 * This is the class you need to modify for your entry. In particular, you need to
 * fill in the getAction() method. Any additional classes you write should either
 * be placed in this package or sub-packages (e.g., game.entries.pacman.mypackage).
 */
public class XMyPacMan extends Controller<MOVE>
{
	private XBayesController my_bayes_controller = null;
	
	////////////////////////////////////////////////////////////////
	private MOVE DirectionChosen;

	// General game state this - not normalized!
	private int mazeIndex;
	private int currentLevel;
	private int pacmanPosition;
	private int pacmanLivesLeft;
	private int currentScore;
	private int totalGameTime;
	private int currentLevelTime;
	private int numOfPillsLeft;
	private int numOfPowerPillsLeft;

	// Ghost this, dir, dist, edible - BLINKY, INKY, PINKY, SUE
	private boolean isBlinkyEdible = false;
	private boolean isInkyEdible = false;
	private boolean isPinkyEdible = false;
	private boolean isSueEdible = false;

	private int blinkyDist = -1;
	private int inkyDist = -1;
	private int pinkyDist = -1;
	private int sueDist = -1;

	private MOVE blinkyDir;
	private MOVE inkyDir;
	private MOVE pinkyDir;
	private MOVE sueDir;

	// Util data - useful for normalization
	private int numberOfNodesInLevel;
	private int numberOfTotalPillsInLevel;
	private int numberOfTotalPowerPillsInLevel;
	private int maximumDistance = 150;
	
	
	//extra info added by dani and carlos
	private int blinkyPosition;
	private int inkyPosition;
	private int pinkyPosition;
	private int suePosition;
	/////////////////////////////////////////////////////////////////
	
	
	private MOVE myMove=MOVE.NEUTRAL;
	
	public MOVE getMove(Game game, long timeDue) 
	{
		//Place your game logic here to play the game as Ms Pac-Man
		
		try {
			myMove = getMoveBayes(game, timeDue);
		} catch (Exception e) {
			// TODO: handle exception
			System.out.println("ERROR: There was an exception: ->" + e.getMessage());
			e.printStackTrace();
		}
		
		return myMove;
	}
	
	private MOVE getMoveBayes(Game game, long timeDue) throws Exception
	{
		if(my_bayes_controller == null)
			my_bayes_controller = new XBayesController();

		double[] vals;

		Instances testingDataset = new Instances(my_bayes_controller.getOriginalDataset());
		
		vals = new double[testingDataset.numAttributes()];

		//////////////////////////////////////////////////////////////////////////////////////
		this.mazeIndex = game.getMazeIndex();
		this.currentLevel = game.getCurrentLevel();
		this.pacmanPosition = game.getPacmanCurrentNodeIndex();
		this.pacmanLivesLeft = game.getPacmanNumberOfLivesRemaining();
		this.currentScore = game.getScore();
		this.totalGameTime = game.getTotalTime();
		this.currentLevelTime = game.getCurrentLevelTime();
		this.numOfPillsLeft = game.getNumberOfActivePills();
		this.numOfPowerPillsLeft = game.getNumberOfActivePowerPills();

		if (game.getGhostLairTime(GHOST.BLINKY) == 0) {
			this.isBlinkyEdible = game.isGhostEdible(GHOST.BLINKY);
			this.blinkyDist = game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.BLINKY));
		}

		if (game.getGhostLairTime(GHOST.INKY) == 0) {
			this.isInkyEdible = game.isGhostEdible(GHOST.INKY);
			this.inkyDist = game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.INKY));
		}

		if (game.getGhostLairTime(GHOST.PINKY) == 0) {
			this.isPinkyEdible = game.isGhostEdible(GHOST.PINKY);
			this.pinkyDist = game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.PINKY));
		}

		if (game.getGhostLairTime(GHOST.SUE) == 0) {
			this.isSueEdible = game.isGhostEdible(GHOST.SUE);
			this.sueDist = game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.SUE));
		}

		this.blinkyDir = game.getNextMoveTowardsTarget(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.BLINKY), DM.PATH);
		this.inkyDir = game.getNextMoveTowardsTarget(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.INKY), DM.PATH);
		this.pinkyDir = game.getNextMoveTowardsTarget(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.PINKY), DM.PATH);
		this.sueDir = game.getNextMoveTowardsTarget(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.SUE), DM.PATH);

		this.numberOfNodesInLevel = game.getNumberOfNodes();
		this.numberOfTotalPillsInLevel = game.getNumberOfPills();
		this.numberOfTotalPowerPillsInLevel = game.getNumberOfPowerPills();
		
		//extra info added by dani and carlos
		this.blinkyPosition = game.getGhostCurrentNodeIndex(GHOST.BLINKY);
		this.inkyPosition = game.getGhostCurrentNodeIndex(GHOST.INKY);
		this.pinkyPosition = game.getGhostCurrentNodeIndex(GHOST.PINKY);
		this.suePosition = game.getGhostCurrentNodeIndex(GHOST.SUE);
		//////////////////////////////////////////////////////////////////////////////////////
		//DataTuple data = new DataTuple(game, MOVE.NEUTRAL);
		vals[0] = testingDataset.attribute("directionChosen").indexOfValue("NEUTRAL");
		vals[1] = this.mazeIndex;
		vals[2] = this.currentLevel;
		vals[3] = this.pacmanPosition;
		vals[4] = this.pacmanLivesLeft;
		vals[5] = this.currentScore;
		vals[6] = this.totalGameTime;
		vals[7] = this.currentLevelTime;
		vals[8] = this.numOfPillsLeft;
		vals[9] = this.numOfPowerPillsLeft;
		vals[10] = testingDataset.attribute("isBlinkyEdible").indexOfValue(this.isBlinkyEdible ? "TRUE" : "FALSE");
		vals[11] = testingDataset.attribute("isInkyEdible").indexOfValue(this.isInkyEdible ? "TRUE" : "FALSE");
		vals[12] = testingDataset.attribute("isPinkyEdible").indexOfValue(this.isPinkyEdible ? "TRUE" : "FALSE");
		vals[13] = testingDataset.attribute("isSueEdible").indexOfValue(this.isSueEdible ? "TRUE" : "FALSE");
		vals[14] = this.blinkyDist;
		vals[15] = this.inkyDist;
		vals[16] = this.pinkyDist;
		vals[17] = this.sueDist;
		vals[18] = testingDataset.attribute("blinkyDir").indexOfValue(this.blinkyDir.toString());
		vals[19] = testingDataset.attribute("inkyDir").indexOfValue(this.inkyDir.toString());
		vals[20] = testingDataset.attribute("pinkyDir").indexOfValue(this.pinkyDir.toString());
		vals[21] = testingDataset.attribute("sueDir").indexOfValue(this.sueDir.toString());
		vals[22] = this.numberOfNodesInLevel;
		vals[23] = this.numberOfTotalPillsInLevel;
		vals[24] = this.numberOfTotalPowerPillsInLevel;
		vals[25] = this.blinkyPosition;
		vals[26] = this.inkyPosition;
		vals[27] = this.pinkyPosition;
		vals[28] = this.suePosition;
		testingDataset.add(new Instance(1.0, vals));
		
		
		
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
		discretize.setInputFormat(testingDataset);
		testingDataset = Filter.useFilter(testingDataset, discretize);
		
		while(testingDataset.firstInstance() != testingDataset.lastInstance())
		{
			testingDataset.delete(0);
		}
		
		
//		Instances trainingDataset = new Instances(my_bayes_controller.getTrainingDataset());
//		
//		int numClasses = trainingDataset.numClasses();
//		//print out class values in the training dataset
//		for(int i = 0; i < numClasses; i++){
//			//get class string value using the class index
//			String classValue = trainingDataset.classAttribute().value(i);
//			System.out.println("Class Value "+i+" is " + classValue);
//		}
		
//		//create and build the classifier
//		J48 nb = new J48();
//		//NaiveBayes nb = new NaiveBayes();
//		nb.buildClassifier(my_bayes_controller.getTrainingDataset());

		
		//load new dataset
//		Instances testDataset = testingDataset;	
//		//set class index to the last attribute
//		testDataset.setClassIndex(0);
		
		//loop through the new dataset and make predictions
		System.out.println("===================");
		System.out.println("Actual Class, NB Predicted");
		String predString = "NEUTRAL";
		BayesNet bn = my_bayes_controller.getBayesNet();
		testingDataset.setClassIndex(0);
//		System.out.println(testingDataset.numAttributes());
//		System.out.println(testingDataset.numAttributes());
		//for (int i = 0; i < testingDataset.numInstances(); i++) {
			//get class double value for current instance
			double actualClass = testingDataset.instance(0).classValue();
			//get class string value using the class index using the class's int value
			String actual = testingDataset.classAttribute().value((int) actualClass);
			//get Instance object of current instance
			Instance newInst = testingDataset.instance(0);
			newInst.setDataset(my_bayes_controller.getTrainingDataset());
//			for (int j = 0; j < newInst.numAttributes(); j++) {
//				System.out.println(newInst.attribute(j));
//			}
//			for (int j = 0; j < newInst.numAttributes(); j++) {
//				System.out.println(newInst.value(j));
//			}
			
			
		
			
			//call classifyInstance, which returns a double value for the class
//			System.out.println(bn.classifyInstance(newInst));
			double predNB = bn.classifyInstance(newInst);
			//use this value to get string value of the predicted class
			predString = testingDataset.classAttribute().value((int) predNB);
			System.out.println(actual+", "+predString);
		//}
		
		myMove = MOVE.valueOf(predString);
		return myMove;
	}
}