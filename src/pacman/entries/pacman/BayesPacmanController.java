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
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.Filter;

public class BayesPacmanController extends Controller<MOVE>
{
	private Instance current_instance;
	private Attribute[] attributes;
	private BayesNet bayesnet = null;
	private Instances dataset;
	private MOVE myMove=MOVE.NEUTRAL;
	
	private Instances dataset1;	
	
	public MOVE getMove(Game game, long timeDue) 
	{
		if(bayesnet == null)
		{
			try {
				BayesNetGenerator gen = new BayesNetGenerator();
				bayesnet = gen.getBayesNet();
				dataset1 = gen.getDataset();
				DataSource source = new DataSource("arff/pacman_attr_selected.arff");
				dataset = source.getDataSet();
				dataset.setClassIndex(dataset.numAttributes()-1);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			CreateInstance();
		}
		
		PopulateInstance(game, timeDue);
		
		double predNB = 4;
		dataset.setClassIndex(dataset.numAttributes()-1);
		int num = dataset.numInstances()-1;
		try {
			predNB = bayesnet.classifyInstance(dataset.instance(dataset.numInstances()-1));
		} catch (Exception e) {
			System.out.println("EVALUATE EXCEPTION: " + e.getMessage());
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		myMove = MOVE.valueOf(dataset.classAttribute().value((int) predNB));
		System.out.println(myMove.toString());
		return myMove;
	}

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
		
		//dataset.add(current_instance);
//		current_instance.dataset().add(current_instance);
		
		Discretize();
	}
	
	private void Discretize()
	{
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

	private void CreateInstance()
	{
		current_instance = new Instance(18); 
		attributes = new Attribute[18];
		
		FastVector direction_nominals = new FastVector(5); 
		direction_nominals.addElement("UP");
		direction_nominals.addElement("RIGHT");
		direction_nominals.addElement("DOWN");
		direction_nominals.addElement("LEFT");
		direction_nominals.addElement("NEUTRAL");
		
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
		
		//current_instance.setDataset(dataset);
	}
}
