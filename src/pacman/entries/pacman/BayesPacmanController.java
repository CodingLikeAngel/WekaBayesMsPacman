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

public class BayesPacmanController extends Controller<MOVE>
{
	private Instance current_instance;
	private Attribute[] attributes;
	private BayesNet bayesnet = null;
	private Instances dataset;
	private MOVE myMove=MOVE.NEUTRAL;
	
	public MOVE getMove(Game game, long timeDue) 
	{
		if(bayesnet == null)
		{
			try {
				BayesNetGenerator gen = new BayesNetGenerator();
				bayesnet = gen.getBayesNet();
				dataset = gen.getDataset();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			CreateInstance();
		}
		
		PopulateInstance(game, timeDue);
		
		double predNB = 4;
		try {
			predNB = bayesnet.classifyInstance(current_instance);
		} catch (Exception e) {
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
					current_instance.setValue(attributes[i].NUMERIC, game.getTotalTime()); 
					break;
				case 1:
					current_instance.setValue(attributes[i].NUMERIC, game.getScore());
					break;
				case 2:
					current_instance.setValue(attributes[i].NUMERIC, game.getNumberOfActivePills());
					break;
				case 3:
					
					current_instance.setValue(attributes[i].NOMINAL, game.getNextMoveTowardsTarget(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.BLINKY), DM.PATH).ordinal());
					break;
				case 4:
					current_instance.setValue(attributes[i].NOMINAL, game.getNextMoveTowardsTarget(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.SUE), DM.PATH).ordinal());
					break;
				case 5:
					current_instance.setValue(attributes[i].NOMINAL, game.getNextMoveTowardsTarget(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.PINKY), DM.PATH).ordinal());
					break;
				case 6:
					current_instance.setValue(attributes[i].NOMINAL, game.getNextMoveTowardsTarget(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.INKY), DM.PATH).ordinal());
					break;
				case 7:
					current_instance.setValue(attributes[i].NUMERIC, game.getCurrentLevel());
					break;
				case 8:
					current_instance.setValue(attributes[i].NUMERIC, game.getPacmanCurrentNodeIndex());
					break;
				case 9:
					current_instance.setValue(attributes[i].NUMERIC, game.getGhostCurrentNodeIndex(GHOST.INKY));
					break;
				case 10:
					current_instance.setValue(attributes[i].NUMERIC, game.getGhostCurrentNodeIndex(GHOST.BLINKY));
					break;
				case 11:
					current_instance.setValue(attributes[i].NUMERIC, game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.INKY)));
					break;
				case 12:
					current_instance.setValue(attributes[i].NUMERIC, game.getGhostCurrentNodeIndex(GHOST.PINKY));
					break;
				case 13:
					current_instance.setValue(attributes[i].NUMERIC, game.getGhostCurrentNodeIndex(GHOST.SUE));
					break;
				case 14:
					current_instance.setValue(attributes[i].NUMERIC, game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.SUE)));
					break;
				case 15:
					current_instance.setValue(attributes[i].NUMERIC, game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.PINKY)));
					break;
				case 16:
					current_instance.setValue(attributes[i].NUMERIC, game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), game.getGhostCurrentNodeIndex(GHOST.BLINKY)));
					break;
				case 17:
					current_instance.setValue(attributes[i].NOMINAL, 4);
					break;
			}
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
		
		current_instance.setDataset(dataset);
	}
}
