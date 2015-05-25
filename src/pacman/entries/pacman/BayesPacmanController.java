package pacman.entries.pacman;

import pacman.controllers.Controller;
import pacman.game.Game;
import pacman.game.Constants.MOVE;

public class BayesPacmanController extends Controller<MOVE>
{
	private BayesNetGenerator bayesnet_generator = null;
	private MOVE myMove=MOVE.NEUTRAL;
	
	public MOVE getMove(Game game, long timeDue) 
	{
		if(bayesnet_generator == null)
		{
			try {
				bayesnet_generator = new BayesNetGenerator();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		//Place your game logic here to play the game as Ms Pac-Man
		
		return myMove;
	}
}
