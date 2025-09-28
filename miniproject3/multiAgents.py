# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        if len(foodList) > 0:
            minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])
        else:
            minFoodDistance = 0  # No food left
        foodScore = 1.0 / (minFoodDistance + 1)

        # Calculate a ghost score based on distance and scared time
        ghost_distances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        ghost_score = 0
        for i, distance in enumerate(ghost_distances):
            if newScaredTimes[i] > 0:
                # If the ghost is scared, chase it
                ghost_score += 1.0 / (distance + 1)
            else:
                # If the ghost is not scared, avoid it
                ghost_score -= 1.0 / (distance + 1)

        score = successorGameState.getScore() + foodScore + ghost_score
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def minimax(gameState, agentIndex, depth):
            # Check state
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            # Max, Pacman turn
            if agentIndex == 0:
                legalActions = gameState.getLegalActions(agentIndex)
                scores = [minimax(gameState.generateSuccessor(agentIndex, action), 1, depth) for action in legalActions]
                return max(scores)
            else: # Min, Ghost turn
                legalActions = gameState.getLegalActions(agentIndex)
                nextAgent = agentIndex + 1
                if nextAgent >= gameState.getNumAgents():
                    nextAgent = 0
                    depth -= 1
                scores = [minimax(gameState.generateSuccessor(agentIndex, action), nextAgent, depth) for action in
                          legalActions]
                return min(scores)

        # Get actions for Pacman
        legalActions = gameState.getLegalActions(0)
        # Scores for action
        scores = [minimax(gameState.generateSuccessor(0, action), 1, self.depth) for action in legalActions]
        #High score
        bestScore = max(scores)
        # Indices of Highscores
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalActions[chosenIndex]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBeta(gameState, agentIndex, depth, alpha, beta):
            # Checking state
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            # Max, Pacman turn
            if agentIndex == 0:
                value = float('-inf')
                legalActions = gameState.getLegalActions(agentIndex)
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = max(value, alphaBeta(successor, 1, depth, alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else: # Min, Ghost turn
                value = float('inf')
                legalActions = gameState.getLegalActions(agentIndex)
                nextAgent = agentIndex + 1
                if nextAgent >= gameState.getNumAgents():
                    nextAgent = 0
                    depth -= 1
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = min(value, alphaBeta(successor, nextAgent, depth, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        legalActions = gameState.getLegalActions(0)

        alpha = float('-inf')
        beta = float('inf')
        # Scores for actions
        bestAction = None
        value = float('-inf')
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = alphaBeta(successor, 1, self.depth, alpha, beta)
            if score > value:
                value = score
                bestAction = action
            if value > beta:
                return action
            alpha = max(alpha, value)

        return bestAction

        util.raiseNotDefined()



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            #Max Pacman
            if agentIndex == 0:
                legalActions = gameState.getLegalActions(agentIndex)
                scores = [expectimax(gameState.generateSuccessor(agentIndex, action), 1, depth) for action in
                          legalActions]
                return max(scores)
            else: #Ghost Turn
                legalActions = gameState.getLegalActions(agentIndex)
                nextAgent = agentIndex + 1
                if nextAgent >= gameState.getNumAgents():
                    nextAgent = 0
                    depth -= 1
                scores = [expectimax(gameState.generateSuccessor(agentIndex, action), nextAgent, depth) for action in
                          legalActions]
                return float(sum(scores)) / len(scores)

        legalActions = gameState.getLegalActions(0)
        # Scores for action
        scores = [expectimax(gameState.generateSuccessor(0, action), 1, self.depth) for action in legalActions]
        # highscore
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalActions[chosenIndex]

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pacmanPosition = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    score = currentGameState.getScore()
    # Distance to the closest point
    foodList = foodGrid.asList()
    if len(foodList) > 0:
        minFoodDistance = min([manhattanDistance(pacmanPosition, food) for food in foodList])
    else:
        minFoodDistance = 0

    numFood = len(foodList)

    # Distance to ghost and timer
    ghostDistances = []
    for i, ghostState in enumerate(ghostStates):
        ghostPos = ghostState.getPosition()
        ghostDist = manhattanDistance(pacmanPosition, ghostPos)
        if scaredTimes[i] > 0:
            # Ghost scared, distance doesnt matter
            ghostDistances.append(-ghostDist)
        else:
            # Ghost not scared, distance does matter
            ghostDistances.append(ghostDist)

    minGhostDistance = min(ghostDistances) if ghostDistances else 0
    evalScore = score + (1.0 / (minFoodDistance + 1)) - (2 * numFood) + minGhostDistance

    return evalScore

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

