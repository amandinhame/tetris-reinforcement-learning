import random
import math
import numpy as np
from copy import copy

class Learner(object):
    """ Learner agent. """

    def __init__(self, total_training_trials, total_testing_trials, success_lines):
        self.total_training_trials = total_training_trials
        self.total_testing_trials = total_testing_trials
        self.success_lines = success_lines

        self.trial = 0                  # Current trial
        self.testing = False            # Flag indicating testing (True) / training (False)
        self.random = False             # Flag indication random test (True) / choosing action test (False)
        self.total_testing_success = 0  # Total number of success on testing
        self.epsilon = 1.0              # Exploration factor
        self.alpha = 0.5                # Learning factor
        self.Q = dict()                 # Create a Q-table
        
        self.state = None               # Current state that it is trying to learn
        self.action = None              # Current action to perform
        self.reward = None              # Reward for the current action / state
        self.last_game_state = None     # Previous game state (used for logging metrics of the last state of trial)
    
    def reset(self):
        """ The reset function is called at the beginning of each trial."""

        if self.testing:
            if self.random:
                self.epsilon = 1.0
            else:             
                self.epsilon = 0.0
        else:
            self.epsilon = 1.0 - (self.trial / float(self.total_training_trials))
            #self.epsilon = math.cos((math.pi / 2) * self.trial / float(self.total_training_trials))
        return None

    def before_action(self, game_state):
        """ Called at each iteration before perform action. Builds state and choose action. """

        self.state = str(game_state.build_state_metrics())
        self.create_q(self.state, game_state)
        self.action = self.choose_action(self.state)
        return copy(self.action)

    def create_q(self, state, game_state):
        """ Create key if it doesn't exist on table. """

        if  not self.Q.has_key(state):
            stone = Stones.get_stone(game_state.stone_id) 
            actions = []
            for i in range(0, stone.n_rotation):
                for j in range(0, len(game_state.board[0]) - stone.list_width_height_rotation[i][0] + 1):
                    actions.append(Action(i, j))
            self.Q[state] = dict.fromkeys(actions, 0.0)
        return
    
    def choose_action(self, state):
        """ Choose an action to perform. """

        if random.random() > self.epsilon:
            maxQ = self.get_maxQ(state)
            return random.choice([k for (k, v) in self.Q.get(state).iteritems() if v == maxQ])
        else:
            return self.get_randomQ(state)

    def get_randomQ(self, state):
        """ Get a random Q action for the state the learner is in. """

        return random.choice(list(self.Q[state]))

    def get_maxQ(self, state):
        """ Get the max Q value for the state the learner is in. """

        return max(self.Q[state].values())

    def after_action(self, game_state_before, game_state_after):
        """ Called at each iteration after the action. Receive reward and learn when the stone changes. """

        reward = self.get_reward(game_state_before, game_state_after)
        self.learn(reward)

    def get_reward(self, game_state_before, game_state_after):
        """ Calculate rewards comparing some metrics of the state of the game before and after the action. """
        
        reward = 0.0            # Initial reward value

        # Low reward if it increases the number of holes
        reward -= (game_state_after.count_holes() - game_state_before.count_holes()) * 5
        
        # Low reward if it increases the max height
        reward -= (game_state_after.get_max_height() - game_state_before.get_max_height()) * 4

        # Low reward if game over
        reward -= -20 if (game_state_after.gameover) else 0

        # Reward if it makes a line
        reward += (game_state_after.lines - game_state_before.lines) * 50

        # Reward if lessen empty spaces below the max height
        diff_empty = game_state_before.count_empty() - game_state_after.count_empty()
        reward += diff_empty * 2 if diff_empty > 0 else 0

        return reward

    def learn(self, reward):
        """ Called to learn the reward of that action in that game state. """

        cur_value = self.Q.get(self.state).get(self.action)
        self.Q[self.state][self.action] = cur_value + self.alpha * (reward - cur_value)


class GameState(object):
    """ Game state and metrics. """

    def __init__(self, stone_id, board, lines, gameover):
        self.stone_id = stone_id
        self.board = np.array(board)
        self.lines = lines
        self.gameover = gameover

    def build_state_metrics(self):
        """ Build state metrics used to learn (Stone id and skyline). """

        skyline = self.get_skyline()
        return (self.stone_id, skyline)

    def get_skyline(self):
        """ Get the skyline format of the pieces of the board. 
        The left-most position is always 0 the others are relative to the height on the left to a max of the 
        stone height. """

        col_range = range(0, len(self.board[0]))      # Which are the columns below
        col_heights = [self.col_height(self.board[:, i]) for i in col_range]    # Columns heights, reference board
        col_heights_norm = [h - col_heights[i - 1 if i > 0 else 0] for i, h in enumerate(col_heights)]            # Columns heights, reference topleft cell
        
        stone_height = Stones.get_stone(self.stone_id).max_height                      # The max cliff is the stone height + 1
        for i, h in enumerate(col_heights_norm):
            if (abs(h) > stone_height and h > 0):
                col_heights_norm[i] = stone_height
            elif (abs(h) > stone_height and h < 0):
                col_heights_norm[i] = -stone_height
        
        return col_heights_norm

    def col_height(self, col):
        """ Get the highest row with a block in a specific column of the board. """

        return len(self.board) - 1 - np.min([i for i, x in enumerate(col) if x > 0])

    def count_holes(self):
        """ Get the number of holes in the board. Holes are every empty cell with a block over it. """

        board_height = len(self.board[0])
        col_holes = [self.col_height(self.board[:, i]) - np.count_nonzero(self.board[:, i]) + 1 for i in range(0, board_height)]
        return sum(col_holes)

    def count_empty(self):
        """ Get the number of empty cells. Empty cell is a cell with no block in a row under the max height. """

        max_height = self.get_max_height()
        board_height = len(self.board)
        board_width = len(self.board[0])
        return (max_height * board_width) - np.count_nonzero(self.board[board_height - max_height - 1:board_height - 1, :])
        
    def get_max_height(self):
        """ Get the max height of blocks in the board."""

        board_width = len(self.board[0])
        col_heights = [self.col_height(self.board[:, i]) for i in range(0, board_width)]
        return max(col_heights)
 

class Stone(object):
    """ Stone, with its possible number of rotation, max height and its width and height at each rotation. """

    def __init__(self, n_rotation, max_height, list_width_height_rotation):
        self.n_rotation = n_rotation
        self.max_height = max_height
        self.list_width_height_rotation = list_width_height_rotation


class Stones(object):
    """ Helper that get/set the stone types and its moves / sizes. """

    stones = []
    stones.append(Stone(1, 1, [(1, 1)]))            # . Stone
    stones.append(Stone(2, 2, [(2, 1), (1, 2)]))    # I Stone
    stones.append(Stone(2, 2, [(2, 2), (2, 2)]))    # / Stone
    stones.append(Stone(1, 2, [(2, 2)]))            # O Stone
    
    @staticmethod
    def get_stone(id):
        return Stones.stones[id]


class Action(object):
    """ Action, how many times should rotate and move right. """

    def __init__(self, n_rotation, n_side):
        self.n_rotation = n_rotation
        self.n_side = n_side
