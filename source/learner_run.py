from learner import *
from tetris import *
import numpy as np

train = True                         # Indicates if learner should train
total_training_trials = 5000         # Total of training trials to perform
total_testing_trials = 100           # Total of testing trials to perform
success_lines = 24                   # Number of lines that indicates success = Level 5

def run():
    """ Run all the simulation (training and testing). """

    learner = Learner(total_training_trials, total_testing_trials, success_lines)

    training_logs = []
    random_logs = []
    testing_logs = []

    # Training
    if train:
        learner.testing = False
        for trial in xrange(learner.total_training_trials):
            trialIteration(learner, trial + 1)
            training_logs.append([trial + 1, learner.last_game_state.count_holes(), learner.last_game_state.lines, learner.epsilon])
        np.save('Qlearning.npy', learner.Q)
        np.savetxt("training.csv", training_logs, delimiter=",", fmt='%.4f')
    print "Summary\nTraining trials: %d" % (learner.total_training_trials)
    
    # Testing random actions
    learner.testing = True
    learner.random = True
    learner.total_testing_success = 0
    for trial in xrange(learner.total_testing_trials):
        trialIteration(learner, trial + 1)
        random_logs.append([trial + 1, learner.last_game_state.count_holes(), learner.last_game_state.lines, learner.total_testing_success])
    np.savetxt("testing_random.csv", random_logs, delimiter=",", fmt='%d')
    print "Testing random: %d / %d" % (learner.total_testing_success, learner.total_testing_trials)

    # Testing after learning
    learner.random = False
    learner.total_testing_success = 0
    learner.Q = np.load('Qlearning.npy').item()
    for trial in xrange(learner.total_testing_trials):
        trialIteration(learner, trial + 1)
        testing_logs.append([trial + 1, learner.last_game_state.count_holes(), learner.last_game_state.lines, learner.total_testing_success])
    np.savetxt("testing.csv", testing_logs, delimiter=",", fmt='%d')
    print "Testing learned: %d / %d" % (learner.total_testing_success, learner.total_testing_trials)
    
    
def trialIteration(learner, trial):
    """ At each trial update values (epsilon) and run the game."""

    learner.trial = trial
    learner.reset()
    
    App = TetrisApp(learner)
    App.run()


if __name__ == '__main__':
    run()
    