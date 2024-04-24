import pandas as pd
import numpy as np
import seaborn as sns
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from s2s.utils import make_dir

class ExplorationModel():
    def __init__(self, exploration_model_name):
        self.model = None
        self.exploration_model_name = exploration_model_name
        self.heatmap_model = None
        self.less_visited_state_x_y = None
        self.less_visited_state = None


    def get_farest_goal(self, path = 'treasure_game_data/transition.pkl'):
        print("Path: {}".format(path))
        with open(path, 'rb') as f:
            transitions = pd.read_pickle(f, compression="gzip")
            df_transition_states = pd.DataFrame(transitions.state.tolist())
            # print(df_transition_states)
            initial_state = None


            max_distance = 0
            best_x_y = [0,0]
            best_state = None
            for index, row in transitions.iterrows():
                state = row['state']

                if not initial_state:
                    best_x_y = [state[0], state[1]] # il primo dato è il punto di partenza
                    best_state = None
                    initial_state = state
                    continue

                max_noise = 0.1
                noise = np.random.normal(0,max_noise,len(state))
                distance = norm(np.array(state)-np.array(initial_state)+noise) # calculate distance
                if distance > max_distance:
                    max_distance = distance
                    best_x_y = [state[0], state[1]]
                    best_state = state

            print("Biggest distance {} in x.y {}".format(max_distance,best_x_y))
            self.less_visited_state_x_y = best_x_y
            self.less_visited_state = best_state

    def get_random_state(self, path = 'treasure_game_data/transition.pkl'):
        print("Path: {}".format(path))
        with open(path, 'rb') as f:
            transitions = pd.read_pickle(f, compression="gzip")
            df_transition_states = pd.DataFrame(transitions.state.tolist())

            if self.exploration_model_name == "GOAL_BABBLING":
                sample_row = transitions.sample()
                state = sample_row["state"].iloc[0]
                state_x_y = [state[0], state[1]]
                self.less_visited_state_x_y = state_x_y
                self.less_visited_state = state
            else:
                raise Exception("Error in self.exploration_model_name selection...")


if __name__ == '__main__':
    exploration_model = ExplorationModel()
