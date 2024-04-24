import sys
import os
from typing import List,Tuple
import cv2
import numpy as np
import pickle
import random
import time
import pygame
import pandas as pd
import random
from copy import deepcopy

from gym.spaces import Discrete

from exploration_strategies import ExplorationModel

from s2s.build_treasure_game import make_video
from s2s.core.build_pddl import build_pddl, find_goal_symbols, find_goal_symbols_from_state
from s2s.core.explore import collect_data
from s2s.core.learn_operators import learn_preconditions, learn_effects, combine_learned_operators
from s2s.core.partition import partition_options
from s2s.env.s2s_env import S2SEnv, S2SWrapper
from s2s.pddl.domain_description import PDDLDomain
from s2s.pddl.problem_description import PDDLProblem
from s2s.pddl.proposition import Proposition
from s2s.render import visualise_partitions, visualise_symbols
from s2s.utils import save, make_dir, show, load

from s2s.core.build_model import build_model
from s2s.env.treasure_game.treasure_game import TreasureGame
from s2s.planner.mgpt_planner import mGPT

class SensePlanAct:

    def __init__(self, options_per_episode_arg = None, num_episodes_arg = None, num_cycles_arg = 10, exploration_arg = None, trial = 0):
        self.env = TreasureGame(generate_options = False)
        self.log_file = open("data/log.txt", "a")
        self.log_file.write("\n\nTreasureGame log file\n")
        self.log_file.close()


        # exploration parameters
        if options_per_episode_arg:
            self.options_per_episode = options_per_episode_arg
        else:
            self.options_per_episode = options_per_episode

        if num_episodes_arg:
            self.n_episodes = num_episodes_arg
        else:
            self.n_episodes = num_episodes

        self.num_cycles = num_cycles_arg

        self.verbose = True
        self.n_jobs = self.n_episodes # threads == episodes

        if exploration_arg:
            self.exploration_model_name = exploration_arg
            self.exploration_model = ExplorationModel(exploration_arg)
        else:
            raise Exception("Exploration strategy missing!")

        # symbolic representation
        self.plan_to_target = None
        self.pddl_domain = None

        self.planner = mGPT(mdpsim_path='/home/gabriele/Github/skills_to_symbols_v2/s2s/planner/mdpsim-1.23/mdpsim',
                   mgpt_path='/home/gabriele/Github/skills_to_symbols_v2/s2s/planner/mgpt/planner',
                   wsl=False)

        # exploration model
        self.save_dir = 'data/treasure_game_data_' + str(self.exploration_model_name) + "_" + str(self.options_per_episode)
        self.writeOnLog("\ntrial_n: {}\noptions_per_episode: {}\nn_episodes: {}\nnum_cycles: {}\nexploration: {}\n".format(trial,self.options_per_episode,self.n_episodes,self.num_cycles,self.exploration_model_name))
        self.writeOnLog("\ncycle;options;trans;init;ops;sym;l_p_target;p_target;l_p_origin;p_origin;time\n")
        self.createInitialRepresentation()


    def resetEnvDiscovery(self):
        self.option_list = self.env.option_list
        self.option_names = self.env.option_names
        self.env = TreasureGame(False, self.option_list, self.option_names, self.plan_to_target)

        for i, opt in enumerate(self.option_list):
            opt.md = self.env._env
            opt.drawer = self.env.drawer
            self.writeOnLog("{}:[{},{}],".format(i,opt.primitive,opt.termination))
        self.writeOnLog(";")
        print("options names: {}".format(self.env.option_names))

    def run(self, cycle = 1, trial = 0):
        """
        1.explore and collect data
        2.create new options
        3.create new abstraction using previous and new data
        4.go to 1, starting the exploration from the last interesting state using planning.
        """
        success = 0 # problem solved?

        self.save_dir = 'data/treasure_game_data_' + str(self.exploration_model_name) + "_" + str(self.options_per_episode) + "_" + str(cycle) + "_" + str(trial)

        print("\n\nCycle {}\n".format(cycle))
        self.writeOnLog("\n{};".format(cycle))
        self.resetEnvDiscovery()
        self.collectData(cycle)
        self.createSymbolicRepresetation()
        self.explorationStrategy()

        self.plan_to_original_goal = None
        try:
            self.getOriginalProblem()
            self.plan_to_original_goal = self.planToOriginalGoal()
        except Exception as ex:
            print("No enough information to plan to end the game!\n{}".format(ex))

        if self.plan_to_target:
            print("\nplan_to_target (length:{}): {}".format(len(self.plan_to_target),self.plan_to_target))
            self.writeOnLog("{};{};".format(len(self.plan_to_target),self.plan_to_target))
        else:
            print("\nplan_to_target (length:0): {}".format(self.plan_to_target))
            self.writeOnLog("0;[];")

        if self.plan_to_original_goal:
            print("\nplan_to_original_goal (length:{}): {}".format(len(self.plan_to_original_goal),self.plan_to_original_goal))
            self.writeOnLog("{};{};".format(len(self.plan_to_original_goal),self.plan_to_original_goal))
            success = self.checkPlan()
        else:
            print("\nnplan_to_original_goal (length:0): {}".format(self.plan_to_original_goal))
            self.writeOnLog("0;[];")
        print("plan_to_target: {}\nplan_to_original_goal: {}".format(self.plan_to_target,self.plan_to_original_goal))

        return success

    def writeOnLog(self, str):
        self.log_file = open("data/log.txt", "a")
        self.log_file.write(str)
        self.log_file.close()

    def collectData(self, cycle = 0):
        """
        It explores the environment collecting data and extending the present one.
        """
        transition_data, initiation_data = collect_data(S2SWrapper(self.env, self.options_per_episode),
                                                    max_episode=self.n_episodes,
                                                    verbose=self.verbose,
                                                    n_jobs=self.n_jobs, options = self.plan_to_target, cycle = cycle)
        print("Len transition_data {}".format(len(transition_data)))
        print("Len initiation_data {}".format(len(initiation_data)))

        # extends dataset
        self.transition_data = pd.concat([transition_data, self.transition_data], ignore_index=True)
        self.initiation_data = pd.concat([initiation_data, self.initiation_data], ignore_index=True)

        print("Then, len transition_data {}".format(len(self.transition_data)))
        print("Then, len initiation_data {}".format(len(self.initiation_data)))

    def createInitialRepresentation(self):
        """
        It creates a first symbolic representation.
        """
        self.save_dir = 'data/treasure_game_data_' + str(self.exploration_model_name) + "_" + str(self.options_per_episode) + "_0"

        transition_data, initiation_data = collect_data(S2SWrapper(self.env, self.options_per_episode),
                                                    max_episode=self.n_episodes,
                                                    verbose=self.verbose,
                                                    n_jobs=self.n_jobs, options = self.plan_to_target)
        self.transition_data = transition_data
        self.initiation_data = initiation_data

        self.createSymbolicRepresetation()


    def createSymbolicRepresetation(self):
        """
        ...
        """
        show("Saving data in {}...".format(self.save_dir), self.verbose)
        make_dir(self.save_dir)
        self.transition_data.to_pickle('{}/transition.pkl'.format(self.save_dir), compression='gzip')
        self.initiation_data.to_pickle('{}/init.pkl'.format(self.save_dir), compression='gzip')

        #  self.writeOnLog("\nUsing transition_data: {}\nUsing initiation_data: {}".format(len(self.transition_data),len(self.initiation_data)))
        self.writeOnLog("{};{};".format(len(self.transition_data),len(self.initiation_data)))

        # 1. Partition options
        self.partitions = partition_options(self.env,
                                       self.transition_data,
                                       verbose=self.verbose,
                                       n_jobs=self.n_jobs)

        # 2. Estimate preconditions
        self.preconditions = learn_preconditions(self.env,
                                            self.initiation_data,
                                            self.partitions,
                                            verbose=self.verbose,
                                            n_jobs=self.n_jobs)

        # 3. Estimate effects
        self.effects = learn_effects(self.partitions, verbose=self.verbose, n_jobs=self.n_jobs)
        self.operators = combine_learned_operators(self.env, self.partitions, self.preconditions, self.effects)

        # 4. Build PDDL
        self.factors, self.vocabulary, self.schemata = build_pddl(self.env, self.transition_data, self.operators, verbose=self.verbose, n_jobs=self.n_jobs)
        self.pddl_domain = PDDLDomain(self.env, self.vocabulary, self.schemata)

        # self.writeOnLog("\nNum operators: {}".format(len(self.schemata)))
        # self.writeOnLog("\nNum symbols: {}".format(len(self.vocabulary)))
        self.writeOnLog("{};{};".format(len(self.schemata),len(self.vocabulary)))

        save(self.pddl_domain, '{}/domain.pkl'.format(self.save_dir))
        save(self.pddl_domain, '{}/domain.pddl'.format(self.save_dir), binary=False)
        # visualise_partitions('{}/vis_partitions'.format(self.save_dir), self.env, self.partitions, verbose=self.verbose,
        #                          option_descriptor=lambda option: self.env.describe_option(option))
        visualise_symbols('{}/vis_symbols'.format(self.save_dir), self.env, self.vocabulary, verbose=True,
                              render=self.env.render_states)


    def explorationStrategy(self):
        # RANDOM ACTION EXPLORATION
        if self.exploration_model_name == "ACTION_BABBLING":
            self.plan_to_target = None

        # RANDOM GOAL EXPLORATION
        if self.exploration_model_name == "GOAL_BABBLING":
            self.exploration_model.get_random_state(self.save_dir + '/transition.pkl')
            goal_state = self.exploration_model.less_visited_state
            self.getProblemFromState(goal_state)
            self.plan_to_target = self.planToGoal()

            trials = 0
            max_trials = 20
            while not self.plan_to_target and trials < max_trials:
                print("Trial number {}".format(trials))
                self.exploration_model.get_random_state(self.save_dir + '/transition.pkl')
                goal_state = self.exploration_model.less_visited_state
                self.getProblemFromState(goal_state)
                self.plan_to_target = self.planToGoal()
                trials = trials + 1

        # BORDER EXPLORATION
        if self.exploration_model_name == "DISTANCE_BABBLING":
            self.exploration_model.get_farest_goal(self.save_dir + '/transition.pkl')
            goal_state = self.exploration_model.less_visited_state
            self.getProblemFromState(goal_state)
            self.plan_to_target = self.planToGoal()

            trials = 0
            max_trials = 20
            while not self.plan_to_target and trials < max_trials:
                print("Trial number {}".format(trials))
                self.exploration_model.get_farest_goal(self.save_dir + '/transition.pkl')
                goal_state = self.exploration_model.less_visited_state
                self.getProblemFromState(goal_state)
                self.plan_to_target = self.planToGoal()
                trials = trials + 1


    def selectRandomOperator(self):
        self.target_operator = None
        self.target_operator = random.sample(self.schemata, 1)[0]


    def getOperatorPlan(self, **kwargs):
        # 6. Build PDDL problem file
        self.pddl_problem_operator = PDDLProblem(kwargs.get('problem_name', 'p1'), self.env.name)
        self.pddl_problem_operator.add_start_proposition(Proposition.not_failed())
        for prop in self.vocabulary.start_predicates:
            self.pddl_problem_operator.add_start_proposition(prop)

        effets = self.target_operator.effects[0]
        goal_symbols = [p for p in effets[1]]
        print(goal_symbols)
        for prop in self.vocabulary.goal_predicates + goal_symbols:
            self.pddl_problem_operator.add_goal_proposition(prop)

        save(self.pddl_problem_operator, '{}/problem_operator.pkl'.format(self.save_dir))
        save(self.pddl_problem_operator, '{}/problem_operator.pddl'.format(self.save_dir), binary=False)

    def planToOperator(self):
        # # Now feed it to a planner
        valid, output = self.planner.find_plan(self.pddl_domain, self.pddl_problem_operator)
        print("Valid: {}\nOutput: {}".format(valid,output))

        if valid:
            plan = list()
            for option in output.path:
                operator_idx = int(option[option.rindex('-') + 1:])
                operator = self.pddl_domain.operators[operator_idx]
                plan.append(operator.option)
            plan.append(self.target_operator.option)
            print("Plan: {}".format(plan))
            return plan
        else:
            return None


    def getOriginalProblem(self,  **kwargs):
        # 6. Build PDDL problem file
        self.pddl_problem_original = PDDLProblem(kwargs.get('problem_name', 'p1'), self.env.name)
        self.pddl_problem_original.add_start_proposition(Proposition.not_failed())
        for prop in self.vocabulary.start_predicates:
            self.pddl_problem_original.add_start_proposition(prop)

        goal_prob, goal_symbols = find_goal_symbols(self.factors, self.vocabulary, self.transition_data, verbose=self.verbose, **kwargs)
        self.pddl_problem_original.add_goal_proposition(Proposition.not_failed())
        for prop in self.vocabulary.goal_predicates + goal_symbols:
            self.pddl_problem_original.add_goal_proposition(prop)

        save(self.pddl_problem_original, '{}/problem_original.pkl'.format(self.save_dir))
        save(self.pddl_problem_original, '{}/problem_original.pddl'.format(self.save_dir), binary=False)


    def planToOriginalGoal(self):
        # # Now feed it to a planner
        valid, output = self.planner.find_plan(self.pddl_domain, self.pddl_problem_original)
        print("Valid: {}\nOutput: {}".format(valid,output))

        if valid:
            plan = list()
            for option in output.path:
                operator_idx = int(option[option.rindex('-') + 1:])
                operator = self.pddl_domain.operators[operator_idx]
                plan.append(operator.option)
            print("Plan: {}".format(plan))
            if plan:
                self.plan_to_original_goal = plan
            make_video(self.pddl_domain, output.path, directory= self.save_dir, file_name = "plan_to_original", env = self.env)
            return plan
        else:
            return None

    def getProblemFromState(self, goal_state, relax = False, **kwargs):
        """
        It moves the agent
        """
        if not relax:
            self.pddl_problem = PDDLProblem(kwargs.get('problem_name', 'p1'), self.env.name)
            self.pddl_problem.add_start_proposition(Proposition.not_failed())
            for prop in self.vocabulary.start_predicates:
                self.pddl_problem.add_start_proposition(prop)

            goal_prob, goal_symbols = find_goal_symbols_from_state(goal_state, self.factors, self.vocabulary, self.transition_data, verbose=self.verbose, **kwargs)
            for prop in self.vocabulary.goal_predicates + goal_symbols:
                self.pddl_problem.add_goal_proposition(prop)

            self.temp_problem = deepcopy(self.pddl_problem)
            self.pddl_problem.add_goal_proposition(Proposition.not_failed())


        if relax:
            print(self.pddl_problem.goal_propositions)
            self.pddl_problem.goal_propositions = random.sample(self.temp_problem.goal_propositions, random.randint(1,len(self.temp_problem.goal_propositions)))
            self.pddl_problem.add_goal_proposition(Proposition.not_failed())
            print(self.pddl_problem.goal_propositions)

        save(self.pddl_problem, '{}/problem_target.pkl'.format(self.save_dir))
        save(self.pddl_problem, '{}/problem_target.pddl'.format(self.save_dir), binary=False)


    def planToGoal(self):
        # Now feed it to a planner
        valid, output = self.planner.find_plan(self.pddl_domain, self.pddl_problem)
        print("Valid: {}\nOutput: {}".format(valid,output))

        if valid:
            plan = list()
            for option in output.path:
                operator_idx = int(option[option.rindex('-') + 1:])
                operator = self.pddl_domain.operators[operator_idx]
                plan.append(operator.option)
            print("Plan: {}".format(plan))
            if plan:
                self.plan_to_target = plan
            make_video(self.pddl_domain, output.path, directory= self.save_dir, file_name = "plan_to_target", env = self.env)
            return plan
        else:
            return None


    def checkPlan(self):
        """
        The function checks if the plan reach the final goal.
        """
        print("checkPlan")

        for i in range(20):
            self.env.reset()
            plan_length = len(self.plan_to_original_goal)

            for action in self.plan_to_original_goal:
                next_state, reward, done, _ = self.env.step(action)
                print(str(next_state) + " " + str(reward) + " " + str(done))
                if done:
                    print("DONE!")
                    return 1
        return 0


if __name__ == '__main__':

    if not "data" in os.listdir("."):
            os.mkdir("data")

    explorations = ["DISTANCE_BABBLING","GOAL_BABBLING","ACTION_BABBLING"]
    options_per_episode_range = [800]
    cycles = 15
    num_episodes = 4
    trials = 1


    for exp in explorations:
        results = np.zeros((trials, cycles))

        for j in range(trials):
            for num_opt in options_per_episode_range:
                file_name = f"data/results_{exp}.npy"
                spa = SensePlanAct(num_opt, num_episodes, cycles, exp, j)
                time_start_system = time.time()
                for i in range(cycles):
                    time_start = time.time()
                    print("\n\nRound {}".format(i+1))

                    success = spa.run(i+1, j)

                    time_round = time.time() - time_start
                    print("\nRound {} finished in {} sec,".format(i+1,time_round))
                    spa.writeOnLog("{}".format(time_round))

                    results[j,i] = success
                    np.save(file_name, results)

                time_system = time.time() - time_start_system
                print("Everything done in {} sec, look at the symbols!".format(time_system))
