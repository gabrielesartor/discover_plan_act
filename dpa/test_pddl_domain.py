import pickle
from s2s.utils import save, make_dir, show, load
from s2s.planner.mgpt_planner import mGPT
from s2s.env.treasure_game.treasure_game import TreasureGame
from time import sleep

version = 1 # or None

planner = mGPT(mdpsim_path='/home/gabriele/Github/skills_to_symbols_v2/s2s/planner/mdpsim-1.23/mdpsim',
           mgpt_path='/home/gabriele/Github/skills_to_symbols_v2/s2s/planner/mgpt/planner',
           wsl=False)

domain_path = "/home/gabriele/Github/skills_to_symbols_v2/dpa/data/05_complete_domain/trial_5/treasure_game_data_DISTANCE_BABBLING_800_15_0/domain.pkl"
problem_path = "/home/gabriele/Github/skills_to_symbols_v2/dpa/data/05_complete_domain/trial_5/treasure_game_data_DISTANCE_BABBLING_800_15_0/problem_original.pkl"

# load pddl_domain
pddl_domain = pickle.load(open(domain_path, 'rb'))
# load pddl_problem
pddl_problem = pickle.load(open(problem_path, 'rb'))

valid, output = planner.find_plan(pddl_domain, pddl_problem)
print("Valid: {}\nOutput: {}".format(valid,output))
plan = list()

if valid:
    for option in output.path:
        operator_idx = int(option[option.rindex('-') + 1:])
        operator = pddl_domain.operators[operator_idx]
        plan.append(operator.option)

print("Plan length: {}\nPlan: {}".format(len(plan),plan))
#
# # executing the plan
# env = TreasureGame()
# env.reset()
# for op in plan:
#     env.render()
#     next_state, reward, done, info = env.step(op)
#     sleep(0.2)
#     env.render()

print("Everything done!")
