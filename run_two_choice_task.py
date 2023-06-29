import os
import numpy as np
import matplotlib.pyplot as plt

from hierarchical_q_learning import HRL_agent
from two_choice_task import TwoChoiceTask

two_choice_env = TwoChoiceTask()
config = {
    'alpha': 0.1,
    'epsilon': 0.1,
    'tau': 0.5,
    'gamma': 1,
    'selectivity': -1,
    'exploration_strategy': 'boltzmann',
    'flip_episode': 200,
    'flip_reward_value': -6,
    'D': 2,
    'directional_D': True,
    'update_lower_levels_only': False,
    'level_arbitration': False,
    'preferred_level': 0,
    'N_dyna': 0,
    'plot_each_run': False,
    'plot_intermediate_outputs': True
}

##################################################################################

## Figure 9(B) ##
config['plot_intermediate_outputs'] = True
config['D'] = 2
config['level_arbitration'] = False
config['preferred_level'] = 2
# config['N_dyna'] = 0

# q_agent = HRL_agent(env=two_choice_env, config=config)
# q_agent.train_nruns(maxeps=400, nruns=10)


config['N_dyna'] = 2
q_agent = HRL_agent(env=two_choice_env, config=config)
q_agent.train_nruns(maxeps=400, nruns=10)

config['N_dyna'] = 0

# ##################################################################################

# ## no figure in text ##
# config['plot_intermediate_outputs'] = False
# config['level_arbitration'] = False

# ## Fifth plot - Food reward
# config['D'] = 0

# phase1_avg_y_Food = []
# phase1_avg_y_Drug = []
# phase1_std_y_Food = []
# phase1_std_y_Drug = []

# phase2_avg_y_Food = []
# phase2_avg_y_Drug = []
# phase2_std_y_Food = []
# phase2_std_y_Drug = []


# for control_level in [2,1,0]:
#     config['preferred_level'] = control_level
#     q_agent = HRL_agent(env=two_choice_env, config=config)

#     q_agent.train_nruns(maxeps=400, nruns=10)

#     q_agent.barstats_phase1_avg
#     q_agent.barstats_phase2_avg
#     q_agent.barstats_phase1_std
#     q_agent.barstats_phase2_std


#     phase1_avg_y_Food.append(q_agent.barstats_phase1_avg[0])
#     phase1_avg_y_Drug.append(q_agent.barstats_phase1_avg[1])
#     phase1_std_y_Food.append(q_agent.barstats_phase1_std[0])
#     phase1_std_y_Drug.append(q_agent.barstats_phase1_std[1])

#     phase2_avg_y_Food.append(q_agent.barstats_phase2_avg[0])
#     phase2_avg_y_Drug.append(q_agent.barstats_phase2_avg[1])
#     phase2_std_y_Food.append(q_agent.barstats_phase2_std[0])
#     phase2_std_y_Drug.append(q_agent.barstats_phase2_std[1])

# ## Saving
# os.makedirs("fifth_plot_food_reward", exist_ok=True)
# dirname = "fifth_plot_food_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Food))
# with open(os.path.join(dirname, 'phase1_avg_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Drug))
# with open(os.path.join(dirname, 'phase1_std_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Food))
# with open(os.path.join(dirname, 'phase1_std_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Drug))

# with open(os.path.join(dirname, 'phase2_avg_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Food))
# with open(os.path.join(dirname, 'phase2_avg_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Drug))
# with open(os.path.join(dirname, 'phase2_std_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Food))
# with open(os.path.join(dirname, 'phase2_std_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Drug))

# ## Loading
# dirname = "fifth_plot_food_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Food.npy'), 'rb') as f:
#     phase1_avg_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase1_avg_y_Drug.npy'), 'rb') as f:
#     phase1_avg_y_Drug = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Food.npy'), 'rb') as f:
#     phase1_std_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Drug.npy'), 'rb') as f:
#     phase1_std_y_Drug = np.load(f)

# with open(os.path.join(dirname, 'phase2_avg_y_Food.npy'), 'rb') as f:
#     phase2_avg_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase2_avg_y_Drug.npy'), 'rb') as f:
#     phase2_avg_y_Drug = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Food.npy'), 'rb') as f:
#     phase2_std_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Drug.npy'), 'rb') as f:
#     phase2_std_y_Drug = np.load(f)


# ## Plotting

# ## Before pairing the goal state with punishment
# ind = np.arange(len(phase1_avg_y_Food))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase1_avg_y_Food, width, yerr=phase1_std_y_Food,
#                 label='Food')
# rects2 = ax.bar(ind + width/2, phase1_avg_y_Drug, width, yerr=phase1_std_y_Drug,
#                 label='Drug')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('Before pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('Top-level control', 'Mid-level control', 'Low-level control'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()


# ## After pairing the goal state with punishment
# ind = np.arange(len(phase2_avg_y_Food))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase2_avg_y_Food, width, yerr=phase2_std_y_Food,
#                 label='Food')
# rects2 = ax.bar(ind + width/2, phase2_avg_y_Drug, width, yerr=phase2_std_y_Drug,
#                 label='Drug')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('After pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('Top-level control', 'Mid-level control', 'Low-level control'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()

# ##################################################################################
## Figure 8(A) ##
# config['plot_intermediate_outputs'] = False
# config['level_arbitration'] = False

# config['D'] = 2

# phase1_avg_y_Food = []
# phase1_avg_y_Drug = []
# phase1_std_y_Food = []
# phase1_std_y_Drug = []

# phase2_avg_y_Food = []
# phase2_avg_y_Drug = []
# phase2_std_y_Food = []
# phase2_std_y_Drug = []


# for control_level in [2,1,0]:
#     config['preferred_level'] = control_level
#     q_agent = HRL_agent(env=two_choice_env, config=config)

#     q_agent.train_nruns(maxeps=400, nruns=10)

#     q_agent.barstats_phase1_avg
#     q_agent.barstats_phase2_avg
#     q_agent.barstats_phase1_std
#     q_agent.barstats_phase2_std


#     phase1_avg_y_Food.append(q_agent.barstats_phase1_avg[0])
#     phase1_avg_y_Drug.append(q_agent.barstats_phase1_avg[1])
#     phase1_std_y_Food.append(q_agent.barstats_phase1_std[0])
#     phase1_std_y_Drug.append(q_agent.barstats_phase1_std[1])

#     phase2_avg_y_Food.append(q_agent.barstats_phase2_avg[0])
#     phase2_avg_y_Drug.append(q_agent.barstats_phase2_avg[1])
#     phase2_std_y_Food.append(q_agent.barstats_phase2_std[0])
#     phase2_std_y_Drug.append(q_agent.barstats_phase2_std[1])

# ## Saving
# os.makedirs("fifth_plot_drug_reward", exist_ok=True)
# dirname = "fifth_plot_drug_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Food))
# with open(os.path.join(dirname, 'phase1_avg_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Drug))
# with open(os.path.join(dirname, 'phase1_std_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Food))
# with open(os.path.join(dirname, 'phase1_std_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Drug))

# with open(os.path.join(dirname, 'phase2_avg_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Food))
# with open(os.path.join(dirname, 'phase2_avg_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Drug))
# with open(os.path.join(dirname, 'phase2_std_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Food))
# with open(os.path.join(dirname, 'phase2_std_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Drug))

# ## Loading
# dirname = "fifth_plot_drug_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Food.npy'), 'rb') as f:
#     phase1_avg_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase1_avg_y_Drug.npy'), 'rb') as f:
#     phase1_avg_y_Drug = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Food.npy'), 'rb') as f:
#     phase1_std_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Drug.npy'), 'rb') as f:
#     phase1_std_y_Drug = np.load(f)

# with open(os.path.join(dirname, 'phase2_avg_y_Food.npy'), 'rb') as f:
#     phase2_avg_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase2_avg_y_Drug.npy'), 'rb') as f:
#     phase2_avg_y_Drug = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Food.npy'), 'rb') as f:
#     phase2_std_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Drug.npy'), 'rb') as f:
#     phase2_std_y_Drug = np.load(f)


# ## Plotting

# ## Before pairing the goal state with punishment
# ind = np.arange(len(phase1_avg_y_Food))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase1_avg_y_Food, width, yerr=phase1_std_y_Food,
#                 label='Food')
# rects2 = ax.bar(ind + width/2, phase1_avg_y_Drug, width, yerr=phase1_std_y_Drug,
#                 label='Drug')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('Before pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('Top-level control', 'Mid-level control', 'Low-level control'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()


# ## After pairing the goal state with punishment
# ind = np.arange(len(phase2_avg_y_Food))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase2_avg_y_Food, width, yerr=phase2_std_y_Food,
#                 label='Food')
# rects2 = ax.bar(ind + width/2, phase2_avg_y_Drug, width, yerr=phase2_std_y_Drug,
#                 label='Drug')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('After pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('Top-level control', 'Mid-level control', 'Low-level control'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()

# ###################################################################################
## Figure 5(A) ##
# config['plot_intermediate_outputs'] = False
# config['level_arbitration'] = True

# ## Second plot - Drug reward
# config['D'] = 2

# phase1_avg_y_Food = []
# phase1_avg_y_Drug = []
# phase1_std_y_Food = []
# phase1_std_y_Drug = []

# phase2_avg_y_Food = []
# phase2_avg_y_Drug = []
# phase2_std_y_Food = []
# phase2_std_y_Drug = []


# for selectivity in [-5,0,5]:
#     config['selectivity'] = selectivity
#     q_agent = HRL_agent(env=two_choice_env, config=config)

#     q_agent.train_nruns(maxeps=400, nruns=10)

#     q_agent.barstats_phase1_avg
#     q_agent.barstats_phase2_avg
#     q_agent.barstats_phase1_std
#     q_agent.barstats_phase2_std


#     phase1_avg_y_Food.append(q_agent.barstats_phase1_avg[0])
#     phase1_avg_y_Drug.append(q_agent.barstats_phase1_avg[1])
#     phase1_std_y_Food.append(q_agent.barstats_phase1_std[0])
#     phase1_std_y_Drug.append(q_agent.barstats_phase1_std[1])

#     phase2_avg_y_Food.append(q_agent.barstats_phase2_avg[0])
#     phase2_avg_y_Drug.append(q_agent.barstats_phase2_avg[1])
#     phase2_std_y_Food.append(q_agent.barstats_phase2_std[0])
#     phase2_std_y_Drug.append(q_agent.barstats_phase2_std[1])

# ## Saving
# os.makedirs("sixth_plot_drug_reward", exist_ok=True)
# dirname = "sixth_plot_drug_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Food))
# with open(os.path.join(dirname, 'phase1_avg_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Drug))
# with open(os.path.join(dirname, 'phase1_std_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Food))
# with open(os.path.join(dirname, 'phase1_std_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Drug))

# with open(os.path.join(dirname, 'phase2_avg_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Food))
# with open(os.path.join(dirname, 'phase2_avg_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Drug))
# with open(os.path.join(dirname, 'phase2_std_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Food))
# with open(os.path.join(dirname, 'phase2_std_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Drug))

# ## Loading
# dirname = "sixth_plot_drug_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Food.npy'), 'rb') as f:
#     phase1_avg_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase1_avg_y_Drug.npy'), 'rb') as f:
#     phase1_avg_y_Drug = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Food.npy'), 'rb') as f:
#     phase1_std_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Drug.npy'), 'rb') as f:
#     phase1_std_y_Drug = np.load(f)

# with open(os.path.join(dirname, 'phase2_avg_y_Food.npy'), 'rb') as f:
#     phase2_avg_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase2_avg_y_Drug.npy'), 'rb') as f:
#     phase2_avg_y_Drug = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Food.npy'), 'rb') as f:
#     phase2_std_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Drug.npy'), 'rb') as f:
#     phase2_std_y_Drug = np.load(f)


# ## Plotting

# ## Before pairing the goal state with punishment
# ind = np.arange(len(phase1_avg_y_Food))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase1_avg_y_Food, width, yerr=phase1_std_y_Food,
#                 label='Food')
# rects2 = ax.bar(ind + width/2, phase1_avg_y_Drug, width, yerr=phase1_std_y_Drug,
#                 label='Drug')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('Before pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('selectivity = -5', 'selectivity = 0', 'selectivity = +5'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()


# ## After pairing the goal state with punishment
# ind = np.arange(len(phase2_avg_y_Food))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase2_avg_y_Food, width, yerr=phase2_std_y_Food,
#                 label='Food')
# rects2 = ax.bar(ind + width/2, phase2_avg_y_Drug, width, yerr=phase2_std_y_Drug,
#                 label='Drug')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('After pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('selectivity = -5', 'selectivity = 0', 'selectivity = +5'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()

# ##################################################################################

### two_choice_env_partial = TwoChoiceTask(partial_flip=True)
### This argument doesn't work, so needs to be manually set in class args in two_choice_task.py
# two_choice_env_partial = TwoChoiceTask()

# ##################################################################################
# ## Figure 8(B) ##
# config['plot_intermediate_outputs'] = False
# config['level_arbitration'] = False

# ## Seventh plot - Drug reward
# config['D'] = 2

# phase1_avg_y_Food = []
# phase1_avg_y_Drug = []
# phase1_std_y_Food = []
# phase1_std_y_Drug = []

# phase2_avg_y_Food = []
# phase2_avg_y_Drug = []
# phase2_std_y_Food = []
# phase2_std_y_Drug = []


# for control_level in [2,1,0]:
#     config['preferred_level'] = control_level
#     q_agent = HRL_agent(env=two_choice_env_partial, config=config)

#     q_agent.train_nruns(maxeps=400, nruns=10)

#     q_agent.barstats_phase1_avg
#     q_agent.barstats_phase2_avg
#     q_agent.barstats_phase1_std
#     q_agent.barstats_phase2_std


#     phase1_avg_y_Food.append(q_agent.barstats_phase1_avg[0])
#     phase1_avg_y_Drug.append(q_agent.barstats_phase1_avg[1])
#     phase1_std_y_Food.append(q_agent.barstats_phase1_std[0])
#     phase1_std_y_Drug.append(q_agent.barstats_phase1_std[1])

#     phase2_avg_y_Food.append(q_agent.barstats_phase2_avg[0])
#     phase2_avg_y_Drug.append(q_agent.barstats_phase2_avg[1])
#     phase2_std_y_Food.append(q_agent.barstats_phase2_std[0])
#     phase2_std_y_Drug.append(q_agent.barstats_phase2_std[1])

# ## Saving
# os.makedirs("seventh_plot_drug_reward", exist_ok=True)
# dirname = "seventh_plot_drug_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Food))
# with open(os.path.join(dirname, 'phase1_avg_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Drug))
# with open(os.path.join(dirname, 'phase1_std_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Food))
# with open(os.path.join(dirname, 'phase1_std_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Drug))

# with open(os.path.join(dirname, 'phase2_avg_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Food))
# with open(os.path.join(dirname, 'phase2_avg_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Drug))
# with open(os.path.join(dirname, 'phase2_std_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Food))
# with open(os.path.join(dirname, 'phase2_std_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Drug))

# ## Loading
# dirname = "seventh_plot_drug_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Food.npy'), 'rb') as f:
#     phase1_avg_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase1_avg_y_Drug.npy'), 'rb') as f:
#     phase1_avg_y_Drug = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Food.npy'), 'rb') as f:
#     phase1_std_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Drug.npy'), 'rb') as f:
#     phase1_std_y_Drug = np.load(f)

# with open(os.path.join(dirname, 'phase2_avg_y_Food.npy'), 'rb') as f:
#     phase2_avg_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase2_avg_y_Drug.npy'), 'rb') as f:
#     phase2_avg_y_Drug = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Food.npy'), 'rb') as f:
#     phase2_std_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Drug.npy'), 'rb') as f:
#     phase2_std_y_Drug = np.load(f)


# ## Plotting

# ## Before pairing the goal state with punishment
# ind = np.arange(len(phase1_avg_y_Food))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase1_avg_y_Food, width, yerr=phase1_std_y_Food,
#                 label='Food')
# rects2 = ax.bar(ind + width/2, phase1_avg_y_Drug, width, yerr=phase1_std_y_Drug,
#                 label='Drug')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('Before pairing the drug goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('Top-level control', 'Mid-level control', 'Low-level control'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()


# ## After pairing the goal state with punishment
# ind = np.arange(len(phase2_avg_y_Food))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase2_avg_y_Food, width, yerr=phase2_std_y_Food,
#                 label='Food')
# rects2 = ax.bar(ind + width/2, phase2_avg_y_Drug, width, yerr=phase2_std_y_Drug,
#                 label='Drug')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('After pairing the drug goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('Top-level control', 'Mid-level control', 'Low-level control'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()

# ###################################################################################
# ## Figure 5(B) ##
# config['plot_intermediate_outputs'] = False
# config['level_arbitration'] = True

# ## Eigth plot - Drug reward
# config['D'] = 2

# phase1_avg_y_Food = []
# phase1_avg_y_Drug = []
# phase1_std_y_Food = []
# phase1_std_y_Drug = []

# phase2_avg_y_Food = []
# phase2_avg_y_Drug = []
# phase2_std_y_Food = []
# phase2_std_y_Drug = []


# for selectivity in [-5,0,5]:
#     config['selectivity'] = selectivity
#     q_agent = HRL_agent(env=two_choice_env_partial, config=config)

#     q_agent.train_nruns(maxeps=400, nruns=10)

#     q_agent.barstats_phase1_avg
#     q_agent.barstats_phase2_avg
#     q_agent.barstats_phase1_std
#     q_agent.barstats_phase2_std


#     phase1_avg_y_Food.append(q_agent.barstats_phase1_avg[0])
#     phase1_avg_y_Drug.append(q_agent.barstats_phase1_avg[1])
#     phase1_std_y_Food.append(q_agent.barstats_phase1_std[0])
#     phase1_std_y_Drug.append(q_agent.barstats_phase1_std[1])

#     phase2_avg_y_Food.append(q_agent.barstats_phase2_avg[0])
#     phase2_avg_y_Drug.append(q_agent.barstats_phase2_avg[1])
#     phase2_std_y_Food.append(q_agent.barstats_phase2_std[0])
#     phase2_std_y_Drug.append(q_agent.barstats_phase2_std[1])

# ## Saving
# os.makedirs("eigth_plot_drug_reward", exist_ok=True)
# dirname = "eigth_plot_drug_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Food))
# with open(os.path.join(dirname, 'phase1_avg_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Drug))
# with open(os.path.join(dirname, 'phase1_std_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Food))
# with open(os.path.join(dirname, 'phase1_std_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Drug))

# with open(os.path.join(dirname, 'phase2_avg_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Food))
# with open(os.path.join(dirname, 'phase2_avg_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Drug))
# with open(os.path.join(dirname, 'phase2_std_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Food))
# with open(os.path.join(dirname, 'phase2_std_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Drug))

# ## Loading
# dirname = "eigth_plot_drug_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Food.npy'), 'rb') as f:
#     phase1_avg_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase1_avg_y_Drug.npy'), 'rb') as f:
#     phase1_avg_y_Drug = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Food.npy'), 'rb') as f:
#     phase1_std_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Drug.npy'), 'rb') as f:
#     phase1_std_y_Drug = np.load(f)

# with open(os.path.join(dirname, 'phase2_avg_y_Food.npy'), 'rb') as f:
#     phase2_avg_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase2_avg_y_Drug.npy'), 'rb') as f:
#     phase2_avg_y_Drug = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Food.npy'), 'rb') as f:
#     phase2_std_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Drug.npy'), 'rb') as f:
#     phase2_std_y_Drug = np.load(f)


# ## Plotting

# ## Before pairing the goal state with punishment
# ind = np.arange(len(phase1_avg_y_Food))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase1_avg_y_Food, width, yerr=phase1_std_y_Food,
#                 label='Food')
# rects2 = ax.bar(ind + width/2, phase1_avg_y_Drug, width, yerr=phase1_std_y_Drug,
#                 label='Drug')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('Before pairing the drug goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('selectivity = -5', 'selectivity = 0', 'selectivity = +5'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()


# ## After pairing the goal state with punishment
# ind = np.arange(len(phase2_avg_y_Food))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase2_avg_y_Food, width, yerr=phase2_std_y_Food,
#                 label='Food')
# rects2 = ax.bar(ind + width/2, phase2_avg_y_Drug, width, yerr=phase2_std_y_Drug,
#                 label='Drug')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('After pairing the drug goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('selectivity = -5', 'selectivity = 0', 'selectivity = +5'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()

# ###################################################################################

# ### CHANGE RESTART STATE to CLOSER TO DRUG ###

# ### two_choice_env_partial = TwoChoiceTask(restart_state=10) or 8 ## SET MANUALLY :(
# ### This argument doesn't work, so needs to be manually set in class args in two_choice_task.py
# two_choice_env_partial = TwoChoiceTask()

# ##################################################################################

## NO Figure ##
# config['plot_intermediate_outputs'] = False
# config['level_arbitration'] = False

# ## Ninth plot - Drug reward
# config['D'] = 2

# phase1_avg_y_Food = []
# phase1_avg_y_Drug = []
# phase1_std_y_Food = []
# phase1_std_y_Drug = []

# phase2_avg_y_Food = []
# phase2_avg_y_Drug = []
# phase2_std_y_Food = []
# phase2_std_y_Drug = []


# for control_level in [1,0]:
#     config['preferred_level'] = control_level
#     q_agent = HRL_agent(env=two_choice_env, config=config)

#     q_agent.train_nruns(maxeps=400, nruns=10)

#     q_agent.barstats_phase1_avg
#     q_agent.barstats_phase2_avg
#     q_agent.barstats_phase1_std
#     q_agent.barstats_phase2_std


#     phase1_avg_y_Food.append(q_agent.barstats_phase1_avg[0])
#     phase1_avg_y_Drug.append(q_agent.barstats_phase1_avg[1])
#     phase1_std_y_Food.append(q_agent.barstats_phase1_std[0])
#     phase1_std_y_Drug.append(q_agent.barstats_phase1_std[1])

#     phase2_avg_y_Food.append(q_agent.barstats_phase2_avg[0])
#     phase2_avg_y_Drug.append(q_agent.barstats_phase2_avg[1])
#     phase2_std_y_Food.append(q_agent.barstats_phase2_std[0])
#     phase2_std_y_Drug.append(q_agent.barstats_phase2_std[1])

# ## Saving
# os.makedirs("ninth_plot_drug_reward", exist_ok=True)
# dirname = "ninth_plot_drug_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Food))
# with open(os.path.join(dirname, 'phase1_avg_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Drug))
# with open(os.path.join(dirname, 'phase1_std_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Food))
# with open(os.path.join(dirname, 'phase1_std_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Drug))

# with open(os.path.join(dirname, 'phase2_avg_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Food))
# with open(os.path.join(dirname, 'phase2_avg_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Drug))
# with open(os.path.join(dirname, 'phase2_std_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Food))
# with open(os.path.join(dirname, 'phase2_std_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Drug))

# ## Loading
# dirname = "ninth_plot_drug_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Food.npy'), 'rb') as f:
#     phase1_avg_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase1_avg_y_Drug.npy'), 'rb') as f:
#     phase1_avg_y_Drug = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Food.npy'), 'rb') as f:
#     phase1_std_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Drug.npy'), 'rb') as f:
#     phase1_std_y_Drug = np.load(f)

# with open(os.path.join(dirname, 'phase2_avg_y_Food.npy'), 'rb') as f:
#     phase2_avg_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase2_avg_y_Drug.npy'), 'rb') as f:
#     phase2_avg_y_Drug = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Food.npy'), 'rb') as f:
#     phase2_std_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Drug.npy'), 'rb') as f:
#     phase2_std_y_Drug = np.load(f)


# ## Plotting

# ## Before pairing the goal state with punishment
# ind = np.arange(len(phase1_avg_y_Food))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase1_avg_y_Food, width, yerr=phase1_std_y_Food,
#                 label='Food')
# rects2 = ax.bar(ind + width/2, phase1_avg_y_Drug, width, yerr=phase1_std_y_Drug,
#                 label='Drug')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('Before pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('Mid-level control', 'Low-level control'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()


# ## After pairing the goal state with punishment
# ind = np.arange(len(phase2_avg_y_Food))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase2_avg_y_Food, width, yerr=phase2_std_y_Food,
#                 label='Food')
# rects2 = ax.bar(ind + width/2, phase2_avg_y_Drug, width, yerr=phase2_std_y_Drug,
#                 label='Drug')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('After pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('Mid-level control', 'Low-level control'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()

# ###################################################################################

## Figure 6(A,B) ##
# config['plot_intermediate_outputs'] = False
# config['level_arbitration'] = True

# ## Second plot - Drug reward
# config['D'] = 2

# phase1_avg_y_Food = []
# phase1_avg_y_Drug = []
# phase1_std_y_Food = []
# phase1_std_y_Drug = []

# phase2_avg_y_Food = []
# phase2_avg_y_Drug = []
# phase2_std_y_Food = []
# phase2_std_y_Drug = []


# for selectivity in [-5,0,5]:
#     config['selectivity'] = selectivity
#     q_agent = HRL_agent(env=two_choice_env, config=config)

#     q_agent.train_nruns(maxeps=400, nruns=10)

#     q_agent.barstats_phase1_avg
#     q_agent.barstats_phase2_avg
#     q_agent.barstats_phase1_std
#     q_agent.barstats_phase2_std


#     phase1_avg_y_Food.append(q_agent.barstats_phase1_avg[0])
#     phase1_avg_y_Drug.append(q_agent.barstats_phase1_avg[1])
#     phase1_std_y_Food.append(q_agent.barstats_phase1_std[0])
#     phase1_std_y_Drug.append(q_agent.barstats_phase1_std[1])

#     phase2_avg_y_Food.append(q_agent.barstats_phase2_avg[0])
#     phase2_avg_y_Drug.append(q_agent.barstats_phase2_avg[1])
#     phase2_std_y_Food.append(q_agent.barstats_phase2_std[0])
#     phase2_std_y_Drug.append(q_agent.barstats_phase2_std[1])

# ## Saving
# os.makedirs("tenth_plot_drug_reward", exist_ok=True)
# dirname = "tenth_plot_drug_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Food))
# with open(os.path.join(dirname, 'phase1_avg_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Drug))
# with open(os.path.join(dirname, 'phase1_std_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Food))
# with open(os.path.join(dirname, 'phase1_std_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Drug))

# with open(os.path.join(dirname, 'phase2_avg_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Food))
# with open(os.path.join(dirname, 'phase2_avg_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Drug))
# with open(os.path.join(dirname, 'phase2_std_y_Food.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Food))
# with open(os.path.join(dirname, 'phase2_std_y_Drug.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Drug))

# ## Loading
# dirname = "tenth_plot_drug_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Food.npy'), 'rb') as f:
#     phase1_avg_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase1_avg_y_Drug.npy'), 'rb') as f:
#     phase1_avg_y_Drug = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Food.npy'), 'rb') as f:
#     phase1_std_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Drug.npy'), 'rb') as f:
#     phase1_std_y_Drug = np.load(f)

# with open(os.path.join(dirname, 'phase2_avg_y_Food.npy'), 'rb') as f:
#     phase2_avg_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase2_avg_y_Drug.npy'), 'rb') as f:
#     phase2_avg_y_Drug = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Food.npy'), 'rb') as f:
#     phase2_std_y_Food = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Drug.npy'), 'rb') as f:
#     phase2_std_y_Drug = np.load(f)


# ## Plotting

# ## Before pairing the goal state with punishment
# ind = np.arange(len(phase1_avg_y_Food))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase1_avg_y_Food, width, yerr=phase1_std_y_Food,
#                 label='Food')
# rects2 = ax.bar(ind + width/2, phase1_avg_y_Drug, width, yerr=phase1_std_y_Drug,
#                 label='Drug')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('Before pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('selectivity = -5', 'selectivity = 0', 'selectivity = +5'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()


# ## After pairing the goal state with punishment
# ind = np.arange(len(phase2_avg_y_Food))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase2_avg_y_Food, width, yerr=phase2_std_y_Food,
#                 label='Food')
# rects2 = ax.bar(ind + width/2, phase2_avg_y_Drug, width, yerr=phase2_std_y_Drug,
#                 label='Drug')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('After pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('selectivity = -5', 'selectivity = 0', 'selectivity = +5'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()

##################################################################################
