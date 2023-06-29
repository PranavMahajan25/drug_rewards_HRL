import os
import matplotlib.pyplot as plt
import numpy as np


from hierarchical_q_learning import HRL_agent
from go_nogo_task import GoNoGoTask

go_nogo_env = GoNoGoTask()
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

# ## First plot - Food reward
# config['D'] = 0
# config['level_arbitration'] = False
# config['preferred_level'] = 0

# q_agent = HRL_agent(env=go_nogo_env, config=config)
# q_agent.train_nruns(maxeps=400, nruns=10)

# ## First plot - Drug reward
# config['D'] = 2
# config['level_arbitration'] = False
# config['preferred_level'] = 0 

# q_agent = HRL_agent(env=go_nogo_env, config=config)
# q_agent.train_nruns(maxeps=400, nruns=10)

# ###################################################################################
# config['plot_intermediate_outputs'] = False
# config['level_arbitration'] = False

# ## Second plot - Food reward
# config['D'] = 0

# phase1_avg_y_Go = []
# phase1_avg_y_NoGo = []
# phase1_std_y_Go = []
# phase1_std_y_NoGo = []

# phase2_avg_y_Go = []
# phase2_avg_y_NoGo = []
# phase2_std_y_Go = []
# phase2_std_y_NoGo = []


# for control_level in [2,1,0]:
#     config['preferred_level'] = control_level
#     q_agent = HRL_agent(env=go_nogo_env, config=config)

#     q_agent.train_nruns(maxeps=400, nruns=10)

#     q_agent.barstats_phase1_avg
#     q_agent.barstats_phase2_avg
#     q_agent.barstats_phase1_std
#     q_agent.barstats_phase2_std


#     phase1_avg_y_Go.append(q_agent.barstats_phase1_avg[0])
#     phase1_avg_y_NoGo.append(q_agent.barstats_phase1_avg[1])
#     phase1_std_y_Go.append(q_agent.barstats_phase1_std[0])
#     phase1_std_y_NoGo.append(q_agent.barstats_phase1_std[1])

#     phase2_avg_y_Go.append(q_agent.barstats_phase2_avg[0])
#     phase2_avg_y_NoGo.append(q_agent.barstats_phase2_avg[1])
#     phase2_std_y_Go.append(q_agent.barstats_phase2_std[0])
#     phase2_std_y_NoGo.append(q_agent.barstats_phase2_std[1])

# ## Saving
# os.makedirs("second_plot_food_reward", exist_ok=True)
# dirname = "second_plot_food_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Go.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Go))
# with open(os.path.join(dirname, 'phase1_avg_y_NoGo.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_NoGo))
# with open(os.path.join(dirname, 'phase1_std_y_Go.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Go))
# with open(os.path.join(dirname, 'phase1_std_y_NoGo.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_NoGo))

# with open(os.path.join(dirname, 'phase2_avg_y_Go.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Go))
# with open(os.path.join(dirname, 'phase2_avg_y_NoGo.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_NoGo))
# with open(os.path.join(dirname, 'phase2_std_y_Go.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Go))
# with open(os.path.join(dirname, 'phase2_std_y_NoGo.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_NoGo))

# ## Loading
# dirname = "second_plot_food_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Go.npy'), 'rb') as f:
#     phase1_avg_y_Go = np.load(f)
# with open(os.path.join(dirname, 'phase1_avg_y_NoGo.npy'), 'rb') as f:
#     phase1_avg_y_NoGo = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Go.npy'), 'rb') as f:
#     phase1_std_y_Go = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_NoGo.npy'), 'rb') as f:
#     phase1_std_y_NoGo = np.load(f)

# with open(os.path.join(dirname, 'phase2_avg_y_Go.npy'), 'rb') as f:
#     phase2_avg_y_Go = np.load(f)
# with open(os.path.join(dirname, 'phase2_avg_y_NoGo.npy'), 'rb') as f:
#     phase2_avg_y_NoGo = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Go.npy'), 'rb') as f:
#     phase2_std_y_Go = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_NoGo.npy'), 'rb') as f:
#     phase2_std_y_NoGo = np.load(f)


# ## Plotting

# ## Before pairing the goal state with punishment
# ind = np.arange(len(phase1_avg_y_Go))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase1_avg_y_Go, width, yerr=phase1_std_y_Go,
#                 label='Go')
# rects2 = ax.bar(ind + width/2, phase1_avg_y_NoGo, width, yerr=phase1_std_y_NoGo,
#                 label='No Go')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('Before pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('Top-level control', 'Mid-level control', 'Low-level control'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()


# ## After pairing the goal state with punishment
# ind = np.arange(len(phase2_avg_y_Go))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase2_avg_y_Go, width, yerr=phase2_std_y_Go,
#                 label='Go')
# rects2 = ax.bar(ind + width/2, phase2_avg_y_NoGo, width, yerr=phase2_std_y_NoGo,
#                 label='No Go')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('After pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('Top-level control', 'Mid-level control', 'Low-level control'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()

# ##################################################################################
# config['plot_intermediate_outputs'] = False
# config['level_arbitration'] = False

# ## Second plot - Drug reward
# config['D'] = 2

# phase1_avg_y_Go = []
# phase1_avg_y_NoGo = []
# phase1_std_y_Go = []
# phase1_std_y_NoGo = []

# phase2_avg_y_Go = []
# phase2_avg_y_NoGo = []
# phase2_std_y_Go = []
# phase2_std_y_NoGo = []


# for control_level in [2,1,0]:
#     config['preferred_level'] = control_level
#     q_agent = HRL_agent(env=go_nogo_env, config=config)

#     q_agent.train_nruns(maxeps=400, nruns=10)

#     q_agent.barstats_phase1_avg
#     q_agent.barstats_phase2_avg
#     q_agent.barstats_phase1_std
#     q_agent.barstats_phase2_std


#     phase1_avg_y_Go.append(q_agent.barstats_phase1_avg[0])
#     phase1_avg_y_NoGo.append(q_agent.barstats_phase1_avg[1])
#     phase1_std_y_Go.append(q_agent.barstats_phase1_std[0])
#     phase1_std_y_NoGo.append(q_agent.barstats_phase1_std[1])

#     phase2_avg_y_Go.append(q_agent.barstats_phase2_avg[0])
#     phase2_avg_y_NoGo.append(q_agent.barstats_phase2_avg[1])
#     phase2_std_y_Go.append(q_agent.barstats_phase2_std[0])
#     phase2_std_y_NoGo.append(q_agent.barstats_phase2_std[1])

# ## Saving
# os.makedirs("second_plot_drug_reward", exist_ok=True)
# dirname = "second_plot_drug_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Go.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_Go))
# with open(os.path.join(dirname, 'phase1_avg_y_NoGo.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_avg_y_NoGo))
# with open(os.path.join(dirname, 'phase1_std_y_Go.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_Go))
# with open(os.path.join(dirname, 'phase1_std_y_NoGo.npy'), 'wb') as f:
#     np.save(f, np.array(phase1_std_y_NoGo))

# with open(os.path.join(dirname, 'phase2_avg_y_Go.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Go))
# with open(os.path.join(dirname, 'phase2_avg_y_NoGo.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_NoGo))
# with open(os.path.join(dirname, 'phase2_std_y_Go.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Go))
# with open(os.path.join(dirname, 'phase2_std_y_NoGo.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_NoGo))

# ## Loading
# dirname = "second_plot_drug_reward"
# with open(os.path.join(dirname, 'phase1_avg_y_Go.npy'), 'rb') as f:
#     phase1_avg_y_Go = np.load(f)
# with open(os.path.join(dirname, 'phase1_avg_y_NoGo.npy'), 'rb') as f:
#     phase1_avg_y_NoGo = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_Go.npy'), 'rb') as f:
#     phase1_std_y_Go = np.load(f)
# with open(os.path.join(dirname, 'phase1_std_y_NoGo.npy'), 'rb') as f:
#     phase1_std_y_NoGo = np.load(f)

# with open(os.path.join(dirname, 'phase2_avg_y_Go.npy'), 'rb') as f:
#     phase2_avg_y_Go = np.load(f)
# with open(os.path.join(dirname, 'phase2_avg_y_NoGo.npy'), 'rb') as f:
#     phase2_avg_y_NoGo = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Go.npy'), 'rb') as f:
#     phase2_std_y_Go = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_NoGo.npy'), 'rb') as f:
#     phase2_std_y_NoGo = np.load(f)

# ## Plotting

# ## Before pairing the goal state with punishment
# ind = np.arange(len(phase1_avg_y_Go))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase1_avg_y_Go, width, yerr=phase1_std_y_Go,
#                 label='Go')
# rects2 = ax.bar(ind + width/2, phase1_avg_y_NoGo, width, yerr=phase1_std_y_NoGo,
#                 label='No Go')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('Before pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('Top-level control', 'Mid-level control', 'Low-level control'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()


# ## After pairing the goal state with punishment
# ind = np.arange(len(phase2_avg_y_Go))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase2_avg_y_Go, width, yerr=phase2_std_y_Go,
#                 label='Go')
# rects2 = ax.bar(ind + width/2, phase2_avg_y_NoGo, width, yerr=phase2_std_y_NoGo,
#                 label='No Go')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('After pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('Top-level control', 'Mid-level control', 'Low-level control'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()

##################################################################################
## Figure 9(A) ##
config['plot_intermediate_outputs'] = True
config['D'] = 2
config['level_arbitration'] = False
config['preferred_level'] = 0
config['N_dyna'] = 0

# q_agent = HRL_agent(env=go_nogo_env, config=config)
# q_agent.train_nruns(maxeps=400, nruns=10)


config['N_dyna'] = 10
q_agent = HRL_agent(env=go_nogo_env, config=config)
q_agent.train_nruns(maxeps=400, nruns=1)

config['N_dyna'] = 0
###################################################################################
# ## Fourth plot - Effect of selectivity - only phase 2
# config['plot_intermediate_outputs'] = False
# config['level_arbitration'] = True

# ## Food reward
# config['D'] = 0

# phase2_avg_y_Go = []
# phase2_avg_y_NoGo = []
# phase2_std_y_Go = []
# phase2_std_y_NoGo = []


# for selectivity in [-5,0,5]:
#     config['selectivity'] = selectivity
#     q_agent = HRL_agent(env=go_nogo_env, config=config)

#     q_agent.train_nruns(maxeps=400, nruns=10)

#     q_agent.barstats_phase1_avg
#     q_agent.barstats_phase2_avg
#     q_agent.barstats_phase1_std
#     q_agent.barstats_phase2_std

#     phase2_avg_y_Go.append(q_agent.barstats_phase2_avg[0])
#     phase2_avg_y_NoGo.append(q_agent.barstats_phase2_avg[1])
#     phase2_std_y_Go.append(q_agent.barstats_phase2_std[0])
#     phase2_std_y_NoGo.append(q_agent.barstats_phase2_std[1])

# ## Saving
# os.makedirs("fourth_plot_food_reward", exist_ok=True)
# dirname = "fourth_plot_food_reward"

# with open(os.path.join(dirname, 'phase2_avg_y_Go.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Go))
# with open(os.path.join(dirname, 'phase2_avg_y_NoGo.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_NoGo))
# with open(os.path.join(dirname, 'phase2_std_y_Go.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Go))
# with open(os.path.join(dirname, 'phase2_std_y_NoGo.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_NoGo))

# ## Loading
# dirname = "fourth_plot_food_reward"
# with open(os.path.join(dirname, 'phase2_avg_y_Go.npy'), 'rb') as f:
#     phase2_avg_y_Go = np.load(f)
# with open(os.path.join(dirname, 'phase2_avg_y_NoGo.npy'), 'rb') as f:
#     phase2_avg_y_NoGo = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Go.npy'), 'rb') as f:
#     phase2_std_y_Go = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_NoGo.npy'), 'rb') as f:
#     phase2_std_y_NoGo = np.load(f)


# ## Plotting

# ## Before pairing the goal state with punishment
# ind = np.arange(len(phase2_avg_y_Go))  # the x locations for the groups
# width = 0.35  # the width of the bars

# ## After pairing the goal state with punishment
# ind = np.arange(len(phase2_avg_y_Go))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase2_avg_y_Go, width, yerr=phase2_std_y_Go,
#                 label='Go')
# rects2 = ax.bar(ind + width/2, phase2_avg_y_NoGo, width, yerr=phase2_std_y_NoGo,
#                 label='No Go')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('After pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('selectivity = -5', 'selectivity = 0', 'selectivity = +5'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()

# ##################################################################################
# config['plot_intermediate_outputs'] = False
# config['level_arbitration'] = True

# ## Second plot - Drug reward
# config['D'] = 2

# phase2_avg_y_Go = []
# phase2_avg_y_NoGo = []
# phase2_std_y_Go = []
# phase2_std_y_NoGo = []

# for selectivity in [-5,0,5]:
#     config['selectivity'] = selectivity
#     q_agent = HRL_agent(env=go_nogo_env, config=config)

#     q_agent.train_nruns(maxeps=400, nruns=10)

#     q_agent.barstats_phase1_avg
#     q_agent.barstats_phase2_avg
#     q_agent.barstats_phase1_std
#     q_agent.barstats_phase2_std

#     phase2_avg_y_Go.append(q_agent.barstats_phase2_avg[0])
#     phase2_avg_y_NoGo.append(q_agent.barstats_phase2_avg[1])
#     phase2_std_y_Go.append(q_agent.barstats_phase2_std[0])
#     phase2_std_y_NoGo.append(q_agent.barstats_phase2_std[1])

# ## Saving
# os.makedirs("fourth_plot_drug_reward", exist_ok=True)
# dirname = "fourth_plot_drug_reward"

# with open(os.path.join(dirname, 'phase2_avg_y_Go.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_Go))
# with open(os.path.join(dirname, 'phase2_avg_y_NoGo.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_avg_y_NoGo))
# with open(os.path.join(dirname, 'phase2_std_y_Go.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_Go))
# with open(os.path.join(dirname, 'phase2_std_y_NoGo.npy'), 'wb') as f:
#     np.save(f, np.array(phase2_std_y_NoGo))

# ## Loading
# dirname = "fourth_plot_drug_reward"
# with open(os.path.join(dirname, 'phase2_avg_y_Go.npy'), 'rb') as f:
#     phase2_avg_y_Go = np.load(f)
# with open(os.path.join(dirname, 'phase2_avg_y_NoGo.npy'), 'rb') as f:
#     phase2_avg_y_NoGo = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_Go.npy'), 'rb') as f:
#     phase2_std_y_Go = np.load(f)
# with open(os.path.join(dirname, 'phase2_std_y_NoGo.npy'), 'rb') as f:
#     phase2_std_y_NoGo = np.load(f)

# ## Plotting

# ## Before pairing the goal state with punishment
# ind = np.arange(len(phase2_avg_y_Go))  # the x locations for the groups
# width = 0.35  # the width of the bars

# ## After pairing the goal state with punishment
# ind = np.arange(len(phase2_avg_y_Go))  # the x locations for the groups
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(ind - width/2, phase2_avg_y_Go, width, yerr=phase2_std_y_Go,
#                 label='Go')
# rects2 = ax.bar(ind + width/2, phase2_avg_y_NoGo, width, yerr=phase2_std_y_NoGo,
#                 label='No Go')

# ax.set_ylabel('Choice aggregate (normalized)')
# ax.set_title('After pairing the goal state with punishment')
# ax.set_xticks(ind)
# ax.set_xticklabels(('selectivity = -5', 'selectivity = 0', 'selectivity = +5'))
# ax.set_ylim([0, 1.2])
# ax.legend()
# plt.show()

