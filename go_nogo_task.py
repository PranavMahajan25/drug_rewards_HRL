import numpy as np
import sys

## state hierarchy table ## useless
# sh_table_6steps = np.array(
#                 [   [0, 6],
#                     [0, 2, 4, 6],
#                     [0, 1, 2, 3, 4, 5, 6]
#                 ])

level0_mdp = np.array(
                [   [0,1], # in state 0 action 0 leads to state 0 and action 1 leads to state 1
                    [0,2], # in state 1 action 0 leads to state 0 and action 1 leads to state 2
                    [1,3], # in state 2 action 0 leads to state 1 and action 1 leads to state 3
                    [2,4], # in state 3 action 0 leads to state 2 and action 1 leads to state 4
                    [3,5], # in state 4 action 0 leads to state 3 and action 1 leads to state 5
                    [4,6], # in state 5 action 0 leads to state 4 and action 1 leads to state 6
                    [5,6]  # in state 6 action 0 leads to state 5 and action 1 leads to state 6
                ])

level1_mdp = np.array(
                [   [0,2], # in state 0 option 0 leads to state 0 and option 1 leads to state 2
                    [np.nan, np.nan], # state 1 not eligible for level 1 options
                    [0,4], # in state 2 option 0 leads to state 0 and option 1 leads to state 4
                    [np.nan, np.nan], # state 3 not eligible for level 1 options
                    [2,6], # in state 4 option 0 leads to state 2 and option 1 leads to state 6
                    [np.nan, np.nan], # state 5 not eligible for level 1 options
                    [4,6]  # in state 6 option 0 leads to state 4 and option 1 leads to state 6
                ])

level2_mdp = np.array(
                [   [0,6], # in state 0 option 0 leads to state 0 and option 1 leads to state 6
                    [np.nan, np.nan], # state 1 not eligible for level 2 options
                    [np.nan, np.nan], # state 2 not eligible for level 2 options
                    [np.nan, np.nan], # state 3 not eligible for level 2 options
                    [np.nan, np.nan], # state 4 not eligible for level 2 options
                    [np.nan, np.nan], # state 5 not eligible for level 2 options
                    [0,6]  # in state 6 option 0 leads to state 0 and option 1 leads to state 6
                ])

## stacked mdp
stacked_mdp_6steps = [level0_mdp, level1_mdp, level2_mdp]


## state-wise option-level eligibility table 
swo_table_6steps = np.array(
                [   [0, 1, 2],  # state 0 
                    [0],        # state 1
                    [0, 1],     # state 2
                    [0],        # state 3
                    [0, 1],     # state 4
                    [0],        # state 5
                    [0, 1, 2]   # state 6
                ])
# in principle this can be easily derived from stacked MDPs 
# but it's useful to have this table for quick checking while arbitration/action selection

level01_state_map =  np.array(
                [   0, # level 0 state 0 mapped to level 1 state 0
                    0, # level 0 state 1 mapped to level 1 state 0
                    2, # level 0 state 2 mapped to level 1 state 2
                    2, # level 0 state 3 mapped to level 1 state 2
                    4, # level 0 state 4 mapped to level 1 state 4
                    4, # level 0 state 5 mapped to level 1 state 4
                    6  # level 0 state 6 mapped to level 1 state 6
                ])

level12_state_map =  np.array(
                [   0, # level 1 state 0 mapped to level 2 state 0
                    np.nan, # level 1 state 1 does not exist
                    0, # level 1 state 2 mapped to level 2 state 0
                    np.nan, # level 1 state 3 does not exist
                    0, # level 1 state 4 mapped to level 2 state 6   # debatable whether 0 or 6 
                    np.nan, # level 1 state 5 does not exist
                    6  # level 1 state 6 mapped to level 2 state 6
                ])


abstract_state_mapping_6steps = [level01_state_map, level12_state_map]

nogo_dues_set_6steps = np.array([1, 2, 6])

class GoNoGoTask:
    def __init__(self, finalstate=6, max_steps=100, goal_reward=10, levels=3, stacked_mdp=stacked_mdp_6steps, swo_table = swo_table_6steps, abstract_state_mapping = abstract_state_mapping_6steps, nogo_dues_set = nogo_dues_set_6steps):
        self.task_name="GoNoGoTask"
        self.finalstate=finalstate
        self.curr_state=0
        self.reset_state=0
        self.step_count=0
        self.max_steps=max_steps
        self.goal_reward=goal_reward # flip from 10 to -16 later
        
        self.num_states = finalstate+1
        self.num_actions = 2 #primitive actions
        self.num_options = 2

        self.levels=levels
        self.stacked_mdp = stacked_mdp
        self.swo_table = swo_table
        self.abstract_state_mapping = abstract_state_mapping
        self.nogo_dues_set = nogo_dues_set

    def take_primitive_action(self, state, action):
        self.step_count+=1
        # print(self.step_count)
        if (state == self.finalstate):
            next_state = state
            done=True
        else:
            if (action == 0):
                next_state = max(0,state-1)
            elif (action == 1):
                next_state = min(self.finalstate, state+1)
            
            if (self.step_count==self.max_steps):
                print("Episode terminated, max_steps reached.")
                done=True
            else:
                done=False
        
        if (state==self.finalstate):
            reward = self.goal_reward
        else:
            reward = 0
        return next_state, reward, done

    def check_if_end_state(self, state):
        if (state == self.finalstate):
            return True
        else: 
            return False

    def take_hierarchical_option(self, state, option, level):
        # Attempt to reduce to several primitive actions, lock until option termination state achieved
        next_state_list = []
        reward_list = []
        done = False

        level_mdp = self.stacked_mdp[level]
        option_termination_state = int(level_mdp[state, option])
        print("option_termination_state = ", option_termination_state)
        
        if (option == 0):
            nogo_dues = 0
            if state == 0:
                # special case to balance Go and No-Go action
                nogo_dues = self.nogo_dues_set[level]

            # lock
            while True:
                if nogo_dues !=0:
                    nogo_dues -= 1
                next_state, reward, done = self.take_primitive_action(state, 0)
                print("state = ", state, "option = ", option, "next_state = ", next_state, "done = ", done)
                next_state_list.append(next_state)
                reward_list.append(reward)
                state = next_state
                print("0")
                if done:
                    break
                elif state==option_termination_state:
                    if nogo_dues == 0:
                        break
            # unlock
        elif (option == 1):
            # lock
            while True:
                next_state, reward, done = self.take_primitive_action(state, 1)
                print("state = ", state, "option = ", option, "next_state = ", next_state, "done = ", done)
                next_state_list.append(next_state)
                reward_list.append(reward)
                state = next_state
                print("1")
                if (state==option_termination_state or done):
                    break    
            # unlock
        else:
            print("Options don't align with actions - terminating simulation")
            sys.exit()
        print(done)
        # The returned next_state_list is a list of primitive states
        return next_state_list, reward_list, done

    def reset(self):
        self.step_count=0
        self.curr_state=0
        return self.curr_state
    
    def flip_reward(self, new_goal_reward):
        self.goal_reward = new_goal_reward
        return 
    
    # def top_level_next_state(self, state, option):
    #     if state == 0:
    #         if option == 0:
    #             next_state = 0
    #         else:
    #             next_state = self.finalstate
    #     elif state == self.finalstate:
    #         if option == 0:
    #             next_state = 0
    #         else:
    #             next_state = self.finalstate
    #     else:
    #         next_state = 0
    #         print("Wrong top level state")
    #         sys.exit()
    #     return next_state
    
    def any_level_next_state(self, state, option, level):
        level_mdp = self.stacked_mdp[level]
        option_termination_state = level_mdp[state, option]
        if np.isnan(option_termination_state):
            print("Wrong state")
            sys.exit()
        option_termination_state = int(option_termination_state)
        return option_termination_state