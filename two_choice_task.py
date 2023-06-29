import numpy as np
import sys

## state hierarchy table ## useless
# sh_table_6steps = np.array(
#                 [   [0, 6, 12],
#                     [0, 2, 4, 6, 8, 10, 12],
#                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#                 ])
# starts at [6] and 6 steps to either end

level0_mdp = np.array(
                [   [0,1],  # in state 0 action 0 leads to state 0 and action 1 leads to state 1
                    [0,2],  # in state 1 action 0 leads to state 0 and action 1 leads to state 2
                    [1,3],  # in state 2 action 0 leads to state 1 and action 1 leads to state 3
                    [2,4],  # in state 3 action 0 leads to state 2 and action 1 leads to state 4
                    [3,5],  # in state 4 action 0 leads to state 3 and action 1 leads to state 5
                    [4,6],  # in state 5 action 0 leads to state 4 and action 1 leads to state 6
                    [5,7],  # in state 6 action 0 leads to state 5 and action 1 leads to state 7
                    [6,8],  # in state 7 action 0 leads to state 6 and action 1 leads to state 8
                    [7,9],  # in state 8 action 0 leads to state 7 and action 1 leads to state 9
                    [8,10], # in state 9 action 0 leads to state 8 and action 1 leads to state 10
                    [9,11], # in state 10 action 0 leads to state 9 and action 1 leads to state 11
                    [10,12],# in state 11 action 0 leads to state 10 and action 1 leads to state 12
                    [11,12] # in state 12 action 0 leads to state 11 and action 1 leads to state 12
                ])

level1_mdp = np.array(
                [   [0,2], # in state 0 option 0 leads to state 0 and option 1 leads to state 2
                    [np.nan, np.nan], # state 1 not eligible for level 1 options
                    [0,4], # in state 2 option 0 leads to state 0 and option 1 leads to state 4
                    [np.nan, np.nan], # state 3 not eligible for level 1 options
                    [2,6], # in state 4 option 0 leads to state 2 and option 1 leads to state 6
                    [np.nan, np.nan], # state 5 not eligible for level 1 options
                    [4,8], # in state 6 option 0 leads to state 4 and option 1 leads to state 8
                    [np.nan, np.nan], # state 7 not eligible for level 1 options
                    [6,10], # in state 8 option 0 leads to state 6 and option 1 leads to state 10
                    [np.nan, np.nan], # state 9 not eligible for level 1 options
                    [8,12], # in state 10 option 0 leads to state 8 and option 1 leads to state 12
                    [np.nan, np.nan], # state 11 not eligible for level 1 options
                    [10,12] # in state 12 option 0 leads to state 10 and option 1 leads to state 12
                ])

level2_mdp = np.array(
                [   [0,6], # in state 0 option 0 leads to state 0 and option 1 leads to state 6
                    [np.nan, np.nan], # state 1 not eligible for level 2 options
                    [np.nan, np.nan], # state 2 not eligible for level 2 options
                    [np.nan, np.nan], # state 3 not eligible for level 2 options
                    [np.nan, np.nan], # state 4 not eligible for level 2 options
                    [np.nan, np.nan], # state 5 not eligible for level 2 options
                    [0,12],  # in state 6 option 0 leads to state 0 and option 1 leads to state 12
                    [np.nan, np.nan], # state 7 not eligible for level 2 options
                    [np.nan, np.nan], # state 8 not eligible for level 2 options
                    [np.nan, np.nan], # state 9 not eligible for level 2 options
                    [np.nan, np.nan], # state 10 not eligible for level 2 options
                    [np.nan, np.nan], # state 11 not eligible for level 2 options
                    [6,12]  # in state 12 option 0 leads to state 6 and option 1 leads to state 12
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
                    [0, 1, 2],  # state 6
                    [0],        # state 7
                    [0, 1],     # state 8
                    [0],        # state 9
                    [0, 1],     # state 10
                    [0],        # state 11
                    [0, 1, 2]   # state 12
                ])
# in principle this can be easily derived from stacked MDPs 
# but it's useful to have this table for quick checking while arbitration/action selection

level01_state_map =  np.array(
                [   0, # level 0 state 0 mapped to level 1 state 0
                    2, # level 0 state 1 mapped to level 1 state 2
                    2, # level 0 state 2 mapped to level 1 state 2
                    4, # level 0 state 3 mapped to level 1 state 4
                    4, # level 0 state 4 mapped to level 1 state 4
                    6, # level 0 state 5 mapped to level 1 state 6
                    6, # level 0 state 6 mapped to level 1 state 6
                    6, # level 0 state 7 mapped to level 1 state 6
                    8, # level 0 state 8 mapped to level 1 state 8
                    8, # level 0 state 9 mapped to level 1 state 8
                    10,# level 0 state 10 mapped to level 1 state 10
                    10,# level 0 state 11 mapped to level 1 state 10
                    12 # level 0 state 12 mapped to level 1 state 12
                ])

level12_state_map =  np.array(
                [   0, # level 1 state 0 mapped to level 2 state 0
                    np.nan, # level 1 state 1 does not exist
                    6, # level 1 state 2 mapped to level 2 state 0
                    np.nan, # level 1 state 3 does not exist
                    6, # level 1 state 4 mapped to level 2 state 6   
                    np.nan, # level 1 state 5 does not exist
                    6, # level 1 state 6 mapped to level 2 state 6
                    np.nan, # level 1 state 7 does not exist
                    6, # level 1 state 8 mapped to level 2 state 0
                    np.nan, # level 1 state 9 does not exist
                    6, # level 1 state 10 mapped to level 2 state 6   
                    np.nan, # level 1 state 11 does not exist
                    12  # level 1 state 12 mapped to level 2 state 6
                ])


abstract_state_mapping_6steps = [level01_state_map, level12_state_map]

class TwoChoiceTask:
    def __init__(self, finalstate1=0, finalstate2=12, max_steps=200, goal_reward=10, levels=3, stacked_mdp=stacked_mdp_6steps, swo_table = swo_table_6steps, abstract_state_mapping = abstract_state_mapping_6steps, partial_flip=False, restart_state=6):
        self.task_name="TwoChoiceTask"
        self.finalstate1=finalstate1
        self.finalstate2=finalstate2
        self.curr_state=restart_state # start state in the middle
        self.reset_state=restart_state
        self.step_count=0
        self.max_steps=max_steps
        self.goal_reward=goal_reward # flip from 10 to -15 later
        self.partial_flip=partial_flip # somehow this doesn't work, needs to be manually set in class args
        self.flipped_drug_reward = goal_reward
        
        self.num_states = finalstate2+1
        self.num_actions = 2 #primitive actions
        self.num_options = 2

        self.levels=levels
        self.stacked_mdp = stacked_mdp
        self.swo_table = swo_table
        self.abstract_state_mapping = abstract_state_mapping

    def take_primitive_action(self, state, action):
        self.step_count+=1
        # print(self.step_count)
        if (state == self.finalstate1 or state == self.finalstate2):
            next_state = state
            done=True
        else:
            if (action == 0):
                next_state = max(self.finalstate1, state-1)
            elif (action == 1):
                next_state = min(self.finalstate2, state+1)
            
            if (self.step_count==self.max_steps):
                print("Episode terminated, max_steps reached.")
                done=True
            else:
                done=False
        
        if self.partial_flip:
            if (state==self.finalstate1):
                reward = self.goal_reward
            elif (state==self.finalstate2):
                reward = self.flipped_drug_reward
            else:
                reward = 0
        else:
            if (state==self.finalstate1 or state==self.finalstate2):
                reward = self.goal_reward
            else:
                reward = 0
        
        return next_state, reward, done

    def check_if_end_state(self, state):
        if (state == self.finalstate1 or state == self.finalstate2):
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
            # lock
            while True:
                next_state, reward, done = self.take_primitive_action(state, 0)
                print("state = ", state, "option = ", option, "next_state = ", next_state, "done = ", done)
                next_state_list.append(next_state)
                reward_list.append(reward)
                state = next_state
                print("0")
                if (state==option_termination_state or done):
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
        self.curr_state=self.reset_state
        return self.curr_state
    
    def flip_reward(self, new_goal_reward):
        if self.partial_flip:
            self.flipped_drug_reward = new_goal_reward
        else:
            self.goal_reward = new_goal_reward
        return 
    
    # def top_level_next_state(self, state, option):
    #     top_level_mdp = self.stacked_mdp[self.levels-1]
    #     if np.isnan(top_level_mdp[state, 0]):
    #         next_state = 0
    #         print("Wrong top level state")
    #         sys.exit()
    #     else:
    #         next_state = int(top_level_mdp[state, option])
    #     return next_state
    
    def any_level_next_state(self, state, option, level):
        level_mdp = self.stacked_mdp[level]
        option_termination_state = level_mdp[state, option]
        if np.isnan(option_termination_state):
            print("Wrong state")
            sys.exit()
        option_termination_state = int(option_termination_state)
        return option_termination_state