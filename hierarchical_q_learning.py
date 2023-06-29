import sys
import random
import numpy as np
import matplotlib.pyplot as plt

class simple_Qlearning_agent:
    def __init__(self, env):
        self.env = env
        self.alpha = 0.1
        self.epsilon = 0.1 # for epsilon-greedy exploration
        self.tau = 1 # temperature hyperparameter for boltzman exploration
        self.gamma = 0.9

        self.exploration_strategy = 'epsilon-greedy' #'boltzmann' #'epsilon-greedy'
        self.flip_episode = 200
        self.flip_reward_value = -5

        self.Q = np.zeros((env.num_states, env.num_actions))
        self.V = np.zeros((env.num_states, 1))
        self.state_counts = np.zeros((env.num_states, 1))

    def sample_primitive_action(self, state, exploration):
        if exploration == 'epsilon-greedy':
            if np.random.random() < self.epsilon:
                action = np.random.randint(0, self.env.num_actions)
            else:
                # Equivalent to action = np.argmax(Q_primitive[state, :]), 
                # but handles the case with multiple max Q actions
                best_actions = np.argwhere(self.Q[state, :] == np.amax(self.Q[state, :]))
                best_actions = best_actions.flatten().tolist()
                action = random.choice(best_actions)

        elif exploration == 'boltzmann':
            Qsa = self.Q[state, :]
            # A more safe and stable implementation of softmax 
            Qsa_shifted = Qsa - np.max(Qsa)
            transformed_Qsa = Qsa_shifted /self.tau
            Q_num = np.exp(transformed_Qsa)
            Q_denom = np.sum(Q_num)
            Q_dist = Q_num / Q_denom

            action_list = np.arange(self.env.num_actions)
            action = random.choices(action_list, weights = Q_dist, k=1)
            action = action[0]
        else:
            print("Please check chosen exploration strategy")
            sys.exit()

        return action

    def train(self, maxeps):
        self.ep_scores = np.zeros(maxeps)
        self.ep_steps = np.zeros(maxeps)
        self.values_per_episode = np.zeros((self.env.num_states, maxeps))

        for i_episode in range(maxeps):
            print("\rEpisode {}/{}, epsilon: {}".format(i_episode, maxeps, self.epsilon), end="")
            sys.stdout.flush() 


            ## flip reward at episode 200
            if i_episode == self.flip_episode:
                self.env.flip_reward(self.flip_reward_value)

            state = self.env.reset()

            ep_score = 0
            j_step = 0
            done = False
            while not done:
                # print("state = ", state)
                action = self.sample_primitive_action(state=state, exploration=self.exploration_strategy)
                next_state, reward, done = self.env.take_primitive_action(state, action)

                if done:
                    self.Q[state, action] += self.alpha * (reward - self.Q[state, action])
                    self.V[state] += self.alpha * (reward - self.V[state])
                else:
                    self.Q[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action])
                    self.V[state] += self.alpha * (reward + self.gamma * self.V[next_state] - self.V[state])

                self.state_counts[state] += 1

                ep_score += reward
                j_step += 1
                state = next_state
            
            self.ep_scores[i_episode] =  ep_score
            self.ep_steps[i_episode] = j_step
            self.values_per_episode[:, i_episode] = self.V[:,0]
        self.plot_values()

    def plot_values(self):
        for state in range(self.env.num_states):
            plt.plot(self.values_per_episode[state, :])
        plt.show()


class HRL_agent:
    def __init__(self, env, config):
        self.env = env
        
        self.alpha = config['alpha']
        self.epsilon = config['epsilon'] # for epsilon-greedy exploration
        self.tau = config['tau'] # temperature hyperparameter for boltzman exploration
        self.gamma = config['gamma']
        self.selectivity = config['selectivity'] #newly introduced parameter

        self.exploration_strategy = config['exploration_strategy'] #'boltzmann' #'epsilon-greedy'
        self.flip_episode = config['flip_episode']
        self.flip_reward_value = config['flip_reward_value']

        self.D = config['D']
        self.directional_D = config['directional_D'] # could be redundant, but setting to True is necessary

        self.update_lower_levels_only = config['update_lower_levels_only']
        self.level_arbitration = config['level_arbitration'] ## Check this!
        self.preferred_level = config['preferred_level']   

        self.N_dyna = config['N_dyna'] # Dyna, but value learning looks weird with Dyna     
        self.plot_each_run = config['plot_each_run']
        self.plot_intermediate_outputs = config['plot_intermediate_outputs']

        self.model_dysfunctional = True

        # instantiate Q and V tables for each level
        self.Q = []
        self.V = []

        for level in range(self.env.levels):
            Q_abstract = np.zeros((self.env.num_states, self.env.num_options))
            V_abstract = np.zeros((self.env.num_states, 1))    # V(s) is basically Q(s, argamax a) here
            if (level == self.env.levels-1):
                self.Observed_sa_bool = np.zeros((self.env.num_states, self.env.num_options)) 
                self.Model_nextstate = np.zeros((self.env.num_states, self.env.num_options))
                self.Model_reward = np.zeros((self.env.num_states, self.env.num_options))

            level_mdp = self.env.stacked_mdp[level]
            for i in range(len(level_mdp)):
                if (np.any(np.isnan(level_mdp[i]))):
                    Q_abstract[i] = np.nan
                    V_abstract[i] = np.nan
                    if (level == self.env.levels-1):
                        self.Observed_sa_bool[i] = np.nan
                        self.Model_nextstate[i] = np.nan
                        self.Model_reward[i] = np.nan

            self.Q.append(Q_abstract)
            self.V.append(V_abstract)

        self.state_counts = np.zeros((env.num_states, 1))

    def reset_values(self):
        self.env.__init__()
        # instantiate Q and V tables for each level
        self.Q = []
        self.V = []

        for level in range(self.env.levels):
            Q_abstract = np.zeros((self.env.num_states, self.env.num_options))
            V_abstract = np.zeros((self.env.num_states, 1))    # V(s) is basically Q(s, argamax a) here
            if (level == self.env.levels-1):
                self.Observed_sa_bool = np.zeros((self.env.num_states, self.env.num_options)) 
                self.Model_nextstate = np.zeros((self.env.num_states, self.env.num_options))
                self.Model_reward = np.zeros((self.env.num_states, self.env.num_options))

            level_mdp = self.env.stacked_mdp[level]
            for i in range(len(level_mdp)):
                if (np.any(np.isnan(level_mdp[i]))):
                    Q_abstract[i] = np.nan
                    V_abstract[i] = np.nan
                    if (level == self.env.levels-1):
                        self.Observed_sa_bool[i] = np.nan
                        self.Model_nextstate[i] = np.nan
                        self.Model_reward[i] = np.nan

            self.Q.append(Q_abstract)
            self.V.append(V_abstract)

        self.state_counts = np.zeros((self.env.num_states, 1))

    # sample options at each level
    def sample_options_within_level(self, state, level, exploration):
        if exploration == 'epsilon-greedy':
            if np.random.random() < self.epsilon:
                option = np.random.randint(0, self.env.num_options)
            else:
                Q_level = self.Q[level]
                print(state, Q_level)
                # Equivalent to action = np.argmax(Q_primitive[state, :]), 
                # but handles the case with multiple max Q actions
                best_options = np.argwhere(Q_level[state, :] == np.amax(Q_level[state, :]))
                best_options = best_options.flatten().tolist()
                option = random.choice(best_options)

        elif exploration == 'boltzmann':
            Q_level = self.Q[level]
            Qsa = Q_level[state, :]
            # A more safe and stable implementation of softmax 
            Qsa_shifted = Qsa - np.max(Qsa)
            transformed_Qsa = Qsa_shifted /self.tau
            Q_num = np.exp(transformed_Qsa)
            Q_denom = np.sum(Q_num)
            Q_dist = Q_num / Q_denom

            print("Q_dist: ", Q_dist)

            option_list = np.arange(self.env.num_options)
            option = random.choices(option_list, weights = Q_dist, k=1)
            option = option[0]
        
        else:
            print("Please check chosen exploration strategy")
            sys.exit()

        return option
    
    def choose_hlevel(self, state):
        # fill this value based arbitration (corollary of the cost-benefit scheme)
        level_indices = self.env.swo_table[state]
        value_arr = np.zeros_like(level_indices)
        for i_level in level_indices:
            Q_abstract = self.Q[i_level]
            value_arr[i_level] = max(Q_abstract[state, :])
        value_arr_shifted = value_arr - np.max(value_arr)
        transformed_value_arr = value_arr_shifted * self.selectivity
        v_num = np.exp(transformed_value_arr)
        v_denom = np.sum(v_num)
        v_dist = v_num / v_denom

        chosen_level = random.choices(level_indices, weights = v_dist, k=1)
        # return self.preferred_level
        return chosen_level[0]

    def train(self, maxeps):
        self.reset_values()
        self.ep_scores = np.zeros(maxeps)
        self.ep_steps = np.zeros(maxeps)
        
        self.values_per_episode = np.zeros((self.env.levels, self.env.num_states, maxeps))
        for i_level in range(len(self.values_per_episode)):
            level_mdp = self.env.stacked_mdp[i_level]
            for j_state in range(len(level_mdp)):
                if (np.any(np.isnan(level_mdp[j_state]))):
                    self.values_per_episode[i_level,j_state,:] = np.nan
        
        self.Qvalues_per_episode = np.zeros((self.env.levels, self.env.num_states, self.env.num_options, maxeps))
        for i_level in range(len(self.Qvalues_per_episode)):
            level_mdp = self.env.stacked_mdp[i_level]
            for j_state in range(len(level_mdp)):
                for k_option in range(self.env.num_options):
                    if (np.any(np.isnan(level_mdp[j_state]))):
                        self.Qvalues_per_episode[i_level,j_state,k_option,:] = np.nan

        self.num_reachgoal_phase1 = 0
        self.num_reachgoal1_phase1 = 0
        self.num_reachgoal2_phase1 = 0
        self.num_totalep_phase1 = 0
        self.num_reachgoal_phase2 = 0
        self.num_reachgoal1_phase2 = 0
        self.num_reachgoal2_phase2 = 0
        self.num_totalep_phase2 = 0
        self.go_nogo_epwiselist = []
        self.two_choice_epwiselist = []

        for i_episode in range(maxeps):
            print("\rEpisode {}/{}, epsilon: {}".format(i_episode, maxeps, self.epsilon), end="")
            sys.stdout.flush() 


            ## flip reward at episode 200
            if i_episode == self.flip_episode:
                self.env.flip_reward(self.flip_reward_value)

            state = self.env.reset()

            ep_score = 0
            j_step = 0
            done = False
            while not done:
                print("\nstate = ", state)     

                if (self.level_arbitration):
                    hlevel = self.choose_hlevel(state)
                else:
                    hlevel = self.preferred_level
                    # if i_episode<50:
                    #     hlevel = 2
                    # else: 
                    #     print("top Q : ", self.Q[2])
                    #     hlevel = 1
                print("hlevel = ", hlevel)

                option = self.sample_options_within_level(state, hlevel, self.exploration_strategy)
                print("option = ", option)
                next_state_list, reward_list, done = self.env.take_hierarchical_option(state, option, hlevel)
                print("took hoption")
                print(next_state_list)
                # Update cycle starts here

                primitive_start_state = state
                primitive_end_state = next_state_list[-1]
                print("primitive_end_state = ", primitive_end_state)
                
                # update top level model for Dyna
                print("Observed_sa_bool = ", self.Observed_sa_bool)
                if (hlevel == self.env.levels-1):
                    self.Observed_sa_bool[primitive_start_state, option] = 1
                    self.Model_nextstate[primitive_start_state,option] = primitive_end_state
                    self.Model_reward[primitive_start_state,option] = sum(reward_list)

                # loop through all levels 
                
                if self.update_lower_levels_only:
                    level_range = hlevel+1
                else:
                    level_range = self.env.levels

                for level in range(level_range):    # This loop is bottom up updates 
                    print("LEVEL = ", level)
                    state_l = primitive_start_state

                    for i in range(len(reward_list)):
                        next_state_l = next_state_list[i]
                        r = reward_list[i]
                        if (level != self.env.levels - 1):
                            r_nextlevel = reward_list[i] # This needs to be checked
                        
                        print("state_l = ", state_l, "next_state_l = ", next_state_l, "r = ", r)

                        Q_abstract = self.Q[level]
                        Qsa = Q_abstract[state_l, option] # base assumption is that options and primitive actions align
                        
                        print("Q_abstract: ", Q_abstract)

                        if np.isnan(Qsa):   # if the abstract state doesn't exist, then skip
                            print("skipped state_l ", state_l)
                            state_l = next_state_l
                            continue
                        
                        if (level == self.env.levels - 1):
                            next_abstract_state_l = self.env.any_level_next_state(state_l, option, self.env.levels-1)
                            Vsa_nextstate = max(Q_abstract[next_abstract_state_l, :])
                        else: 
                            Q_abstract_nextlevel = self.Q[level+1] 
                            state_mapper = self.env.abstract_state_mapping[level]
                            mapped_next_level_state = int(state_mapper[state_l])
                            print("mapped_next_level_state = ", mapped_next_level_state)
                            Qsa_nextlevel = Q_abstract_nextlevel[mapped_next_level_state, option]  
                        
                        
                        if self.directional_D:
                            if option == 1:
                                drug_rew_D = self.D
                            else:
                                drug_rew_D = 0
                        else:
                            drug_rew_D = self.D

                        # compute delta 
                        if not done:
                            
                            if (level == self.env.levels - 1):
                                delta = r + self.gamma*(Vsa_nextstate) - Qsa  + drug_rew_D 
                            else: 
                                delta = r + self.gamma*(Qsa_nextlevel - r_nextlevel) - Qsa + drug_rew_D
                            # should this be gamma or gamma ^ (intermediate primitive steps)
                        else:
                    
                            if (level == self.env.levels - 1):
                                delta = r - Qsa 
                            else: 
                                delta = r - Qsa 

                        # update Q_table of that level
                        Q_abstract[state_l, option] += self.alpha * delta
                        self.Q[level] = Q_abstract
                        print("Q_abstract = ",Q_abstract)

                        if i==0:
                            self.state_counts[state_l] += 1
                            ep_score += r
                            j_step += 1
                        self.values_per_episode[level, state_l, i_episode] = max(Q_abstract[state_l, :])
                        for k_option in range(self.env.num_options):
                            self.Qvalues_per_episode[level, state_l, k_option, i_episode] = Q_abstract[state_l, k_option]
                            # this is not attaching for the level not in the hlevel loop!

                        # end miniloop over reward_list
                        state_l = next_state_l

                # After exit from the update loop
                state = primitive_end_state

                # Do dyna planning on top level
                if (hlevel == self.env.levels-1):
                    for n in range(self.N_dyna):
                        s_rand = 0
                        a_rand = random.randint(0,self.env.num_options-1)
                        if a_rand == 1:
                            model_drug_rew_D = 2
                        else:
                            model_drug_rew_D = 0

                        next_abstract_state_l = int(self.Model_nextstate[s_rand, a_rand])
                        r = self.Model_reward[s_rand,a_rand]
                        Vsa_nextstate = max(Q_abstract[next_abstract_state_l, :])
                        print("DYNAAAAAA")
                        if a_rand ==1:
                            print(next_abstract_state_l, Vsa_nextstate)
                        if a_rand ==1:
                            # delta = r + self.gamma*(Vsa_nextstate) - self.Q[hlevel][s_rand, a_rand] + model_drug_rew_D
                            delta = r + max(self.Q[hlevel][6, :]) - self.Q[hlevel][s_rand, a_rand] + model_drug_rew_D
                            self.Q[hlevel][s_rand, a_rand] += self.alpha * delta
                        else:
                            delta = r + max(self.Q[hlevel][s_rand, :]) - self.Q[hlevel][s_rand, a_rand] + model_drug_rew_D
                            self.Q[hlevel][s_rand, a_rand] += self.alpha * delta
                        if a_rand == 1:
                            print(delta)
                        print(self.Q[hlevel][s_rand, :])

                        # s_rand = random.randint(0,self.env.num_states-1)
                        # a_rand = random.randint(0,self.env.num_options-1)
                        # while(self.Observed_sa_bool[s_rand, a_rand] == 0 or np.isnan(self.Observed_sa_bool[s_rand, a_rand])):
                        #     s_rand = random.randint(0,self.env.num_states-1)
                        #     a_rand = random.randint(0,self.env.num_options-1)
                        
                        # Q_abstract = self.Q[hlevel]
                        # Qsa = Q_abstract[s_rand, a_rand] # base assumption is that options and primitive actions align
                        
                        # next_abstract_state_l = int(self.Model_nextstate[s_rand, a_rand])
                        # r = self.Model_reward[s_rand,a_rand]
                        # Vsa_nextstate = max(Q_abstract[next_abstract_state_l, :])
                        # # print("Vsa_nextstate --", Vsa_nextstate)
                        # done = self.env.check_if_end_state(s_rand)

                        # if self.directional_D:
                        #     if a_rand == 1:
                        #         model_drug_rew_D = self.D
                        #     else:
                        #         model_drug_rew_D = 0
                        # else:
                        #     model_drug_rew_D = self.D
                        # print("LEVEL IS :   ",  level)
                        # print(s_rand, a_rand)
                        # print("model_drug_rew_D: ", model_drug_rew_D)

                        # print("Q(start, go): ", self.Q[2][0, 1])

                        # if not done:
                        #     if self.model_dysfunctional:
                        #         print(Qsa)
                        #         delta = r + self.gamma*(Vsa_nextstate) - Qsa
                        #         print("DELTA: ", delta)
                        #         delta += model_drug_rew_D
                        #         print("DELTA: ", delta)
                                
                        #     else:
                        #         delta = r + self.gamma*(Vsa_nextstate) - Qsa
                        # else:
                        #     delta = r - Qsa
                            
                        # # print("CHECK THIS -- ", s_rand, delta, r, Qsa)
                        # Q_abstract[s_rand, a_rand] += self.alpha * delta
                        # self.Q[level] = Q_abstract        
                        # print("Q(start, go): ", self.Q[2][0, 1])

            self.ep_scores[i_episode] =  ep_score
            self.ep_steps[i_episode] = j_step
            self.add_to_stats(i_episode, state)

        if self.plot_each_run:
            self.plot_hvalues()
            self.plot_solvestats()

        if self.env.task_name == "GoNoGoTask":
            self.barstats_phase1 = [self.num_reachgoal_phase1/self.num_totalep_phase1, 1-(self.num_reachgoal_phase1/self.num_totalep_phase1)]
            self.barstats_phase2 = [self.num_reachgoal_phase2/self.num_totalep_phase2, 1-(self.num_reachgoal_phase2/self.num_totalep_phase2)]
        elif self.env.task_name == "TwoChoiceTask":
            self.barstats_phase1 = [self.num_reachgoal1_phase1/self.num_totalep_phase1, self.num_reachgoal2_phase1/self.num_totalep_phase1]
            self.barstats_phase2 = [self.num_reachgoal1_phase2/self.num_totalep_phase2, self.num_reachgoal2_phase2/self.num_totalep_phase2]
        else:
            print("Unregistered environment")
            sys.exit()

    def train_nruns(self, maxeps=400, nruns=1):
        self.values_list = []
        self.qvalues_list = []
        self.barstats_phase1_list = []
        self.barstats_phase2_list = []
        for i in range(nruns):
            self.train(maxeps)
            self.values_list.append(self.values_per_episode)
            self.qvalues_list.append(self.Qvalues_per_episode)
            self.barstats_phase1_list.append(self.barstats_phase1)
            self.barstats_phase2_list.append(self.barstats_phase2)
        self.values_list = np.array(self.values_list)
        self.qvalues_list = np.array(self.qvalues_list)
        self.values_avg = np.mean(self.values_list, axis=0)
        self.qvalues_avg = np.mean(self.qvalues_list, axis=0)
        self.values_std = np.std(self.values_list, axis=0)
        self.qvalues_std = np.std(self.qvalues_list, axis=0)
        self.barstats_phase1_list = np.array(self.barstats_phase1_list)
        self.barstats_phase2_list = np.array(self.barstats_phase2_list)
        self.barstats_phase1_avg = np.mean(self.barstats_phase1_list, axis=0)
        self.barstats_phase2_avg = np.mean(self.barstats_phase2_list, axis=0)
        self.barstats_phase1_std = np.std(self.barstats_phase1_list, axis=0)
        self.barstats_phase2_std = np.std(self.barstats_phase2_list, axis=0)
        if self.plot_intermediate_outputs:
            self.plot_hvalues_avg()
            self.plot_solvestats_avg()

    def add_to_stats(self, i_episode, state):
        if self.env.task_name == "GoNoGoTask":
            if i_episode<self.flip_episode:
                self.num_totalep_phase1 +=1
                if state==self.env.finalstate: ## write a reach goal func depending on task i/p
                    self.num_reachgoal_phase1 +=1
                    self.go_nogo_epwiselist.append(1)
                else:
                    self.go_nogo_epwiselist.append(0)
            else:
                self.num_totalep_phase2 +=1
                if state==self.env.finalstate:
                    self.num_reachgoal_phase2 +=1
                    self.go_nogo_epwiselist.append(1)
                else:
                    self.go_nogo_epwiselist.append(0)
        elif self.env.task_name == "TwoChoiceTask":
            if i_episode<self.flip_episode:
                self.num_totalep_phase1 +=1
                if state==self.env.finalstate1: # food
                    self.num_reachgoal1_phase1 +=1
                    self.two_choice_epwiselist.append(-1) 
                elif state==self.env.finalstate2: # drug
                    self.num_reachgoal2_phase1 +=1
                    self.two_choice_epwiselist.append(1)
                else:
                    self.two_choice_epwiselist.append(0)
            else:
                self.num_totalep_phase2 +=1
                if state==self.env.finalstate1: # food
                    self.num_reachgoal1_phase2 +=1
                    self.two_choice_epwiselist.append(-1) 
                elif state==self.env.finalstate2: # drug
                    self.num_reachgoal2_phase2 +=1
                    self.two_choice_epwiselist.append(1)
                else:
                    self.two_choice_epwiselist.append(0)
        else:
            print("Unregistered environment")
            sys.exit()
    
    def plot_hvalues(self):
        # plot value of the start state for each h level 
        for level in range(self.env.levels):
            plt.plot(self.values_per_episode[level, self.env.reset_state, :], label = 'Level '+str(level))
        plt.legend()
        plt.title("Value of the start state at different levels of the hierarchy")
        plt.show()

        # plot Q value of each option at the start state for each h level 
        for option in range(self.env.num_options):
            for level in range(self.env.levels):
                plt.plot(self.Qvalues_per_episode[level, self.env.reset_state, option, :], label = 'Level '+str(level))
            plt.legend()
            plt.title("Q value of option " + str(option) +" at the start state at different levels of the hierarchy")
            plt.show()

    def plot_hvalues_avg(self):
        color_list = ['blue', 'green', 'orange']

        # plot value of the start state for each h level 
        for level in range(self.env.levels):
            y = self.values_avg[level, self.env.reset_state, :]
            err = self.values_std[level, self.env.reset_state, :]
            eps_count = len(y)
            t = np.arange(eps_count)
            plt.plot(y, label = 'Level '+str(level), color=color_list[level])
            plt.fill_between(t, y-err, y+err, alpha=0.5, color=color_list[level])
        plt.legend()
        plt.title("Value of the start state at different levels of the hierarchy")
        plt.show()

        # plot Q value of each option at the start state for each h level 
        for option in range(self.env.num_options):
            for level in range(self.env.levels):
                y = self.qvalues_avg[level, self.env.reset_state, option, :]
                err = self.qvalues_std[level, self.env.reset_state, option, :]
                eps_count = len(y)
                t = np.arange(eps_count)
                plt.plot(y, label = 'Level '+str(level), color=color_list[level])
                plt.fill_between(t, y-err, y+err, alpha=0.5, color=color_list[level])
            plt.legend()
            plt.ylim(-8,+18)
            plt.title("Q value of option " + str(option) +" at the start state at different levels of the hierarchy")
            plt.show()

    def plot_solvestats(self):
        if self.env.task_name == "GoNoGoTask":
            plt.plot(self.go_nogo_epwiselist)
            plt.title("Episode-wise Go(1) - NoGo(0) result")
            plt.show()

            result = ['Go', 'NoGo']
            barstats = [self.num_reachgoal_phase1/self.num_totalep_phase1, 1-(self.num_reachgoal_phase1/self.num_totalep_phase1)]
            plt.bar(result,barstats)
            plt.title("Before pairing the goal state with punishment")
            plt.show()

            result = ['Go', 'NoGo']
            barstats = [self.num_reachgoal_phase2/self.num_totalep_phase2, 1-(self.num_reachgoal_phase2/self.num_totalep_phase2)]
            plt.bar(result,barstats)
            plt.title("After pairing the goal state with punishment")
            plt.show()
        elif self.env.task_name == "TwoChoiceTask":
            plt.plot(self.two_choice_epwiselist)
            plt.title("Episode-wise Food(-1) - NoGo(0) - Drug (1) result")
            plt.show()

            # result = ['Food', 'Drug', 'NoGo']
            # barstats = [self.num_reachgoal1_phase1/self.num_totalep_phase1, self.num_reachgoal2_phase1/self.num_totalep_phase1, 1 -(self.num_reachgoal1_phase1/self.num_totalep_phase1) -(self.num_reachgoal2_phase1/self.num_totalep_phase1)]
            result = ['Food', 'Drug']
            barstats = [self.num_reachgoal1_phase1/self.num_totalep_phase1, self.num_reachgoal2_phase1/self.num_totalep_phase1]
            
            plt.bar(result,barstats)
            plt.title("Before pairing the goal state with punishment")
            plt.show()

            # result = ['Food', 'Drug', 'NoGo']
            # barstats = [self.num_reachgoal1_phase2/self.num_totalep_phase2, self.num_reachgoal2_phase2/self.num_totalep_phase2, 1 -(self.num_reachgoal1_phase2/self.num_totalep_phase2) -(self.num_reachgoal2_phase2/self.num_totalep_phase2)]
            result = ['Food', 'Drug']
            barstats = [self.num_reachgoal1_phase2/self.num_totalep_phase2, self.num_reachgoal2_phase2/self.num_totalep_phase2]
            plt.bar(result,barstats)
            plt.title("After pairing the goal state with punishment")
            plt.show()
        else:
            print("Unregistered environment")
            sys.exit()

    def plot_solvestats_avg(self):
        if self.env.task_name == "GoNoGoTask":
            plt.title("Episode-wise Go - NoGo result")
            result = ['Go', 'NoGo']
            plt.bar(result,self.barstats_phase1_avg, yerr=self.barstats_phase1_std)
            plt.title("Before pairing the goal state with punishment")
            plt.ylabel('Choice aggregate (normalized)')
            plt.ylim([0,1.2])
            plt.show()

            plt.title("Episode-wise Go - NoGo result")
            result = ['Go', 'NoGo']
            plt.bar(result,self.barstats_phase2_avg, yerr=self.barstats_phase2_std)
            plt.ylabel('Choice aggregate (normalized)')
            plt.ylim([0,1.2])
            plt.title("After pairing the goal state with punishment")
            plt.show()
        elif self.env.task_name == "TwoChoiceTask":
            plt.title("Episode-wise Food - Drug result")
            result = ['Food', 'Drug']
            plt.bar(result,self.barstats_phase1_avg, yerr=self.barstats_phase1_std)
            plt.ylabel('Choice aggregate (normalized)')
            plt.title("Before pairing the goal state with punishment")
            plt.ylim([0,1.2])
            plt.show()

            plt.title("Episode-wise Food - Drug result")
            result = ['Food', 'Drug']
            plt.bar(result,self.barstats_phase2_avg, yerr=self.barstats_phase2_std)
            plt.ylabel('Choice aggregate (normalized)')
            plt.title("After pairing the goal state with punishment")
            plt.ylim([0,1.2])
            plt.show()
        else:
            print("Unregistered environment")
            sys.exit()