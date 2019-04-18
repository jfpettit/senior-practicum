from collections import defaultdict
import numpy as np

class monte_carlo_policy_iteration:
    def __init__(self, num_states, environment, gamma):
        self.num_states = num_states
        self.env = environment()
        self.gamma = gamma
    
    def find_first_visit(self, state, states):
        for i in range(len(states)):
            if states[i] == state:
                return i

    def generate_episode(self, policy, time_steps):
        episode, state = [], self.env._reset()
        for time_step in range(time_steps):
            action_choice = policy(state[0])
            new_state, reward, over, _ = self.env._step(action_choice)
            episode.append([state, action_choice, reward])
            if over:
                break
            state = new_state
        return episode
    
    def get_control_episode(self, time_steps, Q_, epsilon):
        episode, state = [], self.env._reset()
        policy = self.make_policy(Q_, epsilon, self.env.action_space.n)
        for time_step in range(time_steps):
            pi = policy(state)
            action_choice = np.random.choice(np.arange(len(pi)), p=pi)
            new_state, reward, over, _ = self.env._step(action_choice)
            episode.append([state, action_choice, reward])
            if over:
                break
            state = new_state
        return episode, policy
    
    def first_visit_MC_prediction(self, policy, epochs):
        V_ = defaultdict(float)
        returns = defaultdict(float)
        count = defaultdict(float)
        
        for ep in range(epochs):
            episode = self.generate_episode(policy, 1000)
            states = [tuple(x[0]) for x in episode]
            for state in states:
                first = self.find_first_visit(state, states)
                G = sum([x[2]*(self.gamma**i) for i,x in enumerate(episode[first:])])
                returns[state] += G
                count[state] += 1
                V_[state] = returns[state] / count[state]
        return V_
    
    def make_policy(self, Q_, epsilon, num_actions):
        def policy(obs):
            pi = np.ones(num_actions, dtype=float) * epsilon / num_actions
            optimal_action = np.argmax(Q_[obs])
            pi[optimal_action] += 1 - epsilon
            return pi
        return policy
    
    def state_action_MC_control(self, epochs, epsilon):
        Q_win_rate, pi_win_rate = [], []
        Q_wr_10, pi_wr_10 = [], []
        Q_ = defaultdict(lambda: np.ones(self.env.action_space.n))
        returns = defaultdict(float)
        count = defaultdict(float)
        
        for ep in range(epochs):
            episode, pi = self.get_control_episode(1000, Q_, epsilon)
            states = [(x[0], x[1]) for x in episode]
            for state, action in states:
                state_action = (state, action)
                first = self.find_first_visit(state_action, states)
                G = sum([x[2]*(self.gamma**i) for i,x in enumerate(episode[first:])])
                returns[state_action] += G
                count[state_action] += 1
                Q_[state][action] = returns[state_action] / count[state_action]
            if ep % 100000 == 0:
                Q_win_rate.append(self.record_win_rate_q_func(Q_, 10000))
                pi_win_rate.append(self.record_win_rate_policy(pi, 10000))
            if ep in [0, 10, 100, 1000, 10000, 100000, 100000-1]:
                Q_wr_10.append(self.record_win_rate_q_func(Q_, 10000))
                pi_wr_10.append(self.record_win_rate_policy(pi, 10000))
        return Q_, Q_win_rate, Q_wr_10, pi, pi_win_rate, pi_wr_10
    
    def record_win_rate_policy(self, policy, num_games):
        player_wins, dealer_wins, draw = 0, 0, 0
        for game in range(num_games):
            s_ = blckjck._reset()
            for i in range(100):
                pi = policy(s_)
                action_choice = np.argmax(pi)
                new_state, reward, over, _ = blckjck._step(action_choice)
                if over:
                    break
                s_ = new_state
            if reward == 1: player_wins += 1
            elif reward == -1: dealer_wins += 1
            else: draw += 1
        return player_wins/num_games
    
    def record_win_rate_q_func(self, q, num_games):
        player_wins, dealer_wins, draw = 0, 0, 0
        for game in range(num_games):
            s_ = blckjck._reset()
            for i in range(100):
                q_s = q[s_]
                action_choice = np.argmax(q_s)
                new_state, reward, over, _ = blckjck._step(action_choice)
                if over:
                    break
                s_ = new_state
            if reward == 1: player_wins += 1
            elif reward == -1: dealer_wins += 1
            else: draw += 1
        return player_wins/num_games