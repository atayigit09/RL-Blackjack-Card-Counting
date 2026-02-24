import numpy as np
from collections import defaultdict
import config as cfg
from agents.base_agent import BaseAgent

class SARSAAgent(BaseAgent):
    """
    On-policy tabular SARSA.
    Learns Q for the epsilon-greedy policy it's actually following.
    """
    def __init__(self, alpha=cfg.ALPHA, gamma=cfg.GAMMA,
                 epsilon=cfg.EPSILON_START, epsilon_min=cfg.EPSILON_END,
                 epsilon_decay=cfg.EPSILON_DECAY):
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: defaultdict(float))
        self._next_action = None   # SARSA needs a' chosen before update

    def select_action(self, state, legal_actions, explore=True):
        if explore and np.random.random() < self.epsilon:
            return np.random.choice(legal_actions)
        return max(legal_actions, key=lambda a: self.Q[state][a])

    def update(self, state, action, reward, next_state, next_legal_actions, done):
        current_q = self.Q[state][action]
        if done:
            target = reward
        else:
            # On-policy: bootstrap from the action we're actually going to take
            next_action = self.select_action(next_state, next_legal_actions)
            self._next_action = next_action
            target = reward + self.gamma * self.Q[next_state][next_action]
        self.Q[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_policy(self, state, legal_actions):
        return max(legal_actions, key=lambda a: self.Q[state][a])
