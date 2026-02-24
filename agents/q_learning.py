import numpy as np
from collections import defaultdict
import config as cfg
from agents.base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    """
    Off-policy tabular Q-Learning.
    Learns the optimal Q* regardless of the exploration policy.
    """
    def __init__(self, alpha=cfg.ALPHA, gamma=cfg.GAMMA,
                 epsilon=cfg.EPSILON_START, epsilon_min=cfg.EPSILON_END,
                 epsilon_decay=cfg.EPSILON_DECAY):
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Q[state][action] — defaults to 0.0
        # Optimistic Q-value initialization to 0.5
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.5))

    def select_action(self, state, legal_actions, explore=True):
        if explore and np.random.random() < self.epsilon:
            return np.random.choice(legal_actions)   # explore
        # Exploit: pick legal action with highest Q value
        return max(legal_actions, key=lambda a: self.Q[state][a])

    def update(self, state, action, reward, next_state, next_legal_actions, done):
        current_q = self.Q[state][action]
        if done:
            target = reward
        else:
            # Off-policy: bootstrap from max Q of next state
            best_next = max(self.Q[next_state][a] for a in next_legal_actions)
            target = reward + self.gamma * best_next
        self.Q[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_policy(self, state, legal_actions):
        return max(legal_actions, key=lambda a: self.Q[state][a])
