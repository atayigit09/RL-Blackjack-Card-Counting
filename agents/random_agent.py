import numpy as np
from agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """Uniform random baseline — picks any legal action with equal probability."""
    def select_action(self, state, legal_actions, explore=True):
        return np.random.choice(legal_actions)

    def update(self, *args, **kwargs):
        pass   # nothing to learn
