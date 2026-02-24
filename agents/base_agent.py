from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state, legal_actions, explore=True):
        """Return action index."""
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, next_legal_actions, done):
        """Update internal estimates after a transition."""
        pass

    def decay_epsilon(self):
        pass
