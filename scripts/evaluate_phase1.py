import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import config as cfg
from agents.q_learning import QLearningAgent
from evaluation.evaluate import compare_policies, policy_agreement_rate

if __name__ == "__main__":
    print("=== Phase 1: Evaluation ===\n")
    agent = QLearningAgent()
    # Load Q-table
    with open("results/q_table_phase1.pkl", "rb") as f:
        agent.Q = pickle.load(f)
    compare_policies(agent)
    policy_agreement_rate(agent)
