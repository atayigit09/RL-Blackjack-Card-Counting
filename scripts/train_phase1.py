import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from training.train import train
from evaluation.evaluate import compare_policies, policy_agreement_rate

def smooth(x, window=10_000):
    return np.convolve(x, np.ones(window)/window, mode='valid')

if __name__ == "__main__":
    print("=== Phase 1: Tabular Q-Learning ===\n")

    agent = QLearningAgent()   # swap to SARSAAgent() to compare
    rewards = train(agent, n_episodes=cfg.N_EPISODES)


    # Helper to recursively convert defaultdicts to dicts
    def recursive_dict(d):
        if isinstance(d, dict):
            return {k: recursive_dict(v) for k, v in d.items()}
        else:
            return d

    # Save Q-table
    os.makedirs("results", exist_ok=True)
    with open("results/q_table_phase1.pkl", "wb") as f:
        pickle.dump(recursive_dict(agent.Q), f)
    print("Q-table saved to results/q_table_phase1.pkl")

    # Evaluate
    compare_policies(agent)
    policy_agreement_rate(agent)

    # Plot learning curve
    smoothed = smooth(rewards)
    plt.figure(figsize=(10, 4))
    plt.plot(smoothed, linewidth=0.8)
    plt.axhline(y=-0.005, color='red', linestyle='--', label='Basic strategy target (−0.5%)')
    plt.xlabel("Episode")
    plt.ylabel("Expected Value (smoothed)")
    plt.title("Phase 1: Q-Learning Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/learning_curve_phase1.png", dpi=150)
    print("Learning curve saved to results/learning_curve_phase1.png")
