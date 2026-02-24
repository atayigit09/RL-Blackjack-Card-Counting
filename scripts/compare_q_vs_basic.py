import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config as cfg
from agents.q_learning import QLearningAgent
from env.basic_strategy import basic_strategy_action

def load_q_table(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main(qtable_path):
    agent = QLearningAgent()
    agent.Q = load_q_table(qtable_path)

    mismatches = []
    total = 0
    agree = 0

    # Collect all states in Q-table
    for state in agent.Q:
        legal = [a for a in agent.Q[state]]
        if not legal:
            continue
        learned = max(legal, key=lambda a: agent.Q[state][a])
        optimal = basic_strategy_action(state, legal_actions=legal)
        if learned == optimal:
            agree += 1
        else:
            mismatches.append((state, learned, optimal, legal))
        total += 1

    print(f"\nTotal unique states in Q-table: {total}")
    print(f"Agreement with basic strategy: {agree/total*100:.2f}%")
    print(f"Mismatches: {len(mismatches)}\n")
    print("Sample mismatches (up to 20):")
    for i, (state, learned, optimal, legal) in enumerate(mismatches[:20]):
        print(f"  State: {state}\n    Learned: {cfg.ACTION_NAMES[learned]} | Basic: {cfg.ACTION_NAMES[optimal]} | Legal: {[cfg.ACTION_NAMES[a] for a in legal]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare learned Q-table policy to basic strategy.")
    parser.add_argument('--qtable', type=str, required=True, help='Path to Q-table pickle file')
    args = parser.parse_args()
    main(args.qtable)