import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config as cfg
from env.blackjack_env import BlackjackEnv
from env.basic_strategy import basic_strategy_action
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.random_agent import RandomAgent

STRATEGIES = {
    'basic': basic_strategy_action,
    'random': lambda state, legal: RandomAgent().select_action(state, legal, explore=False),
    'q': None,   # will be set below if Q-table is loaded
    'sarsa': None, # will be set below if SARSA Q-table is loaded
}

def print_state(state, tokens, bet, hand_num):
    print(f"\n--- Hand {hand_num} ---")
    print(f"Tokens: {tokens}")
    print(f"Player total: {state.player_total} (Usable Ace: {state.usable_ace})")
    print(f"Dealer upcard: {state.dealer_upcard}")
    print(f"Legal actions: {[cfg.ACTION_NAMES[a] for a in env.legal_actions(state)]}")
    print(f"Current bet: {bet}")

def main():
    parser = argparse.ArgumentParser(description="Simulate Blackjack play with a given strategy.")
    parser.add_argument('--strategy', type=str, default='basic', choices=['basic', 'random', 'q', 'sarsa'], help='Strategy to use')
    parser.add_argument('--delay', type=float, default=1.0, help='Seconds between plays')
    parser.add_argument('--tokens', type=int, default=1000, help='Starting tokens')
    parser.add_argument('--bet', type=int, default=10, help='Bet per hand')
    parser.add_argument('--qtable', type=str, default=None, help='Path to Q-table pickle (for q/sarsa)')
    args = parser.parse_args()

    global env
    env = BlackjackEnv()
    tokens = args.tokens
    bet = args.bet
    hand_num = 1

    # Setup agent/strategy
    if args.strategy == 'q':
        agent = QLearningAgent()
        if args.qtable:
            import pickle
            with open(args.qtable, 'rb') as f:
                agent.Q = pickle.load(f)
        STRATEGIES['q'] = lambda state, legal: agent.select_action(state, legal, explore=False)
    elif args.strategy == 'sarsa':
        agent = SARSAAgent()
        if args.qtable:
            import pickle
            with open(args.qtable, 'rb') as f:
                agent.Q = pickle.load(f)
        STRATEGIES['sarsa'] = lambda state, legal: agent.select_action(state, legal, explore=False)

    strategy_fn = STRATEGIES[args.strategy]

    print(f"Starting simulation with {tokens} tokens, bet {bet} per hand, strategy: {args.strategy}\n")
    while tokens >= bet:
        state = env.reset()
        print_state(state, tokens, bet, hand_num)
        done = False
        total_reward = 0.0
        while not done:
            legal = env.legal_actions(state)
            if args.strategy == 'basic':
                action = strategy_fn(state)
            else:
                action = strategy_fn(state, legal)
            print(f"Action: {cfg.ACTION_NAMES[action]}")
            next_state, reward, done = env.step(action)
            state = next_state if not done else state
            time.sleep(args.delay)
        tokens += int(reward * bet)
        print(f"Result: {'Win' if reward > 0 else 'Loss' if reward < 0 else 'Push'} | Reward: {reward} | Tokens left: {tokens}")
        hand_num += 1
        time.sleep(args.delay)
        if tokens < bet:
            print("\nOut of tokens! Simulation ended.")
            break

if __name__ == "__main__":
    main()
