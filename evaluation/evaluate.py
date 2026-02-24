import numpy as np
from tqdm import tqdm
from env.blackjack_env import BlackjackEnv
from env.basic_strategy import basic_strategy_action
import config as cfg

def compute_ev(agent, n_episodes=cfg.N_EVAL_EPISODES, use_basic_strategy=False):
    """
    Run n_episodes with no exploration. Returns expected value per hand.
    If use_basic_strategy=True, ignores agent and uses the lookup table instead.
    """
    env = BlackjackEnv()
    total = 0.0

    for _ in tqdm(range(n_episodes), desc="Evaluating", leave=False):
        state = env.reset()
        while True:
            legal = env.legal_actions(state)
            if use_basic_strategy:
                action = basic_strategy_action(state, legal_actions=legal)
            else:
                action = agent.select_action(state, legal, explore=False)
            next_state, reward, done = env.step(action)
            total += reward
            if done:
                break
            state = next_state

    return total / n_episodes

def compare_policies(agent, n_episodes=cfg.N_EVAL_EPISODES):
    """Print EV comparison table."""
    print("\n── Policy Comparison ──────────────────────────────")
    
    from agents.random_agent import RandomAgent
    random_agent = RandomAgent()
    
    ev_random = compute_ev(random_agent, n_episodes)
    ev_basic  = compute_ev(None, n_episodes, use_basic_strategy=True)
    ev_learned = compute_ev(agent, n_episodes)

    print(f"  Random policy      EV: {ev_random:+.4f}  ({ev_random*100:+.2f}%)")
    print(f"  Basic strategy     EV: {ev_basic:+.4f}  ({ev_basic*100:+.2f}%)")
    print(f"  Learned policy     EV: {ev_learned:+.4f}  ({ev_learned*100:+.2f}%)")
    print(f"  Target             EV: −0.0050  (−0.50%)")
    print(f"\n  Gap from optimal: {abs(ev_learned - ev_basic)*100:.3f}%")
    print("────────────────────────────────────────────────────\n")
    return ev_random, ev_basic, ev_learned

def policy_agreement_rate(agent, n_samples=10_000):
    """
    What % of states does the learned policy agree with basic strategy?
    Samples random states and compares decisions.
    """
    env = BlackjackEnv()
    matches = 0

    for _ in range(n_samples):
        state = env.reset()
        legal = env.legal_actions(state)
        learned = agent.select_action(state, legal, explore=False)
        optimal = basic_strategy_action(state)
        if optimal not in legal:
            optimal = cfg.HIT
        if learned == optimal:
            matches += 1

    rate = matches / n_samples
    print(f"  Policy agreement with basic strategy: {rate*100:.1f}%")
    return rate
