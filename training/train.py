from tqdm import tqdm
from env.blackjack_env import BlackjackEnv
from collections import defaultdict
state_action_counts = defaultdict(int)

def run_episode(env, agent, explore=True):
    """Run one hand. Returns total reward."""
    state = env.reset()
    total_reward = 0.0

    while True:
        legal = env.legal_actions(state)
        action = agent.select_action(state, legal, explore=explore)
        # Log visitation
        state_action_counts[(state, action)] += 1
        next_state, reward, done = env.step(action)
        next_legal = env.legal_actions(next_state) if not done else []
        agent.update(state, action, reward, next_state, next_legal, done)
        total_reward += reward
        if done:
            break
        state = next_state

    return total_reward

def train(agent, n_episodes, log_every=100_000):
    env = BlackjackEnv()
    rewards = []
    running_total = 0.0

    for ep in tqdm(range(1, n_episodes + 1), desc="Training"):
        r = run_episode(env, agent, explore=True)
        running_total += r
        rewards.append(r)
        agent.decay_epsilon()

        if ep % log_every == 0:
            ev = running_total / log_every
            print(f"  Episode {ep:>8,} | EV (last {log_every//1000}k hands): {ev:+.4f} "
                  f"| ε={agent.epsilon:.4f}")
            # Print visitation stats for rare actions
            from config import HIT, STAND, DOUBLE, SPLIT, SURRENDER, ACTION_NAMES
            action_counts = {a: 0 for a in [HIT, STAND, DOUBLE, SPLIT, SURRENDER]}
            for (s, a), count in state_action_counts.items():
                action_counts[a] += count
            print("    Action visitation counts (cumulative):")
            for a in [HIT, STAND, DOUBLE, SPLIT, SURRENDER]:
                print(f"      {ACTION_NAMES[a]:8}: {action_counts[a]:,}")
            running_total = 0.0

    return rewards
