"""
Published basic strategy for 6-deck, S17, late surrender.
Used as ground-truth benchmark to evaluate the learned policy.
Keys: (player_total, dealer_upcard, usable_ace, is_pair)
Values: optimal action
"""
import config as cfg


# Split table (pairs): (pair_value, dealer_upcard) → action
SPLIT = {
    # Always split Aces and 8s
    **{(11, u): cfg.SPLIT for u in range(2, 12)},
    **{(8, u): cfg.SPLIT for u in range(2, 12)},
    # Never split 5s, 10s
    **{(5, u): cfg.HIT for u in range(2, 12)},
    **{(10, u): cfg.STAND for u in range(2, 12)},
    # 2s and 3s: split vs 2-7, otherwise hit
    **{(2, u): cfg.SPLIT if u in range(2, 8) else cfg.HIT for u in range(2, 12)},
    **{(3, u): cfg.SPLIT if u in range(2, 8) else cfg.HIT for u in range(2, 12)},
    # 4s: split vs 5-6, otherwise hit
    **{(4, u): cfg.SPLIT if u in [5, 6] else cfg.HIT for u in range(2, 12)},
    # 6s: split vs 2-6, otherwise hit
    **{(6, u): cfg.SPLIT if u in range(2, 7) else cfg.HIT for u in range(2, 12)},
    # 7s: split vs 2-7, otherwise hit
    **{(7, u): cfg.SPLIT if u in range(2, 8) else cfg.HIT for u in range(2, 12)},
    # 9s: split vs 2-6, 8-9; stand vs 7, 10, 11
    **{(9, u): cfg.SPLIT if u in [2,3,4,5,6,8,9] else cfg.STAND for u in range(2, 12)},
}

# Hard totals (usable_ace=False, is_pair=False)
HARD = {
    # (total, upcard): action
    **{(t, u): cfg.HIT for t in range(4, 12) for u in range(2, 12)},
    (12, 2): cfg.HIT,  (12, 3): cfg.HIT,  (12, 4): cfg.STAND,
    (12, 5): cfg.STAND,(12, 6): cfg.STAND,(12, 7): cfg.HIT,
    (12, 8): cfg.HIT,  (12, 9): cfg.HIT,  (12,10): cfg.HIT, (12,11): cfg.HIT,
    **{(t, u): cfg.STAND for t in [13,14,15,16] for u in [2,3,4,5,6]},
    **{(t, u): cfg.HIT   for t in [13,14,15,16] for u in [7,8,9,10,11]},
    (15,10): cfg.SURRENDER, (16, 9): cfg.SURRENDER,
    (16,10): cfg.SURRENDER, (16,11): cfg.SURRENDER,
    **{(t, u): cfg.STAND for t in range(17, 22) for u in range(2, 12)},
    # Doubles (first action)
    (11, 2): cfg.DOUBLE,  (11, 3): cfg.DOUBLE, (11, 4): cfg.DOUBLE,
    (11, 5): cfg.DOUBLE,  (11, 6): cfg.DOUBLE, (11, 7): cfg.DOUBLE,
    (11, 8): cfg.DOUBLE,  (11, 9): cfg.DOUBLE, (11,10): cfg.DOUBLE,
    (11,11): cfg.HIT,
    (10, 2): cfg.DOUBLE,  (10, 3): cfg.DOUBLE, (10, 4): cfg.DOUBLE,
    (10, 5): cfg.DOUBLE,  (10, 6): cfg.DOUBLE, (10, 7): cfg.DOUBLE,
    (10, 8): cfg.DOUBLE,  (10, 9): cfg.DOUBLE,
    (10,10): cfg.HIT,     (10,11): cfg.HIT,
    (9,  3): cfg.DOUBLE,  (9, 4): cfg.DOUBLE, (9, 5): cfg.DOUBLE,
    (9,  6): cfg.DOUBLE,
}

# Soft totals (usable_ace=True)
SOFT = {
    (13, 5): cfg.DOUBLE, (13, 6): cfg.DOUBLE,
    (14, 5): cfg.DOUBLE, (14, 6): cfg.DOUBLE,
    (15, 4): cfg.DOUBLE, (15, 5): cfg.DOUBLE, (15, 6): cfg.DOUBLE,
    (16, 4): cfg.DOUBLE, (16, 5): cfg.DOUBLE, (16, 6): cfg.DOUBLE,
    (17, 3): cfg.DOUBLE, (17, 4): cfg.DOUBLE, (17, 5): cfg.DOUBLE,
    (17, 6): cfg.DOUBLE,
    (18, 3): cfg.DOUBLE, (18, 4): cfg.DOUBLE, (18, 5): cfg.DOUBLE,
    (18, 6): cfg.DOUBLE,
    (18, 2): cfg.STAND,  (18, 7): cfg.STAND, (18, 8): cfg.STAND,
    (18, 9): cfg.HIT,    (18,10): cfg.HIT,   (18,11): cfg.HIT,
    **{(t, u): cfg.STAND for t in [19,20,21] for u in range(2, 12)},
    **{(t, u): cfg.HIT   for t in [13,14,15,16,17] for u in range(2, 12)
       if (t, u) not in {(13,5),(13,6),(14,5),(14,6),(15,4),(15,5),(15,6),
                         (16,4),(16,5),(16,6),(17,3),(17,4),(17,5),(17,6)}},
}

def basic_strategy_action(state, legal_actions=None):
    """Return the basic strategy action for a given State namedtuple. legal_actions is used to mask unavailable actions."""
    t, u, soft, is_pair = state.player_total, state.dealer_upcard, state.usable_ace, state.is_pair

    def clamp(action):
        if legal_actions is None:
            return action
        if action in legal_actions:
            return action
        # Double not available → hit; surrender not available → hit
        if action == cfg.DOUBLE:
            return cfg.HIT
        if action == cfg.SURRENDER:
            return cfg.HIT
        if action == cfg.SPLIT:
            return cfg.HIT
        return action

    # Split logic if pair
    if is_pair:
        pair_val = t // 2 if t // 2 in range(2, 12) else t // 2
        action = SPLIT.get((pair_val, u))
        if action is not None:
            return clamp(action)
    # Soft totals
    if soft:
        action = SOFT.get((t, u))
        if action is not None:
            return clamp(action)
        return cfg.HIT if t < 18 else cfg.STAND
    # Hard totals
    action = HARD.get((t, u))
    if action is not None:
        return clamp(action)
    return cfg.STAND if t >= 17 else cfg.HIT
