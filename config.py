# ── Shoe ──────────────────────────────────────────────────────────────────
# 6-deck Vegas rules. Dealer stands on soft 17. Blackjack pays 3:2.
# Double after split and resplitting pairs (up to 3 splits, 4 hands total) supported.
NUM_DECKS        = 6
RESHUFFLE_PENETRATION = 0.75   # reshuffle when 75% of shoe is dealt

# ── Rewards ───────────────────────────────────────────────────────────────
REWARD_BLACKJACK  =  1.5
REWARD_WIN        =  1.0
REWARD_PUSH       =  0.0
REWARD_SURRENDER  = -0.5
REWARD_LOSS       = -1.0

# ── Actions ───────────────────────────────────────────────────────────────
HIT       = 0
STAND     = 1
DOUBLE    = 2
SPLIT     = 3
SURRENDER = 4
ACTION_NAMES = {0: "Hit", 1: "Stand", 2: "Double", 3: "Split", 4: "Surrender"}

# ── Training ──────────────────────────────────────────────────────────────

# Increased training episodes and slower epsilon decay for better exploration
N_EPISODES   = 20_000_000
ALPHA        = 0.1      # learning rate
GAMMA        = 1.0      # no discounting — single-stage decision
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.9999998   # slower decay, more exploration

# ── Evaluation ────────────────────────────────────────────────────────────
N_EVAL_EPISODES = 1_000_000
