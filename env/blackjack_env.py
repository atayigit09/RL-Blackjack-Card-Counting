
from collections import namedtuple
from env.shoe import Shoe
import config as cfg

# State visible to the agent — only what is strategically meaningful
State = namedtuple("State", [
    "player_total",   # int: current hand total (best non-bust value)
    "dealer_upcard",  # int: dealer's visible card (1-10, ace=11→shown as 11)
    "usable_ace",     # bool: player has an ace counted as 11
    "is_pair",        # bool: player has a pair (for split logic)
])

def hand_total(cards):
    """Return best non-bust total for a list of card values."""
    total = sum(cards)
    aces = cards.count(11)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total

def has_usable_ace(cards):
    return 11 in cards and sum(cards) <= 21

def is_bust(cards):
    return hand_total(cards) > 21



class BlackjackEnv:
    """
    6-deck blackjack environment.
    Rules: dealer stands on soft 17, blackjack pays 3:2, late surrender allowed.
    Double after split and resplitting up to 3 times (4 hands total) supported.
    No card counting — shoe info not exposed in base state.
    """
    def __init__(self):
        self.shoe = Shoe()
        self.dealer_cards = []
        self.split_hands = []  # List of hands to play (each is a list of cards)
        self.current_hand = 0  # Index of current hand in split_hands
        self.doubled = []      # Track if each hand was doubled
        self.surrendered = []  # Track if each hand was surrendered
        self.max_splits = 3    # Standard Vegas: up to 3 splits (4 hands)

    def reset(self):
        first_hand = [self.shoe.deal(), self.shoe.deal()]
        self.split_hands = [first_hand]
        self.current_hand = 0
        self.dealer_cards = [self.shoe.deal(), self.shoe.deal()]
        self.doubled = [False]
        self.surrendered = [False]
        return self._get_state()

    @property
    def player_cards(self):
        return self.split_hands[self.current_hand]

    @player_cards.setter
    def player_cards(self, value):
        self.split_hands[self.current_hand] = value

    def _get_state(self):
        return State(
            player_total=hand_total(self.player_cards),
            dealer_upcard=self.dealer_cards[0],
            usable_ace=has_usable_ace(self.player_cards),
            is_pair=(len(self.player_cards) == 2 and self.player_cards[0] == self.player_cards[1]),
        )

    def legal_actions(self, state=None):
        # Action availability tracked via env state, not the State tuple
        actions = [cfg.HIT, cfg.STAND]
        # Only allow double, surrender, split on first action of a hand
        if len(self.player_cards) == 2:
            actions.append(cfg.DOUBLE)
            actions.append(cfg.SURRENDER)
            if self.player_cards[0] == self.player_cards[1] and len(self.split_hands) <= self.max_splits:
                actions.append(cfg.SPLIT)
        return actions



    def step(self, action):
        assert action in self.legal_actions()

        # Surrender: hand ends immediately
        if action == cfg.SURRENDER:
            self.surrendered[self.current_hand] = True
            reward = cfg.REWARD_SURRENDER
            # End this hand, do not allow further actions
            return self._next_hand(reward)

        # Double: hand ends after one card
        if action == cfg.DOUBLE:
            self.player_cards.append(self.shoe.deal())
            self.doubled[self.current_hand] = True
            # After double, hand always ends (win/loss/push)
            if is_bust(self.player_cards):
                reward = cfg.REWARD_LOSS * 2
            else:
                reward = self._dealer_play() * 2
            return self._next_hand(reward)

        # Split (resplitting supported)
        if action == cfg.SPLIT:
            card = self.player_cards[0]
            self.split_hands[self.current_hand] = [card, self.shoe.deal()]
            self.doubled[self.current_hand] = False
            self.surrendered[self.current_hand] = False
            self.split_hands.append([card, self.shoe.deal()])
            if hasattr(self, 'hand_rewards'):
                self.hand_rewards.append(0.0)
            self.doubled.append(False)
            self.surrendered.append(False)
            return self._get_state(), 0.0, False

        # Hit
        if action == cfg.HIT:
            self.player_cards.append(self.shoe.deal())
            if is_bust(self.player_cards):
                reward = cfg.REWARD_LOSS * (2 if self.doubled[self.current_hand] else 1)
                return self._next_hand(reward)
            if hand_total(self.player_cards) == 21:
                reward = self._dealer_play() * (2 if self.doubled[self.current_hand] else 1)
                return self._next_hand(reward)
            return self._get_state(), 0.0, False

        # Stand
        if action == cfg.STAND:
            reward = self._dealer_play() * (2 if self.doubled[self.current_hand] else 1)
            return self._next_hand(reward)

    def _next_hand(self, reward):
        # Store reward for this hand
        if not hasattr(self, 'hand_rewards'):
            self.hand_rewards = [0.0 for _ in self.split_hands]
        self.hand_rewards[self.current_hand] = reward
        # Move to next hand if any
        self.current_hand += 1
        if self.current_hand < len(self.split_hands):
            return self._get_state(), 0.0, False
        # All hands played: sum rewards
        total = sum(self.hand_rewards)
        del self.hand_rewards
        return None, total, True

    def _dealer_play(self):
        """Dealer draws to 17+, stands on soft 17. Returns reward for current hand."""
        dealer_bj = (hand_total(self.dealer_cards) == 21 and len(self.dealer_cards) == 2)
        player_bj = (hand_total(self.player_cards) == 21 and len(self.player_cards) == 2)

        if player_bj and dealer_bj:
            return cfg.REWARD_PUSH
        if player_bj:
            return cfg.REWARD_BLACKJACK
        if dealer_bj:
            return cfg.REWARD_LOSS

        # Dealer draws: stands on hard 17+, also stands on soft 17
        while True:
            total = hand_total(self.dealer_cards)
            soft = has_usable_ace(self.dealer_cards)
            if total > 17:
                break
            if total == 17 and not soft:   # hard 17 — stand
                break
            if total == 17 and soft:       # soft 17 — stand (S17 rule)
                break
            self.dealer_cards.append(self.shoe.deal())

        dealer_total = hand_total(self.dealer_cards)
        player_total = hand_total(self.player_cards)

        if dealer_total > 21 or player_total > dealer_total:
            return cfg.REWARD_WIN
        if player_total == dealer_total:
            return cfg.REWARD_PUSH
        return cfg.REWARD_LOSS

    def _dealer_play(self):
        """Dealer draws to 17+, stands on soft 17. Returns reward."""
        # Check dealer blackjack first
        dealer_bj = (hand_total(self.dealer_cards) == 21 and len(self.dealer_cards) == 2)
        player_bj = (hand_total(self.player_cards) == 21 and len(self.player_cards) == 2)

        if player_bj and dealer_bj:
            return cfg.REWARD_PUSH
        if player_bj:
            return cfg.REWARD_BLACKJACK
        if dealer_bj:
            return cfg.REWARD_LOSS

        # Dealer draws: stands on hard 17+, also stands on soft 17
        while True:
            total = hand_total(self.dealer_cards)
            soft = has_usable_ace(self.dealer_cards)
            if total > 17:
                break
            if total == 17 and not soft:   # hard 17 — stand
                break
            if total == 17 and soft:       # soft 17 — stand (S17 rule)
                break
            self.dealer_cards.append(self.shoe.deal())

        dealer_total = hand_total(self.dealer_cards)
        player_total = hand_total(self.player_cards)

        if dealer_total > 21 or player_total > dealer_total:
            return cfg.REWARD_WIN
        if player_total == dealer_total:
            return cfg.REWARD_PUSH
        return cfg.REWARD_LOSS
