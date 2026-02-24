import random
from config import NUM_DECKS, RESHUFFLE_PENETRATION

CARD_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]  # 2-9, 10/J/Q/K, A

class Shoe:
    def __init__(self, num_decks=NUM_DECKS):
        self.num_decks = num_decks
        self.cards = []
        self.shuffle()

    def shuffle(self):
        # 13 card types per deck, 4 suits — but we only need values
        # Four 10-value cards per suit → weight 10s accordingly
        single_deck = [2,3,4,5,6,7,8,9,10,10,10,10,11] * 4
        self.cards = single_deck * self.num_decks
        random.shuffle(self.cards)
        self.total_cards = len(self.cards)

    def deal(self):
        if len(self.cards) / self.total_cards < (1 - RESHUFFLE_PENETRATION):
            self.shuffle()
        return self.cards.pop()

    def needs_reshuffle(self):
        return len(self.cards) / self.total_cards < (1 - RESHUFFLE_PENETRATION)
