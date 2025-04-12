import copy
import random


class LeducHoldem:
    def __init__(self):
        self.reset()

    def init_deck(self):
        """
        Initialize the Leduc deck.
        In Leduc Hold'em, there are 6 cards: two copies each of 'J', 'Q', and 'K'.
        Each card is represented as a tuple (rank, copy_id).
        """
        ranks = ["J", "Q", "K"]
        deck = []
        for rank in ranks:
            deck.append((rank, 1))
            deck.append((rank, 2))
        return deck

    def reset(self):
        """
        Reset the game to the initial state.
        This method shuffles the deck, deals one private card to each player,
        and resets the pot, betting history, and game phase.
        """
        self.deck = self.init_deck()
        random.shuffle(self.deck)

        # Deal private cards to two players; player indices: 0 and 1.
        self.private_cards = {0: self.deck.pop(), 1: self.deck.pop()}
        self.public_card = None

        # Initialize betting and game state variables.
        self.pot = 0
        self.bet_history = []  # e.g., [(player, action)]
        self.current_player = 0  # whose turn it is
        self.current_round = (
            1  # betting round: 1 for pre-public card, 2 for post-public card
        )
        self.phase = (
            "pre-flop"  # game phase: can be "pre-flop", "post-flop", or "terminal"
        )

        # For fixed-limit betting, track the last bet and each player's bet for the round.
        self.last_bet = 0
        self.player_bets = {0: 0, 1: 0}

    def get_perfect_information(self):
        """
        Return full state information as a dictionary.
        This includes private cards, public card, remaining deck, pot, betting history,
        current round and phase, current bets, and the active player.
        """
        state = {
            "private_cards": self.private_cards.copy(),
            "public_card": self.public_card,
            "deck": self.deck.copy(),
            "pot": self.pot,
            "bet_history": self.bet_history.copy(),
            "current_round": self.current_round,
            "phase": self.phase,
            "player_bets": self.player_bets.copy(),
            "current_player": self.current_player,
        }
        return state

    def get_legal_actions(self):
        """
        Return a list of legal actions for the current player based on the betting state.
        For simplicity, we consider a set of actions:
          - If no outstanding bet exists, the player can "check" or "bet".
          - If there is an outstanding bet, legal actions are "fold", "call", or "raise".

        Note: In a more detailed simulation, you might also consider the available chips,
        limits on the number of raises per round, etc.
        """
        if self.last_bet == 0:
            return ["check", "bet"]
        else:
            return ["fold", "call", "raise"]

    def take_action(self, action, amount=0):
        """
        Update the game state based on the player's action.
        This method logs the action and updates the pot, bets, and active player.

        Parameters:
          action: a string among 'bet', 'raise', 'call', 'check', or 'fold'
          amount: the bet/raise amount (used when action is 'bet' or 'raise')

        For simplicity, this implementation is basic. You can extend it to include
        full betting logic, round termination, and game termination conditions.
        """
        self.bet_history.append((self.current_player, action))
        if action in ["bet", "raise"]:
            self.last_bet = amount
            self.player_bets[self.current_player] += amount
            self.pot += amount
        elif action == "call":
            call_amount = self.last_bet - self.player_bets[self.current_player]
            self.player_bets[self.current_player] += call_amount
            self.pot += call_amount
        elif action == "check":
            # No chips are moved in a check.
            pass
        elif action == "fold":
            # In a fold, we mark the game phase as terminal.
            self.phase = "terminal"
            # Additional logic to determine the winner can be added here.

        # Alternate the turn to the other player.
        self.current_player = 1 - self.current_player

        # Optionally, add logic here to transition rounds/phases if both players have acted.

    def deal_next_card(self, card=None):
        """
        Deal the public card from the deck.
        This method is used during the chance node of the expectimax.

        Parameters:
          card: (optional) a specific card tuple to set as the public card.
                If provided, it is removed from the deck; otherwise, a random card is chosen.

        Raises an exception if the public card has already been dealt.
        """
        if self.public_card is not None:
            raise Exception("Public card already dealt")

        if card is not None:
            if card in self.deck:
                self.deck.remove(card)
                self.public_card = card
            else:
                raise Exception("Specified card not available in deck")
        else:
            # Randomly select and remove a card from the deck.
            idx = random.randrange(len(self.deck))
            self.public_card = self.deck.pop(idx)

        # Transition the game state: update phase, round, and reset betting parameters.
        self.phase = "post-flop"
        self.current_round = 2
        self.last_bet = 0
        self.player_bets = {0: 0, 1: 0}
        self.current_player = 0  # Optionally, decide who starts the second round.

    def get_chance_distribution(self):
        """
        Compute and return a probability distribution over the next card to be dealt.
        The distribution is returned as a dictionary where keys are card tuples (e.g., ('J', 1))
        and values are their corresponding probabilities (uniform over the remaining deck).
        """
        distribution = {}
        total = len(self.deck)
        for card in self.deck:
            distribution[card] = distribution.get(card, 0) + 1 / total
        return distribution

    def clone(self):
        """
        Return a deep copy of the current game state.
        This is useful for expectimax simulations where you need to simulate actions
        on a copy of the state without affecting the original.
        """
        return copy.deepcopy(self)
