import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define maximum hand size (assuming it rarely exceeds 10)
        self.max_hand_size = 10

        # Define action and observation spaces
        # Actions are indices of cards in the player's hand
        self.action_space = spaces.Discrete(self.max_hand_size)

        # Observation: Player's hand (max_hand_size), player's life, opponent's life
        self.observation_space = spaces.Box(
            low=0, high=20, shape=(self.max_hand_size + 2,), dtype=np.int32
        )

        # Game constants
        self.HAND_LIMIT = 5  # Maximum hand size to draw up to
        self.STARTING_LIFE = 20  # Starting life points for both players

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Create a deck with four copies of cards numbered 1 to 9
        card_numbers = list(range(1, 10)) * 4  # 4 copies of each card 1-9
        self.deck = card_numbers.copy()
        np.random.shuffle(self.deck)

        # Initialize the discard pile
        self.discard_pile = []

        # Set life points
        self.player_life = self.STARTING_LIFE
        self.opponent_life = self.STARTING_LIFE

        # Deal initial hands of 5 cards to each player
        self.player_hand = [self.draw_card() for _ in range(5)]
        self.opponent_hand = [self.draw_card() for _ in range(5)]

        # Game state
        self.done = False

        # Return initial observation and info
        return self.get_observation(), {}

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        # Validate action
        if (
            action < 0
            or action >= len(self.player_hand)
            or self.player_hand[action] is None
        ):
            self.done = True
            return self.get_observation(), -10, True, False, {}  # Invalid action

        # Player's Attack Phase
        attack_card = self.player_hand.pop(action)
        # Remove None entries from player_hand
        self.player_hand = [card for card in self.player_hand if card is not None]

        # Opponent's Defense Phase
        if attack_card in self.opponent_hand:
            # Opponent has matching card
            self.opponent_hand.remove(attack_card)
            # Both cards are discarded
            self.discard_pile.extend([attack_card, attack_card])
        else:
            # Opponent cannot defend
            self.opponent_life -= attack_card
            # Only attack card is discarded
            self.discard_pile.append(attack_card)
            # Opponent keeps their hand intact

        # Both players draw cards up to HAND_LIMIT unless they have HAND_LIMIT or more cards
        self.draw_up_to_hand_size()

        # Check for opponent's defeat
        if self.opponent_life <= 0:
            self.done = True
            return self.get_observation(), 1, True, False, {}  # Player wins

        # Opponent's Turn
        if self.opponent_hand:
            opponent_action = self.opponent_select_attack()
            opponent_attack_card = self.opponent_hand.pop(opponent_action)

            # Player's Defense Phase
            if opponent_attack_card in self.player_hand:
                # Player has matching card
                self.player_hand.remove(opponent_attack_card)
                # Both cards are discarded
                self.discard_pile.extend([opponent_attack_card, opponent_attack_card])
            else:
                # Player cannot defend
                self.player_life -= opponent_attack_card
                # Only attack card is discarded
                self.discard_pile.append(opponent_attack_card)
                # Player keeps their hand intact

            # Both players draw cards up to HAND_LIMIT unless they have HAND_LIMIT or more cards
            self.draw_up_to_hand_size()

            # Check for player's defeat
            if self.player_life <= 0:
                self.done = True
                return self.get_observation(), -1, True, False, {}  # Player loses

        # Continue the game
        return self.get_observation(), 0, False, False, {}

    def render(self):
        # Visual representation of the game state
        print("=================================")
        print(f"Player Life Points: {self.player_life}")
        print(f"Opponent Life Points: {self.opponent_life}")
        print(f"Player Hand: {self.player_hand}")
        print(f"Opponent Hand Size: {len(self.opponent_hand)}")
        print(f"Deck Size: {len(self.deck)}")
        print(f"Discard Pile Size: {len(self.discard_pile)}")
        print("=================================")

    def valid_moves(self):
        # Return indices of valid moves (cards present in player's hand)
        return [
            i for i in range(len(self.player_hand)) if self.player_hand[i] is not None
        ]

    def draw_card(self):
        # Draw a card from the deck; reshuffle discard pile if deck is empty
        if not self.deck:
            if not self.discard_pile:
                # No cards left to draw
                return None
            else:
                # Reshuffle discard pile into deck
                self.deck = self.discard_pile.copy()
                self.discard_pile = []
                np.random.shuffle(self.deck)
        return self.deck.pop()

    def draw_up_to_hand_size(self):
        # Player draws up to HAND_LIMIT cards
        while len(self.player_hand) < self.HAND_LIMIT:
            card = self.draw_card()
            if card is not None:
                self.player_hand.append(card)
            else:
                break  # No more cards to draw

        # Opponent draws up to HAND_LIMIT cards
        while len(self.opponent_hand) < self.HAND_LIMIT:
            card = self.draw_card()
            if card is not None:
                self.opponent_hand.append(card)
            else:
                break  # No more cards to draw

    def get_observation(self):
        # Construct the observation array
        hand_array = np.zeros(self.max_hand_size, dtype=np.int32)
        for i, card in enumerate(self.player_hand):
            hand_array[i] = card if card is not None else 0

        life_points = np.array([self.player_life, self.opponent_life], dtype=np.int32)
        observation = np.concatenate([hand_array, life_points])

        return observation

    def opponent_select_attack(self):
        # Opponent selects a card to attack (randomly)
        return np.random.randint(len(self.opponent_hand))
