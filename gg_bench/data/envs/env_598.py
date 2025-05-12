import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0: Keep the drawn digit, 1: Give the drawn digit to opponent
        self.action_space = spaces.Discrete(2)

        # Observations: [
        #   own hand (3 digits, -1 for empty slots),
        #   opponent's hand size (0-3),
        #   digit just drawn (0-9)
        # ]
        self.observation_space = spaces.Box(low=-1, high=9, shape=(5,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the deck with 4 copies of digits 0-9
        self.deck = [digit for digit in range(10) for _ in range(4)]
        np.random.shuffle(self.deck)

        self.discard_pile = []

        # Hands for players (lists of digits)
        self.hands = {1: [], 2: []}

        # Round scores for players
        self.scores = {1: 0, 2: 0}

        # Current player (1 or 2)
        self.current_player = 1

        self.other_player = 2

        # Flags
        self.done = False  # Game over
        self.round_over = False  # Round over
        self.winner = None  # Winner of the game

        self.ties_in_a_row = 0  # To avoid infinite loops in case of repeated ties

        # Draw a digit for the current player to start
        self.digit_drawn = self.draw_digit()

        # Build initial observation
        observation = self.get_observation()

        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        # Apply the action
        if action == 0:  # Keep
            # Add digit to current player's hand if not full
            if len(self.hands[self.current_player]) < 3:
                self.hands[self.current_player].append(self.digit_drawn)
            else:
                # Hand is full, discard the digit
                self.discard_pile.append(self.digit_drawn)
        elif action == 1:  # Give
            # Give digit to opponent
            if len(self.hands[self.other_player]) < 3:
                self.hands[self.other_player].append(self.digit_drawn)
            else:
                # Opponent's hand is full, discard the digit
                self.discard_pile.append(self.digit_drawn)
        else:
            # Invalid action
            return self.get_observation(), -10, True, False, {}

        # Check if both players have full hands
        if len(self.hands[1]) == 3 and len(self.hands[2]) == 3:
            # Round over
            self.round_over = True
            # Evaluate hands to determine round winner
            self.evaluate_round()
            # Check if game is over
            if self.scores[1] == 2 or self.scores[2] == 2:
                self.done = True
                reward = 1 if self.scores[self.current_player] == 2 else -1
                return self.get_observation(), reward, True, False, {}
            else:
                # Start new round
                self.round_over = False
                # Reset hands
                self.hands = {1: [], 2: []}
                # Shuffle discard pile back into deck if necessary
                # (This will be handled in draw_digit())
                # Draw a new digit for the current player
                self.digit_drawn = self.draw_digit()
        else:
            # Switch to next player's turn
            self.current_player, self.other_player = (
                self.other_player,
                self.current_player,
            )
            # Draw a digit for the next player
            self.digit_drawn = self.draw_digit()

        observation = self.get_observation()
        return observation, 0, False, False, {}

    def render(self):
        # Return a string representing the game state
        s = f"Player {self.current_player}'s turn.\n"
        s += f"Digit drawn: {self.digit_drawn}\n"
        s += f"Your hand: {self.hands[self.current_player]}\n"
        s += f"Opponent's hand size: {len(self.hands[self.other_player])}\n"
        s += f"Scores: Player 1: {self.scores[1]}, Player 2: {self.scores[2]}\n"
        s += f"Cards remaining in deck: {len(self.deck)}\n"
        return s

    def valid_moves(self):
        # In this game, both actions are always valid
        return [0, 1]

    def draw_digit(self):
        if not self.deck:
            # Reshuffle discard pile into deck
            self.deck = self.discard_pile
            self.discard_pile = []
            np.random.shuffle(self.deck)
        # Draw a digit from the deck
        return self.deck.pop()

    def get_observation(self):
        # Build observation array
        own_hand = self.hands[self.current_player] + [-1] * (
            3 - len(self.hands[self.current_player])
        )
        observation = np.array(
            own_hand + [len(self.hands[self.other_player]), self.digit_drawn],
            dtype=np.int8,
        )
        return observation

    def evaluate_round(self):
        # For both players, form the highest possible number
        def highest_number(digits):
            digits_sorted = sorted(digits, reverse=True)
            number_str = "".join(map(str, digits_sorted))
            return int(number_str)

        number1 = highest_number(self.hands[1])
        number2 = highest_number(self.hands[2])

        if number1 > number2:
            winner = 1
        elif number2 > number1:
            winner = 2
        else:
            # Tie: round is replayed with remaining cards
            # Reset hands but keep scores unchanged
            winner = None

        if winner:
            self.scores[winner] += 1
            self.ties_in_a_row = 0
        else:
            # Tie occurred
            self.ties_in_a_row += 1
            if self.ties_in_a_row > 10:
                # To prevent infinite ties, we can declare a draw or handle appropriately
                pass  # For simplicity, we'll proceed without handling infinite ties

        # Discard used digits
        self.discard_pile.extend(self.hands[1])
        self.discard_pile.extend(self.hands[2])

    def close(self):
        pass
