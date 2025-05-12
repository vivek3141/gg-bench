import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - Add to own score, 1 - Subtract from opponent's score, 2 - Discard
        self.action_space = spaces.Discrete(3)

        # Define observation space
        # [own_score, opponent_score, current_player(0 or 1), drawn_digit]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 1]),
            high=np.array([50, 50, 1, 9]),
            dtype=np.int32,
        )

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize deck: digits 1-9, each appearing 4 times
        self.deck_counts = {digit: 4 for digit in range(1, 10)}
        self.discard_pile = []

        # Initialize player scores
        self.scores = [0, 0]

        # Decide starting player (0 or 1)
        self.current_player = 0

        # Draw initial digit
        self.drawn_digit = self._draw_digit()

        self.done = False

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}
        else:
            reward = 0
            # Perform action
            own_score = self.scores[self.current_player]
            opponent = 1 - self.current_player
            opponent_score = self.scores[opponent]

            if action == 0:
                # Add to own score
                self.scores[self.current_player] += self.drawn_digit
            elif action == 1:
                # Subtract from opponent's score
                self.scores[opponent] = max(0, opponent_score - self.drawn_digit)
            elif action == 2:
                # Discard the digit (no action)
                pass

            # Check for winning condition
            if self.scores[self.current_player] == 50:
                reward = 1
                self.done = True
                return self._get_observation(), reward, True, False, {}

            # If the player's score exceeds 50, move is invalid
            if self.scores[self.current_player] > 50:
                reward = -10
                self.done = True
                return self._get_observation(), reward, True, False, {}

            # Add drawn digit to discard pile
            self.discard_pile.append(self.drawn_digit)

            # Switch player
            self.current_player = 1 - self.current_player

            # Draw new digit
            self.drawn_digit = self._draw_digit()

            observation = self._get_observation()
            info = {}
            return observation, reward, False, False, info

    def render(self):
        board_str = f"Player 0 Score: {self.scores[0]}\n"
        board_str += f"Player 1 Score: {self.scores[1]}\n"
        board_str += f"Current Player: {self.current_player}\n"
        board_str += f"Drawn Digit: {self.drawn_digit}\n"
        return board_str

    def valid_moves(self):
        valid_actions = []

        own_score = self.scores[self.current_player]
        opponent = 1 - self.current_player
        opponent_score = self.scores[opponent]

        # Check if adding the digit to own score is valid
        if own_score + self.drawn_digit <= 50:
            valid_actions.append(0)

        # Check if subtracting the digit from opponent's score is valid
        if opponent_score - self.drawn_digit >= 0:
            valid_actions.append(1)

        # If neither adding nor subtracting is possible, discarding is valid
        if not valid_actions:
            valid_actions.append(2)

        return valid_actions

    def _draw_digit(self):
        # If deck is empty, reshuffle discard pile
        total_cards_left = sum(self.deck_counts.values())
        if total_cards_left == 0:
            self._reshuffle_deck()

        # Create a list of available digits based on counts
        available_digits = []
        for digit, count in self.deck_counts.items():
            available_digits.extend([digit] * count)

        # Draw a random digit
        drawn_digit = self.np_random.choice(available_digits)

        # Decrease the count of the drawn digit
        self.deck_counts[drawn_digit] -= 1

        return drawn_digit

    def _reshuffle_deck(self):
        # Reset deck counts based on discard pile
        self.deck_counts = {digit: 0 for digit in range(1, 10)}
        for digit in self.discard_pile:
            self.deck_counts[digit] += 1
        self.discard_pile = []

    def _get_observation(self):
        own_score = self.scores[self.current_player]
        opponent = 1 - self.current_player
        opponent_score = self.scores[opponent]
        return np.array(
            [own_score, opponent_score, self.current_player, self.drawn_digit],
            dtype=np.int32,
        )
