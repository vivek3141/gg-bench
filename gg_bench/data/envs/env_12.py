import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 = Keep the current card, 1 = Discard the current card
        self.action_space = spaces.Discrete(2)

        # Observations:
        # [0]: Player's cumulative score (normalized between 0 and 1)
        # [1]: Opponent's cumulative score (normalized between 0 and 1)
        # [2]: Current card value (normalized between 0 and 1)
        # [3]: First draw flag (1 if first draw, 0 if second draw)
        # [4]: Discard allowed flag (1 if discard is allowed, 0 otherwise)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize game parameters
        self.scores = [0, 0]  # [Player 1 score, Player 2 score]
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.extra_turn = False  # Indicates if the player earned an extra turn
        self.first_draw = True  # Indicates if it's the first draw in the turn
        self.game_over = False  # Indicates if the game is over

        # Create and shuffle the deck
        self.deck = [i for i in range(1, 11)] * 4  # Four copies of cards 1-10
        random.shuffle(self.deck)
        self.discard_pile = []  # Initialize discard pile

        # Draw the first card
        self.current_card = self.draw_card()

        # Prepare initial observation
        observation = self.get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        # Check if the action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            reward = -10
            terminated = True
            truncated = False
            info = {"reason": "Invalid action"}
            observation = self.get_observation()
            return observation, reward, terminated, truncated, info

        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Get current player's score
        player_score = self.scores[self.current_player]

        if action == 0:
            # Keep the current card
            total_score = player_score + self.current_card
            if total_score > 50:
                # Invalid move: cannot keep a card that causes score to exceed 50
                reward = -10
                terminated = True
                info = {"reason": "Score exceeds 50"}
                observation = self.get_observation()
                return observation, reward, terminated, truncated, info
            else:
                # Valid move: update player's score
                self.scores[self.current_player] = total_score

                # Check for win condition
                if total_score == 50:
                    reward = 1
                    terminated = True
                    observation = self.get_observation()
                    return observation, reward, terminated, truncated, info

                # Check for prime cumulative score
                if self.is_prime(total_score):
                    self.extra_turn = True
                else:
                    self.extra_turn = False

                # Discard the used card
                self.discard_pile.append(self.current_card)

                # Prepare for next draw or switch player
                if self.extra_turn:
                    # Player gets an extra turn
                    self.first_draw = True
                    self.current_card = self.draw_card()
                else:
                    # Switch to next player
                    self.current_player = 1 - self.current_player
                    self.first_draw = True
                    self.current_card = self.draw_card()
        elif action == 1 and self.first_draw and player_score + self.current_card <= 50:
            # Discard the current card (on first draw only)
            self.discard_pile.append(self.current_card)
            self.first_draw = False
            # Draw the second card, which must be kept if possible
            self.current_card = self.draw_card()
            total_score = player_score + self.current_card
            if total_score > 50:
                # Cannot keep the second card, discard it
                self.discard_pile.append(self.current_card)
                self.current_card = None
                self.extra_turn = False
                # Switch to next player
                self.current_player = 1 - self.current_player
                self.first_draw = True
                self.current_card = self.draw_card()
            else:
                # Keep the second card
                self.scores[self.current_player] = total_score

                # Check for win condition
                if total_score == 50:
                    reward = 1
                    terminated = True
                    observation = self.get_observation()
                    return observation, reward, terminated, truncated, info

                # Check for prime cumulative score
                if self.is_prime(total_score):
                    self.extra_turn = True
                else:
                    self.extra_turn = False

                # Discard the used card
                self.discard_pile.append(self.current_card)

                # Prepare for next draw or switch player
                if self.extra_turn:
                    # Player gets an extra turn
                    self.first_draw = True
                    self.current_card = self.draw_card()
                else:
                    # Switch to next player
                    self.current_player = 1 - self.current_player
                    self.first_draw = True
                    self.current_card = self.draw_card()
        else:
            # Invalid action
            reward = -10
            terminated = True
            info = {"reason": "Invalid action"}
            observation = self.get_observation()
            return observation, reward, terminated, truncated, info

        observation = self.get_observation()
        return observation, reward, terminated, truncated, info

    def render(self):
        # Return a string representation of the game state
        player = self.current_player + 1
        opponent = 2 if player == 1 else 1
        state = f"Player {player} Turn:\n"
        state += (
            f"Current Scores - Player 1: {self.scores[0]}, Player 2: {self.scores[1]}\n"
        )
        state += f"Card drawn: {self.current_card}\n"
        if self.first_draw:
            state += "This is the first draw.\n"
        else:
            state += "This is the second draw (must keep the card if possible).\n"
        state += f"Discard allowed: {'Yes' if self.first_draw and self.current_card else 'No'}\n"
        return state

    def valid_moves(self):
        # Return a list of valid actions: [0] for Keep, [1] for Discard
        valid_actions = []

        player_score = self.scores[self.current_player]
        total_score = player_score + self.current_card

        if self.first_draw:
            if total_score <= 50:
                # Can choose to Keep or Discard
                valid_actions = [0, 1]
            else:
                # Cannot keep, must Discard
                valid_actions = [1]
        else:
            # Second draw, must keep if possible
            if total_score <= 50:
                valid_actions = [0]
            else:
                # Cannot keep, no valid actions (but agent must proceed)
                valid_actions = []
        return valid_actions

    def draw_card(self):
        # Draw a card from the deck
        if not self.deck:
            # Reshuffle discard pile into deck
            self.deck = self.discard_pile
            self.discard_pile = []
            random.shuffle(self.deck)
        return self.deck.pop()

    def get_observation(self):
        # Prepare observation
        player_score = self.scores[self.current_player]
        opponent_score = self.scores[1 - self.current_player]
        current_card = self.current_card if self.current_card else 0
        first_draw_flag = 1 if self.first_draw else 0
        discard_allowed_flag = (
            1 if self.first_draw and (player_score + current_card) <= 50 else 0
        )

        observation = np.array(
            [
                player_score / 50.0,
                opponent_score / 50.0,
                current_card / 10.0,
                first_draw_flag,
                discard_allowed_flag,
            ],
            dtype=np.float32,
        )

        return observation

    def is_prime(self, n):
        # Check if a number n is prime
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(np.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
