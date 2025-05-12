import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 0 = Keep the card, 1 = Discard the card
        self.action_space = spaces.Discrete(2)

        # Observation space: array of size 24
        # Index 0: current player's score
        # Index 1: opponent's score
        # Index 2: number of cards remaining in deck
        # Index 3: current card (-10 to +10, excluding zero, 0 if no card drawn yet)
        # Indices 4-23: Discard pile (up to 20 cards, zero-padded)

        low_obs = np.array([-55, -55, 0, -10] + [-10] * 20, dtype=np.int32)
        high_obs = np.array([55, 55, 20, 10] + [10] * 20, dtype=np.int32)

        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Create a custom deck of 20 number cards ranging from -10 to -1 and 1 to 10, excluding zero
        self.deck = list(range(-10, 0)) + list(range(1, 11))
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.np_random.shuffle(self.deck)

        # Players' scores
        self.player_scores = [0, 0]  # Player 0 and Player 1

        # Discard pile
        self.discard_pile = []

        # Current player (0 or 1)
        self.current_player = 0

        # Current card drawn by current player (None to start)
        self.current_card = None

        # Game over flag
        self.game_over = False

        # Return the initial observation and info
        observation = self.get_observation()
        return observation, {}

    def get_observation(self):
        # Returns the observation for the current player

        # Observation is a numpy array of size 24:
        # [current player's score, opponent's score, number of cards remaining, current card, discard_pile (20 slots)]

        observation = np.zeros(24, dtype=np.int32)

        observation[0] = self.player_scores[self.current_player]
        observation[1] = self.player_scores[1 - self.current_player]
        observation[2] = len(self.deck)  # number of cards remaining in deck

        if self.current_card is None:
            observation[3] = 0  # Zero indicates no card drawn yet
        else:
            observation[3] = self.current_card

        # Fill in the discard pile, up to 20 cards, zero-padded
        for i, card in enumerate(self.discard_pile):
            observation[4 + i] = card

        return observation

    def step(self, action):
        # Check if action is valid
        valid_actions = [0, 1]

        if action not in valid_actions:
            # Invalid action
            reward = -10
            done = True
            info = {}
            observation = self.get_observation()
            return observation, reward, done, info

        if self.game_over:
            # Game is already over
            reward = 0
            done = True
            info = {}
            observation = self.get_observation()
            return observation, reward, done, info

        # If current_card is None, draw a card
        if self.current_card is None:
            if len(self.deck) == 0:
                # No more cards to draw, game over
                self.game_over = True
                return self.end_game()
            else:
                # Draw the top card
                self.current_card = self.deck.pop(0)

        if action == 0:
            # Keep the card
            self.player_scores[self.current_player] += self.current_card
            self.current_card = None  # Reset current_card

            # Check if deck is empty after move
            if len(self.deck) == 0:
                self.game_over = True
                return self.end_game()
            else:
                # Switch player
                self.current_player = 1 - self.current_player

                reward = -10  # Valid move
                done = False
                info = {}
                observation = self.get_observation()
                return observation, reward, done, info

        elif action == 1:
            # Discard the card
            self.discard_pile.append(self.current_card)
            self.current_card = None  # Reset current_card

            # Mandatory draw
            if len(self.deck) == 0:
                # No more cards to draw, cannot proceed
                self.game_over = True
                return self.end_game()
            else:
                # Draw the next card
                mandatory_card = self.deck.pop(0)
                self.player_scores[self.current_player] += mandatory_card

                # Check if deck is empty after move
                if len(self.deck) == 0:
                    self.game_over = True
                    return self.end_game()
                else:
                    # Switch player
                    self.current_player = 1 - self.current_player

                    reward = -10  # Valid move
                    done = False
                    info = {}
                    observation = self.get_observation()
                    return observation, reward, done, info

    def end_game(self):
        # Determine winner
        p0_score = self.player_scores[0]
        p1_score = self.player_scores[1]

        if p0_score > p1_score:
            winner = 0
        elif p1_score > p0_score:
            winner = 1
        else:
            winner = None  # It's a draw

        # Set reward
        if winner == self.current_player:
            reward = 1
        else:
            reward = 0

        done = True
        info = {"winner": winner}
        observation = self.get_observation()
        return observation, reward, done, info

    def render(self):
        render_str = ""

        render_str += f"Current Player: Player {self.current_player +1}\n"
        render_str += "Scores:\n"
        render_str += f"  Player 1: {self.player_scores[0]}\n"
        render_str += f"  Player 2: {self.player_scores[1]}\n"
        render_str += f"Cards remaining in deck: {len(self.deck)}\n"
        render_str += f"Discard pile: {self.discard_pile}\n"

        if self.current_card is not None:
            render_str += f"Current card: {self.current_card}\n"
        else:
            render_str += "No current card drawn.\n"

        return render_str

    def valid_moves(self):
        if self.game_over:
            return []
        else:
            return [0, 1]
