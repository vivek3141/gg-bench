import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 (place on own stack), 1 (place on opponent's stack)
        self.action_space = spaces.Discrete(2)

        # Observation space includes both stacks (20 cards each), plus current card
        # Each stack is represented by an array of 20 integers (cards or zeros)
        # Total observation space is (41,)
        # Cards range from 0 to 10 (0 for empty slots)
        self.observation_space = spaces.Box(low=0, high=10, shape=(41,), dtype=np.int32)

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        # Seed the random number generator
        super().reset(seed=seed)

        # Create the deck and shuffle
        self.deck = [i for i in range(1, 11)] * 2  # Numbers 1-10, each appearing twice
        self.np_random.shuffle(self.deck)

        # Initialize player stacks as empty lists
        self.player_stacks = {0: [], 1: []}

        # Set current player (0 or 1)
        self.current_player = 0

        # Draw the first card
        if len(self.deck) > 0:
            self.current_card = self.deck.pop(0)
        else:
            self.current_card = None  # Should not happen at the start

        # Game not done
        self.done = False

        # Return initial observation and info dict
        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        # Prepare observation array of length 41
        obs = np.zeros(41, dtype=np.int32)

        # Get stacks for both players, padded to length 20
        stack0 = self.player_stacks[0] + [0] * (20 - len(self.player_stacks[0]))
        stack1 = self.player_stacks[1] + [0] * (20 - len(self.player_stacks[1]))

        # Fill in the observation
        obs[:20] = stack0
        obs[20:40] = stack1
        obs[40] = self.current_card if self.current_card is not None else 0

        return obs

    def step(self, action):
        # Check if game is over
        if self.done:
            return self._get_observation(), -10, True, False, {}

        # Check if action is valid
        if action not in [0, 1]:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Place the current card on the chosen player's stack
        if action == 0:
            # Place on own stack
            self.player_stacks[self.current_player].append(self.current_card)
        elif action == 1:
            # Place on opponent's stack
            opponent = 1 - self.current_player
            self.player_stacks[opponent].append(self.current_card)

        # Reset the current card
        self.current_card = None

        # Check if deck is empty after the move
        if len(self.deck) == 0:
            # Game over
            self.done = True
            # Compute final scores and reward
            reward = self._compute_reward()
            observation = self._get_observation()
            info = {}
            return observation, reward, True, False, info

        # Switch to next player
        self.current_player = 1 - self.current_player

        # Draw the next card
        if len(self.deck) > 0:
            self.current_card = self.deck.pop(0)
        else:
            # Deck is empty, game over
            self.done = True
            # Compute final scores and reward
            reward = self._compute_reward()
            observation = self._get_observation()
            info = {}
            return observation, reward, True, False, info

        # Reward is zero for ongoing game
        reward = 0

        # Return observation, reward, done, truncated, info
        observation = self._get_observation()
        info = {}
        return observation, reward, self.done, False, info

    def _compute_reward(self):
        # Compute scores for both players
        scores = [0, 0]

        for player in [0, 1]:
            stack = self.player_stacks[player]
            total = sum(stack)
            num_fives = stack.count(5)
            penalty = num_fives * 5
            score = total - penalty
            scores[player] = score

        # Determine winner
        if scores[self.current_player] > scores[1 - self.current_player]:
            reward = 1  # Current player wins
        else:
            reward = 0  # Current player does not win
        return reward

    def render(self):
        output = ""
        output += f"Current Player: Player {self.current_player + 1}\n"
        output += f"Current Card: {self.current_card}\n"
        output += f"Player 1's Stack: {self.player_stacks[0]}\n"
        output += f"Player 2's Stack: {self.player_stacks[1]}\n"
        output += f"Cards Remaining in Deck: {len(self.deck)}\n"
        return output

    def valid_moves(self):
        if self.done:
            return []
        else:
            return [0, 1]
