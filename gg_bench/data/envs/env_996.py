import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    """
    Number Duel Environment for self-play reinforcement learning.
    Two players take turns picking numbers from either end of a sequence.
    The goal is to accumulate a higher score than the opponent.
    The environment manages turns and game state internally.
    """

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 (pick left), 1 (pick right)
        self.action_space = spaces.Discrete(2)

        # Define observation space:
        # First 9 elements: current sequence (padded with zeros)
        # 10th element: current player's score
        # 11th element: opponent's score
        self.observation_space = spaces.Box(low=0, high=45, shape=(11,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        Returns:
            observation (np.array): The initial observation of the environment.
            info (dict): Additional information (empty in this case).
        """
        super().reset(seed=seed)

        # Initialize the sequence with a random permutation of numbers from 1 to 9
        self.sequence = list(np.random.permutation(np.arange(1, 10)))

        # Initialize player scores
        self.current_player = 1  # Player 1: 1, Player 2: -1
        self.scores = {1: 0, -1: 0}

        # Game is not done
        self.done = False

        # Prepare initial observation
        observation = self._get_observation()

        return observation, {}  # Return observation and empty info

    def step(self, action):
        """
        Take an action in the environment.
        Args:
            action (int): Action taken by the agent (0 for left, 1 for right).
        Returns:
            observation (np.array): The observation after the action.
            reward (float): The reward obtained from the action.
            done (bool): Whether the game has ended.
            info (dict): Additional information (empty in this case).
        """
        if self.done or action not in [0, 1]:
            # Invalid move since the game is over or action is invalid
            return self._get_observation(), -10, True, False, {}

        if not self.sequence:
            # The game is over; no moves can be made
            self.done = True
            # Determine winner
            if self.scores[self.current_player] > self.scores[-self.current_player]:
                reward = 1  # Current player wins
            else:
                reward = 0  # Current player loses
            return self._get_observation(), reward, True, False, {}

        # Perform the action
        if action == 0:
            # Pick leftmost number
            number_picked = self.sequence.pop(0)
        elif action == 1:
            # Pick rightmost number
            number_picked = self.sequence.pop(-1)

        # Update current player's score
        self.scores[self.current_player] += number_picked

        # Check if the game has ended after this move
        if not self.sequence:
            self.done = True
            # Determine winner
            if self.scores[self.current_player] > self.scores[-self.current_player]:
                reward = 1  # Current player wins
            else:
                reward = 0  # Current player loses
            return self._get_observation(), reward, True, False, {}
        else:
            # Game continues; switch to the next player
            reward = 0
            self.current_player *= -1
            return self._get_observation(), reward, False, False, {}

    def render(self):
        """
        Return a visual representation of the current state of the game.
        """
        sequence_str = "[" + ", ".join(str(num) for num in self.sequence) + "]"
        print(f"Current sequence: {sequence_str}")
        print(f"Player {1 if self.current_player ==1 else 2}'s turn.")
        print(f"Scores: Player 1: {self.scores[1]}, Player 2: {self.scores[-1]}")

    def valid_moves(self):
        """
        Return a list of valid moves for the current state.
        Returns:
            moves (list): List of valid action indices (0 or 1).
        """
        if self.done or not self.sequence:
            return []
        else:
            return [0, 1]

    def _get_observation(self):
        """
        Generate the current observation of the environment.
        Returns:
            obs (np.array): The observation array.
        """
        # Prepare the sequence part of the observation
        obs_sequence = [num for num in self.sequence]
        # Pad with zeros to ensure the sequence has length 9
        obs_sequence += [0] * (9 - len(obs_sequence))
        # Combine sequence with current player's score and opponent's score
        obs = np.array(
            obs_sequence
            + [self.scores[self.current_player], self.scores[-self.current_player]],
            dtype=np.int32,
        )
        return obs
