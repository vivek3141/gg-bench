import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers 1 to 5 (indices 0 to 4)
        self.action_space = spaces.Discrete(5)

        # Define observation space: 10 elements indicating available numbers for both players
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the available numbers for both players (1 for available, 0 for used)
        self.agent_numbers = np.ones(5, dtype=np.int32)
        self.opponent_numbers = np.ones(5, dtype=np.int32)

        self.agent_score = 0
        self.opponent_score = 0

        self.round_number = 1

        self.done = False

        # Optionally track the moves (history)
        self.agent_moves = []
        self.opponent_moves = []

        # Observation is concatenation of both players' available numbers
        observation = np.concatenate([self.agent_numbers, self.opponent_numbers])

        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Map action to number to play (1 to 5)
        number_to_play = action + 1

        # Check if the number is available
        if self.agent_numbers[action] == 0:
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Agent plays the number
        self.agent_numbers[action] = 0
        self.agent_moves.append(number_to_play)

        # Opponent selects a number from their available numbers
        opponent_available_indices = np.where(self.opponent_numbers == 1)[0]
        if len(opponent_available_indices) == 0:
            # Should not happen
            self.done = True
            return self._get_obs(), 0, True, False, {}

        opponent_action = np.random.choice(opponent_available_indices)
        opponent_number = opponent_action + 1
        self.opponent_numbers[opponent_action] = 0
        self.opponent_moves.append(opponent_number)

        # Resolve the round
        if number_to_play > opponent_number:
            # Agent wins the round
            points = number_to_play - opponent_number
            self.agent_score += points
        elif number_to_play < opponent_number:
            # Opponent wins the round
            points = opponent_number - number_to_play
            self.opponent_score += points
        else:
            # Tie, no points awarded
            pass

        # Update round number
        self.round_number += 1

        # Check if game is over (after 5 rounds)
        if self.round_number > 5:
            self.done = True
            # Determine winner
            if self.agent_score > self.opponent_score:
                # Agent wins the game
                reward = 1
            else:
                reward = 0
            terminated = True
        else:
            reward = 0
            terminated = False

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return np.concatenate([self.agent_numbers, self.opponent_numbers])

    def render(self):
        output = f"Round {self.round_number - 1}\n"
        output += f"Agent's available numbers: {[i+1 for i, available in enumerate(self.agent_numbers) if available == 1]}\n"
        output += f"Opponent's available numbers: {[i+1 for i, available in enumerate(self.opponent_numbers) if available == 1]}\n"
        output += f"Agent's moves: {self.agent_moves}\n"
        output += f"Opponent's moves: {self.opponent_moves}\n"
        output += f"Agent's score: {self.agent_score}\n"
        output += f"Opponent's score: {self.opponent_score}\n"
        print(output)

    def valid_moves(self):
        return [i for i in range(5) if self.agent_numbers[i] == 1]
