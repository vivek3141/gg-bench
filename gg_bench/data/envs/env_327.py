import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(5)  # Actions correspond to numbers 1-5
        self.observation_space = spaces.Box(low=0, high=15, shape=(25,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_round = 1
        self.current_player = 0  # Player 0 or Player 1

        # Initialize numbers remaining for each player (1 if available, 0 if used)
        self.numbers_remaining = np.ones((2, 5), dtype=np.int32)
        # Record numbers used by each player in each round
        self.numbers_used = np.zeros((2, 5), dtype=np.int32)

        # Initialize rounds won and total sums for each player
        self.rounds_won = np.zeros(2, dtype=np.int32)
        self.total_sums = np.zeros(2, dtype=np.int32)

        # Store actions (numbers selected) in the current round
        self.current_actions = [None, None]

        # Game over flag
        self.done = False

        # Prepare the initial observation
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        # Check for invalid action
        reward = 0
        terminated = False
        truncated = False

        if self.done:
            # Game is already over
            return self._get_observation(), 0, True, False, {}

        if action < 0 or action >= 5:
            # Invalid action index
            reward = -10
            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, {}

        number_chosen = action + 1  # Actions are 0-4, numbers are 1-5

        # Check if the number has already been used by the current player
        if self.numbers_remaining[self.current_player, action] == 0:
            # Invalid move
            reward = -10
            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, {}

        # Valid action
        self.numbers_remaining[self.current_player, action] = 0  # Mark number as used
        self.numbers_used[self.current_player, self.current_round - 1] = number_chosen
        self.total_sums[self.current_player] += number_chosen
        self.current_actions[self.current_player] = number_chosen

        # Check if both players have selected their numbers for the round
        if self.current_actions[0] is not None and self.current_actions[1] is not None:
            # Resolve the round
            number_p0 = self.current_actions[0]
            number_p1 = self.current_actions[1]
            if number_p0 > number_p1:
                # Player 0 wins the round
                self.rounds_won[0] += 1
            elif number_p1 > number_p0:
                # Player 1 wins the round
                self.rounds_won[1] += 1
            else:
                # Draw, no points awarded
                pass

            # Reset for next round
            self.current_actions = [None, None]
            self.current_round += 1

            # Check if game is over
            if self.current_round > 5:
                # Game over
                self.done = True
                terminated = True
                # No additional reward at game end
            else:
                # Continue to next round
                pass

        # Switch to next player
        self.current_player = 1 - self.current_player  # Switch between 0 and 1

        # Prepare the observation
        observation = self._get_observation()

        return observation, reward, terminated, truncated, {}

    def render(self):
        # Create a visual representation of the environment state
        output = f"Round: {min(self.current_round, 5)}\n"
        output += f"Current Player: Player {self.current_player}\n"
        output += f"Rounds Won - Player 0: {self.rounds_won[0]}, Player 1: {self.rounds_won[1]}\n"
        output += f"Total Sum Used - Player 0: {self.total_sums[0]}, Player 1: {self.total_sums[1]}\n"
        output += f"Numbers Remaining - Player 0: {self._numbers_list(0)}\n"
        output += f"Numbers Remaining - Player 1: {self._numbers_list(1)}\n"
        output += f"Numbers Used - Player 0: {self.numbers_used[0]}\n"
        output += f"Numbers Used - Player 1: {self.numbers_used[1]}\n"

        if self.done:
            output += "Game Over.\n"
            # Determine the winner
            if self.rounds_won[0] > self.rounds_won[1]:
                output += "Player 0 wins the game!\n"
            elif self.rounds_won[1] > self.rounds_won[0]:
                output += "Player 1 wins the game!\n"
            else:
                # Tie-breaker
                if self.total_sums[0] < self.total_sums[1]:
                    output += "Game is tied in rounds won. Player 0 wins by lower total sum of numbers used!\n"
                elif self.total_sums[1] < self.total_sums[0]:
                    output += "Game is tied in rounds won. Player 1 wins by lower total sum of numbers used!\n"
                else:
                    output += "Game is tied in rounds won and total sums. Proceed to sudden death (not implemented).\n"
        return output

    def valid_moves(self):
        # Return a list of valid moves (indices) for the current player
        return [
            i for i in range(5) if self.numbers_remaining[self.current_player, i] == 1
        ]

    def _get_observation(self):
        obs = np.zeros(25, dtype=np.int32)
        # Current player's numbers remaining
        obs[0:5] = self.numbers_remaining[self.current_player]
        # Opponent's numbers remaining
        obs[5:10] = self.numbers_remaining[1 - self.current_player]
        # Current player's numbers used
        obs[10:15] = self.numbers_used[self.current_player]
        # Opponent's numbers used
        obs[15:20] = self.numbers_used[1 - self.current_player]
        # Current round number
        obs[20] = min(self.current_round, 5)
        # Current player's rounds won
        obs[21] = self.rounds_won[self.current_player]
        # Opponent's rounds won
        obs[22] = self.rounds_won[1 - self.current_player]
        # Current player's total sum of numbers used
        obs[23] = self.total_sums[self.current_player]
        # Opponent's total sum of numbers used
        obs[24] = self.total_sums[1 - self.current_player]
        return obs

    def _numbers_list(self, player):
        return [i + 1 for i in range(5) if self.numbers_remaining[player, i] == 1]
