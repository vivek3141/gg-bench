import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)

        # Observations:
        # - observations[0-8]: numbers 1-9, values:
        #   - 0: number available
        #   - 1: current player selected this number
        #   - -1: opponent selected this number
        # - observations[9]: current player's cumulative sum (0-15)
        # - observations[10]: opponent's cumulative sum (0-15)

        self.observation_space = spaces.Box(
            low=np.array([-1] * 9 + [0, 0]),
            high=np.array([1] * 9 + [15, 15]),
            dtype=np.int32,
        )

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.numbers_available = [True] * 9  # Numbers 1 to 9
        self.player_turn = 1  # Current player: 1 or -1
        self.first_player = 1  # The player who moves first
        self.player_sums = {1: 0, -1: 0}
        self.player_nums_chosen = {1: [], -1: []}
        self.done = False

        # Initialize observations
        # observations[0-8]: numbers 1-9 status
        # observations[9]: current player's sum
        # observations[10]: opponent's sum
        self.observation = np.array([0] * 9 + [0, 0], dtype=np.int32)

        return self.observation, {}

    def step(self, action):
        if self.done:
            # Game already over
            return self.observation, -10, True, False, {}

        # Check if action is valid
        if self.observation[action] != 0:
            # Invalid move
            self.done = True
            return self.observation, -10, True, False, {}

        # Valid move
        number_selected = action + 1  # Numbers are from 1 to 9

        # Update observation
        self.observation[action] = 1  # Current player selected this number

        # Update current player's cumulative sum
        self.player_sums[self.player_turn] += number_selected

        # Update current player's numbers chosen
        self.player_nums_chosen[self.player_turn].append(number_selected)

        # Update observations[9] and observations[10]
        # observations[9]: current player's cumulative sum
        # observations[10]: opponent's cumulative sum
        self.observation[9] = self.player_sums[self.player_turn]
        self.observation[10] = self.player_sums[-self.player_turn]

        # Check for win/loss conditions
        reward = 0
        if self.player_sums[self.player_turn] == 15:
            # Current player wins
            self.done = True
            reward = 1
            return self.observation, reward, True, False, {}
        elif self.player_sums[self.player_turn] > 15:
            # Current player loses
            self.done = True
            reward = -1
            return self.observation, reward, True, False, {}
        elif len(self.valid_moves()) == 0:
            # All numbers have been selected, determine winner
            self.done = True

            player_total = self.player_sums[self.player_turn]
            opponent_total = self.player_sums[-self.player_turn]

            if player_total > opponent_total and player_total <= 15:
                # Current player wins
                reward = 1
            elif opponent_total > player_total and opponent_total <= 15:
                # Current player loses
                reward = -1
            elif player_total == opponent_total:
                # Tie-breaker
                player_numbers_selected = len(self.player_nums_chosen[self.player_turn])
                opponent_numbers_selected = len(
                    self.player_nums_chosen[-self.player_turn]
                )

                if player_numbers_selected < opponent_numbers_selected:
                    # Current player wins
                    reward = 1
                elif opponent_numbers_selected < player_numbers_selected:
                    # Current player loses
                    reward = -1
                else:
                    # Still tied, second player wins
                    if self.player_turn != self.first_player:
                        # Current player wins
                        reward = 1
                    else:
                        # Current player loses
                        reward = -1
            return self.observation, reward, True, False, {}
        else:
            # Game continues
            reward = 0

            # Switch player
            self.player_turn *= -1

            # Adjust observations to reflect current player's perspective
            # Swap observations[0-8]: change 1 to -1 and -1 to 1
            self.observation[0:9] *= -1

            # Swap observations[9] and observations[10]
            self.observation[9], self.observation[10] = (
                self.observation[10],
                self.observation[9],
            )

            return self.observation, reward, False, False, {}

    def render(self):
        output = "Available Numbers: "
        available_numbers = [str(i + 1) for i in range(9) if self.observation[i] == 0]
        output += " ".join(available_numbers) + "\n"
        output += f"Current Player's Total: {self.player_sums[self.player_turn]}\n"
        output += f"Opponent's Total: {self.player_sums[-self.player_turn]}\n"
        output += f"Numbers Selected by Current Player: {self.player_nums_chosen[self.player_turn]}\n"
        output += f"Numbers Selected by Opponent: {self.player_nums_chosen[-self.player_turn]}\n"
        return output

    def valid_moves(self):
        return [i for i in range(9) if self.observation[i] == 0]
