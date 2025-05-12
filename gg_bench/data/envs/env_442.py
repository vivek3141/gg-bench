import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 9 numbers (1-9) and 2 actions (Attack, Defend) for each
        self.action_space = spaces.Discrete(18)

        # Observation space: [current player's HP, opponent's HP, available_numbers[1-9]]
        # player_hp and opponent_hp range from 0 to 10
        # available_numbers: 9 binary indicators (1 if available, 0 if used)
        self.observation_space = spaces.Box(low=0, high=10, shape=(11,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_hp = [10, 10]  # Player 0 and Player 1 HP
        self.current_player = 0  # Start with player 0
        self.available_numbers = np.ones(9, dtype=np.int32)  # Numbers 1-9 available
        self.done = False
        self.info = {}
        observation = self._get_observation()
        return observation, self.info  # Return observation and info

    def _get_observation(self):
        # Observation is [current player's HP, opponent's HP, available_numbers[1-9]]
        opponent = 1 - self.current_player
        obs = np.concatenate(
            (
                np.array(
                    [self.player_hp[self.current_player], self.player_hp[opponent]]
                ),
                self.available_numbers.copy(),
            )
        )
        return obs

    def step(self, action):
        if self.done:
            # If the game is already done, return the current observation
            observation = self._get_observation()
            return observation, 0, True, False, self.info

        # First, map action index to number and action_type (Attack/Defend)
        number, action_type = self._decode_action(action)

        # Check if action is valid
        if number is None or action_type is None:
            # Invalid action
            self.done = True
            observation = self._get_observation()
            return observation, -10, True, False, self.info

        # Check if number is available
        if self.available_numbers[number - 1] == 0:
            # Invalid move: number not available
            self.done = True
            observation = self._get_observation()
            return observation, -10, True, False, self.info

        # Valid move, proceed
        self.available_numbers[number - 1] = 0  # Remove number from available numbers

        opponent = 1 - self.current_player

        if action_type == "Attack":
            # Deal damage equal to full value of the number to opponent
            self.player_hp[opponent] -= number
        elif action_type == "Defend":
            # Increase own HP by half of the number (rounded down)
            gain = number // 2
            self.player_hp[self.current_player] += gain
            if self.player_hp[self.current_player] > 10:
                self.player_hp[self.current_player] = 10  # Max HP is 10

        # Check if opponent is defeated
        if self.player_hp[opponent] <= 0:
            self.done = True
            observation = self._get_observation()
            return observation, 1, True, False, self.info  # Current player wins

        # Check if all numbers are used
        if np.all(self.available_numbers == 0):
            self.done = True
            # Determine winner based on remaining HP
            if self.player_hp[self.current_player] > self.player_hp[opponent]:
                # Current player wins
                observation = self._get_observation()
                return observation, 1, True, False, self.info
            elif self.player_hp[self.current_player] < self.player_hp[opponent]:
                # Current player loses
                observation = self._get_observation()
                return observation, -1, True, False, self.info
            else:
                # Tie-breaker: Player who took the last turn loses
                # Since current player just took the turn, current player loses
                observation = self._get_observation()
                return observation, -1, True, False, self.info

        # Switch to next player
        self.current_player = opponent

        # No reward for valid move
        observation = self._get_observation()
        return (
            observation,
            0,
            False,
            False,
            self.info,
        )  # Observation, reward, terminated, truncated, info

    def _decode_action(self, action):
        # Action is an integer from 0 to 17
        # Map to number and action_type
        if action < 0 or action >= 18:
            return None, None
        number = action // 2 + 1  # Numbers from 1 to 9
        action_type = "Attack" if action % 2 == 0 else "Defend"
        return number, action_type

    def render(self):
        opponent = 1 - self.current_player
        s = f"Player {self.current_player + 1}'s turn\n"
        s += f"Your HP: {self.player_hp[self.current_player]}\n"
        s += f"Opponent's HP: {self.player_hp[opponent]}\n"
        s += "Available Numbers: "
        available_numbers = [
            str(i + 1) for i in range(9) if self.available_numbers[i] == 1
        ]
        s += ", ".join(available_numbers) + "\n"
        return s

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        for i in range(9):
            if self.available_numbers[i]:
                attack_action = i * 2  # Attack action index
                defend_action = i * 2 + 1  # Defend action index
                valid_actions.extend([attack_action, defend_action])
        return valid_actions
