import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space:
        # 0 to 4: Probe with numbers 1 to 5
        # 5: Unlock
        self.action_space = spaces.Discrete(6)

        # Define observation space:
        # [0]: Current player's Lock Strength (0 to 10)
        # [1]: Opponent's Lock Strength (0 to 10)
        # [2]: Indicator if current player has correctly identified opponent's Key Number (0 or 1)
        # [3-7]: For numbers 1-5, indicating whether they have been probed and found incorrect (1 if probed and incorrect, 0 otherwise)
        self.observation_space = spaces.Box(low=0, high=10, shape=(8,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_lock_strength = [10, 10]  # [Player 1, Player 2]
        self.player_key_number = [
            np.random.randint(1, 6),
            np.random.randint(1, 6),
        ]  # Secret Key Numbers
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.probed_correctly = [
            False,
            False,
        ]  # If each player has correctly probed the opponent's Key Number
        self.probed_numbers = [
            [],
            [],
        ]  # Numbers each player has probed and found incorrect
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def _get_observation(self):
        # Construct the observation for the current player
        opponent = 1 - self.current_player
        obs = np.zeros(8, dtype=np.int32)
        obs[0] = self.player_lock_strength[self.current_player]
        obs[1] = self.player_lock_strength[opponent]
        obs[2] = int(self.probed_correctly[self.current_player])

        # Mark numbers that have been probed and found incorrect
        probed_incorrect = np.zeros(5, dtype=np.int32)
        for number in self.probed_numbers[self.current_player]:
            probed_incorrect[number - 1] = 1  # Mark as probed and incorrect
        obs[3:8] = probed_incorrect
        return obs

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        if action not in self.valid_moves():
            # Invalid action
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}
        else:
            reward = 0
            opponent = 1 - self.current_player

            if 0 <= action <= 4:
                # Probe action
                guess = action + 1  # Map action to guess number (1-5)
                if guess == self.player_key_number[opponent]:
                    # Correct guess
                    self.probed_correctly[self.current_player] = True
                else:
                    # Incorrect guess
                    if guess not in self.probed_numbers[self.current_player]:
                        self.probed_numbers[self.current_player].append(guess)
            elif action == 5:
                # Unlock action
                if self.probed_correctly[self.current_player]:
                    # Successful unlock
                    self.player_lock_strength[opponent] -= 5
                    if self.player_lock_strength[opponent] <= 0:
                        self.player_lock_strength[opponent] = 0
                        reward = 1
                        self.done = True
                        return self._get_observation(), reward, True, False, {}
                    else:
                        # Opponent selects a new Key Number
                        self.player_key_number[opponent] = np.random.randint(1, 6)
                        # Reset probe status for current player
                        self.probed_correctly[self.current_player] = False
                        self.probed_numbers[self.current_player] = []
                else:
                    # Attempted to unlock without correctly probing
                    reward = -10
                    self.done = True
                    return self._get_observation(), reward, True, False, {}
            # Switch turns
            self.current_player = opponent
            return self._get_observation(), reward, False, False, {}

    def render(self):
        opponent = 1 - self.current_player
        state = f"--- Player {self.current_player + 1}'s Turn ---\n"
        state += (
            f"Your Lock Strength: {self.player_lock_strength[self.current_player]}\n"
        )
        state += f"Opponent's Lock Strength: {self.player_lock_strength[opponent]}\n"
        state += f"Probed Correctly: {self.probed_correctly[self.current_player]}\n"
        state += f"Numbers probed and incorrect: {self.probed_numbers[self.current_player]}\n"
        return state

    def valid_moves(self):
        # Return valid actions for the current player
        valid_actions = [0, 1, 2, 3, 4]  # Probing numbers 1-5
        if self.probed_correctly[self.current_player]:
            valid_actions.append(5)  # Unlock action
        return valid_actions
