import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: indices 0-18 correspond to numbers 2-20, 19 is 'pass'
        self.action_space = spaces.Discrete(
            20
        )  # Actions: 0-18 for numbers 2-20, 19 for 'pass'

        # Observation space: array of shape (28,)
        # Positions:
        # [0-18]: Pool availability (1 if available, 0 if not) for numbers 2-20
        # [19-22]: Player 1's sequence (numbers, 0 if empty)
        # [23-26]: Player 2's sequence
        # [27]: current_player (1 or 2)
        self.observation_space = spaces.Box(low=0, high=20, shape=(28,), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pool = np.ones(19, dtype=np.int32)  # Numbers 2 to 20 are available
        self.player_sequences = {
            1: np.zeros(4, dtype=np.int32),  # Player 1's sequence
            -1: np.zeros(4, dtype=np.int32),  # Player 2's sequence
        }
        self.current_player = 1  # Player 1 starts
        self.last_valid_player = None  # For tie-breaker
        self.done = False
        # Return initial observation and info
        observation = self._get_observation()
        return observation, {}

    def _get_observation(self):
        observation = np.zeros(28, dtype=np.int32)
        observation[0:19] = self.pool  # Pool availability
        observation[19:23] = self.player_sequences[1]
        observation[23:27] = self.player_sequences[-1]
        observation[27] = 1 if self.current_player == 1 else 2
        return observation

    def _switch_player(self):
        self.current_player *= -1

    def valid_moves(self):
        valid_actions = []
        # Find indices of available numbers in the pool
        available_numbers = np.where(self.pool == 1)[0]  # indices in 0..18
        seq = self.player_sequences[self.current_player]
        seq_numbers = seq[seq > 0]
        if len(seq_numbers) == 0:
            if len(available_numbers) > 0:
                valid_actions = list(available_numbers)
            else:
                valid_actions = [19]  # Only 'pass' action
        else:
            last_number = seq_numbers[-1]
            for idx in available_numbers:
                number = idx + 2
                if last_number % number == 0 or number % last_number == 0:
                    valid_actions.append(idx)
            if len(valid_actions) == 0:
                valid_actions = [19]  # Only 'pass' action
        return valid_actions

    def step(self, action):
        if self.done:
            raise Exception("Game is over")

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, self.done, {}
        else:
            if action == 19:
                # 'Pass' action
                self._switch_player()
                # Check if opponent can move
                if len(self.valid_moves()) == 0:
                    # Game over, determine winner
                    len_self = np.count_nonzero(
                        self.player_sequences[self.current_player]
                    )
                    len_other = np.count_nonzero(
                        self.player_sequences[-self.current_player]
                    )
                    if len_self > len_other:
                        reward = 1  # Current player wins
                    elif len_self < len_other:
                        reward = -1  # Current player loses
                    else:
                        # Tie-breaker
                        if self.last_valid_player == self.current_player:
                            reward = 1
                        else:
                            reward = -1
                    self.done = True
                    observation = self._get_observation()
                    return observation, reward, self.done, {}
                else:
                    # Continue game
                    reward = 0
                    observation = self._get_observation()
                    return observation, reward, self.done, {}
            else:
                action_number = action + 2  # Convert to number 2-20
                # Remove number from pool
                self.pool[action] = 0
                # Add number to current player's sequence
                seq = self.player_sequences[self.current_player]
                next_idx = np.where(seq == 0)[0][0]  # Next empty position
                seq[next_idx] = action_number
                # Update last valid player
                self.last_valid_player = self.current_player
                # Check for immediate victory
                if next_idx == 3:  # Sequence length 4
                    reward = 1
                    self.done = True
                    observation = self._get_observation()
                    return observation, reward, self.done, {}
                else:
                    # Switch player
                    self._switch_player()
                    # Check if opponent can move
                    if len(self.valid_moves()) == 0:
                        # Switch back to self
                        self._switch_player()
                        if len(self.valid_moves()) == 0:
                            # Game over, determine winner
                            len_self = np.count_nonzero(
                                self.player_sequences[self.current_player]
                            )
                            len_other = np.count_nonzero(
                                self.player_sequences[-self.current_player]
                            )
                            if len_self > len_other:
                                reward = 1
                            elif len_self < len_other:
                                reward = -1
                            else:
                                if self.last_valid_player == self.current_player:
                                    reward = 1
                                else:
                                    reward = -1
                            self.done = True
                            observation = self._get_observation()
                            return observation, reward, self.done, {}
                    # Continue game
                    reward = 0
                    observation = self._get_observation()
                    return observation, reward, self.done, {}

    def render(self):
        pool_numbers = [str(i + 2) for i in range(19) if self.pool[i] == 1]
        pool_str = "Shared Pool: " + ", ".join(pool_numbers)

        seq1_numbers = [str(num) for num in self.player_sequences[1] if num > 0]
        seq1_str = "Player 1 Sequence: " + ", ".join(seq1_numbers)

        seq2_numbers = [str(num) for num in self.player_sequences[-1] if num > 0]
        seq2_str = "Player 2 Sequence: " + ", ".join(seq2_numbers)

        current_player_str = (
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"
        )

        return "\n".join([pool_str, seq1_str, seq2_str, current_player_str])
