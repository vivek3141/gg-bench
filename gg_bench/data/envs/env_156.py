import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to selecting numbers from 1 to 9
        self.action_space = spaces.Discrete(
            9
        )  # Actions: 0 to 8, representing numbers 1 to 9

        # Observation space consists of the sequence (padded to length 20) and current player (0 or 1)
        self.observation_space = spaces.Dict(
            {
                "sequence": spaces.Box(low=0, high=9, shape=(20,), dtype=np.int32),
                "current_player": spaces.Discrete(2),  # 0 or 1
            }
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = []
        self.current_player = 0  # Player 0 starts
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is over, any action is invalid
            return (
                self._get_obs(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        number = action + 1  # Convert action to number (1 to 9)

        # Check if action is valid
        temp_sequence = self.sequence + [number]

        if len(temp_sequence) >= 3:
            last_three = temp_sequence[-3:]

            # Check for increasing sequence
            if last_three[0] < last_three[1] < last_three[2]:
                # Invalid move
                self.done = True
                return self._get_obs(), -10, True, False, {}

            # Check for decreasing sequence
            if last_three[0] > last_three[1] > last_three[2]:
                # Invalid move
                self.done = True
                return self._get_obs(), -10, True, False, {}

            # Check for constant sequence
            if last_three[0] == last_three[1] == last_three[2]:
                # Invalid move
                self.done = True
                return self._get_obs(), -10, True, False, {}

        # Move is valid, update sequence
        self.sequence.append(number)

        # Check if the opponent has any valid moves
        opponent_has_move = False
        for opp_action in range(9):
            opp_number = opp_action + 1
            opp_temp_sequence = self.sequence + [opp_number]
            if len(opp_temp_sequence) >= 3:
                last_three = opp_temp_sequence[-3:]

                # Check for increasing sequence
                if last_three[0] < last_three[1] < last_three[2]:
                    continue  # Invalid move for opponent
                # Check for decreasing sequence
                if last_three[0] > last_three[1] > last_three[2]:
                    continue  # Invalid move for opponent
                # Check for constant sequence
                if last_three[0] == last_three[1] == last_three[2]:
                    continue  # Invalid move for opponent
            # Move is valid for opponent
            opponent_has_move = True
            break

        if not opponent_has_move:
            # Opponent has no valid moves, current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Switch to the next player
        self.current_player = 1 - self.current_player

        return self._get_obs(), 0, False, False, {}

    def render(self):
        sequence_str = " ".join(map(str, self.sequence))
        return f"Sequence: [{sequence_str}]\nCurrent Player: Player {self.current_player + 1}"

    def valid_moves(self):
        valid_actions = []
        for action in range(9):
            number = action + 1
            temp_sequence = self.sequence + [number]
            if len(temp_sequence) >= 3:
                last_three = temp_sequence[-3:]

                # Check for increasing sequence
                if last_three[0] < last_three[1] < last_three[2]:
                    continue  # Invalid move
                # Check for decreasing sequence
                if last_three[0] > last_three[1] > last_three[2]:
                    continue  # Invalid move
                # Check for constant sequence
                if last_three[0] == last_three[1] == last_three[2]:
                    continue  # Invalid move
            # Move is valid
            valid_actions.append(action)
        return valid_actions

    def _get_obs(self):
        # Prepare the observation
        obs_sequence = np.zeros(20, dtype=np.int32)
        seq_len = min(len(self.sequence), 20)
        obs_sequence[:seq_len] = self.sequence[-20:]  # Take the last 20 numbers
        obs_current_player = self.current_player
        return {"sequence": obs_sequence, "current_player": obs_current_player}
