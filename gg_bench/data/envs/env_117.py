import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=10):
        super(CustomEnv, self).__init__()

        self.N = N
        self.action_space = spaces.Discrete(
            N
        )  # Actions are numbers from 1 to N (indices 0 to N-1)
        # Observation includes:
        # - Player's remaining numbers (binary vector of length N)
        # - Opponent's remaining numbers (binary vector of length N)
        # - Current role (0 for attacker, 1 for defender)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2 * self.N + 1,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player_numbers = np.ones(
            self.N, dtype=np.float32
        )  # Player's numbers
        self.opponent_numbers = np.ones(self.N, dtype=np.float32)  # Opponent's numbers
        self.current_role = 0  # 0 for attacker, 1 for defender
        self.done = False

        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        if action < 0 or action >= self.N:
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {"info": "Invalid action, out of bounds"},
            )

        if self.current_player_numbers[action] != 1:
            # Invalid move
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {"info": "Invalid action, number not available"},
            )

        player_number = action + 1

        # Select opponent's action
        valid_opponent_actions = np.where(self.opponent_numbers == 1)[0]
        if len(valid_opponent_actions) == 0:
            # Opponent has no numbers left, current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}

        opponent_action = np.random.choice(valid_opponent_actions)
        opponent_number = opponent_action + 1

        # Resolve the round
        player_number_captured = False
        opponent_number_captured = False

        if self.current_role == 0:
            # Attacker's turn
            if player_number > opponent_number:
                # Attack successful
                self.opponent_numbers[opponent_action] = 0  # Opponent loses the number
                opponent_number_captured = True
            else:
                # Defense successful
                self.current_player_numbers[action] = (
                    0  # Current player loses the number
                )
                player_number_captured = True
        else:
            # Defender's turn
            if opponent_number > player_number:
                # Attack successful
                self.current_player_numbers[action] = (
                    0  # Current player loses the number
                )
                player_number_captured = True
            else:
                # Defense successful
                self.opponent_numbers[opponent_action] = 0  # Opponent loses the number
                opponent_number_captured = True

        # Check for win conditions
        current_player_numbers_left = np.sum(self.current_player_numbers)
        opponent_numbers_left = np.sum(self.opponent_numbers)

        if current_player_numbers_left == 0 and opponent_numbers_left == 0:
            # Both players have no numbers left
            if player_number_captured and opponent_number_captured:
                # Both numbers were captured in the same round
                if self.current_role == 1:
                    # Defender wins
                    self.done = True
                    return self._get_observation(), 1, True, False, {}
                else:
                    # Attacker loses
                    self.done = True
                    return self._get_observation(), -1, True, False, {}
            else:
                # This situation should not occur
                pass
        elif opponent_numbers_left == 0:
            # Opponent has no numbers left, current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}
        elif current_player_numbers_left == 0:
            # Current player has no numbers left, opponent wins
            self.done = True
            return self._get_observation(), -1, True, False, {}

        # Switch roles
        self.current_role = 1 - self.current_role
        return self._get_observation(), 0, False, False, {}

    def render(self):
        output = []
        output.append(
            f"Current role: {'Attacker' if self.current_role == 0 else 'Defender'}"
        )
        output.append(
            f"Your remaining numbers: {np.nonzero(self.current_player_numbers)[0]+1}"
        )
        output.append(
            f"Opponent's remaining numbers: {np.nonzero(self.opponent_numbers)[0]+1}"
        )
        return "\n".join(output)

    def valid_moves(self):
        valid_actions = np.nonzero(self.current_player_numbers)[0]
        return valid_actions.tolist()

    def _get_observation(self):
        obs = np.concatenate(
            [
                self.current_player_numbers,
                self.opponent_numbers,
                np.array([self.current_role], dtype=np.float32),
            ]
        )
        return obs
