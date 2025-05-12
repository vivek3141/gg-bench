import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 20 possible actions corresponding to numbers 1 to 20
        self.action_space = spaces.Discrete(20)

        # Define observation space
        # Observation consists of:
        # - Number pool: 20 elements (1 if available, 0 if taken)
        # - Player 1 total sum (scalar)
        # - Player 1 parity (-1 for odd, 1 for even, 0 if undefined)
        # - Player 2 total sum (scalar)
        # - Player 2 parity (-1 for odd, 1 for even, 0 if undefined)
        # - Current player (1 or -1)
        low_obs = np.array([0] * 20 + [0, -1, 0, -1, -1], dtype=np.int32)
        high_obs = np.array([1] * 20 + [100, 1, 100, 1, 1], dtype=np.int32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the number pool: 1 if number is available, 0 if taken
        self.number_pool = np.ones(20, dtype=np.int32)

        # Initialize player sequences and total sums
        self.player_sums = {1: 0, -1: 0}  # Player 1 and Player -1 (Player 2)
        self.player_parities = {1: 0, -1: 0}  # 0 if undefined, -1 for odd, 1 for even

        # Current player: 1 or -1
        self.current_player = 1

        self.done = False

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # Check if game is already over
        if self.done:
            observation = self._get_observation()
            return observation, 0, True, False, {}

        # Map action to selected number (1-based)
        selected_number = action + 1

        # Check if selected number is available in the number pool
        if self.number_pool[action] == 0:
            # Invalid action: number already taken
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Check if selected number matches player's required parity
        player_parity = self.player_parities[self.current_player]
        if player_parity == 0:
            # First move, set player's parity
            self.player_parities[self.current_player] = (
                -1 if selected_number % 2 == 1 else 1
            )
            player_parity = self.player_parities[self.current_player]
        else:
            required_parity = player_parity
            if (selected_number % 2 == 1 and required_parity != -1) or (
                selected_number % 2 == 0 and required_parity != 1
            ):
                # Invalid action: parity mismatch
                reward = -10
                self.done = True
                observation = self._get_observation()
                return observation, reward, True, False, {}

        # Valid move: update game state
        self.number_pool[action] = 0  # Remove number from pool
        self.player_sums[
            self.current_player
        ] += selected_number  # Update player's total sum

        # Check for victory condition
        if self.player_sums[self.current_player] == 30:
            reward = 1
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}
        elif self.player_sums[self.current_player] > 30:
            # Player loses if total exceeds 30
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Check if opponent has valid moves
        self.current_player *= -1  # Switch current player

        opponent_parity = self.player_parities[self.current_player]
        valid_numbers = self._get_valid_numbers(opponent_parity)
        if len(valid_numbers) == 0:
            # Opponent has no valid moves; current player wins
            reward = 1
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Continue game
        reward = -10  # Penalty to encourage shorter games
        observation = self._get_observation()
        return observation, reward, False, False, {}

    def render(self):
        number_pool_list = [i + 1 for i in range(20) if self.number_pool[i] == 1]
        p1_sequence = f"Player 1 Total Sum: {self.player_sums[1]}, Parity: {'Odd' if self.player_parities[1]==-1 else ('Even' if self.player_parities[1]==1 else 'Undefined')}"
        p2_sequence = f"Player 2 Total Sum: {self.player_sums[-1]}, Parity: {'Odd' if self.player_parities[-1]==-1 else ('Even' if self.player_parities[-1]==1 else 'Undefined')}"
        current_player_str = (
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"
        )
        render_str = (
            "--- Parity Chase Game State ---\n"
            f"Available Numbers: {number_pool_list}\n"
            f"{p1_sequence}\n"
            f"{p2_sequence}\n"
            f"{current_player_str}\n"
            "--------------------------------\n"
        )
        return render_str

    def valid_moves(self):
        # Return a list of valid actions (indices) for the current player
        player_parity = self.player_parities[self.current_player]
        valid_moves = []
        for i in range(20):
            if self.number_pool[i] == 1:
                number = i + 1
                if player_parity == 0:
                    # First move, any number is valid
                    valid_moves.append(i)
                else:
                    required_parity = player_parity
                    if (number % 2 == 1 and required_parity == -1) or (
                        number % 2 == 0 and required_parity == 1
                    ):
                        valid_moves.append(i)
        return valid_moves

    def _get_observation(self):
        # Create observation array
        observation = np.zeros(25, dtype=np.int32)
        observation[0:20] = self.number_pool
        observation[20] = self.player_sums[1]
        observation[21] = self.player_parities[1]
        observation[22] = self.player_sums[-1]
        observation[23] = self.player_parities[-1]
        observation[24] = self.current_player
        return observation

    def _get_valid_numbers(self, player_parity):
        valid_numbers = []
        for i in range(20):
            if self.number_pool[i] == 1:
                number = i + 1
                if player_parity == 0:
                    # Player's first move, any number is valid
                    valid_numbers.append(number)
                else:
                    required_parity = player_parity
                    if (number % 2 == 1 and required_parity == -1) or (
                        number % 2 == 0 and required_parity == 1
                    ):
                        valid_numbers.append(number)
        return valid_numbers
