import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 -> move 1 step, 1 -> move 2 steps, 2 -> move 3 steps
        self.action_space = spaces.Discrete(3)

        # Observation space:
        # For each position (0-10), we have 4 features:
        # [player_token, opponent_token, own_hurdle, revealed_opponent_hurdle]
        # Flattened into a vector of length 44 (positions 0 to 10 inclusive)
        self.observation_space = spaces.Box(low=0, high=1, shape=(44,), dtype=np.int8)

        self.track_length = 11  # Positions 0 to 10 inclusive
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.player_positions = [0, 0]
        self.done = False

        # Randomly place hurdles for both players at positions 3 to 9 inclusive
        available_positions = list(range(3, 10))
        np.random.shuffle(available_positions)
        self.hurdles = [
            sorted(available_positions[:3]),
            sorted(available_positions[3:6]),
        ]
        # Initialize revealed hurdles for each player
        self.revealed_hurdles = [[], []]

        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        steps = action + 1  # Map action 0,1,2 to steps 1,2,3
        player = self.current_player
        opponent = 1 - self.current_player
        new_position = self.player_positions[player] + steps

        if new_position > 10:
            # Move beyond position 10 is invalid
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check for opponent's hurdle at new_position
        if new_position in self.hurdles[opponent]:
            # Hit a hurdle
            # Player stays at previous position
            # Remove the hurdle from opponent's hurdles and add to revealed
            self.hurdles[opponent].remove(new_position)
            self.revealed_hurdles[player].append(new_position)
            # Turn ends, switch player
            reward = 0
            terminated = False
        else:
            # Successful move
            self.player_positions[player] = new_position
            # Check for win
            if new_position == 10:
                self.done = True
                reward = 1
                terminated = True
            else:
                reward = 0
                terminated = False

        # Switch current player
        self.current_player = opponent

        return self._get_observation(), reward, terminated, False, {}

    def render(self):
        board_str = ""
        for pos in range(self.track_length):
            tokens = []
            if self.player_positions[0] == pos:
                tokens.append("P1")
            if self.player_positions[1] == pos:
                tokens.append("P2")
            if pos in self.hurdles[0]:
                tokens.append("H1")
            if pos in self.hurdles[1]:
                tokens.append("H2")
            if pos in self.revealed_hurdles[0] or pos in self.revealed_hurdles[1]:
                tokens.append("RH")
            pos_str = f"Position {pos}: " + ", ".join(tokens)
            board_str += pos_str + "\n"
        return board_str

    def valid_moves(self):
        # Return list of valid actions (0,1,2) that do not move beyond position 10
        current_pos = self.player_positions[self.current_player]
        valid_actions = []
        for action in range(3):
            steps = action + 1
            new_position = current_pos + steps
            if new_position <= 10:
                valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        # Observation is a flattened array of shape (44,)
        # For positions 0 to 10, we have 4 features:
        # [player_token, opponent_token, own_hurdle, revealed_opponent_hurdle]
        obs = np.zeros((self.track_length, 4), dtype=np.int8)

        player = self.current_player
        opponent = 1 - self.current_player

        # Player token
        obs[self.player_positions[player], 0] = 1
        # Opponent token
        obs[self.player_positions[opponent], 1] = 1
        # Own hurdles
        for pos in self.hurdles[player]:
            obs[pos, 2] = 1
        # Revealed opponent's hurdles
        for pos in self.revealed_hurdles[player]:
            obs[pos, 3] = 1

        return obs.flatten()
