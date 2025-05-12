import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - move -2, 1 - move -1, 2 - move +1, 3 - move +2
        self.action_space = spaces.Discrete(4)
        # Observation: 3 layers of size 15 (positions 1 to 15)
        # Layer 0: current player's token position (0 or 1)
        # Layer 1: opponent's token position (0 or 1)
        # Layer 2: revealed traps (0 or 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, 15), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize positions of tokens
        self.player_positions = {1: 0, 2: 14}  # Positions 0 to 14 correspond to 1 to 15

        # Randomly place traps for both players (positions 1 to 13 correspond to 2 to 14)
        positions = np.arange(1, 14)
        np.random.shuffle(positions)
        self.player_traps = {
            1: positions[:3].tolist(),  # Player 1's traps
            2: positions[3:6].tolist(),  # Player 2's traps
        }

        # Initialize revealed traps
        self.revealed_traps = []

        # Set the current player (1 or 2)
        self.current_player = 1

        self.done = False

        return self.get_observation(), {}

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        move_dict = {0: -2, 1: -1, 2: 1, 3: 2}
        move = move_dict[action]

        current_pos = self.player_positions[self.current_player]
        new_pos = current_pos + move

        # Check move validity
        if new_pos < 0 or new_pos > 14:
            # Invalid move, beyond number line boundaries
            self.done = True
            reward = -10
            return self.get_observation(), reward, True, False, {}

        # Check for trap
        opponent = 1 if self.current_player == 2 else 2
        opponent_traps = self.player_traps[opponent]

        if (new_pos in opponent_traps) and (new_pos not in self.revealed_traps):
            # Trap is sprung
            self.revealed_traps.append(new_pos)
            # Send token back to base
            if self.current_player == 1:
                self.player_positions[1] = 0
            else:
                self.player_positions[2] = 14
            reward = 0
            done = False
        else:
            # Move token to new position
            self.player_positions[self.current_player] = new_pos

            # Check for win condition
            if (self.current_player == 1 and new_pos == 14) or (
                self.current_player == 2 and new_pos == 0
            ):
                # Current player wins
                self.done = True
                reward = 1
                return self.get_observation(), reward, True, False, {}
            else:
                reward = 0
                done = False

        # Switch current player
        self.current_player = 1 if self.current_player == 2 else 2

        return self.get_observation(), reward, done, False, {}

    def render(self):
        # Create a string representation of the number line
        line = ""
        for i in range(15):
            pos_str = ""
            if self.player_positions[1] == i and self.player_positions[2] == i:
                pos_str += "B"  # Both players on the same position
            elif self.player_positions[1] == i:
                pos_str += "1"  # Player 1's token
            elif self.player_positions[2] == i:
                pos_str += "2"  # Player 2's token
            else:
                pos_str += "."

            if i in self.revealed_traps:
                pos_str += "T"  # Revealed trap
            else:
                pos_str += "  "

            line += f"{i+1}:{pos_str} "

        return line.strip()

    def valid_moves(self):
        current_pos = self.player_positions[self.current_player]
        valid_actions = []
        move_dict = {0: -2, 1: -1, 2: 1, 3: 2}
        for action, move in move_dict.items():
            new_pos = current_pos + move
            if 0 <= new_pos <= 14:
                valid_actions.append(action)
        return valid_actions

    def get_observation(self):
        obs = np.zeros((3, 15), dtype=np.int8)

        # Layer 0: current player's token position
        player_pos = self.player_positions[self.current_player]
        obs[0, player_pos] = 1

        # Layer 1: opponent's token position
        opponent = 1 if self.current_player == 2 else 2
        opponent_pos = self.player_positions[opponent]
        obs[1, opponent_pos] = 1

        # Layer 2: revealed traps
        for pos in self.revealed_traps:
            obs[2, pos] = 1

        return obs
