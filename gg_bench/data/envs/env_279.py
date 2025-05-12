import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - move by 1, 1 - move by 2, 2 - move by 3
        self.action_space = spaces.Discrete(3)
        # Observation: current_player, position of player1, position of player2
        self.observation_space = spaces.Box(
            low=np.array([-1, 0, 0], dtype=np.int32),
            high=np.array([1, 20, 20], dtype=np.int32),
            dtype=np.int32,
        )

        # Game setup
        self.mines = [5, 10, 15, 18]
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_positions = {1: 0, -1: 0}
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(
            [self.current_player, self.player_positions[1], self.player_positions[-1]],
            dtype=np.int32,
        )

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        valid_moves = self.valid_moves()

        if action not in valid_moves:
            # Invalid move
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        move_distance = action + 1  # Actions are 0,1,2 corresponding to moving by 1,2,3
        new_position = self.player_positions[self.current_player] + move_distance

        # Ensure new_position does not exceed 20
        if new_position > 20:
            new_position = 20

        # Update player's position
        self.player_positions[self.current_player] = new_position

        reward = -10  # Penalty per move

        if new_position in self.mines:
            # Player lands on a mine, loses
            reward = -1  # Negative reward for losing
            self.done = True
            return self._get_obs(), reward, True, False, {}
        elif new_position == 20:
            # Player reaches position 20, wins
            reward += 1  # Reward = -10 + 1 = -9
            self.done = True
            return self._get_obs(), reward, True, False, {}
        else:
            # Continue game
            pass

        # Switch current player
        self.current_player *= -1

        return self._get_obs(), reward, False, False, {}

    def render(self):
        state_str = f"Player 1 is at position {self.player_positions[1]}.\n"
        state_str += f"Player 2 is at position {self.player_positions[-1]}.\n"
        state_str += f"Current player: {'Player 1' if self.current_player == 1 else 'Player 2'}.\n"
        state_str += f"Mines are at positions: {', '.join(map(str, self.mines))}.\n"
        return state_str

    def valid_moves(self):
        valid_moves = []
        current_position = self.player_positions[self.current_player]
        for action in range(3):  # Actions 0,1,2
            move_distance = action + 1
            new_position = current_position + move_distance
            if new_position > 20:
                continue
            valid_moves.append(action)
        return valid_moves
