import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: move forward by 1, 2, or 3 positions
        self.action_space = spaces.Discrete(3)

        # Observation space: positions of Player 1 and Player 2, current player indicator
        self.observation_space = spaces.Box(
            low=np.array([-5, -5, -1], dtype=np.int8),
            high=np.array([5, 5, 1], dtype=np.int8),
            shape=(3,),
            dtype=np.int8,
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_positions = np.array(
            [-5, 5], dtype=np.int8
        )  # Player 1 at -5, Player 2 at +5
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        return self._get_observation(), {}  # Return observation and info

    def _get_observation(self):
        # Observation includes positions of both players and current player indicator
        return np.array(
            [self.player_positions[0], self.player_positions[1], self.current_player],
            dtype=np.int8,
        )

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if action is valid
        if action not in [0, 1, 2]:
            reward = -10
            self.done = True
            return (
                self._get_observation(),
                reward,
                True,
                False,
                {"reason": "Invalid action"},
            )

        move = action + 1  # Mapping action to move (0->1, 1->2, 2->3)
        if self.current_player == 1:
            player_index = 0
            opponent_index = 1
            direction = 1
            opponent_flag = 5
        else:
            player_index = 1
            opponent_index = 0
            direction = -1
            opponent_flag = -5

        # Calculate the maximum allowed move without exceeding the opponent's flag
        distance_to_flag = abs(opponent_flag - self.player_positions[player_index])
        max_move = min(3, distance_to_flag)

        # Validate move
        if move > max_move:
            reward = -10
            self.done = True
            return (
                self._get_observation(),
                reward,
                True,
                False,
                {"reason": "Invalid move: Exceeds opponent's flag"},
            )

        # Update player's position
        self.player_positions[player_index] += direction * move

        # Check for win conditions
        if self.player_positions[player_index] == opponent_flag:
            # Captured the opponent's flag
            reward = 1
            self.done = True
            return (
                self._get_observation(),
                reward,
                True,
                False,
                {"result": "Captured opponent's flag"},
            )
        elif (
            self.player_positions[player_index] == self.player_positions[opponent_index]
        ):
            # Landed on the opponent's position
            reward = 1
            self.done = True
            return (
                self._get_observation(),
                reward,
                True,
                False,
                {"result": "Captured opponent on same position"},
            )
        else:
            # Continue the game
            self.current_player *= -1  # Switch turns
            reward = 0
            return self._get_observation(), reward, False, False, {}

    def render(self):
        # Create a visual representation of the board state
        board_repr = ""
        for pos in range(-5, 6):
            if pos == self.player_positions[0] and pos == self.player_positions[1]:
                board_repr += "[B]"  # Both players on the same position
            elif pos == self.player_positions[0]:
                board_repr += "[1]"
            elif pos == self.player_positions[1]:
                board_repr += "[2]"
            else:
                board_repr += "[ ]"
        board_repr += f"\nPlayer 1 position: {self.player_positions[0]}, Player 2 position: {self.player_positions[1]}"
        board_repr += f'\nCurrent player: {"Player 1" if self.current_player == 1 else "Player 2"}'
        return board_repr

    def valid_moves(self):
        if self.done:
            return []
        if self.current_player == 1:
            player_index = 0
            opponent_flag = 5
        else:
            player_index = 1
            opponent_flag = -5

        # Calculate valid moves based on distance to opponent's flag
        distance_to_flag = abs(opponent_flag - self.player_positions[player_index])
        max_move = min(3, distance_to_flag)
        valid_actions = [i for i in range(max_move)]
        return valid_actions
