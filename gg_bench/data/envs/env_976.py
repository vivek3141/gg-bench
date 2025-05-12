import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)

        # Observation space: 5x5 grid with integer encoding
        # 0: Empty
        # 1: Current player's position
        # 2: Opponent's position
        # 3: Obstacle placed by current player
        # 4: Revealed obstacle placed by opponent
        self.observation_space = spaces.Box(low=0, high=4, shape=(5, 5), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the grid
        self.grid = np.zeros((5, 5), dtype=np.int8)

        # Place the players
        self.p1_pos = [0, 0]  # (1,1)
        self.p2_pos = [4, 4]  # (5,5)

        # Place obstacles
        self.p1_obstacles = []  # Obstacles placed by Player 1
        self.p2_obstacles = []  # Obstacles placed by Player 2

        # Obstacles revealed to each player
        self.p1_revealed_obstacles = []
        self.p2_revealed_obstacles = []

        # Randomly assign obstacles
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        available_positions = [
            (i, j)
            for i in range(5)
            for j in range(5)
            if (i, j) not in [(0, 0), (4, 4), (2, 2)]
        ]

        # Player 1's obstacles
        p1_obs_positions = self.np_random.choice(
            len(available_positions), 3, replace=False
        )
        for idx in p1_obs_positions:
            pos = available_positions[idx]
            self.p1_obstacles.append(pos)

        # Remove used positions
        available_positions = [
            pos
            for idx, pos in enumerate(available_positions)
            if idx not in p1_obs_positions
        ]

        # Player 2's obstacles
        p2_obs_positions = self.np_random.choice(
            len(available_positions), 3, replace=False
        )
        for idx in p2_obs_positions:
            pos = available_positions[idx]
            self.p2_obstacles.append(pos)

        # Game state
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Turn skipping flags
        self.missed_turn_p1 = False
        self.missed_turn_p2 = False

        return self._get_observation(), {}  # Observation, info

    def _get_observation(self):
        # Observation includes known obstacles and positions
        grid_obs = np.zeros((5, 5), dtype=np.int8)

        # Current player's position
        if self.current_player == 1:
            grid_obs[self.p1_pos[0]][self.p1_pos[1]] = 1
            grid_obs[self.p2_pos[0]][self.p2_pos[1]] = 2
            # Own obstacles
            for pos in self.p1_obstacles:
                grid_obs[pos[0]][pos[1]] = 3
            # Revealed opponent obstacles
            for pos in self.p1_revealed_obstacles:
                grid_obs[pos[0]][pos[1]] = 4
        else:
            grid_obs[self.p2_pos[0]][self.p2_pos[1]] = 1
            grid_obs[self.p1_pos[0]][self.p1_pos[1]] = 2
            # Own obstacles
            for pos in self.p2_obstacles:
                grid_obs[pos[0]][pos[1]] = 3
            # Revealed opponent obstacles
            for pos in self.p2_revealed_obstacles:
                grid_obs[pos[0]][pos[1]] = 4

        return grid_obs

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Turn skipping due to obstacle encounter
        if self.current_player == 1 and self.missed_turn_p1:
            self.missed_turn_p1 = False
            self.current_player = 2
            return self._get_observation(), 0, False, False, {}
        if self.current_player == 2 and self.missed_turn_p2:
            self.missed_turn_p2 = False
            self.current_player = 1
            return self._get_observation(), 0, False, False, {}

        # Map action to movement
        move_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        if action not in move_map:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        dx, dy = move_map[action]

        # Get current player state
        (
            curr_pos,
            opponent_pos,
            own_obstacles,
            opponent_obstacles,
            revealed_opponent_obstacles,
            missed_turn,
        ) = self._get_player_state(self.current_player)

        new_x, new_y = curr_pos[0] + dx, curr_pos[1] + dy

        # Check grid boundaries
        if not (0 <= new_x < 5 and 0 <= new_y < 5):
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check if moving onto opponent
        if [new_x, new_y] == opponent_pos:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check for obstacles
        pos_tuple = (new_x, new_y)
        if pos_tuple in own_obstacles:
            # Move successfully onto own obstacle
            pass
        elif pos_tuple in opponent_obstacles:
            # Encountered opponent's obstacle
            if self.current_player == 1:
                self.p1_revealed_obstacles.append(pos_tuple)
                self.missed_turn_p1 = True
            else:
                self.p2_revealed_obstacles.append(pos_tuple)
                self.missed_turn_p2 = True
            # Remain in place
            self.current_player = 2 if self.current_player == 1 else 1
            return self._get_observation(), 0, False, False, {}

        # Move is valid
        if self.current_player == 1:
            self.p1_pos = [new_x, new_y]
        else:
            self.p2_pos = [new_x, new_y]

        # Check for win condition
        if [new_x, new_y] == [2, 2]:  # Center position (3,3)
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch turns
        self.current_player = 2 if self.current_player == 1 else 1
        return self._get_observation(), 0, False, False, {}

    def _get_player_state(self, player):
        if player == 1:
            curr_pos = self.p1_pos
            opponent_pos = self.p2_pos
            own_obstacles = self.p1_obstacles
            opponent_obstacles = self.p2_obstacles
            revealed_opponent_obstacles = self.p1_revealed_obstacles
            missed_turn = self.missed_turn_p1
        else:
            curr_pos = self.p2_pos
            opponent_pos = self.p1_pos
            own_obstacles = self.p2_obstacles
            opponent_obstacles = self.p1_obstacles
            revealed_opponent_obstacles = self.p2_revealed_obstacles
            missed_turn = self.missed_turn_p2
        return (
            curr_pos,
            opponent_pos,
            own_obstacles,
            opponent_obstacles,
            revealed_opponent_obstacles,
            missed_turn,
        )

    def render(self):
        grid_obs = self._get_observation()
        grid_str = ""
        symbols = {0: ".", 1: "A", 2: "B", 3: "X", 4: "O"}
        for i in range(5):
            for j in range(5):
                cell = grid_obs[i][j]
                grid_str += f"{symbols[cell]} "
            grid_str += "\n"
        return grid_str

    def valid_moves(self):
        moves = []
        move_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        (
            curr_pos,
            opponent_pos,
            own_obstacles,
            opponent_obstacles,
            revealed_opponent_obstacles,
            missed_turn,
        ) = self._get_player_state(self.current_player)

        if missed_turn:
            return []

        for action, (dx, dy) in move_map.items():
            new_x, new_y = curr_pos[0] + dx, curr_pos[1] + dy
            # Check grid boundaries
            if not (0 <= new_x < 5 and 0 <= new_y < 5):
                continue
            # Check if moving onto opponent
            if [new_x, new_y] == opponent_pos:
                continue
            moves.append(action)
        return moves
