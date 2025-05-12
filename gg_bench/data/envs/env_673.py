import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define grid size
        self.grid_size = 5

        # Define action and observation space
        # There are 8 movement actions and 25 trap placement actions (for each cell in the grid)
        self.action_space = spaces.Discrete(33)

        # Observation space: 25 cells with values from -2 to 3
        # -2: opponent's flag
        # -1: opponent's agent
        # 0: empty cell
        # 1: own agent
        # 2: own flag
        # 3: own trap
        self.observation_space = spaces.Box(low=-2, high=3, shape=(25,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board as a 5x5 grid
        self.board = np.zeros((5, 5), dtype=np.int8)

        # Flag positions
        self.player_flags = {
            1: (0, 0),  # Player 1's flag at (0,0)
            -1: (4, 4),  # Player 2's flag at (4,4)
        }

        # Agent positions
        self.agent_positions = {
            1: self.player_flags[1],
            -1: self.player_flags[-1],
        }

        # Traps for each player
        self.traps = {
            1: [],  # Player 1's traps
            -1: [],  # Player 2's traps
        }

        # Traps remaining for each player
        self.traps_remaining = {
            1: 2,
            -1: 2,
        }

        # Place flags on the board
        self.board[self.player_flags[1][1], self.player_flags[1][0]] = (
            2  # Player 1's flag
        )
        self.board[self.player_flags[-1][1], self.player_flags[-1][0]] = (
            -2
        )  # Player 2's flag

        # Place agents on the board
        self.board[self.agent_positions[1][1], self.agent_positions[1][0]] = 1
        self.board[self.agent_positions[-1][1], self.agent_positions[-1][0]] = -1

        # Set current player
        self.current_player = 1

        # Game over flag
        self.done = False

        # Return the initial observation and info
        return self.get_observation(), {}

    def get_observation(self):
        # Build observation from current player's perspective
        obs = np.zeros((5, 5), dtype=np.int8)

        # Own flag
        own_flag_pos = self.player_flags[self.current_player]
        obs[own_flag_pos[1], own_flag_pos[0]] = 2

        # Own agent
        own_agent_pos = self.agent_positions[self.current_player]
        obs[own_agent_pos[1], own_agent_pos[0]] = 1

        # Own traps
        for pos in self.traps[self.current_player]:
            obs[pos[1], pos[0]] = 3

        # Opponent's flag
        opp_player = -self.current_player
        opp_flag_pos = self.player_flags[opp_player]
        obs[opp_flag_pos[1], opp_flag_pos[0]] = -2

        # Opponent's agent
        opp_agent_pos = self.agent_positions[opp_player]
        obs[opp_agent_pos[1], opp_agent_pos[0]] = -1

        # Flatten the observation to a 1D array
        obs = obs.flatten()

        return obs

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            return self.get_observation(), -10, True, False, {}

        # Execute the action
        if action >= 0 and action <= 7:
            # Movement action
            dx, dy = self.get_movement_direction(action)
            self.move_agent(dx, dy)
        else:
            # Trap placement action
            self.place_trap(action - 8)

        # Check for win condition
        reward = 0
        own_agent_pos = self.agent_positions[self.current_player]
        opp_flag_pos = self.player_flags[-self.current_player]

        if own_agent_pos == opp_flag_pos:
            # Current player wins
            self.done = True
            reward = 1
            return self.get_observation(), reward, True, False, {}

        # Swap current player
        self.current_player *= -1

        # Return observation, reward, done, info
        return self.get_observation(), reward, False, False, {}

    def get_movement_direction(self, action):
        # Map action IDs 0-7 to movement directions
        movements = [
            (-1, -1),  # NW
            (0, -1),  # N
            (1, -1),  # NE
            (-1, 0),  # W
            (1, 0),  # E
            (-1, 1),  # SW
            (0, 1),  # S
            (1, 1),  # SE
        ]
        return movements[action]

    def move_agent(self, dx, dy):
        # Get current agent position
        x, y = self.agent_positions[self.current_player]

        # Calculate new position
        new_x = x + dx
        new_y = y + dy

        # Remove agent from current position on the board
        self.board[y, x] = 0

        # Temporary position before checking for traps
        temp_pos = (new_x, new_y)

        # Check for traps
        opp_player = -self.current_player
        if temp_pos in self.traps[opp_player]:
            # Agent triggers opponent's trap
            self.traps[opp_player].remove(temp_pos)
            # Send agent back to own flag
            new_x, new_y = self.player_flags[self.current_player]
            self.agent_positions[self.current_player] = (new_x, new_y)
        else:
            # No trap triggered, agent moves to new position
            self.agent_positions[self.current_player] = temp_pos

        # Place agent on the board
        self.board[new_y, new_x] = 1 if self.current_player == 1 else -1

    def place_trap(self, cell_index):
        # Convert cell index to coordinates
        x = cell_index % 5
        y = cell_index // 5

        # Place the trap
        self.traps[self.current_player].append((x, y))
        self.traps_remaining[self.current_player] -= 1

    def valid_moves(self):
        valid_actions = []

        x, y = self.agent_positions[self.current_player]

        # Movement actions
        for action in range(8):
            dx, dy = self.get_movement_direction(action)
            new_x = x + dx
            new_y = y + dy

            if 0 <= new_x < 5 and 0 <= new_y < 5:
                # Check if new position is not occupied by own agent or own trap, and within grid
                opp_agent_pos = self.agent_positions[-self.current_player]
                own_traps = self.traps[self.current_player]
                if (new_x, new_y) != self.agent_positions[1] and (
                    new_x,
                    new_y,
                ) != self.agent_positions[-1]:
                    if (new_x, new_y) not in own_traps:
                        valid_actions.append(action)

        # Trap placement actions
        if self.traps_remaining[self.current_player] > 0:
            for cell_index in range(25):
                x_cell = cell_index % 5
                y_cell = cell_index // 5
                # Skip flag positions
                if (x_cell, y_cell) == self.player_flags[1] or (
                    x_cell,
                    y_cell,
                ) == self.player_flags[-1]:
                    continue
                # Skip occupied positions
                if (x_cell, y_cell) == self.agent_positions[1] or (
                    x_cell,
                    y_cell,
                ) == self.agent_positions[-1]:
                    continue
                # Skip if cell already has one of own traps
                if (x_cell, y_cell) in self.traps[self.current_player]:
                    continue
                # Valid trap placement
                valid_actions.append(8 + cell_index)

        return valid_actions

    def render(self):
        # Build a representation of the board from current player's perspective
        board_repr = np.zeros((5, 5), dtype=np.int8)

        # Own flag
        own_flag_pos = self.player_flags[self.current_player]
        board_repr[own_flag_pos[1], own_flag_pos[0]] = 2

        # Own agent
        own_agent_pos = self.agent_positions[self.current_player]
        board_repr[own_agent_pos[1], own_agent_pos[0]] = 1

        # Own traps
        for pos in self.traps[self.current_player]:
            board_repr[pos[1], pos[0]] = 3

        # Opponent's flag
        opp_player = -self.current_player
        opp_flag_pos = self.player_flags[opp_player]
        board_repr[opp_flag_pos[1], opp_flag_pos[0]] = -2

        # Opponent's agent
        opp_agent_pos = self.agent_positions[opp_player]
        board_repr[opp_agent_pos[1], opp_agent_pos[0]] = -1

        # Build the board string
        board_str = "  A B C D E\n"
        for y in range(5):
            row_str = str(y + 1) + " "
            for x in range(5):
                cell = board_repr[y, x]
                if cell == 0:
                    row_str += ". "
                elif cell == 1:
                    row_str += "A "  # Own agent
                elif cell == -1:
                    row_str += "B "  # Opponent's agent
                elif cell == 2:
                    row_str += "F "  # Own flag
                elif cell == -2:
                    row_str += "f "  # Opponent's flag
                elif cell == 3:
                    row_str += "T "  # Own trap
            board_str += row_str + "\n"
        print("Current player:", "Player 1" if self.current_player == 1 else "Player 2")
        print(board_str)
