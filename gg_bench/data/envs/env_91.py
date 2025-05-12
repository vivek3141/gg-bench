import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 8 possible actions
        # 0: Move up, no challenge
        # 1: Move down, no challenge
        # 2: Move left, no challenge
        # 3: Move right, no challenge
        # 4: Move up, challenge
        # 5: Move down, challenge
        # 6: Move left, challenge
        # 7: Move right, challenge
        self.action_space = spaces.Discrete(8)

        # Define observation space
        # The observation is a 5x5x4 array
        # Channel 0: Cell values (1-5)
        # Channel 1: Visited by Player 1 (0 or 1)
        # Channel 2: Visited by Player 2 (0 or 1)
        # Channel 3: Player positions (0: empty, 1: P1, 2: P2)
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(5, 5, 4), dtype=np.int8
        )

        self.grid_size = 5
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the grid with random cell values between 1 and 5
        self.grid_values = np.random.randint(
            1, 6, size=(self.grid_size, self.grid_size)
        )

        # Initialize visited cells for both players
        self.visited_p1 = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.visited_p2 = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Mark starting positions as visited
        self.p1_pos = [0, 0]
        self.p2_pos = [4, 4]
        self.visited_p1[tuple(self.p1_pos)] = 1
        self.visited_p2[tuple(self.p2_pos)] = 1

        # Initialize player positions
        self.player_positions = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.int8
        )
        self.player_positions[tuple(self.p1_pos)] = 1
        self.player_positions[tuple(self.p2_pos)] = 2

        # Initialize scores
        self.score_p1 = 0
        self.score_p2 = 0
        # Starting positions don't add to score

        # Set current player: Player 1 starts
        self.current_player = 1

        self.done = False

        observation = self._get_observation()
        return observation, {}  # observation, info

    def step(self, action):
        if self.done:
            # If game is over, do nothing
            return self._get_observation(), 0, True, False, {}

        # Map action to movement and challenge decision
        move_action = action % 4
        challenge = action >= 4

        # Get movement direction
        if move_action == 0:  # Move up
            move = (-1, 0)
        elif move_action == 1:  # Move down
            move = (1, 0)
        elif move_action == 2:  # Move left
            move = (0, -1)
        elif move_action == 3:  # Move right
            move = (0, 1)
        else:
            # Invalid action
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Get current player position
        if self.current_player == 1:
            current_pos = self.p1_pos
            opponent_pos = self.p2_pos
            visited = self.visited_p1
            opponent_visited = self.visited_p2
            score = self.score_p1
            opponent_score = self.score_p2
            starting_pos = [0, 0]
            opponent_starting_pos = [4, 4]
        else:
            current_pos = self.p2_pos
            opponent_pos = self.p1_pos
            visited = self.visited_p2
            opponent_visited = self.visited_p1
            score = self.score_p2
            opponent_score = self.score_p1
            starting_pos = [4, 4]
            opponent_starting_pos = [0, 0]

        new_row = current_pos[0] + move[0]
        new_col = current_pos[1] + move[1]

        # Check if the move is within grid boundaries
        if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Check if the cell has been visited by the player
        if visited[new_row, new_col] == 1:
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Check if the cell is occupied by opponent
        if [new_row, new_col] == opponent_pos:
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Move is valid, update position
        current_pos[0], current_pos[1] = new_row, new_col
        visited[new_row, new_col] = 1

        # Update player's score with the cell's value
        cell_value = self.grid_values[new_row, new_col]
        score += cell_value

        # Update player_positions grid
        self.player_positions = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.int8
        )
        self.player_positions[tuple(self.p1_pos)] = 1
        self.player_positions[tuple(self.p2_pos)] = 2

        # Check for victory condition
        if current_pos == opponent_starting_pos:
            # Current player wins
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Check if opponent is adjacent
        adjacent = self._is_adjacent(current_pos, opponent_pos)

        # Handle challenge
        if challenge:
            if adjacent:
                # Initiate challenge
                if score > opponent_score:
                    # Current player wins challenge
                    # Opponent returns to starting position and score resets
                    opponent_pos[0], opponent_pos[1] = opponent_starting_pos
                    opponent_score = 0
                    opponent_visited[tuple(opponent_pos)] = 1

                    # Update player_positions grid
                    self.player_positions = np.zeros(
                        (self.grid_size, self.grid_size), dtype=np.int8
                    )
                    self.player_positions[tuple(self.p1_pos)] = 1
                    self.player_positions[tuple(self.p2_pos)] = 2

                    # Scores updated
                    if self.current_player == 1:
                        self.score_p1 = score
                        self.score_p2 = opponent_score
                    else:
                        self.score_p2 = score
                        self.score_p1 = opponent_score

                elif score < opponent_score:
                    # Opponent wins challenge
                    # Current player returns to starting position and score resets
                    current_pos[0], current_pos[1] = starting_pos
                    score = 0
                    visited[...] = 0  # Visited cells remain as is
                    visited[tuple(current_pos)] = 1

                    # Update player_positions grid
                    self.player_positions = np.zeros(
                        (self.grid_size, self.grid_size), dtype=np.int8
                    )
                    self.player_positions[tuple(self.p1_pos)] = 1
                    self.player_positions[tuple(self.p2_pos)] = 2

                    # Scores updated
                    if self.current_player == 1:
                        self.score_p1 = score
                        self.score_p2 = opponent_score
                    else:
                        self.score_p2 = score
                        self.score_p1 = opponent_score

                else:
                    # Tie, no effect
                    if self.current_player == 1:
                        self.score_p1 = score
                    else:
                        self.score_p2 = score
            else:
                # Invalid action: challenge when opponent not adjacent
                reward = -10
                self.done = True
                return self._get_observation(), reward, True, False, {}
        else:
            # No challenge
            if self.current_player == 1:
                self.score_p1 = score
            else:
                self.score_p2 = score

        # After current player's turn, check if opponent can move
        opponent_valid_moves = self._get_valid_moves(
            opponent_pos, opponent_visited, current_pos
        )
        if len(opponent_valid_moves) == 0:
            # Opponent cannot move, current player wins
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Switch current player
        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

        # No reward yet
        reward = 0
        return self._get_observation(), reward, False, False, {}

    def render(self):
        # Create a visual representation of the grid
        grid_visual = ""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_value = self.grid_values[i, j]
                player_here = ""
                if self.player_positions[i, j] == 1:
                    player_here = "P1"
                elif self.player_positions[i, j] == 2:
                    player_here = "P2"

                grid_visual += f"({i},{j}):{cell_value}"
                if player_here != "":
                    grid_visual += f"[{player_here}]"
                else:
                    grid_visual += "     "
                grid_visual += "  "
            grid_visual += "\n"
        return grid_visual

    def valid_moves(self):
        # Return a list of valid actions (as indices of action_space)
        valid_actions = []
        # For the current player
        if self.current_player == 1:
            current_pos = self.p1_pos
            visited = self.visited_p1
            opponent_pos = self.p2_pos
        else:
            current_pos = self.p2_pos
            visited = self.visited_p2
            opponent_pos = self.p1_pos

        for action in range(8):
            move_action = action % 4
            challenge = action >= 4

            # Get movement direction
            if move_action == 0:  # Up
                move = (-1, 0)
            elif move_action == 1:  # Down
                move = (1, 0)
            elif move_action == 2:  # Left
                move = (0, -1)
            elif move_action == 3:  # Right
                move = (0, 1)

            new_row = current_pos[0] + move[0]
            new_col = current_pos[1] + move[1]

            # Check movement validity
            if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
                continue  # Move is off the grid, invalid

            if visited[new_row, new_col] == 1:
                continue  # Already visited by player

            if [new_row, new_col] == opponent_pos:
                continue  # Occupied by opponent

            # If challenge is attempted, check if opponent would be adjacent after move
            if challenge:
                temp_pos = [new_row, new_col]
                if not self._is_adjacent(temp_pos, opponent_pos):
                    continue  # Cannot challenge, opponent not adjacent

            valid_actions.append(action)

        return valid_actions

    def _get_observation(self):
        # Construct the observation
        obs = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.int8)
        obs[..., 0] = self.grid_values
        obs[..., 1] = self.visited_p1
        obs[..., 2] = self.visited_p2
        obs[..., 3] = self.player_positions
        return obs

    def _is_adjacent(self, pos1, pos2):
        # Check if two positions are adjacent (not diagonal)
        row_diff = abs(pos1[0] - pos2[0])
        col_diff = abs(pos1[1] - pos2[1])
        return (row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)

    def _get_valid_moves(self, position, visited, opponent_pos):
        # Return list of valid moves for a player at position
        valid_moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
        for move in directions:
            new_row = position[0] + move[0]
            new_col = position[1] + move[1]
            if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
                continue  # Move is off the grid
            if visited[new_row, new_col] == 1:
                continue  # Already visited
            if [new_row, new_col] == opponent_pos:
                continue  # Occupied by opponent
            valid_moves.append((new_row, new_col))
        return valid_moves
