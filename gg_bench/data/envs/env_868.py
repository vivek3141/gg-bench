import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: Move forward 1, 2, or 3 steps
        self.action_space = spaces.Discrete(3)

        # Observation space: Positions of both players and mines on the number line
        # For each of the 20 positions, we have 4 features:
        # [Player A here, Player B here, Player A's mine here, Player B's mine here]
        self.observation_space = spaces.Box(low=0, high=1, shape=(20, 4), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize positions
        self.player_positions = {"A": 1, "B": 1}

        # Randomly place mines for both players between positions 2 and 19
        available_positions = list(range(2, 20))
        self.player_mines = {}

        np.random.shuffle(available_positions)
        self.player_mines["A"] = sorted(available_positions[:3])
        self.player_mines["B"] = sorted(available_positions[3:6])

        # Initialize game variables
        self.current_player = "A"
        self.done = False
        self.winner = None

        # Build initial observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, self.done, False, {}

        # Map action to movement (1, 2, or 3 steps)
        movement = action + 1  # action is 0,1,2 corresponding to moves of 1,2,3 steps

        # Get current player
        player = self.current_player
        opponent = "B" if player == "A" else "A"

        # Compute new position
        new_position = self.player_positions[player] + movement

        # Check if movement is valid (cannot move beyond position 20)
        if new_position > 20:
            new_position = 20  # Cap at 20

        # Update the player's position
        self.player_positions[player] = new_position

        reward = -10  # Default reward for a valid move

        # Check for win condition
        if new_position == 20:
            self.done = True
            self.winner = player
            reward = 1  # Reward for winning
        else:
            # Check if stepped on opponent's mine
            if new_position in self.player_mines[opponent]:
                self.done = True
                self.winner = opponent  # Opponent wins
                # Reward remains -10 for valid move (no additional penalty)
            else:
                # The game continues
                pass  # No change to reward

        # Switch to next player if game is not over
        if not self.done:
            self.current_player = opponent

        observation = self._get_observation()

        return observation, reward, self.done, False, {}

    def render(self):
        # Build visual representation of the number line
        representation = ""
        for position in range(1, 21):
            pos_str = f"{position:2d}: "

            # Player positions
            if (
                self.player_positions["A"] == position
                and self.player_positions["B"] == position
            ):
                pos_str += "[A&B]"
            elif self.player_positions["A"] == position:
                pos_str += "[ A ]"
            elif self.player_positions["B"] == position:
                pos_str += "[ B ]"
            else:
                pos_str += "[   ]"

            # Mines (hidden, so not displayed)

            representation += pos_str + "\n"

        return representation

    def valid_moves(self):
        # Return list of valid moves (indices of action_space)
        valid_actions = []
        current_pos = self.player_positions[self.current_player]

        for i in range(1, 4):  # Movement options: 1,2,3 steps
            if current_pos + i <= 20:
                valid_actions.append(i - 1)  # Actions are 0-based indices

        return valid_actions

    def _get_observation(self):
        # Initialize the observation array
        observation = np.zeros((20, 4), dtype=np.int8)

        # Set player positions
        observation[self.player_positions["A"] - 1, 0] = 1  # Player A
        observation[self.player_positions["B"] - 1, 1] = 1  # Player B

        # Set mines positions
        for mine_pos in self.player_mines["A"]:
            observation[mine_pos - 1, 2] = 1  # Player A's mines
        for mine_pos in self.player_mines["B"]:
            observation[mine_pos - 1, 3] = 1  # Player B's mines

        # Flatten the observation to shape (80,)
        observation = observation.flatten()

        return observation
