import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the Target Number
        self.TARGET_NUMBER = 15

        # Define the action space: 18 possible actions (move to number 1-9 and add/subtract)
        self.action_space = spaces.Discrete(18)

        # Define the observation space
        # [current_position, previous_position, current_total, opponent_position, opponent_total, target_number]
        # All integers between appropriate ranges
        self.observation_space = spaces.Box(
            low=np.array([1, 0, 0, 1, 0, 10]),
            high=np.array([9, 9, self.TARGET_NUMBER, 9, self.TARGET_NUMBER, 20]),
            dtype=np.int32,
        )

        # Precompute the action map
        self.action_map = []
        for number in range(1, 10):
            self.action_map.append(
                {"move_to": number, "operation": "add"}
            )  # action index = (number - 1)*2
            self.action_map.append(
                {"move_to": number, "operation": "subtract"}
            )  # action index = (number -1)*2 +1

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Players start at number 1
        self.player_positions = [1, 1]
        self.player_prev_positions = [0, 0]  # No previous position at start
        self.player_totals = [0, 0]

        self.current_player = 0  # Player 0 and Player 1
        self.done = False

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map action index to move_to and operation
        if action < 0 or action >= 18:
            # Invalid action index
            return self._get_observation(), -10, True, False, {}

        action_info = self.action_map[action]
        move_to = action_info["move_to"]
        operation = action_info["operation"]

        # Get current player info
        player_idx = self.current_player
        opponent_idx = 1 - self.current_player

        current_position = self.player_positions[player_idx]
        previous_position = self.player_prev_positions[player_idx]

        # Check if move_to is adjacent to current_position
        adjacent_positions = [
            (current_position % 9) + 1,
            (current_position - 2) % 9 + 1,
        ]
        if previous_position != 0:
            # Exclude backtracking
            adjacent_positions = [
                pos for pos in adjacent_positions if pos != previous_position
            ]

        if move_to not in adjacent_positions:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Get the number at the move_to position
        number = move_to

        # Check if operation is valid
        current_total = self.player_totals[player_idx]
        if operation == "add":
            new_total = current_total + number
        elif operation == "subtract":
            new_total = current_total - number
        else:
            # Invalid operation
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check if new_total overshoots the Target Number
        if new_total > self.TARGET_NUMBER:
            # Cannot perform this operation
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Update player state
        self.player_prev_positions[player_idx] = current_position
        self.player_positions[player_idx] = move_to
        self.player_totals[player_idx] = new_total

        # Check for win condition
        if new_total == self.TARGET_NUMBER:
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch to next player
        self.current_player = 1 - self.current_player

        # Game continues
        return self._get_observation(), 0, False, False, {}

    def render(self):
        # Return a string representation of the game state
        circle_numbers = [str(i) for i in range(1, 10)]
        pos_p0 = self.player_positions[0]
        pos_p1 = self.player_positions[1]

        # Mark player positions
        for idx, num_str in enumerate(circle_numbers):
            num = idx + 1
            markers = ""
            if self.player_positions[0] == num:
                markers += "A"
            if self.player_positions[1] == num:
                markers += "B"
            if markers:
                circle_numbers[idx] = "{}({})".format(num_str, markers)
            else:
                circle_numbers[idx] = num_str

        circle_layout = [
            ("      {}      ".format(circle_numbers[8])),
            ("     / \\     "),
            ("   {}     {}   ".format(circle_numbers[7], circle_numbers[1])),
            ("   |       |   "),
            (" {}         {} ".format(circle_numbers[6], circle_numbers[2])),
            ("   \\       /   "),
            ("    {}-----{}   ".format(circle_numbers[5], circle_numbers[3])),
            ("           |    "),
            ("           {}    ".format(circle_numbers[4])),
        ]

        circle_str = "\n".join(circle_layout)

        totals_str = "Player A Total: {} | Player B Total: {}\nTarget Number: {}\nCurrent Player: {}\n".format(
            self.player_totals[0],
            self.player_totals[1],
            self.TARGET_NUMBER,
            "A" if self.current_player == 0 else "B",
        )
        return circle_str + "\n" + totals_str

    def valid_moves(self):
        # Return list of valid action indices
        valid_actions = []
        player_idx = self.current_player
        current_position = self.player_positions[player_idx]
        previous_position = self.player_prev_positions[player_idx]

        adjacent_positions = [
            (current_position % 9) + 1,
            (current_position - 2) % 9 + 1,
        ]
        if previous_position != 0:
            # Exclude backtracking
            adjacent_positions = [
                pos for pos in adjacent_positions if pos != previous_position
            ]

        for pos in adjacent_positions:
            number = pos
            for op in ["add", "subtract"]:
                # Check if operation is valid
                current_total = self.player_totals[player_idx]
                if op == "add":
                    new_total = current_total + number
                elif op == "subtract":
                    new_total = current_total - number

                if new_total > self.TARGET_NUMBER:
                    continue  # Operation invalid, overshoots target number

                # Map move_to and operation to action index
                action_idx = (pos - 1) * 2 + (0 if op == "add" else 1)
                valid_actions.append(action_idx)

        return valid_actions

    def _get_observation(self):
        # Observation is [current_position, previous_position, current_total, opponent_position, opponent_total, target_number]
        player_idx = self.current_player
        opponent_idx = 1 - self.current_player
        observation = np.array(
            [
                self.player_positions[player_idx],
                self.player_prev_positions[player_idx],
                self.player_totals[player_idx],
                self.player_positions[opponent_idx],
                self.player_totals[opponent_idx],
                self.TARGET_NUMBER,
            ],
            dtype=np.int32,
        )
        return observation
