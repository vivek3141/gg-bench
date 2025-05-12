import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(36)  # 9 Action Numbers * 4 Operations
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([100, 100, 9, 9, 9]),
            shape=(5,),
            dtype=np.int32,
        )

        # Initialize game state variables
        self.current_number = 0
        self.target_number = 15  # Default target number; can be changed in reset()
        self.player_action_numbers = [
            [],
            [],
        ]  # Action Numbers for Player 1 and Player 2
        self.used_action_numbers = [
            [],
            [],
        ]  # Used Action Numbers for Player 1 and Player 2
        self.current_player = 0  # Player 0 and Player 1
        self.done = False

        # Mapping of operation indices to symbols
        self.operations = ["+", "-", "*", "/"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly select Target Number between 10 and 30
        self.target_number = np.random.randint(10, 31)
        self.current_number = 0

        # Randomly select unique Action Numbers between 1 and 9 for each player
        numbers = np.arange(1, 10)
        np.random.shuffle(numbers)
        self.player_action_numbers[0] = numbers[:3].tolist()
        self.player_action_numbers[1] = numbers[3:6].tolist()

        # Reset used Action Numbers
        self.used_action_numbers = [[], []]

        # Start with Player 0
        self.current_player = 0
        self.done = False

        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Decode action into Action Number and Operation
        action_number = (action // 4) + 1  # Action Numbers are from 1 to 9
        operation_index = action % 4
        operation = self.operations[operation_index]

        # Check if the Action Number is available for the current player
        if action_number not in self.player_action_numbers[self.current_player]:
            # Invalid move: Action Number not available
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Perform the operation
        prev_number = self.current_number
        try:
            self.current_number = self._calculate(
                self.current_number, action_number, operation
            )
        except ValueError:
            # Invalid operation (e.g., division by zero)
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Remove the used Action Number
        self.player_action_numbers[self.current_player].remove(action_number)
        self.used_action_numbers[self.current_player].append(action_number)

        # Check game end conditions
        if self.current_number == self.target_number:
            # Current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}
        elif self.current_number > self.target_number:
            # Current player loses
            self.done = True
            return self._get_observation(), -1, True, False, {}

        # Check if both players have used all their Action Numbers
        if (
            len(self.player_action_numbers[0]) == 0
            and len(self.player_action_numbers[1]) == 0
        ):
            # Determine winner based on proximity to Target Number
            player0_diff = abs(self.current_number - self.target_number)
            # Simulate opponent's last number as 0 since they have no moves left
            opponent_number = self.current_number  # Assuming opponent made no change
            player1_diff = abs(opponent_number - self.target_number)

            if player0_diff < player1_diff:
                # Current player wins
                self.done = True
                return self._get_observation(), 1, True, False, {}
            elif player1_diff < player0_diff:
                # Current player loses
                self.done = True
                return self._get_observation(), -1, True, False, {}
            else:
                # Tie: Second player wins
                if self.current_player == 1:
                    self.done = True
                    return self._get_observation(), 1, True, False, {}
                else:
                    self.done = True
                    return self._get_observation(), -1, True, False, {}

        # Switch to the other player
        self.current_player = 1 - self.current_player

        return self._get_observation(), 0, False, False, {}

    def render(self):
        player_numbers = ", ".join(
            str(num) for num in self.player_action_numbers[self.current_player]
        )
        return (
            f"Target Number: {self.target_number}\n"
            f"Current Number: {self.current_number}\n"
            f"Player {self.current_player + 1}'s turn.\n"
            f"Available Action Numbers: {player_numbers}\n"
        )

    def valid_moves(self):
        valid_actions = []
        for action_number in self.player_action_numbers[self.current_player]:
            action_number_index = action_number - 1  # Adjusting to 0-based index
            for operation_index in range(4):
                action = action_number_index * 4 + operation_index
                valid_actions.append(action)
        return valid_actions

    def _calculate(self, current_number, action_number, operation):
        if operation == "+":
            result = current_number + action_number
        elif operation == "-":
            result = current_number - action_number
        elif operation == "*":
            result = current_number * action_number
        elif operation == "/":
            if action_number == 0:
                raise ValueError("Division by zero")
            # Integer division rounding towards zero
            result = int(current_number / action_number)
        else:
            raise ValueError("Invalid operation")
        return result

    def _get_observation(self):
        # For the current player, provide their remaining Action Numbers
        remaining_numbers = self.player_action_numbers[self.current_player]
        numbers = remaining_numbers + [0] * (3 - len(remaining_numbers))
        observation = np.array(
            [
                self.current_number,
                self.target_number,
                numbers[0],
                numbers[1],
                numbers[2],
            ],
            dtype=np.int32,
        )
        return observation
