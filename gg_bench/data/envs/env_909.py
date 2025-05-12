import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the target number (change as needed between 15 and 31)
        self.target_number = 23

        # Define action and observation space
        # Actions: 0 = '+1', 1 = '*2'
        self.action_space = spaces.Discrete(2)

        # Observation space: [current_number, player_add1_remaining, player_mul2_remaining,
        # opponent_add1_remaining, opponent_mul2_remaining]
        # All values are integers
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array(
                [
                    self.target_number,  # current_number cannot exceed target_number
                    5,  # Max 'Add 1' operations per player
                    4,  # Max 'Multiply by 2' operations per player
                    5,
                    4,
                ]
            ),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the game state
        self.current_number = 1

        # Remaining operations for each player
        self.player1_ops = {"+1": 5, "*2": 4}
        self.player2_ops = {"+1": 5, "*2": 4}

        # Set the current player (1 or 2)
        self.current_player = 1

        self.done = False

        # Check if starting player has any valid moves (should always be true at game start)
        if len(self.valid_moves()) == 0:
            self.done = True
            reward = -10
            observation = self._get_observation()
            return observation, {}

        observation = self._get_observation()
        return observation, {}  # Observation and info

    def _get_observation(self):
        if self.current_player == 1:
            player_ops = self.player1_ops
            opp_ops = self.player2_ops
        else:
            player_ops = self.player2_ops
            opp_ops = self.player1_ops

        observation = np.array(
            [
                self.current_number,
                player_ops["+1"],
                player_ops["*2"],
                opp_ops["+1"],
                opp_ops["*2"],
            ],
            dtype=np.int32,
        )

        return observation

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Apply the action
        if self.current_player == 1:
            player_ops = self.player1_ops
        else:
            player_ops = self.player2_ops

        operation = "+1" if action == 0 else "*2"

        player_ops[operation] -= 1  # Decrement operation count

        # Compute new current number
        if operation == "+1":
            self.current_number += 1
        else:  # '*2'
            self.current_number *= 2

        # Check if new current number exceeds target
        if self.current_number > self.target_number:
            # Current player loses
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Check for win condition
        if self.current_number == self.target_number:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, {}

        # Switch to the next player
        prev_player = self.current_player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if next player has any valid moves
        next_valid_moves = self.valid_moves()
        if len(next_valid_moves) == 0:
            # Next player has no valid moves, current player wins
            self.done = True
            reward = 1
            # Switch back to previous player's observation
            self.current_player = prev_player
            return self._get_observation(), reward, True, False, {}

        # Game continues
        reward = 0
        observation = self._get_observation()
        return observation, reward, False, False, {}

    def valid_moves(self):
        if self.done:
            return []

        if self.current_player == 1:
            player_ops = self.player1_ops
        else:
            player_ops = self.player2_ops

        valid_actions = []

        for idx, operation in enumerate(["+1", "*2"]):
            if player_ops[operation] > 0:
                # Check if applying this operation would not exceed the target number
                if operation == "+1":
                    new_current_number = self.current_number + 1
                else:  # '*2'
                    new_current_number = self.current_number * 2

                if new_current_number <= self.target_number:
                    valid_actions.append(idx)

        return valid_actions

    def render(self):
        state_str = "---------------------------\n"
        state_str += f"Current Number: {self.current_number}\n"
        state_str += f"Target Number: {self.target_number}\n"
        state_str += f"Player {self.current_player}'s Turn\n"

        if self.current_player == 1:
            player_ops = self.player1_ops
            opp_player = 2
            opp_ops = self.player2_ops
        else:
            player_ops = self.player2_ops
            opp_player = 1
            opp_ops = self.player1_ops

        state_str += f"Remaining Operations for Player {self.current_player}:\n"
        state_str += f"  Add 1 [+1]: {player_ops['+1']} uses left\n"
        state_str += f"  Multiply by 2 [*2]: {player_ops['*2']} uses left\n"
        state_str += f"Remaining Operations for Player {opp_player}:\n"
        state_str += f"  Add 1 [+1]: {opp_ops['+1']} uses left\n"
        state_str += f"  Multiply by 2 [*2]: {opp_ops['*2']} uses left\n"
        state_str += "---------------------------\n"

        return state_str
