import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - Add, 1 - Subtract, 2 - Pass
        self.action_space = spaces.Discrete(3)

        # Observation space:
        # [player_total, opponent_total, passed_number_to_player,
        # counts of numbers 1-9 remaining in draw pile (9 elements)]
        low = np.array([0, 0, 0] + [0] * 9, dtype=np.int32)
        high = np.array([21, 21, 9] + [2] * 9, dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Create the draw pile: numbers 1-9 each appearing twice
        numbers = [i for i in range(1, 10)] * 2
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.draw_pile = self.np_random.permutation(numbers).tolist()

        # Counts of remaining numbers in the draw pile
        self.remaining_counts = [2] * 9  # Index 0 corresponds to number 1

        # Player totals
        self.player_totals = [0, 0]

        # Passed numbers to players (None if no passed number)
        self.passed_numbers = [0, 0]  # Index 0 - Player 1, Index 1 - Player 2

        # Current player (0 - Player 1, 1 - Player 2)
        self.current_player = 0

        # Game over flag
        self.done = False

        # Observation
        observation = self._get_observation()

        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        reward = 0

        # Current player's index
        idx = self.current_player
        opponent_idx = 1 - idx

        # Apply passed number if there is one
        if self.passed_numbers[idx]:
            self.player_totals[idx] += self.passed_numbers[idx]
            # Check for exceeding 21
            if self.player_totals[idx] > 21:
                self.done = True
                reward = -1  # Loss
                return self._get_observation(), reward, True, False, {}

            # Check for reaching exactly 21
            if self.player_totals[idx] == 21:
                self.done = True
                reward = 1  # Win
                return self._get_observation(), reward, True, False, {}

            self.passed_numbers[idx] = 0  # Clear the passed number

        # Draw a number from the draw pile
        if not self.draw_pile:
            # Draw pile is empty
            self.done = True
            # Determine winner based on who is closer to 21 without exceeding
            player_total = self.player_totals[idx]
            opponent_total = self.player_totals[opponent_idx]
            if player_total > opponent_total:
                reward = 1
            elif player_total < opponent_total:
                reward = -1
            else:
                reward = 0  # Tie
            return self._get_observation(), reward, True, False, {}

        drawn_number = self.draw_pile.pop(0)
        self.remaining_counts[drawn_number - 1] -= 1

        # Validate action
        valid_actions = self.valid_moves(drawn_number)
        if action not in valid_actions:
            self.done = True
            reward = -10  # Invalid move
            return self._get_observation(), reward, True, False, {}

        # Process action
        if action == 0:  # Add
            self.player_totals[idx] += drawn_number
            if self.player_totals[idx] > 21:
                self.done = True
                reward = -1  # Loss
                return self._get_observation(), reward, True, False, {}

            if self.player_totals[idx] == 21:
                self.done = True
                reward = 1  # Win
                return self._get_observation(), reward, True, False, {}

        elif action == 1:  # Subtract
            self.player_totals[idx] -= drawn_number
            if self.player_totals[idx] < 0:
                self.done = True
                reward = -10  # Invalid move
                return self._get_observation(), reward, True, False, {}

        elif action == 2:  # Pass
            self.passed_numbers[opponent_idx] = drawn_number

        else:
            self.done = True
            reward = -10  # Invalid action
            return self._get_observation(), reward, True, False, {}

        # Switch to the next player
        self.current_player = opponent_idx

        return self._get_observation(), reward, self.done, False, {}

    def render(self):
        idx = self.current_player
        opponent_idx = 1 - idx
        s = f"Player {idx + 1}'s Turn:\n"
        s += f"Your Total: {self.player_totals[idx]} | Opponent's Total: {self.player_totals[opponent_idx]}\n"
        s += f"Your Passed Number: {self.passed_numbers[idx]}\n"
        s += f"Numbers Remaining in Draw Pile: {sum(self.remaining_counts)}\n"
        counts_str = ", ".join([f"{i+1}: {self.remaining_counts[i]}" for i in range(9)])
        s += f"Counts of Numbers 1-9 Remaining: {counts_str}\n"
        return s

    def valid_moves(self, drawn_number=None):
        # Return a list of valid actions
        idx = self.current_player
        valid_actions = [0, 1, 2]  # Start with all actions

        # If subtracting would result in negative total, remove subtract action
        if drawn_number is None:
            return [0, 1, 2]  # No drawn number, all actions are valid

        if (self.player_totals[idx] - drawn_number) < 0:
            valid_actions.remove(1)  # Remove 'Subtract' action

        return valid_actions

    def _get_observation(self):
        idx = self.current_player
        opponent_idx = 1 - idx
        observation = np.array(
            [
                self.player_totals[idx],
                self.player_totals[opponent_idx],
                self.passed_numbers[idx],
            ]
            + self.remaining_counts,
            dtype=np.int32,
        )
        return observation
