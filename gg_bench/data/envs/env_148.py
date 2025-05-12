import numpy as np
import gymnasium as gym
from gymnasium import spaces
import gymnasium.utils.seeding


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 27 possible actions (digits 1-9 and slots 0-2)
        self.action_space = spaces.Discrete(27)

        # Observation space: target number, available digits, current player's slots, opponent's slots
        # Shape: (16,), values between 0 and 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(16,), dtype=np.float32
        )

        self.np_random = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        else:
            self.np_random, seed = gymnasium.utils.seeding.np_random()

        # Generate target number between 100 and 999
        self.target = self.np_random.randint(100, 1000)

        # Initialize available digits: digits 1 to 9
        self.available_digits = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Initialize players' numbers slots as empty
        self.player_numbers = {
            1: [0, 0, 0],  # slots for player 1
            -1: [0, 0, 0],  # slots for player 2 (represented as -1)
        }

        # Set current player
        self.current_player = 1  # Player 1 starts

        # Game over flag
        self.done = False

        # Track total steps
        self.total_steps = 0

        # Track when each player completes their number
        self.player_completion_step = {1: None, -1: None}

        return self._get_obs(), {}

    def _get_obs(self):
        # Normalize target number to between 0 and 1
        target_norm = self.target / 999.0

        # Available digits: 1 if available, 0 if not
        avail_digits_array = np.zeros(9)
        for d in self.available_digits:
            avail_digits_array[d - 1] = 1.0

        # Current player's slots, normalized
        current_player_slots = np.array(self.player_numbers[self.current_player]) / 9.0

        # Opponent's slots, normalized
        opponent_slots = np.array(self.player_numbers[-self.current_player]) / 9.0

        obs = np.concatenate(
            ([target_norm], avail_digits_array, current_player_slots, opponent_slots)
        )

        return obs.astype(np.float32)

    def _get_number(self, player):
        slots = self.player_numbers[player]
        h, t, o = slots
        number = h * 100 + t * 10 + o
        return number

    def step(self, action):
        # Check if game is over
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Extract digit and slot from action
        digit = (action // 3) + 1  # action index to digit
        slot = action % 3  # slot: 0 (hundreds), 1 (tens), 2 (ones)

        reward = 0
        info = {}

        # Check if digit is available, and slot is unfilled
        if (
            digit not in self.available_digits
            or self.player_numbers[self.current_player][slot] != 0
        ):
            # Invalid move
            self.done = True  # Ends the game
            reward = -10
            info["reason"] = "Invalid move"
            terminated = True
            truncated = False
            return self._get_obs(), reward, terminated, truncated, info

        # Valid move
        # Remove digit from available digits
        self.available_digits.remove(digit)
        # Assign digit to current player's slot
        self.player_numbers[self.current_player][slot] = digit
        self.total_steps += 1

        # Check if player's number is complete
        if self.player_completion_step[self.current_player] is None and all(
            v != 0 for v in self.player_numbers[self.current_player]
        ):
            self.player_completion_step[self.current_player] = self.total_steps

        # Check if game is over (both players have filled their numbers)
        total_slots_filled = sum([1 for v in self.player_numbers[1] if v != 0]) + sum(
            [1 for v in self.player_numbers[-1] if v != 0]
        )

        if total_slots_filled == 6:
            self.done = True
            # Game over, determine winner
            player1_number = self._get_number(1)
            player2_number = self._get_number(-1)
            target = self.target

            p1_diff = abs(player1_number - target)
            p2_diff = abs(player2_number - target)

            if p1_diff < p2_diff:
                winner = 1
            elif p2_diff < p1_diff:
                winner = -1
            else:
                # Tie, player who completed their number first wins
                if self.player_completion_step[1] < self.player_completion_step[-1]:
                    winner = 1
                else:
                    winner = -1

            if winner == self.current_player:
                reward = 1
            else:
                reward = -1

            terminated = True
            truncated = False

        else:
            # Switch current player
            self.current_player = -self.current_player
            terminated = False
            truncated = False

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        lines = []
        lines.append("--- Digits Duel Game State ---")
        lines.append(f"Target Number: {self.target}")
        lines.append(f"Available Digits: {' '.join(map(str,self.available_digits))}")
        lines.append("")
        lines.append("Player 1:")
        lines.append(f"  Number Slots: {self._format_slots(self.player_numbers[1])}")
        lines.append("Player 2:")
        lines.append(f"  Number Slots: {self._format_slots(self.player_numbers[-1])}")
        lines.append(
            f"Current Player: Player {self.current_player if not self.done else 'None (Game Over)'}"
        )

        if self.done:
            player1_number = self._get_number(1)
            player2_number = self._get_number(-1)
            lines.append("")
            lines.append("--- Game Over ---")
            lines.append(f"Player 1's Number: {player1_number}")
            lines.append(f"Player 2's Number: {player2_number}")

            # Determine winner
            player1_diff = abs(player1_number - self.target)
            player2_diff = abs(player2_number - self.target)
            if player1_diff < player2_diff:
                winner = "Player 1"
            elif player2_diff < player1_diff:
                winner = "Player 2"
            else:
                # Tie-breaker based on completion step
                if self.player_completion_step[1] < self.player_completion_step[-1]:
                    winner = "Player 1"
                else:
                    winner = "Player 2"
            lines.append(f"Winner: {winner}")
        return "\n".join(lines)

    def _format_slots(self, slots):
        slot_names = ["Hundreds", "Tens", "Ones"]
        slot_lines = []
        for name, value in zip(slot_names, slots):
            if value == 0:
                v_str = "_"
            else:
                v_str = str(value)
            slot_lines.append(f"{name}: {v_str}")
        return ", ".join(slot_lines)

    def valid_moves(self):
        valid_actions = []
        for digit in self.available_digits:
            for slot in range(3):
                if self.player_numbers[self.current_player][slot] == 0:
                    action = (digit - 1) * 3 + slot
                    valid_actions.append(action)
        return valid_actions
