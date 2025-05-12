import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 (Take), 1 (Skip)
        self.action_space = spaces.Discrete(2)
        # Observation: [Top Number, Numbers Left in Stack, Player's Numbers Collected,
        # Player's Total Score, Opponent's Numbers Collected, Opponent's Total Score]
        self.observation_space = spaces.Box(low=0, high=100, shape=(6,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate the stack with 15 random integers between 1 and 9
        self.stack = np.random.randint(1, 10, size=15).tolist()
        self.current_player = 1  # Player 1 starts the game
        self.done = False
        self.consecutive_skips = 0
        # Store collected numbers for both players
        self.player_scores = {1: [], -1: []}

        # Return the initial observation and info
        return self.get_observation(), {}

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        if action not in [0, 1]:
            # Invalid action
            self.done = True
            return self.get_observation(), -10, True, False, {}

        if action == 0:  # Take
            if len(self.stack) == 0:
                # Invalid move: cannot take from empty stack
                self.done = True
                return self.get_observation(), -10, True, False, {}
            else:
                # Remove top number from stack and add to current player's collected numbers
                number_taken = self.stack.pop(0)
                self.player_scores[self.current_player].append(number_taken)
                self.consecutive_skips = 0  # Reset consecutive skips
        elif action == 1:  # Skip
            self.consecutive_skips += 1
            if self.consecutive_skips == 2:
                # Both players skipped consecutively: discard top number
                if len(self.stack) > 0:
                    self.stack.pop(0)
                self.consecutive_skips = 0

        # Check if game is over
        if len(self.stack) == 0:
            self.done = True
            # Determine winner
            player_score = sum(self.player_scores[self.current_player])
            opponent_score = sum(self.player_scores[-self.current_player])
            if player_score > opponent_score:
                reward = 1  # Current player wins
            elif player_score < opponent_score:
                reward = -1  # Current player loses
            else:
                # Tie: player with more numbers collected wins
                player_numbers = len(self.player_scores[self.current_player])
                opponent_numbers = len(self.player_scores[-self.current_player])
                if player_numbers > opponent_numbers:
                    reward = 1  # Current player wins
                else:
                    reward = -1  # Current player loses
            return self.get_observation(), reward, True, False, {}

        # Swap current player
        self.current_player *= -1

        # No reward yet
        return self.get_observation(), 0, False, False, {}

    def get_observation(self):
        if len(self.stack) > 0:
            top_number = self.stack[0]
        else:
            top_number = 0

        numbers_left_in_stack = len(self.stack)
        player_collected_numbers = len(self.player_scores[self.current_player])
        player_total_score = sum(self.player_scores[self.current_player])
        opponent_collected_numbers = len(self.player_scores[-self.current_player])
        opponent_total_score = sum(self.player_scores[-self.current_player])

        observation = np.array(
            [
                top_number,
                numbers_left_in_stack,
                player_collected_numbers,
                player_total_score,
                opponent_collected_numbers,
                opponent_total_score,
            ],
            dtype=np.int32,
        )
        return observation

    def render(self):
        output = ""
        output += f"Top Number: {self.stack[0] if len(self.stack) > 0 else 'None'}\n"
        output += f"Numbers Left in Stack: {len(self.stack)}\n"
        output += f"Player {self.current_player}'s Turn\n"
        output += f"Player 1's Collected Numbers: {self.player_scores[1]}, Total Score: {sum(self.player_scores[1])}\n"
        output += f"Player -1's Collected Numbers: {self.player_scores[-1]}, Total Score: {sum(self.player_scores[-1])}\n"
        output += f"Consecutive Skips: {self.consecutive_skips}\n"
        return output

    def valid_moves(self):
        if self.done:
            return []
        elif len(self.stack) == 0:
            # No valid actions when stack is empty and game is over
            return []
        else:
            # Both Take and Skip are valid when stack is not empty
            return [0, 1]
