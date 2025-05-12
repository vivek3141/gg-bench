import numpy as np
import gymnasium as gym
from gymnasium import spaces


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Game parameters
        self.STARTING_NUMBER = 15
        self.MAX_NUMBER = 20  # Maximum value any number in the Numbers List can have
        self.MAX_NUMBERS = 10  # Maximum length of the Numbers List
        self.MAX_ACTIONS = 500  # Maximum number of possible actions

        # Define action and observation space
        self.action_space = spaces.Discrete(self.MAX_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=self.MAX_NUMBER, shape=(self.MAX_NUMBERS,), dtype=np.int32
        )

        # Precompute all possible actions
        self.all_actions = []
        action_index = 0
        composite_numbers = [
            n for n in range(4, self.MAX_NUMBER + 1) if not is_prime(n)
        ]
        self.composite_numbers = composite_numbers  # For quick access

        for n in composite_numbers:
            for a in range(1, n):
                b = n - a
                if a >= 1 and b >= 1:
                    self.all_actions.append((n, a, b))
                    action_index += 1
                    if action_index >= self.MAX_ACTIONS:
                        break
            if action_index >= self.MAX_ACTIONS:
                break
        self.num_actions = len(self.all_actions)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.numbers_list = [self.STARTING_NUMBER]
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = self.get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        # Check if the current player has any valid moves
        valid_moves = self.valid_moves()
        if not valid_moves:
            # Current player loses
            reward = -1
            self.done = True
            return self.get_observation(), reward, True, False, {}

        # Check if action is valid
        if action < 0 or action >= self.num_actions or action not in valid_moves:
            # Invalid action
            reward = -10
            self.done = True
            return self.get_observation(), reward, True, False, {}

        # Apply the action
        composite_number, a, b = self.all_actions[action]

        # Remove the composite number and add the split numbers
        self.numbers_list.remove(composite_number)
        self.numbers_list.extend([a, b])

        # Check if there are any composite numbers left
        has_composites = any(n for n in self.numbers_list if not is_prime(n) and n > 1)

        if not has_composites:
            # Current player wins
            reward = 1
            self.done = True
            return self.get_observation(), reward, True, False, {}

        # Otherwise, switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1
        reward = 0
        return self.get_observation(), reward, False, False, {}

    def render(self):
        numbers_str = "Numbers List: " + str(self.numbers_list)
        current_player_str = f"Current Player: Player {self.current_player}"
        return numbers_str + "\n" + current_player_str

    def valid_moves(self):
        valid_moves = []
        composite_numbers_in_list = [
            n for n in self.numbers_list if not is_prime(n) and n > 1
        ]
        for idx, (n, a, b) in enumerate(self.all_actions):
            if n in composite_numbers_in_list:
                valid_moves.append(idx)
        return valid_moves

    def get_observation(self):
        # Return the Numbers List padded to MAX_NUMBERS
        obs = np.zeros(self.MAX_NUMBERS, dtype=np.int32)
        obs[: len(self.numbers_list)] = self.numbers_list[: self.MAX_NUMBERS]
        return obs
