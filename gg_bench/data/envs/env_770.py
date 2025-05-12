import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, max_number=20, starting_number=15):
        super(CustomEnv, self).__init__()
        self.max_number = max_number
        self.starting_number = starting_number

        # Initialize the current number and done flag
        self.current_number = self.starting_number
        self.done = False

        # Precompute all possible actions
        self.action_mapping, self.n_action_mapping = self.precompute_action_mapping()

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.action_mapping))
        self.observation_space = spaces.Box(
            low=1, high=self.max_number, shape=(1,), dtype=np.int32
        )

    def precompute_action_mapping(self):
        action_mapping = []
        n_action_mapping = {}
        action_index = 0

        for N in range(2, self.max_number + 1):
            n_action_indices = []
            for a in range(1, N):
                b = N - a
                # Option 1: Keep 'a', discard 'b', discarded number must be >1
                if b > 1:
                    action_mapping.append((N, a, b, "keep_a"))
                    n_action_indices.append(action_index)
                    action_index += 1
                # Option 2: Keep 'b', discard 'a', discarded number must be >1
                if a > 1:
                    action_mapping.append((N, a, b, "keep_b"))
                    n_action_indices.append(action_index)
                    action_index += 1
            n_action_mapping[N] = n_action_indices
        return action_mapping, n_action_mapping

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.done = False
        return self.get_observation(), {}

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}
        # Agent's turn
        reward, done, info = self.process_player_action(action)
        if done:
            # Game over after agent's move
            self.done = True
            return self.get_observation(), reward, self.done, False, info
        # Check if opponent can make a move
        opponent_actions = self.get_valid_actions(self.current_number)
        if not opponent_actions:
            # Opponent cannot move, agent wins
            self.done = True
            reward = 1
            return self.get_observation(), reward, self.done, False, {}
        # Opponent's turn
        opponent_action = self.get_opponent_action()
        self.process_opponent_action(opponent_action)
        # Check if agent can make a move
        valid_actions = self.valid_moves()
        if not valid_actions:
            # Agent cannot make a move, loses
            self.done = True
            reward = -1
            return self.get_observation(), reward, self.done, False, {}
        # Game continues
        reward = 0
        return self.get_observation(), reward, self.done, False, {}

    def process_player_action(self, action):
        # Returns reward, done, info
        if action < 0 or action >= len(self.action_mapping):
            # Invalid action index
            reward = -10
            done = True
            return reward, done, {"invalid_action": "Invalid action index"}
        action_details = self.action_mapping[action]
        action_N, a, b, keep_choice = action_details
        if action_N != self.current_number:
            # Action does not match current number
            reward = -10
            done = True
            return (
                reward,
                done,
                {"invalid_action": "Action does not match current number"},
            )
        # Process the action
        if keep_choice == "keep_a":
            keep_number = a
            discarded_number = b
        else:
            keep_number = b
            discarded_number = a
        # Validate discarded number
        if discarded_number <= 1:
            # Cannot discard 1
            reward = -10
            done = True
            return reward, done, {"invalid_action": "Cannot discard the number 1"}
        # Update current number
        self.current_number = keep_number
        done = False
        reward = 0
        return reward, done, {}

    def get_opponent_action(self):
        valid_actions = self.get_valid_actions(self.current_number)
        if not valid_actions:
            return None  # Opponent cannot make a move
        # Opponent selects a random valid action
        opponent_action = self.np_random.choice(valid_actions)
        return opponent_action

    def process_opponent_action(self, opponent_action):
        action_details = self.action_mapping[opponent_action]
        action_N, a, b, keep_choice = action_details
        # Process the opponent's action
        if keep_choice == "keep_a":
            keep_number = a
        else:
            keep_number = b
        # Update current number
        self.current_number = keep_number

    def get_valid_actions(self, number):
        if number in self.n_action_mapping:
            return self.n_action_mapping[number]
        else:
            return []

    def get_observation(self):
        return np.array([self.current_number], dtype=np.int32)

    def render(self):
        return f"Current Number: {self.current_number}"

    def valid_moves(self):
        return self.get_valid_actions(self.current_number)
