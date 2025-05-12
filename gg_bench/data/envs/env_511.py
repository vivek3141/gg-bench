import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # Elements mapping
        self.elements = {0: "Fire", 1: "Water", 2: "Earth", 3: "Air", 4: "Lightning"}

        # Element strengths
        self.element_strengths = {
            0: [2, 4],  # Fire beats Earth(2), Lightning(4)
            1: [0, 3],  # Water beats Fire(0), Air(3)
            2: [1, 4],  # Earth beats Water(1), Lightning(4)
            3: [2, 0],  # Air beats Earth(2), Fire(0)
            4: [1, 3],  # Lightning beats Water(1), Air(3)
        }

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_lp = 15
        self.opponent_lp = 15
        self.done = False

        # Elements available to both players (1 if available, 0 if used)
        self.player_elements = np.ones(5, dtype=np.float32)
        self.opponent_elements = np.ones(5, dtype=np.float32)

        # Prepare observation
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # Check if game is already over
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Validate action
        if not self._is_valid_action(action):
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Player uses the element
        self.player_elements[action] = 0

        # Opponent selects an element (randomly from available elements)
        opponent_action = self._opponent_action()

        # Opponent uses the element
        self.opponent_elements[opponent_action] = 0

        # Resolve round
        outcome = self._determine_outcome(action, opponent_action)

        # Apply damage
        if outcome == "player_win":
            self.opponent_lp -= 3
        elif outcome == "opponent_win":
            self.player_lp -= 3
        else:  # Draw
            self.player_lp -= 1
            self.opponent_lp -= 1

        # Check for victory conditions
        if self.player_lp <= 0 and self.opponent_lp <= 0:
            # Tie-breaker logic (not specified in reward)
            self.done = True
            reward = -1  # Considered as loss
        elif self.opponent_lp <= 0:
            self.done = True
            reward = 1  # Win
        elif self.player_lp <= 0:
            self.done = True
            reward = -1  # Loss
        else:
            self.done = False
            reward = 0  # Continue

        observation = self._get_observation()
        info = {
            "player_element": action,
            "opponent_element": opponent_action,
            "outcome": outcome,
        }

        return observation, reward, self.done, False, info

    def render(self):
        player_elements = [
            self.elements[i] for i in range(5) if self.player_elements[i] == 1
        ]
        opponent_elements = [
            self.elements[i] for i in range(5) if self.opponent_elements[i] == 1
        ]

        render_str = f"Player LP: {self.player_lp}\n"
        render_str += f"Player Available Elements: {player_elements}\n"
        render_str += f"Opponent LP: {self.opponent_lp}\n"
        render_str += f"Opponent Available Elements: {opponent_elements}\n"

        return render_str

    def valid_moves(self):
        return [i for i in range(5) if self.player_elements[i] == 1]

    def _get_observation(self):
        # Normalize LPs to [0,1]
        player_lp_norm = self.player_lp / 15.0
        opponent_lp_norm = self.opponent_lp / 15.0

        observation = np.concatenate(
            (
                np.array([player_lp_norm, opponent_lp_norm], dtype=np.float32),
                self.player_elements.copy(),
                self.opponent_elements.copy(),
            )
        )

        return observation

    def _is_valid_action(self, action):
        return self.player_elements[action] == 1

    def _opponent_action(self):
        valid_actions = [i for i in range(5) if self.opponent_elements[i] == 1]
        if valid_actions:
            return self.np_random.choice(valid_actions)
        else:
            # No elements left; opponent cannot act
            return None

    def _determine_outcome(self, player_action, opponent_action):
        # Handle case where opponent has no elements left
        if opponent_action is None:
            return "player_win"

        if opponent_action in self.element_strengths[player_action]:
            return "player_win"
        elif player_action in self.element_strengths[opponent_action]:
            return "opponent_win"
        else:
            return "draw"
