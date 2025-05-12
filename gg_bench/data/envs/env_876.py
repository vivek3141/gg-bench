import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Players can choose attacks from 1 to 10
        # Actions are indices from 0 to 9 corresponding to attack numbers 1 to 10
        self.action_space = spaces.Discrete(10)

        # Observation space consists of:
        # [agent_hp, opponent_hp, agent_prev_attack, opponent_prev_attack]
        # HP ranges from 0 to 100
        # Previous attacks range from 0 to 10 (0 means no previous attack)
        low = np.array([0, 0, 0, 0], dtype=np.int32)
        high = np.array([100, 100, 10, 10], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_hp = 100
        self.opponent_hp = 100
        self.agent_prev_attack = 0  # No previous attack
        self.opponent_prev_attack = 0  # No previous attack

        self.done = False

        observation = np.array(
            [
                self.agent_hp,
                self.opponent_hp,
                self.agent_prev_attack,
                self.opponent_prev_attack,
            ],
            dtype=np.int32,
        )
        return observation, {}

    def step(self, action):
        if self.done:
            return (
                self._get_observation(),
                0,
                True,
                False,
                {},
            )

        # Validate action
        if not self.action_space.contains(action):
            # Invalid action
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        agent_attack = action + 1  # Map action index to attack number (1-10)

        # Check No Repeat Rule for agent
        if agent_attack == self.agent_prev_attack:
            # Invalid move, agent loses
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Opponent's turn (select action)
        opponent_valid_actions = [
            i for i in range(1, 11) if i != self.opponent_prev_attack
        ]
        opponent_attack = self.np_random.choice(opponent_valid_actions)

        # Resolve turn
        if agent_attack > opponent_attack:
            damage = agent_attack - opponent_attack
            self.opponent_hp -= damage
        elif opponent_attack > agent_attack:
            damage = opponent_attack - agent_attack
            self.agent_hp -= damage
        else:
            # No damage if attacks are equal
            pass

        # Update previous attacks
        self.agent_prev_attack = agent_attack
        self.opponent_prev_attack = opponent_attack

        # Check for win/loss conditions
        if self.opponent_hp <= 0 and self.agent_hp > 0:
            self.done = True
            reward = 1  # Agent wins
        elif self.agent_hp <= 0 and self.opponent_hp > 0:
            self.done = True
            reward = -1  # Agent loses
        elif self.agent_hp <= 0 and self.opponent_hp <= 0:
            self.done = True
            reward = 0  # Draw
        else:
            reward = 0
            self.done = False

        return self._get_observation(), reward, self.done, False, {}

    def render(self):
        return (
            f"Agent HP: {self.agent_hp}, Opponent HP: {self.opponent_hp}\n"
            f"Agent previous attack: {self.agent_prev_attack}, Opponent previous attack: {self.opponent_prev_attack}"
        )

    def valid_moves(self):
        # Valid actions are indices from 0 to 9, excluding previous attack - 1
        if self.agent_prev_attack == 0:
            return list(range(10))
        else:
            previous_attack_index = self.agent_prev_attack - 1
            return [i for i in range(10) if i != previous_attack_index]

    def _get_observation(self):
        return np.array(
            [
                self.agent_hp,
                self.opponent_hp,
                self.agent_prev_attack,
                self.opponent_prev_attack,
            ],
            dtype=np.int32,
        )
