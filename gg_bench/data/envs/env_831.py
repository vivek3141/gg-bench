import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    ACTION_CHARGE = 0
    ACTION_SHIELD = 1
    ACTION_ATTACK = 2

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Charge, 1: Shield, 2: Attack

        # Observation space:
        # [current_player IP (0-10), EP (0-5), Shield (0 or 1),
        #  opponent IP (0-10), EP (0-5), Shield (0 or 1)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([10, 5, 1, 10, 5, 1]),
            shape=(6,),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_states = {
            1: {"ip": 10, "ep": 0, "shield_active": False, "shield_blocked": False},
            2: {"ip": 10, "ep": 0, "shield_active": False, "shield_blocked": False},
        }
        self.current_player = 1
        self.other_player = 2
        self.done = False

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # At the start of the player's turn, manage shield status
        current_state = self.player_states[self.current_player]
        if current_state["shield_active"] and not current_state["shield_blocked"]:
            current_state["shield_active"] = False  # Shield expires
        current_state["shield_blocked"] = False  # Reset shield blocked status

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )  # Invalid action penalty

        # Execute action
        reward = 0
        opponent_state = self.player_states[self.other_player]
        if action == self.ACTION_CHARGE:
            if current_state["ep"] < 5:
                current_state["ep"] += 1
                current_state["ep"] = min(current_state["ep"], 5)
        elif action == self.ACTION_SHIELD:
            current_state["ep"] -= 1
            current_state["shield_active"] = True
            current_state["shield_blocked"] = False
        elif action == self.ACTION_ATTACK:
            current_state["ep"] -= 2
            if opponent_state["shield_active"]:
                # Attack is blocked
                opponent_state["shield_active"] = False
                opponent_state["shield_blocked"] = True
            else:
                opponent_state["ip"] -= 2
                opponent_state["ip"] = max(opponent_state["ip"], 0)

        # Check for win condition
        if opponent_state["ip"] <= 0:
            self.done = True
            reward = 1
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Swap current player
        self.current_player, self.other_player = self.other_player, self.current_player

        # At the start of the new current player's turn, manage shield status
        current_state = self.player_states[self.current_player]
        if current_state["shield_active"] and not current_state["shield_blocked"]:
            current_state["shield_active"] = False  # Shield expires
        current_state["shield_blocked"] = False  # Reset shield blocked status

        observation = self._get_observation()
        return observation, reward, False, False, {}

    def render(self):
        current_state = self.player_states[self.current_player]
        opponent_state = self.player_states[self.other_player]
        output = f"Current Player: Player {self.current_player}\n"
        output += f"Player {self.current_player} - Crystal IP: {current_state['ip']}, EP: {current_state['ep']}, Shield: {'ACTIVE' if current_state['shield_active'] else 'INACTIVE'}\n"
        output += f"Player {self.other_player} - Crystal IP: {opponent_state['ip']}, EP: {opponent_state['ep']}, Shield: {'ACTIVE' if opponent_state['shield_active'] else 'INACTIVE'}\n"
        return output

    def valid_moves(self):
        current_state = self.player_states[self.current_player]
        valid_actions = []
        if current_state["ep"] < 5:
            valid_actions.append(self.ACTION_CHARGE)
        if current_state["ep"] >= 1:
            valid_actions.append(self.ACTION_SHIELD)
        if current_state["ep"] >= 2:
            valid_actions.append(self.ACTION_ATTACK)
        return valid_actions

    def _get_observation(self):
        current_state = self.player_states[self.current_player]
        opponent_state = self.player_states[self.other_player]
        observation = np.array(
            [
                current_state["ip"],
                current_state["ep"],
                1 if current_state["shield_active"] else 0,
                opponent_state["ip"],
                opponent_state["ep"],
                1 if opponent_state["shield_active"] else 0,
            ],
            dtype=np.int32,
        )
        return observation
