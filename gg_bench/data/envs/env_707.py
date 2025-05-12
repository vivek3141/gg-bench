import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space

        # Actions:
        # 0-9: harvest 1-10 SUs (actions 0 corresponds to harvest 1 SU)
        # 10: spy
        # 11: sabotage
        # 12: fortify
        # 13: do nothing

        self.action_space = spaces.Discrete(14)

        # Observation space:
        # [SU, PP, is_fortified]
        # SU: from 0 to max_SU
        # PP: from 0 to max_PP
        # is_fortified: 0 or 1

        self.max_SU = 20
        self.max_PP = 100

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([self.max_SU, self.max_PP, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_states = {
            1: {"SU": 3, "PP": 0, "is_fortified": False, "fortified_until": 0},
            -1: {"SU": 3, "PP": 0, "is_fortified": False, "fortified_until": 0},
        }
        self.current_player = 1
        self.opponent = -1
        self.done = False
        self.turn_counter = 0
        return self._get_observation(), {}

    def _get_observation(self):
        state = self.player_states[self.current_player]
        obs = np.array(
            [state["SU"], state["PP"], 1 if state["is_fortified"] else 0],
            dtype=np.float32,
        )
        return obs

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        reward = 0
        info = {}
        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid action
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, info

        current_state = self.player_states[self.current_player]
        opponent_state = self.player_states[self.opponent]

        # Process current player's action
        if action >= 0 and action <= 9:
            # Harvest action
            n = action + 1
            if current_state["SU"] >= n:
                current_state["SU"] -= n
                current_state["PP"] += n * 20
                if current_state["PP"] > 100:
                    reward = -10
                    self.done = True
                    return self._get_observation(), reward, True, False, info
                elif current_state["PP"] == 100:
                    reward = 1
                    self.done = True
                    return self._get_observation(), reward, True, False, info
            else:
                self.done = True
                reward = -10
                return self._get_observation(), reward, True, False, info
        elif action == 10:
            # Spy
            if current_state["SU"] >= 1:
                current_state["SU"] -= 1
                info = {"opponent_PP": opponent_state["PP"]}
            else:
                self.done = True
                reward = -10
                return self._get_observation(), reward, True, False, info
        elif action == 11:
            # Sabotage
            if current_state["SU"] >= 2:
                current_state["SU"] -= 2
                if not opponent_state["is_fortified"]:
                    opponent_state["PP"] = max(0, opponent_state["PP"] - 30)
            else:
                self.done = True
                reward = -10
                return self._get_observation(), reward, True, False, info
        elif action == 12:
            # Fortify
            if current_state["SU"] >= 1:
                current_state["SU"] -= 1
                current_state["is_fortified"] = True
                current_state["fortified_until"] = self.turn_counter + 2
            else:
                self.done = True
                reward = -10
                return self._get_observation(), reward, True, False, info
        elif action == 13:
            # Do nothing
            pass

        self.turn_counter += 1

        # Reset fortification if expired
        if current_state["fortified_until"] <= self.turn_counter:
            current_state["is_fortified"] = False

        # Switch players
        self.current_player, self.opponent = self.opponent, self.current_player

        # Opponent's turn
        opponent_state = self.player_states[self.current_player]
        current_state = self.player_states[self.opponent]

        # Opponent gains 1 SU
        opponent_state["SU"] += 1

        # Reset fortification if expired
        if opponent_state["fortified_until"] <= self.turn_counter:
            opponent_state["is_fortified"] = False

        # Opponent selects random valid action
        opponent_valid_actions = self.get_valid_actions(self.current_player)
        opponent_action = np.random.choice(opponent_valid_actions)

        # Process opponent's action
        if opponent_action >= 0 and opponent_action <= 9:
            # Harvest action
            n = opponent_action + 1
            if opponent_state["SU"] >= n:
                opponent_state["SU"] -= n
                opponent_state["PP"] += n * 20
                if opponent_state["PP"] > 100:
                    reward = 1
                    self.done = True
                    return self._get_observation(), reward, True, False, info
                elif opponent_state["PP"] == 100:
                    reward = -1
                    self.done = True
                    return self._get_observation(), reward, True, False, info
            else:
                reward = 1
                self.done = True
                return self._get_observation(), reward, True, False, info
        elif opponent_action == 10:
            # Spy
            if opponent_state["SU"] >= 1:
                opponent_state["SU"] -= 1
            else:
                reward = 1
                self.done = True
                return self._get_observation(), reward, True, False, info
        elif opponent_action == 11:
            # Sabotage
            if opponent_state["SU"] >= 2:
                opponent_state["SU"] -= 2
                if not current_state["is_fortified"]:
                    current_state["PP"] = max(0, current_state["PP"] - 30)
            else:
                reward = 1
                self.done = True
                return self._get_observation(), reward, True, False, info
        elif opponent_action == 12:
            # Fortify
            if opponent_state["SU"] >= 1:
                opponent_state["SU"] -= 1
                opponent_state["is_fortified"] = True
                opponent_state["fortified_until"] = self.turn_counter + 2
            else:
                reward = 1
                self.done = True
                return self._get_observation(), reward, True, False, info
        elif opponent_action == 13:
            # Do nothing
            pass

        self.turn_counter += 1

        # Reset fortification if expired
        if opponent_state["fortified_until"] <= self.turn_counter:
            opponent_state["is_fortified"] = False

        # Switch back to original player
        self.current_player, self.opponent = self.opponent, self.current_player

        return self._get_observation(), reward, self.done, False, info

    def render(self):
        state = self.player_states[self.current_player]
        opponent_state = self.player_states[self.opponent]
        state_str = f"Player {self.current_player}'s Turn\n"
        state_str += "------------------\n"
        state_str += f"You have {state['SU']} SUs and {state['PP']} PP.\n"
        if state["is_fortified"]:
            state_str += "You are fortified.\n"
        else:
            state_str += "You are not fortified.\n"
        state_str += f"Opponent has {opponent_state['SU']} SUs.\n"
        return state_str

    def valid_moves(self):
        return self.get_valid_actions(self.current_player)

    def get_valid_actions(self, player):
        state = self.player_states[player]
        su = state["SU"]
        valid_actions = []

        # Harvest actions
        max_harvest = min(su, 10)
        for n in range(1, max_harvest + 1):
            valid_actions.append(n - 1)  # Actions 0-9 correspond to harvest 1-10

        # Spy
        if su >= 1:
            valid_actions.append(10)
        # Sabotage
        if su >= 2:
            valid_actions.append(11)
        # Fortify
        if su >= 1:
            valid_actions.append(12)
        # Do nothing
        valid_actions.append(13)

        return valid_actions
