import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Attack, 1 - Defend, 2 - Paradox
        self.action_space = spaces.Discrete(3)

        # Observation space:
        # [player_energy, opponent_energy, player_paradox_used, opponent_paradox_used]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([10, 10, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_energy = 10
        self.opponent_energy = 10
        self.player_paradox_used = 0  # 0 means Paradox not used yet
        self.opponent_paradox_used = 0
        self.done = False
        self.current_player = 1  # Player 1 starts
        self.info = {}
        observation = self._get_observation()
        return observation, self.info  # observation, info

    def step(self, action):
        if self.done:
            raise Exception("Game is over. Please reset the environment.")

        # Check if action is valid
        if action not in self.valid_moves():
            reward = -10
            self.done = True
            observation = self._get_observation()
            return (
                observation,
                reward,
                True,
                False,
                self.info,
            )  # observation, reward, terminated, truncated, info

        # Map action index to action name
        action_mapping = {0: "Attack", 1: "Defend", 2: "Paradox"}
        player_action = action_mapping[action]

        # Check if player has already used Paradox
        if player_action == "Paradox" and self.player_paradox_used == 1:
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, self.info

        # Generate opponent's action
        opponent_action = self._opponent_policy()

        # Resolve actions
        self._resolve_actions(player_action, opponent_action)

        # Check for win/loss conditions
        if self.player_energy <= 0:
            reward = -1  # Player loses
            self.done = True
        elif self.opponent_energy <= 0:
            reward = 1  # Player wins
            self.done = True
        elif self.done:  # Paradox outcome resulting in loss
            if self.player_energy > 0 and self.opponent_energy > 0:
                reward = -1  # Both players lose due to Paradox vs. Paradox
            else:
                reward = 1 if self.opponent_energy <= 0 else -1
        else:
            reward = -10  # Valid move, game continues

        observation = self._get_observation()
        return observation, reward, self.done, False, self.info

    def render(self):
        output = (
            f"Player Energy: {self.player_energy}\n"
            f"Opponent Energy: {self.opponent_energy}\n"
            f"Player Paradox Used: {'Yes' if self.player_paradox_used else 'No'}\n"
            f"Opponent Paradox Used: {'Yes' if self.opponent_paradox_used else 'No'}\n"
        )
        return output

    def valid_moves(self):
        moves = [0, 1]  # Attack and Defend are always available
        if self.player_paradox_used == 0:
            moves.append(2)  # Paradox is available if not used yet
        return moves

    def _get_observation(self):
        return np.array(
            [
                self.player_energy,
                self.opponent_energy,
                self.player_paradox_used,
                self.opponent_paradox_used,
            ],
            dtype=np.float32,
        )

    def _opponent_policy(self):
        # Randomly choose a valid action for the opponent
        opponent_moves = [0, 1]  # Attack and Defend
        if self.opponent_paradox_used == 0:
            opponent_moves.append(2)  # Paradox
        opponent_action = np.random.choice(opponent_moves)
        action_mapping = {0: "Attack", 1: "Defend", 2: "Paradox"}
        return action_mapping[opponent_action]

    def _resolve_actions(self, player_action, opponent_action):
        # Paradox usage flags
        player_paradox_this_round = False
        opponent_paradox_this_round = False

        # Update Paradox usage
        if player_action == "Paradox":
            self.player_paradox_used = 1
            player_paradox_this_round = True
        if opponent_action == "Paradox":
            self.opponent_paradox_used = 1
            opponent_paradox_this_round = True

        # Resolve actions
        if player_action == "Attack" and opponent_action == "Attack":
            # Both players lose 2 energy units
            self.player_energy -= 2
            self.opponent_energy -= 2

        elif player_action == "Attack" and opponent_action == "Defend":
            # Player steals 1 energy unit from opponent
            transfer = min(1, self.opponent_energy)
            self.player_energy = min(10, self.player_energy + transfer)
            self.opponent_energy -= transfer

        elif player_action == "Defend" and opponent_action == "Attack":
            # Opponent steals 1 energy unit from player
            transfer = min(1, self.player_energy)
            self.opponent_energy = min(10, self.opponent_energy + transfer)
            self.player_energy -= transfer

        elif player_action == "Defend" and opponent_action == "Defend":
            # No change
            pass

        elif player_paradox_this_round or opponent_paradox_this_round:
            # Paradox cases
            if player_paradox_this_round and opponent_paradox_this_round:
                # Both players lose
                self.player_energy = 0
                self.opponent_energy = 0
                self.done = True
            elif player_paradox_this_round and opponent_action == "Attack":
                # Player wins immediately
                self.opponent_energy = 0
                self.done = True
            elif opponent_paradox_this_round and player_action == "Attack":
                # Opponent wins immediately
                self.player_energy = 0
                self.done = True
            else:
                # Paradox vs. Defend or Paradox vs. Paradox (already handled)
                pass
        else:
            # Any remaining combinations (e.g., Paradox vs. Defend)
            pass

        # Ensure energy levels are within bounds
        self.player_energy = max(0, min(10, self.player_energy))
        self.opponent_energy = max(0, min(10, self.opponent_energy))
