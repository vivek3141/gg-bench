import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Charge, 1 - Shield, 2 - Blast
        # Since actions are simultaneous, we consider all combinations
        # Mapping action indices to action pairs (player1_action, player2_action)
        self.action_space = spaces.Discrete(9)
        # Observation: [Player1_EP, Player2_EP, Player1_cannot_charge, Player2_cannot_charge]
        self.observation_space = spaces.Box(
            low=np.array([-20, -20, 0, 0]),
            high=np.array([40, 40, 1, 1]),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1_EP = 10
        self.player2_EP = 10
        self.player1_cannot_charge = False
        self.player2_cannot_charge = False
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(
            [
                self.player1_EP,
                self.player2_EP,
                int(self.player1_cannot_charge),
                int(self.player2_cannot_charge),
            ],
            dtype=np.int32,
        )

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Map the action index to player actions
        action_map = {
            0: ("Charge", "Charge"),
            1: ("Charge", "Shield"),
            2: ("Charge", "Blast"),
            3: ("Shield", "Charge"),
            4: ("Shield", "Shield"),
            5: ("Shield", "Blast"),
            6: ("Blast", "Charge"),
            7: ("Blast", "Shield"),
            8: ("Blast", "Blast"),
        }

        player1_action, player2_action = action_map[action]

        reward = 0
        info = {}
        invalid_action = False

        # Check for action validity
        if player1_action == "Charge" and self.player1_cannot_charge:
            invalid_action = True
            reward = -10
            info["error"] = "Player 1 cannot Charge after Shield."

        if player2_action == "Charge" and self.player2_cannot_charge:
            invalid_action = True
            reward = -10
            info["error"] = "Player 2 cannot Charge after Shield."

        if player1_action == "Blast" and self.player1_EP < 2:
            invalid_action = True
            reward = -10
            info["error"] = "Player 1 does not have enough EP to Blast."

        if player2_action == "Blast" and self.player2_EP < 2:
            invalid_action = True
            reward = -10
            info["error"] = "Player 2 does not have enough EP to Blast."

        if invalid_action:
            self.done = True
            return self._get_obs(), reward, self.done, False, info

        # Initialize EP changes
        p1_EP_change = 0
        p2_EP_change = 0

        # Reset cannot_charge flags
        p1_cannot_charge_next = False
        p2_cannot_charge_next = False

        # Process actions
        if player1_action == "Charge":
            p1_EP_change += 1
        elif player1_action == "Shield":
            p1_cannot_charge_next = True
        elif player1_action == "Blast":
            self.player1_EP -= 2  # Cost of Blast

        if player2_action == "Charge":
            p2_EP_change += 1
        elif player2_action == "Shield":
            p2_cannot_charge_next = True
        elif player2_action == "Blast":
            self.player2_EP -= 2  # Cost of Blast

        # Resolve damage
        if player1_action == "Blast":
            if player2_action != "Shield":
                p2_EP_change -= 3  # Damage to Player 2

        if player2_action == "Blast":
            if player1_action != "Shield":
                p1_EP_change -= 3  # Damage to Player 1

        # Update EPs
        self.player1_EP += p1_EP_change
        self.player2_EP += p2_EP_change

        # Update cannot_charge flags
        self.player1_cannot_charge = p1_cannot_charge_next
        self.player2_cannot_charge = p2_cannot_charge_next

        # Check for victory conditions
        if self.player1_EP <= 0 and self.player2_EP <= 0:
            # Determine who caused the final reduction
            if player1_action == "Blast" and player2_action != "Blast":
                reward = -10  # Player 1 loses
            elif player2_action == "Blast" and player1_action != "Blast":
                reward = 1  # Player 1 wins
            else:
                reward = -10  # Both lose
            self.done = True
        elif self.player1_EP <= 0:
            reward = -10  # Player 1 loses
            self.done = True
        elif self.player2_EP <= 0:
            reward = 1  # Player 1 wins
            self.done = True
        else:
            reward = 0  # Game continues

        return self._get_obs(), reward, self.done, False, info

    def render(self):
        p1_status = f"Player 1 EP: {self.player1_EP}, "
        p1_status += (
            "Cannot Charge next turn."
            if self.player1_cannot_charge
            else "Can act freely."
        )

        p2_status = f"Player 2 EP: {self.player2_EP}, "
        p2_status += (
            "Cannot Charge next turn."
            if self.player2_cannot_charge
            else "Can act freely."
        )

        return f"{p1_status}\n{p2_status}\n"

    def valid_moves(self):
        valid_actions = []
        action_map = {
            0: ("Charge", "Charge"),
            1: ("Charge", "Shield"),
            2: ("Charge", "Blast"),
            3: ("Shield", "Charge"),
            4: ("Shield", "Shield"),
            5: ("Shield", "Blast"),
            6: ("Blast", "Charge"),
            7: ("Blast", "Shield"),
            8: ("Blast", "Blast"),
        }

        for action in range(9):
            player1_action, player2_action = action_map[action]
            valid = True

            if player1_action == "Charge" and self.player1_cannot_charge:
                valid = False
            if player1_action == "Blast" and self.player1_EP < 2:
                valid = False
            if player2_action == "Charge" and self.player2_cannot_charge:
                valid = False
            if player2_action == "Blast" and self.player2_EP < 2:
                valid = False

            if valid:
                valid_actions.append(action)

        return valid_actions
