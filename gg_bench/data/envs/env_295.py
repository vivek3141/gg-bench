import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0: Attack, 1: Defend, 2: Heal
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([10, 10, 1, 1, 1]),
            dtype=np.float32,
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player1 = {
            "hp": 10,
            "defending": False,
            "healing": False,
            "attack_restricted": False,
            "last_action": None,
        }
        self.player2 = {
            "hp": 10,
            "defending": False,
            "healing": False,
            "attack_restricted": False,
            "last_action": None,
        }
        self.current_player = 1
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        reward = 0

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        # Get current player and opponent
        if self.current_player == 1:
            player = self.player1
            opponent = self.player2
        else:
            player = self.player2
            opponent = self.player1

        # Process action
        if action == 0:  # Attack
            if opponent["defending"]:
                damage = 0
                opponent["defending"] = False  # Defend only blocks one attack
            elif opponent["healing"]:
                damage = 4  # Double damage
            else:
                damage = 2
            opponent["hp"] = max(0, opponent["hp"] - damage)
        elif action == 1:  # Defend
            player["defending"] = True
            player["attack_restricted"] = True  # Cannot attack next turn
        elif action == 2:  # Heal
            player["hp"] = min(10, player["hp"] + 1)
            player["healing"] = True

        # Check if opponent has lost
        if opponent["hp"] <= 0:
            self.done = True
            reward = 1  # Current player wins
            return self._get_obs(), reward, True, False, {}

        # Before switching turns, update statuses
        # Reset attack_restricted if player had it and has just taken their next turn
        if player["attack_restricted"]:
            player["attack_restricted"] = False
        # Reset player's healing status
        player["healing"] = False  # Healing effect applies for one turn

        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1

        # At the start of new player's turn, reset their 'defending' status if it was not reset earlier
        # Get the new current player
        if self.current_player == 1:
            player = self.player1
        else:
            player = self.player2

        # If player's 'defending' status is still True (i.e., they didn't block an attack), reset it
        if player["defending"]:
            player["defending"] = False

        # Reset player's 'healing' status
        player["healing"] = False

        return self._get_obs(), reward, False, False, {}

    def _get_obs(self):
        # Return observation array
        if self.current_player == 1:
            return np.array(
                [
                    self.player1["hp"],
                    self.player2["hp"],
                    int(self.player1["defending"]),
                    int(self.player2["defending"]),
                    int(self.player1["attack_restricted"]),
                ],
                dtype=np.float32,
            )
        else:
            # Switch perspectives
            return np.array(
                [
                    self.player2["hp"],
                    self.player1["hp"],
                    int(self.player2["defending"]),
                    int(self.player1["defending"]),
                    int(self.player2["attack_restricted"]),
                ],
                dtype=np.float32,
            )

    def render(self):
        # Return a string representing the current state
        state_str = (
            f"Player 1 HP: {self.player1['hp']} "
            + ("(Defending) " if self.player1["defending"] else "")
            + ("(Healing) " if self.player1["healing"] else "")
            + ("(Attack Restricted) " if self.player1["attack_restricted"] else "")
            + "\n"
            f"Player 2 HP: {self.player2['hp']} "
            + ("(Defending) " if self.player2["defending"] else "")
            + ("(Healing) " if self.player2["healing"] else "")
            + ("(Attack Restricted) " if self.player2["attack_restricted"] else "")
            + "\n"
            f"Current Player: Player {self.current_player}\n"
        )
        return state_str

    def valid_moves(self):
        # Return list of valid actions for the current player
        if self.current_player == 1:
            player = self.player1
        else:
            player = self.player2

        valid_actions = [0, 1, 2]  # All actions

        if player["attack_restricted"]:
            if 0 in valid_actions:
                valid_actions.remove(0)  # Cannot attack

        return valid_actions
