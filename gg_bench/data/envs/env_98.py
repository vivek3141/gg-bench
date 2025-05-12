import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 (Activate Ether's Revive), 1-50 (Attacks)
        self.action_space = spaces.Discrete(51)

        # Observation: 21-dimensional vector
        # Index 0-9: Player 1's elements (alive, special used)
        # Index 10-19: Player 2's elements (alive, special used)
        # Index 20: Current player (1 or 2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(21,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.elements = {
            1: {
                "Fire": {"alive": True, "special_used": False},
                "Water": {"alive": True, "special_used": False},
                "Earth": {"alive": True, "special_used": False},
                "Air": {"alive": True, "special_used": False},
                "Ether": {"alive": True, "special_used": False},
            },
            2: {
                "Fire": {"alive": True, "special_used": False},
                "Water": {"alive": True, "special_used": False},
                "Earth": {"alive": True, "special_used": False},
                "Air": {"alive": True, "special_used": False},
                "Ether": {"alive": True, "special_used": False},
            },
        }
        self.current_player = 1
        self.done = False
        return self._get_observation(), {}  # Observation, info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}
        reward = 0

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self._get_observation(), -10, True, False, {}  # Invalid move

        # Process action
        opponent = 2 if self.current_player == 1 else 1

        if action == 0:
            # Activate Ether's Revive ability
            # Revive one of the player's eliminated elements
            eliminated_elements = [
                elem
                for elem, status in self.elements[self.current_player].items()
                if not status["alive"]
            ]
            if eliminated_elements:
                # Revive the first eliminated element
                revived_element = eliminated_elements[0]
                self.elements[self.current_player][revived_element]["alive"] = True
                self.elements[self.current_player]["Ether"]["special_used"] = True
            else:
                # No elements to revive (should not happen due to valid_moves check)
                pass
        else:
            # Attack action
            action -= 1
            attacking_element_index = action // 10
            action %= 10
            defending_element_index = action // 2
            special_ability_usage = action % 2 == 1  # True if using special ability

            # Map indices to element names
            elements_list = ["Fire", "Water", "Earth", "Air", "Ether"]
            attacking_element_name = elements_list[attacking_element_index]
            defending_element_name = elements_list[defending_element_index]

            # Check special ability usage
            attacker_special_used = self.elements[self.current_player][
                attacking_element_name
            ]["special_used"]
            if special_ability_usage and attacker_special_used:
                # Should not happen due to valid_moves check
                pass
            # Attacker's base strength
            strengths = {"Fire": 3, "Water": 4, "Earth": 5, "Air": 2, "Ether": 1}
            attack_strength = strengths[attacking_element_name]
            defense_strength = strengths[defending_element_name]

            # Apply special abilities
            # Attacker's special ability
            if special_ability_usage:
                if attacking_element_name == "Fire":
                    # Fire – Blaze (+2 attack strength)
                    attack_strength += 2
                    self.elements[self.current_player][attacking_element_name][
                        "special_used"
                    ] = True
                elif attacking_element_name == "Air":
                    # Air – Gale (Swap strengths)
                    attack_strength, defense_strength = (
                        defense_strength,
                        attack_strength,
                    )
                    self.elements[self.current_player][attacking_element_name][
                        "special_used"
                    ] = True

            # Defender's special ability (automatically applied if beneficial)
            defender_special_used = self.elements[opponent][defending_element_name][
                "special_used"
            ]
            # Decide whether defender will use special ability
            defender_uses_special = False
            if defending_element_name == "Water" and not defender_special_used:
                # Water – Surge (Nullify attacker's special ability)
                if special_ability_usage:
                    defense_strength = strengths[
                        defending_element_name
                    ]  # Reset strengths
                    if (
                        attacking_element_name == "Fire"
                        and attacking_element_name != "Air"
                    ):
                        attack_strength = strengths[attacking_element_name]
                    elif attacking_element_name == "Air":
                        attack_strength, defense_strength = (
                            strengths[attacking_element_name],
                            strengths[defending_element_name],
                        )
                    self.elements[opponent][defending_element_name][
                        "special_used"
                    ] = True
                    defender_uses_special = True
            elif defending_element_name == "Earth" and not defender_special_used:
                # Earth – Fortify (Prevent elimination)
                # Use it only if Earth would be eliminated
                if attack_strength > defense_strength:
                    self.elements[opponent][defending_element_name][
                        "special_used"
                    ] = True
                    defender_uses_special = True
            # Note: Ether's and Fire's special abilities can't be used in defense

            # Resolve attack
            if attack_strength > defense_strength:
                # Attacker wins
                if not (defending_element_name == "Earth" and defender_uses_special):
                    self.elements[opponent][defending_element_name]["alive"] = False
            elif attack_strength < defense_strength:
                # Defender wins
                self.elements[self.current_player][attacking_element_name][
                    "alive"
                ] = False
            else:
                # Tie
                self.elements[self.current_player][attacking_element_name][
                    "alive"
                ] = False
                if not (defending_element_name == "Earth" and defender_uses_special):
                    self.elements[opponent][defending_element_name]["alive"] = False

        # Check for game over
        if all(not status["alive"] for status in self.elements[opponent].values()):
            # Current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, {}

        # Switch current player
        self.current_player = opponent

        return self._get_observation(), reward, False, False, {}

    def render(self):
        elements_list = ["Fire", "Water", "Earth", "Air", "Ether"]
        render_str = f"Current Player: Player {self.current_player}\n"
        for player in [1, 2]:
            render_str += f"Player {player}:\n"
            for elem in elements_list:
                status = self.elements[player][elem]
                alive_str = "Alive" if status["alive"] else "Eliminated"
                special_str = "Used" if status["special_used"] else "Available"
                render_str += f"  {elem}: {alive_str}, Special: {special_str}\n"
        return render_str

    def valid_moves(self):
        valid_actions = []
        if self.done:
            return valid_actions
        opponent = 2 if self.current_player == 1 else 1
        elements_list = ["Fire", "Water", "Earth", "Air", "Ether"]

        # Check if Ether's Revive can be used
        ether_status = self.elements[self.current_player]["Ether"]
        if (
            ether_status["alive"]
            and not ether_status["special_used"]
            and any(
                not status["alive"]
                for elem, status in self.elements[self.current_player].items()
                if elem != "Ether"
            )
        ):
            valid_actions.append(0)  # Action to activate Ether's Revive

        # Check all possible attack actions
        for atk_idx, atk_elem in enumerate(elements_list):
            atk_status = self.elements[self.current_player][atk_elem]
            if not atk_status["alive"]:
                continue
            for def_idx, def_elem in enumerate(elements_list):
                def_status = self.elements[opponent][def_elem]
                if not def_status["alive"]:
                    continue
                # Determine if special ability can be used
                # Special ability can only be used if not used yet
                special_available = not atk_status["special_used"] and atk_elem in [
                    "Fire",
                    "Air",
                ]
                if special_available:
                    # Action with special ability
                    action = 1 + (atk_idx * 10) + (def_idx * 2) + 1
                    valid_actions.append(action)
                # Action without special ability
                action = 1 + (atk_idx * 10) + (def_idx * 2) + 0
                valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        obs = np.zeros(21, dtype=np.int32)
        elements_list = ["Fire", "Water", "Earth", "Air", "Ether"]
        # Player 1's elements
        for idx, elem in enumerate(elements_list):
            status = self.elements[1][elem]
            obs[idx * 2] = int(status["alive"])
            obs[idx * 2 + 1] = int(status["special_used"])
        # Player 2's elements
        for idx, elem in enumerate(elements_list):
            status = self.elements[2][elem]
            obs[10 + idx * 2] = int(status["alive"])
            obs[10 + idx * 2 + 1] = int(status["special_used"])
        # Current player
        obs[20] = self.current_player - 1  # 0 for Player 1, 1 for Player 2
        return obs
