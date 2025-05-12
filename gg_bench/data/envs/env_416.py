import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 (Pass), 1-9 (Select number 1-9)
        self.action_space = spaces.Discrete(10)
        # Observation space:
        # [Number Pool (9), Current Player Central (1), Current Player Petals (4),
        # Opponent Central (1), Opponent Petals (4)]
        self.observation_space = spaces.Box(low=0, high=9, shape=(19,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Number pool: 1-9 (1 if available, 0 if not)
        self.number_pool = np.ones(9, dtype=np.int32)

        # Blossoms: Each player has a dict with 'central' and 'petals' list
        self.player_blossoms = {
            1: {"central": 0, "petals": []},
            2: {"central": 0, "petals": []},
        }

        # Current player: Start with Player 1
        self.current_player = 1

        # Flags to indicate if central numbers are selected
        self.phase = "central_selection"

        # Game over flag
        self.done = False

        # Info dictionary
        info = {}

        # Return initial observation
        observation = self._get_observation()
        return observation, info

    def step(self, action):
        # Check if game is already over
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map action to number
        if action < 0 or action > 9:
            # Invalid action
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        if action == 0:
            # Player passes their turn
            pass_action = True
        else:
            pass_action = False
            number_chosen = action  # Numbers are 1-9

        # Perform action based on phase
        reward = 0
        terminated = False

        if self.phase == "central_selection":
            # Central number selection phase
            if pass_action:
                # Cannot pass during central number selection
                reward = -10
                self.done = True
                return self._get_observation(), reward, True, False, {}
            else:
                # Validate central number selection
                if self.number_pool[number_chosen - 1] == 0:
                    # Number not in pool (already taken)
                    reward = -10
                    self.done = True
                    return self._get_observation(), reward, True, False, {}
                else:
                    # Valid central number selection
                    self.player_blossoms[self.current_player]["central"] = number_chosen
                    self.number_pool[number_chosen - 1] = 0
                    # Check if both players have selected their central numbers
                    if (
                        self.player_blossoms[1]["central"] != 0
                        and self.player_blossoms[2]["central"] != 0
                    ):
                        self.phase = "petal_selection"
        else:
            # Petal selection phase
            if pass_action:
                # Player passes their turn
                pass
            else:
                # Validate petal selection
                if self.number_pool[number_chosen - 1] == 0:
                    # Number not in pool
                    reward = -10
                    self.done = True
                    return self._get_observation(), reward, True, False, {}
                else:
                    central_number = self.player_blossoms[self.current_player][
                        "central"
                    ]
                    gcd = math.gcd(central_number, number_chosen)
                    if gcd <= 1:
                        # GCD rule violated
                        reward = -10
                        self.done = True
                        return self._get_observation(), reward, True, False, {}
                    else:
                        # Valid petal selection
                        self.player_blossoms[self.current_player]["petals"].append(
                            number_chosen
                        )
                        self.number_pool[number_chosen - 1] = 0
                        # Check for victory condition
                        if (
                            len(self.player_blossoms[self.current_player]["petals"])
                            >= 4
                        ):
                            # Current player wins
                            reward = 1
                            self.done = True
                            return self._get_observation(), reward, True, False, {}
        # Switch to next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Provide observation to the agent from the perspective of the current player
        observation = self._get_observation()
        info = {}

        return observation, reward, False, False, info

    def render(self):
        # Visual representation of the game state
        pool_numbers = [str(i + 1) for i in range(9) if self.number_pool[i] == 1]
        pool_str = "Number Pool: " + ", ".join(pool_numbers) + "\n"

        # Current player blossom
        cp = self.current_player
        op = 2 if cp == 1 else 1
        cp_blossom = self.player_blossoms[cp]
        op_blossom = self.player_blossoms[op]

        cp_central = cp_blossom["central"]
        cp_petals = cp_blossom["petals"]
        op_central = op_blossom["central"]
        op_petals = op_blossom["petals"]

        cp_str = f"Player {cp} Blossom:\n"
        if cp_central != 0:
            cp_str += f"  Central Number: {cp_central}\n"
        else:
            cp_str += "  Central Number: Not selected\n"
        cp_str += f"  Petals: {', '.join(map(str, cp_petals))}\n"

        op_str = f"Player {op} Blossom:\n"
        if op_central != 0:
            op_str += f"  Central Number: {op_central}\n"
        else:
            op_str += "  Central Number: Not selected\n"
        op_str += f"  Petals: {', '.join(map(str, op_petals))}\n"

        phase_str = f"Current Phase: {self.phase.replace('_', ' ').title()}\n"
        player_str = f"Current Player: Player {self.current_player}\n"

        return phase_str + player_str + pool_str + cp_str + op_str

    def valid_moves(self):
        valid_actions = []

        if self.done:
            return valid_actions

        if self.phase == "central_selection":
            # Valid actions are numbers in the pool
            valid_actions = [i + 1 for i in range(9) if self.number_pool[i] == 1]
        else:
            # Petal selection phase
            central_number = self.player_blossoms[self.current_player]["central"]
            pool_numbers = [i + 1 for i in range(9) if self.number_pool[i] == 1]
            for number in pool_numbers:
                if math.gcd(central_number, number) > 1:
                    valid_actions.append(number)
            # Passing is always an option
            valid_actions.append(0)

        return valid_actions

    def _get_observation(self):
        # Observation includes:
        # Number Pool (9 elements): 1 if number is in pool, 0 if not
        # Current Player Central Number (1 element)
        # Current Player Petals (4 elements)
        # Opponent Central Number (1 element)
        # Opponent Petals (4 elements)
        obs = np.zeros(19, dtype=np.int32)
        # Number Pool
        obs[0:9] = self.number_pool
        # Current Player's Blossom
        cp = self.current_player
        op = 2 if cp == 1 else 1
        cp_central = self.player_blossoms[cp]["central"]
        cp_petals = self.player_blossoms[cp]["petals"]
        op_central = self.player_blossoms[op]["central"]
        op_petals = self.player_blossoms[op]["petals"]
        obs[9] = cp_central
        obs[10:14] = self._pad_list(cp_petals, 4)
        # Opponent's Blossom
        obs[14] = op_central
        obs[15:19] = self._pad_list(op_petals, 4)
        return obs

    def _pad_list(self, lst, size):
        # Pads the list to the desired size with zeros
        return lst + [0] * (size - len(lst))
