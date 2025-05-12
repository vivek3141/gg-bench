import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: numbers 1-9 (actions 0-8), and 'pass' action (action 9)
        self.action_space = spaces.Discrete(10)

        # Observation space: [current player's stack total, opponent's stack total, current player indicator (1 or 2)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 1]),
            high=np.array([21, 21, 2]),
            shape=(3,),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_stacks = {1: 0, 2: 0}
        self.current_player = 1
        self.done = False
        self.info = {}
        observation = self._get_observation()
        return observation, self.info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, self.info

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, self.info

        # Handle 'pass' action
        if action == 9:
            # Check if the player must pass
            if len(valid_actions) > 1:
                # Player chose to pass when moves are available
                self.done = True
                reward = -10
                return self._get_observation(), reward, True, False, self.info
            else:
                # Player passes turn
                reward = 0
                self._switch_player()
                return self._get_observation(), reward, False, False, self.info

        # Add selected number to current player's stack
        number = action + 1  # action 0 corresponds to number 1
        self.player_stacks[self.current_player] += number

        # Check if stack total exceeds 21 (should not happen due to valid_moves(), but check for safety)
        if self.player_stacks[self.current_player] > 21:
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, self.info

        # Check for win condition: reaching exactly 21
        if self.player_stacks[self.current_player] == 21:
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, self.info

        # Check if both players cannot make a move
        next_player_valid_actions = self._valid_moves(self._opponent())
        current_player_valid_actions = self.valid_moves()

        if (
            len(current_player_valid_actions) == 1
            and len(next_player_valid_actions) == 1
        ):
            # Both players must pass, determine winner
            self.done = True
            current_total = self.player_stacks[self.current_player]
            opponent_total = self.player_stacks[self._opponent()]
            if current_total == opponent_total:
                # Tie, player who reached total first loses
                if self.current_player == 2:
                    # Player 1 reached total first
                    reward = -1
                else:
                    reward = 1
            elif current_total > opponent_total:
                reward = 1
            else:
                reward = -1
            return self._get_observation(), reward, True, False, self.info

        # Continue game
        reward = 0
        self._switch_player()
        return self._get_observation(), reward, False, False, self.info

    def render(self):
        state_str = "Player 1 Stack Total: {}\n".format(self.player_stacks[1])
        state_str += "Player 2 Stack Total: {}\n".format(self.player_stacks[2])
        state_str += "Current Player: Player {}\n".format(self.current_player)
        return state_str

    def valid_moves(self):
        return self._valid_moves(self.current_player)

    def _valid_moves(self, player):
        stack_total = self.player_stacks[player]
        valid_actions = []
        can_play = False
        for action in range(9):
            number = action + 1
            if stack_total + number <= 21:
                valid_actions.append(action)
                can_play = True
        if not can_play:
            valid_actions.append(9)  # 'pass' action
        return valid_actions

    def _switch_player(self):
        self.current_player = self._opponent()

    def _opponent(self):
        return 1 if self.current_player == 2 else 2

    def _get_observation(self):
        return np.array(
            [
                self.player_stacks[self.current_player],
                self.player_stacks[self._opponent()],
                self.current_player,
            ],
            dtype=np.int32,
        )
