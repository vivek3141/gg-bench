import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions:
        # 0-6: Observe cell 0-6 (cells 1-7)
        # 7: Do not challenge
        # 8: Challenge
        self.action_space = spaces.Discrete(9)

        # Observation space: 7 cells + move prompt + challenge prompt
        # Cells can be -1 (Player 2), 0 (Unobserved), or 1 (Player 1)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(
            7, dtype=np.int32
        )  # Cells 0-6 correspond to positions 1-7
        self.current_player = 1  # 1 for Player 1 ('X'), -1 for Player 2 ('O')
        self.done = False
        self.move_prompt = 1  # 1 when agent should make a move, 0 otherwise
        self.challenge_prompt = (
            0  # 1 when agent should decide on challenge, 0 otherwise
        )
        self.last_action = None  # Keep track of last action taken
        self.info = {}
        return self._get_observation(), self.info  # Return observation and info

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        # Validate action
        if self.move_prompt == 1:
            # Expecting a move action (0-6)
            if action not in range(7):
                # Invalid action
                reward = -10
                terminated = True
                return self._get_observation(), reward, terminated, truncated, self.info

            if self.board[action] != 0 or self.done:
                # Invalid move (cell already observed or game over)
                reward = -10
                terminated = True
                return self._get_observation(), reward, terminated, truncated, self.info

            # Place the player's symbol on the board
            self.board[action] = self.current_player

            # Check for challenge opportunity by opponent
            challenge_possible = self._check_adjacent_own_symbols(
                action, self.current_player
            )

            if challenge_possible:
                # Opponent decides whether to challenge
                opponent_challenges = self._opponent_challenge_decision(action)

                if opponent_challenges:
                    # Resolve challenge
                    challenge_success = self._resolve_challenge()
                    if not challenge_success:
                        # Observation fails; cell reverts to unobserved
                        self.board[action] = 0

            # Check for win
            if self._check_win(self.current_player):
                reward = 1  # Current player wins
                terminated = True
                return self._get_observation(), reward, terminated, truncated, self.info

            # Switch to opponent's turn
            self.current_player *= -1  # Switch player
            self.move_prompt = 0
            self.challenge_prompt = 0

            # Opponent's move
            opponent_action = self._opponent_move()
            if opponent_action is None:
                # No valid moves left
                reward = 0
                terminated = True
                return self._get_observation(), reward, terminated, truncated, self.info

            # Check for challenge opportunity by agent
            challenge_possible = self._check_adjacent_own_symbols(
                opponent_action, self.current_player
            )
            if challenge_possible:
                # Prompt agent to decide whether to challenge
                self.move_prompt = 0
                self.challenge_prompt = 1
                self.last_action = opponent_action
                return self._get_observation(), reward, terminated, truncated, self.info
            else:
                # Switch back to agent's turn
                self.current_player *= -1
                self.move_prompt = 1
                self.challenge_prompt = 0
                return self._get_observation(), reward, terminated, truncated, self.info

        elif self.challenge_prompt == 1:
            # Expecting a challenge decision (7 or 8)
            if action not in [7, 8]:
                # Invalid action
                reward = -10
                terminated = True
                return self._get_observation(), reward, terminated, truncated, self.info

            if action == 8:
                # Agent decides to challenge
                # Resolve challenge
                challenge_success = self._resolve_challenge()
                if not challenge_success:
                    # Observation fails; opponent's cell reverts to unobserved
                    self.board[self.last_action] = 0

            # Check for win after opponent's move
            if self._check_win(self.current_player):
                reward = -1  # Opponent wins
                terminated = True
                return self._get_observation(), reward, terminated, truncated, self.info

            # Switch back to agent's turn
            self.current_player *= -1
            self.move_prompt = 1
            self.challenge_prompt = 0
            self.last_action = None
            return self._get_observation(), reward, terminated, truncated, self.info

        else:
            # Invalid state (should not happen)
            reward = -10
            terminated = True
            return self._get_observation(), reward, terminated, truncated, self.info

    def render(self):
        board_str = ""
        symbols = {1: "X", -1: "O", 0: "_"}
        for idx, cell in enumerate(self.board):
            board_str += f"[ {symbols[cell]} ] "
        board_str += "\n  1     2     3     4     5     6     7"
        return board_str

    def valid_moves(self):
        if self.move_prompt == 1:
            return [i for i in range(7) if self.board[i] == 0]
        elif self.challenge_prompt == 1:
            return [7, 8]  # 7: Do not challenge, 8: Challenge
        else:
            return []

    def _get_observation(self):
        # Build observation array
        obs = np.zeros(9, dtype=np.int32)
        obs[0:7] = self.board  # Board state
        obs[7] = self.move_prompt  # Move prompt
        obs[8] = self.challenge_prompt  # Challenge prompt
        return obs

    def _check_adjacent_own_symbols(self, action, player):
        # Check if the cell is adjacent to one or more of the player's own symbols
        adjacent_indices = []
        if action > 0:
            adjacent_indices.append(action - 1)
        if action < 6:
            adjacent_indices.append(action + 1)
        for idx in adjacent_indices:
            if self.board[idx] == player:
                return True
        return False

    def _opponent_challenge_decision(self, action):
        # Simple opponent challenge policy:
        # Opponent always challenges when possible
        return True

    def _resolve_challenge(self):
        # Randomly resolve challenge (coin flip)
        result = self.np_random.choice([0, 1])
        if result == 1:
            return True  # Observation succeeds
        else:
            return False  # Observation fails
        # Note: self.np_random is initialized in the base class

    def _check_win(self, player):
        # Check for an unbroken chain of four of the player's symbols
        chain_length = 0
        for cell in self.board:
            if cell == player:
                chain_length += 1
                if chain_length == 4:
                    return True
            else:
                chain_length = 0
        return False

    def _opponent_move(self):
        # Simple opponent move policy:
        # Opponent chooses a random valid move
        valid_moves = [i for i in range(7) if self.board[i] == 0]
        if not valid_moves:
            return None  # No valid moves left
        action = self.np_random.choice(valid_moves)
        self.board[action] = self.current_player

        # Check for opponent's victory
        if self._check_win(self.current_player):
            return action  # Opponent's winning move

        # Check if agent can challenge
        # (Challenge decision is handled in the main step function)
        return action
