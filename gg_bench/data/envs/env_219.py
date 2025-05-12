import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action_space
        # Actions are:
        # 0 - 97: selecting numbers 2 to 99 (number = action + 2)
        # 98: selecting number 100
        # 99: challenge opponent's last selection
        self.action_space = spaces.Discrete(100)

        # Define observation_space
        # Observation is an array representing the status of numbers from 2 to 100
        # 0: number is available in pool
        # 1: number has been discarded (composite)
        # 2: current player has collected this prime number
        # 3: opponent has collected this prime number
        self.observation_space = spaces.Box(low=0, high=3, shape=(99,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize number pool (indices 0 to 98 represent numbers 2 to 100)
        self.number_pool = np.zeros(99, dtype=np.int8)  # 0: available

        # Players' collections of prime numbers
        self.player_collections = {1: set(), -1: set()}

        # Keep track of the last selection of each player (action indices)
        self.last_selection = {1: None, -1: None}

        # Penalties: number of turns to skip for each player
        self.skip_turn = {1: 0, -1: 0}

        # Current player: 1 or -1
        self.current_player = 1  # Player 1 starts

        # Keep track of whether the last action was a penalty
        self.last_action_was_penalty = {1: False, -1: False}

        # Set of numbers that have been challenged
        self.challenged_numbers = set()

        # Game over flag
        self.done = False

        return self._get_observation(), {}  # Return observation and info

    def _get_observation(self):
        # Return a copy of the number pool as the observation
        return self.number_pool.copy()

    def _is_prime(self, n):
        if n < 2:
            return False
        elif n == 2:
            return True
        elif n % 2 == 0:
            return False
        else:
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return False
        return True

    def _can_challenge(self):
        opponent = -self.current_player
        # Can challenge if opponent's last action was a valid selection and not penalized
        if (
            self.last_selection[opponent] is not None
            and not self.last_action_was_penalty[opponent]
        ):
            number = self.last_selection[opponent] + 2  # Map index to number
            if number not in self.challenged_numbers:
                return True
        return False

    def _challenge_opponent(self):
        opponent = -self.current_player
        number = self.last_selection[opponent] + 2  # Map index to number
        self.challenged_numbers.add(number)
        is_prime = self._is_prime(number)
        if not is_prime:
            # Successful challenge
            # Remove number from opponent's collection
            if number in self.player_collections[opponent]:
                self.player_collections[opponent].remove(number)
                # Update number pool to discarded
                self.number_pool[self.last_selection[opponent]] = 1
            # Penalty for opponent: they lose next turn
            self.skip_turn[opponent] += 1
            # Reward: steal one prime from opponent's collection (if any)
            if len(self.player_collections[opponent]) > 0:
                stolen_prime = self.player_collections[opponent].pop()
                self.player_collections[self.current_player].add(stolen_prime)
                # Update number pool to reflect ownership change
                prime_idx = stolen_prime - 2
                self.number_pool[prime_idx] = (
                    2 if self.current_player == 1 else 3
                )  # Mark as owned by current player
            # Opponent loses next turn; current player continues
            return True
        else:
            # Failed challenge
            # Penalty: current player loses a prime number (if any)
            if len(self.player_collections[self.current_player]) > 0:
                lost_prime = self.player_collections[self.current_player].pop()
                # Discard the lost prime from the game
                prime_idx = lost_prime - 2
                self.number_pool[prime_idx] = 1  # Mark as discarded
            # Reward for opponent: they gain an extra turn immediately
            # Switch current player to opponent
            self.current_player = opponent
            return False

    def _next_player(self):
        # Determine who should play next based on penalties and extra turns
        if self.skip_turn[self.current_player] > 0:
            self.skip_turn[self.current_player] -= 1
            # Current player skips turn; continue with same player in next step
            self.last_action_was_penalty[self.current_player] = True
        else:
            # Switch to opponent
            self.current_player = -self.current_player
            self.last_action_was_penalty[self.current_player] = False

            # Check if opponent needs to skip turn
            if self.skip_turn[self.current_player] > 0:
                self.skip_turn[self.current_player] -= 1
                # Opponent skips turn; switch back to current player
                self.last_action_was_penalty[self.current_player] = True
                self.current_player = -self.current_player

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if action is valid
        if action not in self.valid_moves():
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        info = {}
        reward = 0

        if action == 99:
            # Challenge opponent's last selection
            success = self._challenge_opponent()
            if success:
                # Successful challenge; current player continues
                pass
            else:
                # Failed challenge; current player switched to opponent
                pass
        else:
            # Selecting a number
            number = action + 2  # Map action to number
            is_prime = self._is_prime(number)
            if is_prime:
                # Add to current player's collection
                self.player_collections[self.current_player].add(number)
                self.number_pool[action] = 2 if self.current_player == 1 else 3
                # Check for winning condition
                if len(self.player_collections[self.current_player]) >= 5:
                    self.done = True
                    reward = 1  # Current player wins
            else:
                # Composite number; discard and apply penalty
                self.number_pool[action] = 1  # Mark as discarded
                self.skip_turn[self.current_player] += 1
                self.last_action_was_penalty[self.current_player] = True

            # Update last selection
            self.last_selection[self.current_player] = action

        if not self.done:
            self._next_player()

        return self._get_observation(), reward, self.done, False, info

    def render(self):
        # Display the current game state
        number_pool_state = ""
        for idx, status in enumerate(self.number_pool):
            number = idx + 2
            if status == 0:
                number_pool_state += f"{number} "
            elif status == 1:
                number_pool_state += "X "
            elif status == 2:
                number_pool_state += f"P1({number}) "
            elif status == 3:
                number_pool_state += f"P2({number}) "
        player1_primes = sorted(self.player_collections[1])
        player2_primes = sorted(self.player_collections[-1])
        result = f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        result += f"Available Numbers: {number_pool_state}\n"
        result += f"Player 1's Primes: {player1_primes}\n"
        result += f"Player 2's Primes: {player2_primes}\n"
        return result

    def valid_moves(self):
        if self.done:
            return []
        moves = []
        # Check if current player must skip their turn
        if self.skip_turn[self.current_player] == 0:
            # Can select any available number
            available_numbers = np.where(self.number_pool == 0)[0]
            moves.extend(available_numbers.tolist())
            # Can challenge if eligible
            if self._can_challenge():
                moves.append(99)  # Action 99 is challenge
        return moves
