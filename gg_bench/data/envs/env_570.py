import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 = Place the drawn card, 1 = Discard the drawn card
        self.action_space = spaces.Discrete(2)

        # Observations:
        # Index 0-4: Player's own stack (numbers 1-5), 1 if placed, 0 otherwise
        # Index 5-9: Opponent's stack
        # Index 10-14: Player's blocked numbers, 1 if blocked, 0 otherwise
        # Index 15-19: Opponent's blocked numbers
        # Index 20: Current drawn card (1-5)
        self.observation_space = spaces.Box(low=0, high=5, shape=(21,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize deck: two copies of each number 1-5
        self.deck = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        self.np_random.shuffle(self.deck)

        # Initialize players' stacks
        self.player_stacks = {1: [], -1: []}

        # Initialize blocked numbers for each player
        self.blocked_numbers = {1: [], -1: []}

        # Current player: 1 or -1
        self.current_player = 1

        # Draw a card for the current player
        self.drawn_card = self._draw_card()

        # Game done flag
        self.done = False

        # Build initial observation
        observation = self._get_observation()

        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        reward = 0

        # Action: 0 = Place, 1 = Discard
        if action not in [0, 1]:
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        player = self.current_player
        opponent = -1 * self.current_player

        if action == 0:
            # Attempt to place the drawn card
            can_place = self._can_place(player, self.drawn_card)
            if can_place:
                # Place the card
                self.player_stacks[player].append(self.drawn_card)

                # Check for win condition
                if self._check_win(player):
                    reward = 1
                    self.done = True

            else:
                # Invalid action
                reward = -10
                self.done = True
                return self._get_observation(), reward, True, False, {}

        elif action == 1:
            # Discard the drawn card
            self.blocked_numbers[opponent].append(self.drawn_card)
            # Remove all instances of the discarded card from the opponent's deck
            self.blocked_numbers[opponent] = list(set(self.blocked_numbers[opponent]))

        # Switch player
        self.current_player = -1 * self.current_player

        # Draw a new card for the new current player
        self.drawn_card = self._draw_card()

        # Check if game is over due to inability to proceed
        if self._is_game_over():
            winner = self._determine_winner()
            if winner == player:
                reward = 1
            elif winner == opponent:
                reward = -1
            else:
                reward = 0  # Tie
            self.done = True

        observation = self._get_observation()

        return (
            observation,
            reward,
            self.done,
            False,
            {},
        )  # Observation, reward, done, truncated, info

    def render(self):
        player = self.current_player
        opponent = -1 * self.current_player
        player_stack = self.player_stacks[player]
        opponent_stack = self.player_stacks[opponent]
        player_blocked = self.blocked_numbers[player]
        opponent_blocked = self.blocked_numbers[opponent]

        render_str = "Current Player: {}\n".format(
            "Player 1" if player == 1 else "Player 2"
        )
        render_str += "Your Stack: {}\n".format(player_stack)
        render_str += "Opponent's Stack: {}\n".format(opponent_stack)
        render_str += "Your Blocked Numbers: {}\n".format(player_blocked)
        render_str += "Opponent's Blocked Numbers: {}\n".format(opponent_blocked)
        render_str += "Drawn Card: {}\n".format(self.drawn_card)
        render_str += "Remaining Cards in Deck: {}\n".format(len(self.deck))
        return render_str

    def valid_moves(self):
        # Returns valid moves [0, 1] based on whether placing is valid
        valid_actions = [1]  # Discard is always valid
        if self._can_place(self.current_player, self.drawn_card):
            valid_actions.append(0)  # Can place
        return valid_actions

    def _draw_card(self):
        if not self.deck:
            # Reshuffle discard pile excluding blocked numbers if deck is empty
            reshuffle_deck = [num for num in range(1, 6) for _ in range(2)]

            # Remove cards in players' stacks and blocked numbers
            used_numbers = (
                self.player_stacks[1]
                + self.player_stacks[-1]
                + self.blocked_numbers[1]
                + self.blocked_numbers[-1]
            )
            for num in used_numbers:
                reshuffle_deck.remove(num)

            # If no cards left to reshuffle, game is over
            if not reshuffle_deck:
                return None

            self.deck = reshuffle_deck
            self.np_random.shuffle(self.deck)

        return self.deck.pop()

    def _can_place(self, player, card):
        if card in self.blocked_numbers[player]:
            return False
        required_next_number = len(self.player_stacks[player]) + 1
        if card == required_next_number:
            return True
        return False

    def _check_win(self, player):
        return len(self.player_stacks[player]) == 5

    def _is_game_over(self):
        # Game is over if both players cannot proceed
        for p in [1, -1]:
            required_next_number = len(self.player_stacks[p]) + 1
            if (
                required_next_number <= 5
                and required_next_number not in self.blocked_numbers[p]
            ):
                # Player p can still proceed
                return False
        return True

    def _determine_winner(self):
        len_player = len(self.player_stacks[self.current_player])
        len_opponent = len(self.player_stacks[-1 * self.current_player])

        if len_player > len_opponent:
            return self.current_player
        elif len_player < len_opponent:
            return -1 * self.current_player
        else:
            return 0  # Tie

    def _get_observation(self):
        # Build observation vector
        observation = np.zeros(21, dtype=np.int32)

        # Player's own stack
        own_stack_numbers = self.player_stacks[self.current_player]
        for num in own_stack_numbers:
            observation[num - 1] = 1  # Indices 0-4

        # Opponent's stack
        opp_stack_numbers = self.player_stacks[-1 * self.current_player]
        for num in opp_stack_numbers:
            observation[5 + num - 1] = 1  # Indices 5-9

        # Player's blocked numbers
        own_blocked_numbers = self.blocked_numbers[self.current_player]
        for num in own_blocked_numbers:
            observation[10 + num - 1] = 1  # Indices 10-14

        # Opponent's blocked numbers
        opp_blocked_numbers = self.blocked_numbers[-1 * self.current_player]
        for num in opp_blocked_numbers:
            observation[15 + num - 1] = 1  # Indices 15-19

        # Drawn card
        if self.drawn_card is not None:
            observation[20] = self.drawn_card
        else:
            observation[20] = (
                0  # No card drawn (deck is empty and no reshuffle possible)
            )

        return observation
