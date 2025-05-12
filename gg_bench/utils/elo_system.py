from collections import defaultdict
from typing import Dict, Union


class EloSystem:
    """
    A class to implement the Elo rating system for calculating relative skill levels of players.
    """

    def __init__(self, initial_rating: int = 1500) -> None:
        """
        Initialize the EloSystem with a default initial rating.

        Args:
            initial_rating (int): The starting rating for new players. Defaults to 1500.
        """
        self.initial_rating: int = initial_rating
        self.ratings: Dict[str, float] = defaultdict(lambda: float(self.initial_rating))

    def update_ratings(
        self, player1: str, player2: str, result: Union[int, float]
    ) -> None:
        """
        Update the Elo ratings of two players based on a game result.

        Args:
            player1 (str): Identifier for the first player.
            player2 (str): Identifier for the second player.
            result (Union[int, float]): The game result. 1 for player1 win, 0 for player2 win, 0.5 for draw.

        Raises:
            ValueError: If the result is not 1, 0, or 0.5.
        """
        if result not in {1, 0, 0.5}:
            raise ValueError("Result must be 1, 0, or 0.5")

        r1, r2 = self.ratings[player1], self.ratings[player2]
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 - e1
        k = 32.0

        s1 = result
        s2 = 1 - result

        self.ratings[player1] += k * (s1 - e1)
        self.ratings[player2] += k * (s2 - e2)

    def get_rating(self, player: str) -> float:
        """
        Get the current rating of a player.

        Args:
            player (str): Identifier for the player.

        Returns:
            float: The current Elo rating of the player.
        """
        return self.ratings[player]

    def reset_ratings(self) -> None:
        """
        Reset all player ratings to the initial rating.
        """
        self.ratings.clear()
