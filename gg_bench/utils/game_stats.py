import json
from dataclasses import asdict, dataclass
from typing import Dict, Union


@dataclass
class GameStats:
    agent_wins: int
    gpt_wins: int
    draws: int

    agent_faults: int
    gpt_faults: int
    env_faults: int

    total_games: int
    avg_turns_per_game: float

    def __init__(self):
        self.agent_wins = 0
        self.gpt_wins = 0
        self.draws = 0
        self.agent_faults = 0
        self.gpt_faults = 0
        self.env_faults = 0
        self.avg_turns_per_game = 0
        self.total_games = 0
        self.total_envs = 1

    def update(self, reward: float, last_move: str, turns: int):
        assert last_move in ["agent", "gpt"], f"Invalid last_move: {last_move}"
        assert reward in [-10, -1, 0, 1], f"Invalid reward: {reward}"

        if last_move == "agent":
            self.agent_wins += reward == 1
            self.gpt_wins += reward == -1
            self.agent_faults += reward == -10
        else:
            self.agent_wins += reward == -1
            self.gpt_wins += reward == 1
            self.gpt_faults += reward == -10

        self.total_games += 1
        self.avg_turns_per_game = (
            self.avg_turns_per_game * (self.total_games - 1) + turns
        ) / self.total_games
        self.draws += reward == 0

    def update_with_other(self, game_stats: "GameStats"):
        self.agent_wins += game_stats.agent_wins
        self.gpt_wins += game_stats.gpt_wins
        self.draws += game_stats.draws
        self.agent_faults += game_stats.agent_faults
        self.gpt_faults += game_stats.gpt_faults
        self.env_faults += game_stats.env_faults
        self.avg_turns_per_game = (
            self.avg_turns_per_game * self.total_games
            + game_stats.avg_turns_per_game * game_stats.total_games
        ) / (self.total_games + game_stats.total_games)
        self.total_games += game_stats.total_games
        self.total_envs += game_stats.total_envs

    def __str__(self):
        return json.dumps(asdict(self), indent=2)

    def save(self, path):
        with open(path, "w") as f:
            f.write(str(self))

    @staticmethod
    def load(path):
        game_stats = GameStats()
        with open(path, "r") as f:
            data = json.load(f)
            for key, value in data.items():
                setattr(game_stats, key, value)
        return game_stats

    @staticmethod
    def from_json(data: Union[Dict, str]):
        game_stats = GameStats()
        if isinstance(data, str):
            data = json.loads(data)
        assert isinstance(data, dict), f"Invalid data type: {type(data)}"
        for key, value in data.items():
            setattr(game_stats, key, value)
        return game_stats
