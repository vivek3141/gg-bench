import json

with open("results/analysis/mcts_ranking_results.json") as f:
    no_mcts_results = json.load(f)

found_envs = []

for env_id in no_mcts_results:
    results = no_mcts_results[env_id]
    if not results["results"]:
        continue

    model_indices = [250000, 500000, 750000, 1000000]
    max_wr, max_m1, max_m2 = 0, -1, -1

    for m1_idx in range(len(model_indices)):
        for m2_idx in range(m1_idx + 1, len(model_indices)):
            m1 = model_indices[m1_idx]
            m2 = model_indices[m2_idx]
            wr = results["results"][str(m1)][str(m2)]

            if wr > max_wr:
                max_wr = wr
                max_m1 = m1
                max_m2 = m2
            elif 1 - wr > max_wr:
                max_wr = 1 - wr
                max_m1 = m2
                max_m2 = m1

    if max_wr > 0.8:
        found_envs.append((env_id, max_m1, max_m2, max_wr))

with open("gg_bench/data/splits/valid_envs.json", "w") as f:
    json.dump([(env_id, m2) for env_id, _, m2, _ in found_envs], f)

print(f"Number of environments with >=80% win rate: {len(found_envs)}")
print(f"Avg win rate: {sum(wr for _, _, _, wr in found_envs) / len(found_envs):.5f}")
