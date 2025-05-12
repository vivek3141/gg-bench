import json
import openai
import tqdm
from typing import List, Optional, Dict, Union
from gg_bench.utils.chat_completion import chat_completion, UsageTracker

try:
    import pulp
except ImportError:
    pulp = None


# ---------------------------------------------------------------------
# PASClusterer: Implements the GOALEX PAS (Propose-Assign-Select) algorithm.
# ---------------------------------------------------------------------
class PASClusterer:
    def __init__(
        self,
        proposer_model: str = "gpt-4o-mini",
        assigner_model: str = "gpt-4o-mini",
        candidate_count: Optional[int] = None,
        use_ilp: bool = True,
        verbose: bool = False,
    ):
        """
        :param proposer_model: OpenAI model name for proposing candidate explanations.
        :param assigner_model: OpenAI model name for assigning texts to explanations.
        :param candidate_count: Number of candidate explanations to generate (if None, defaults to 10 or 2*K if K is provided).
        :param use_ilp: Whether to use ILP to select the final clusters (if False, uses a simple heuristic).
        :param verbose: Whether to print verbose output.
        """
        self.proposer_model = proposer_model
        self.assigner_model = assigner_model
        self.candidate_count = candidate_count
        self.use_ilp = use_ilp
        self.verbose = verbose
        self.usage_tracker = UsageTracker()

    def _call_openai(self, model: str, prompt: str) -> str:
        """Calls OpenAI's API and returns the response text."""
        if self.verbose:
            print(f"[{model}] Prompt (len={len(prompt)}): {prompt[:200]}...")

        result = chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            usage_tracker=self.usage_tracker,
        )
        if self.verbose:
            print(f"[{model}] Response: {result[:200]}...")
        return result

    def propose_explanations(
        self, texts: List[str], goal: str, K: Optional[int] = None
    ) -> List[str]:
        """
        Uses the proposer LLM to generate candidate cluster explanations.
        :param texts: The list of texts.
        :param goal: The clustering goal.
        :param K: If provided, we may propose more than K candidates.
        :return: A list of candidate explanation strings.
        """
        # determine number of candidate explanations
        if self.candidate_count:
            num_candidates = self.candidate_count
        else:
            num_candidates = 10 if K is None else max(10, 2 * K)
        # using a small subset of texts to include as examples (to control prompt length)
        subset = texts[:64]
        examples = "\n".join(f"{i+1}. {t}" for i, t in enumerate(subset))
        prompt = (
            f"Below are a few examples of game descriptions:\n{examples}\n\n"
            f"Goal: {goal}\n\n"
            f"Please brainstorm a list of {num_candidates} candidate explanations for clustering these texts. "
            "I envision the following examples as valid themes: Card Game, Board Game, Word Game, Abstract Strategy Game\n"
            "Return the list as numbered items."
        )
        raw_output = self._call_openai(self.proposer_model, prompt)
        # parsing the output: split by lines, remove numbering and extra punctuation.
        explanations = []
        for line in raw_output.splitlines():
            line = line.strip(" -0123456789.()")
            if line:
                explanations.append(line.strip())
        if self.verbose:
            print(f"Proposer generated {len(explanations)} candidate explanations.")
        # Save the explanations into JSON
        with open("explanations.json", "w", encoding="utf-8") as out_f:
            json.dump(clusters_to_save, out_f, indent=2, ensure_ascii=False)

        print(f"Usage Tracker:\n{self.usage_tracker}")
        return explanations

    def assign_texts(
        self, texts: List[str], explanations: List[str], goal: str
    ) -> List[List[int]]:
        """
        For each candidate explanation, determines for each text whether it belongs.
        Returns an assignment matrix (list of lists) of size [num_texts x num_explanations].
        """
        assignments = [[0 for _ in explanations] for _ in texts]
        for j, expl in enumerate(
            tqdm.tqdm(explanations, desc="Assigning candidate explanations")
        ):
            if self.verbose:
                print(f"Assigning texts for candidate explanation {j}: {expl}")
            for i, text in enumerate(
                tqdm.tqdm(
                    texts, desc=f"Processing texts for explanation {j}", leave=False
                )
            ):
                prompt = (
                    f"Cluster Explanation: {expl}\n"
                    f"Text: {text}\n"
                    f"Question: Does the text belong to the cluster described above? "
                    "Answer with 'Yes' or 'No'."
                )
                answer = self._call_openai(self.assigner_model, prompt)
                # Consider a simple check: if answer starts with 'yes'
                if answer.strip().lower().startswith("yes"):
                    assignments[i][j] = 1
                else:
                    assignments[i][j] = 0
        # Save the assignments
        try:
            with open("assignment_matrix.json", "w", encoding="utf-8") as out_f:
                json.dump(clusters_to_save, out_f, indent=2, ensure_ascii=False)
        except:
            print("Error saving the assignment matrix")

        print(f"Usage Tracker:\n{self.usage_tracker}")
        return assignments

    def select_clusters(
        self, assignments: List[List[int]], K: Optional[int] = None
    ) -> List[int]:
        """
        Selects the optimal subset of candidate clusters.
        If K is provided, exactly K clusters are selected.
        Otherwise, an ILP formulation with a penalty term is used to choose a subset automatically.
        Returns a list of indices of selected candidate explanations.
        """
        num_texts = len(assignments)
        num_candidates = len(assignments[0]) if num_texts > 0 else 0
        if num_candidates == 0:
            return []
        if not self.use_ilp or pulp is None:
            if self.verbose:
                print(
                    "ILP not enabled or PuLP not installed; using heuristic selection."
                )
            if K is None:
                return list(range(num_candidates))
            else:
                return list(range(min(num_candidates, K)))
        # Set up ILP using PuLP
        problem = pulp.LpProblem("ClusterSelection", pulp.LpMinimize)
        # Binary variable s_j: 1 if candidate j is selected
        s_vars = [
            pulp.LpVariable(f"s_{j}", cat="Binary") for j in range(num_candidates)
        ]
        # For each text, let m_i be the number of selected clusters covering it.
        m_vars = [
            pulp.LpVariable(f"m_{i}", lowBound=0, cat="Integer")
            for i in range(num_texts)
        ]
        # For each text, we enforce: m_i == sum_j (assignments[i][j] * s_j)
        for i in range(num_texts):
            problem += (
                m_vars[i]
                == pulp.lpSum(
                    assignments[i][j] * s_vars[j] for j in range(num_candidates)
                ),
                f"Cover_{i}",
            )
            # Enforce non-overlap: ideally, each text should be covered at most once.
            problem += m_vars[i] <= 1, f"NonOverlap_{i}"
        # If K is predefined, add the constraint sum_j s_j = K
        if K is not None:
            problem += (
                pulp.lpSum(s_vars[j] for j in range(num_candidates)) == K,
                "NumClustersConstraint",
            )
        # Objective: minimize uncovered texts.
        # Each text that is not covered contributes a penalty of 1.
        # Since m_i is either 0 or 1 under our constraints, sum(1 - m_i) is the number of uncovered texts.
        objective = pulp.lpSum(1 - m_vars[i] for i in range(num_texts))
        # Additionally, if K is not defined, add a penalty per selected cluster to discourage too many clusters.
        if K is None:
            alpha = (
                0.5  # hyperparameter to control cluster count penalty; adjust as needed
            )
            objective += alpha * pulp.lpSum(s_vars[j] for j in range(num_candidates))
        problem.setObjective(objective)
        if self.verbose:
            print("Solving ILP for cluster selection...")
        problem.solve(pulp.PULP_CBC_CMD(msg=False))
        selected = [j for j in range(num_candidates) if pulp.value(s_vars[j]) == 1]
        if self.verbose:
            print(f"ILP selected {len(selected)} clusters.")
        return selected

    def cluster_texts(
        self, texts: List[str], goal: str, K: Optional[int] = None
    ) -> Union[List[Dict], "pandas.DataFrame"]:
        """
        Runs the full PAS pipeline:
          1. Propose candidate cluster explanations.
          2. Assign each text to each candidate explanation.
          3. Select an optimal subset of candidate clusters (using ILP if desired).
        :param texts: List of game description texts.
        :param goal: The goal string for clustering.
        :param K: Optional desired number of clusters.
        :return: Final clusters, each a dict with 'explanation' and 'texts'.
                 Also returned as a pandas DataFrame if pandas is installed.
        """
        # 1. Propose
        propose_bool = False
        if propose_bool:
            candidate_explanations = self.propose_explanations(texts, goal, K)
            if self.verbose:
                print("Candidate Explanations:", candidate_explanations)
        # 2. Assign
        # Read from explanations.json
        candidate_explanations_file = "explanations.json"
        with open(candidate_explanations_file, "r", encoding="utf-8") as f:
            candidate_explanations = json.load(f)
        assignment_matrix = self.assign_texts(texts, candidate_explanations, goal)
        # 3. Select
        selected_indices = self.select_clusters(assignment_matrix, K)
        # Build the final clusters.
        clusters = []
        for j in selected_indices:
            cluster_texts = [
                texts[i] for i in range(len(texts)) if assignment_matrix[i][j] == 1
            ]
            clusters.append(
                {"explanation": candidate_explanations[j], "texts": cluster_texts}
            )
        # Optionally return as a pandas DataFrame.
        try:
            import pandas as pd

            df = pd.DataFrame(
                [
                    {
                        "Cluster Explanation": c["explanation"],
                        "Num Texts": len(c["texts"]),
                        "Texts": c["texts"],
                    }
                    for c in clusters
                ]
            )
            return df
        except ImportError:
            return clusters


# ---------------------------------------------------------------------
# Main function: Read compiled JSON and perform clustering.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    input_json_path = "gg_bench/scripts/experiments/categories/compiled_descriptions.json"

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    goal = data.get(
        "goal",
        "Cluster the texts based on topic; each cluster should have a description of 'has a topic of <something>'",
    )
    texts = data.get("texts", [])

    # choose among: "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "o3-mini", etc.
    pas_clusterer = PASClusterer(
        proposer_model="o1",
        assigner_model="o3-mini",
        candidate_count=10,  # or None to let it default
        use_ilp=True,
        verbose=True,
    )

    # Option: set K to a fixed number of clusters, e.g. K=3. Otherwise, pass K=None for automatic selection.
    K = None

    final_clusters = pas_clusterer.cluster_texts(texts, goal, K)

    # Print or save the final clusters.
    try:
        import pandas as pd

        print("Final Clusters (as DataFrame):")
        print(final_clusters)
    except ImportError:
        print("Final Clusters:")
        print(final_clusters)

    # save the final clusters to JSON.
    output_json_path = "final_clusters.json"
    if isinstance(final_clusters, list):
        clusters_to_save = final_clusters
    else:
        clusters_to_save = final_clusters.to_dict(orient="records")
    with open(output_json_path, "w", encoding="utf-8") as out_f:
        json.dump(clusters_to_save, out_f, indent=2, ensure_ascii=False)
    print(f"Clusters saved to {output_json_path}")
