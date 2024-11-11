import torch

Clusters = list[list[int]]


def associate(preds: torch.Tensor) -> Clusters:
    """
    Return fragments given a boolean matrix indicating whether pairs of samples
    belong to the same group.

    Args:
        preds: [n, n] boolean matrix

    Returns:
        fragments: list of list of int
    """
    n = preds.size(0)

    # Initialize groups
    merged_groups = []
    tracklet_to_group = {}

    preds = {(i, j): preds[i, j].item() for i in range(n) for j in range(n)}

    # Iterate through all tracklet pairs
    for (t1, t2), pred in preds.items():
        if pred:
            # If both tracklets are not in any group, create a new group
            if t1 not in tracklet_to_group and t2 not in tracklet_to_group:
                new_group = {t1, t2}
                merged_groups.append(new_group)
                tracklet_to_group[t1] = new_group
                tracklet_to_group[t2] = new_group
            # If one of the tracklets is already in a group, add the other to the group
            elif t1 in tracklet_to_group and t2 not in tracklet_to_group:
                group = tracklet_to_group[t1]
                group.add(t2)
                tracklet_to_group[t2] = group
            elif t2 in tracklet_to_group and t1 not in tracklet_to_group:
                group = tracklet_to_group[t2]
                group.add(t1)
                tracklet_to_group[t1] = group
            # If both tracklets are in different groups, merge the two groups
            elif (
                t1 in tracklet_to_group
                and t2 in tracklet_to_group
                and tracklet_to_group[t1] != tracklet_to_group[t2]
            ):
                group1 = tracklet_to_group[t1]
                group2 = tracklet_to_group[t2]
                merged_groups.remove(group2)  # Remove the smaller group
                group1.update(group2)  # Merge both groups
                for t in group2:
                    tracklet_to_group[t] = (
                        group1  # Update all tracklets to point to the new merged group
                    )

    # Add tracklets that were not merged into their own groups
    for tracklet in list(range(n)):
        if tracklet not in tracklet_to_group:
            merged_groups.append({tracklet})

    return [list(g) for g in merged_groups]


def get_clustering(clustering_type: str, parameters=None):
    clustering_cls = {
        "threshold": ThresholdClustering,
        "agglomerative": AgglomerativeClustering,
        "dbscan": DbscanClustering,
        "affinity_propagation": AffinityPropagationClustering,
        "temporal": TemporalClustering,
    }[clustering_type]

    return clustering_cls(parameters)


class Clustering:
    def parametrize(self) -> list:
        """
        Return a list of parameters to be tested.

        Returns:
            parameters: list of dict
        """
        pass

    def fit_predict(self, feats: torch.Tensor, parameters: dict) -> torch.Tensor:
        """
        Args:
            feats: [c, n, n] matrix of features with c channels where the first
              one is [n, n] similarity matrix, the second is [n, n] gaps matrix
            parameters: dict of algorithm related parameters

        Returns:
            preds: [n, n] boolean matrix
        """
        pass


class ThresholdClustering(Clustering):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters

    def parametrize(self):
        if self.parameters:
            return [self.parameters]
        thresholds = torch.linspace(0, 1, 1000).tolist()
        return [{"threshold": t} for t in thresholds]

    def fit_predict(self, feats, parameters):
        scores = feats[0]
        thresh = parameters["threshold"]

        clusters = associate(scores >= thresh)

        return _clusters2preds(clusters, n=scores.size(0), device=scores.device)


class AgglomerativeClustering(Clustering):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters

    def parametrize(self):
        if self.parameters:
            return [self.parameters]

        return [
            {
                "n_clusters": None,
                "distance_threshold": d_tresh,
                "linkage": "average",
            }
            for d_tresh in torch.linspace(0, 3, 1000).tolist()
        ]

    def fit_predict(self, feats, parameters):
        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(
            # n_clusters=None, distance_threshold=d_tresh, linkage="average"
            **parameters
        )

        scores = feats[0]

        preds = torch.tensor(clustering.fit_predict(scores))
        preds = preds.unsqueeze(0) == preds.unsqueeze(1)

        return preds


class DbscanClustering(Clustering):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters

    def parametrize(self):
        if self.parameters:
            return [self.parameters]

        return [
            {
                "eps": eps,
                "min_samples": 1,
            }
            for eps in torch.linspace(0.0001, 2, 1000).tolist()
        ]

    def fit_predict(self, feats, parameters):
        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(**parameters)

        scores = feats[0]

        preds = torch.tensor(clustering.fit_predict(scores))
        preds = preds.unsqueeze(0) == preds.unsqueeze(1)

        return preds


class AffinityPropagationClustering(Clustering):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters

    def parametrize(self):
        if self.parameters:
            return [self.parameters]

        return [
            {
                "damping": damping,
                "max_iter": int(max_iter),
                "convergence_iter": 15,
            }
            # damping must be in [0.5, 1), so we remove 1 from the list
            for damping in torch.linspace(0.5, 1, 101).tolist()[:-1]
            for max_iter in torch.linspace(1, 200, 10).tolist()
        ]

    def fit_predict(self, feats, parameters):
        from sklearn.cluster import AffinityPropagation

        clustering = AffinityPropagation(**parameters)

        scores = feats[0]

        preds = torch.tensor(clustering.fit_predict(scores))
        preds = preds.unsqueeze(0) == preds.unsqueeze(1)

        return preds


class TemporalClustering(Clustering):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters

    def parametrize(self):
        if self.parameters:
            return [self.parameters]

        thresholds = torch.linspace(0, 1, 100).tolist()
        temperatures = torch.linspace(0.1, 1, 10).tolist()
        lambdas = torch.linspace(0, 1, 10).tolist()

        return [
            {
                "threshold": thresh,
                "temperature": temp,
                "lambda": lam,
                "fn": "sum_dist_exp",
            }
            for lam in lambdas
            for temp in temperatures
            for thresh in thresholds
        ]
        # + [
        #     {"threshold": thresh, "temperature": temp, "fn": "mult_dist_exp"}
        #     for temp in temperatures
        #     for thresh in thresholds
        # ]

    def fit_predict(self, feats, parameters):
        thresh = parameters["threshold"]
        score_fn = self._score_fn(parameters["fn"])

        scores = feats[0]
        gaps = feats[1]

        scores = score_fn(scores, gaps, **parameters)
        preds = scores >= thresh
        clusters = associate(preds)
        return _clusters2preds(clusters, n=scores.size(0), device=scores.device)

    def _score_fn(self, fn_name):
        if fn_name == "sum_dist_exp":
            return self.sum_dist_exp
        if fn_name == "mult_dist_exp":
            return self.mult_dist_exp
        raise ValueError(f"Unknown score function: {fn_name}")

    def sum_dist_exp(self, scores, gaps, **kwargs):
        t = kwargs["temperature"]
        lam = kwargs["lambda"]
        return lam * scores + (1 - lam) * torch.exp(-gaps / t)

    def mult_dist_exp(self, scores, gaps, **kwargs):
        t = kwargs["temperature"]
        return scores * torch.exp(-gaps / t)


def _clusters2preds(clusters: Clusters, n: int, device: str):
    preds = torch.zeros(n, n)
    for c in clusters:
        for i in c:
            for j in c:
                preds[i, j] = 1
    return preds.bool().to(device)
