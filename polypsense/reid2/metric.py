# Credits: https://github.com/SerezD/information_retrieval_scores/blob/main/scores.py

import torch


def get_cmc_k_score(labels: torch.IntTensor, predictions: torch.IntTensor, k: int):
    """
    :param labels: (Q) query id labels, where Q is the length of query set
    :param predictions: (Q, G) of id labels, ranked from most likely.
                        Q = length of query set, G = length of gallery set
    :param k: int. if k > G --> k = G
    :return: cmc@k

    Example:
    labels = [1, 4]
    predictions = [[2, 4, 1, 5, 3],[1, 3, 2, 5, 4]]

    Q = 2, G = 5

    cmc@1 = 0.0
    cmc@2 = 0.0
    cmc@3 = 0.5
    cmc@4 = 0.5
    cmc@5 = 1.0
    """

    assert k > 0, f"k: {k} must be an integer greater than 0"

    labels = labels.type(torch.int64)
    predictions = predictions.type(torch.int64)

    q, g = predictions.shape
    k = k if k < g else g

    # q, k
    predictions = predictions[:, :k]

    # scores (1 if label in pred_k, 0 otherwise)
    scores = (torch.sum(predictions == labels.unsqueeze(1), dim=1) > 0).type(
        torch.float32
    )
    cmc_k = torch.mean(scores)
    return cmc_k


def get_map_score(
    labels: torch.IntTensor, predictions: torch.IntTensor, g_labels: torch.IntTensor
):
    """
    :param labels: (Q) query id labels, where Q is the length of query set
    :param predictions: (Q, G) of id labels, ranked from most likely.
                        Q = length of query set, G = length of gallery set
    :param g_labels: (G) gallery id labels, where G is the length of gallery set
    :return: mAP score

    Example:
    labels = [1, 4]
    g_labels = [1, 2, 3, 4, 5]
    predictions = [[2, 4, 1, 5, 3],[1, 3, 2, 5, 4]]

    Q = 2, G = 5

    AP_1 = (0 + 0 + 1/3 + 0 + 0) / 1 = 1/3
    AP_2 = (0 + 0 + 0 + 0 + 1/5) / 1 = 1/5
    mAP = (1/3 + 1/5) / 2 = 0.26667
    """

    # convert all to int 64
    labels = labels.type(torch.int64)
    g_labels = g_labels.type(torch.int64)
    predictions = predictions.type(torch.int64)

    g = g_labels.shape[0]
    device = labels.device

    # relevant docs in gallery for each label
    # (q)
    n_gtp = torch.sum((labels.unsqueeze(1) == g_labels), dim=1)

    # boolean mask on predictions
    # (q, g)
    mask = labels.unsqueeze(1) == predictions

    # number of retrieved documents at each position
    # (q, g)
    retrieved = torch.cumsum(mask, dim=1)

    # denominators (index if found retrieved, 0 otherwise)
    denominators = torch.arange(1, g + 1, device=device)
    denominators = torch.mul(mask, denominators)

    # precisions and average_precisions
    precisions = torch.nan_to_num(retrieved / denominators, nan=0, posinf=0, neginf=0)
    avg_precisions = torch.sum(precisions, dim=1)
    avg_precisions = avg_precisions / n_gtp

    mean_ap = torch.mean(avg_precisions)
    return mean_ap
