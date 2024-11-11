import torch


class SimCLR(torch.nn.Module):
    """
    Adapted from https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/simclr.py
    """

    def __init__(self, n_views=2, temperature=0.07):
        super().__init__()
        self.n_views = n_views
        self.temperature = temperature

    def forward(self, features):
        # features  [v * b, d]
        bs = features.shape[0] // self.n_views
        device = features.device

        labels = torch.cat([torch.arange(bs) for i in range(self.n_views)], dim=0)  # [v * b]
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [v * b, v * b]
        labels = labels.to(device)

        features = torch.nn.functional.normalize(features, dim=1)  # [v * b, d]

        similarity_matrix = torch.matmul(features, features.T)  # [v * b, v * b]
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)  # [v * b, v * b]
        labels = labels[~mask].view(labels.shape[0], -1)  # [v * b, v * b - 1]
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )  # [v * b, v * b - 1]
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [v * b, 1]

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )  # [v * b, v * b - 2], -2 accounts for diag and positive

        logits = torch.cat([positives, negatives], dim=1)  # [v * b, v * b - 1]
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)  # [v * b]

        logits = logits / self.temperature
        return logits, labels
