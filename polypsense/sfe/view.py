import numpy as np


class ViewGenerator:
    def sample(self, view1_idx, view2_candidates):
        raise NotImplementedError()


class GaussianTemporalViewGenerator(ViewGenerator):
    def __init__(self, std=30, seed=42):
        self.std = std
        self.rng = np.random.default_rng(seed)

    def sample(self, view1_idx, view2_candidates):
        for _ in range(1000):
            view2_idx = self._sample(view1_idx, view2_candidates)

            if view2_idx != view1_idx:
                break
        else:
            raise ValueError(
                "Could not sample view2 index. Reached maximum number of attempts (1000)."
            )

        return view2_idx

    def _sample(self, view1_idx, view2_candidates):
        # sample from a gaussian centered in the position of `view1_idx` with a
        # std of window_thresh to ensure that 68.2% of samples are within the
        # window
        view1_pos = np.where(view2_candidates == view1_idx)[0][0]
        view2_pos = int(self.rng.normal(view1_pos, self.std))

        # sanitize the sampled position to ensure it is within the bounds of the
        # candidates, and it is not the same as view1_pos
        view2_pos = np.clip(view2_pos, 0, len(view2_candidates) - 1)

        view2_idx = view2_candidates[view2_pos]

        return view2_idx


class UniformTemporalViewGenerator(ViewGenerator):
    def __init__(self, window=None, seed=42):
        self.window = window
        self.rng = np.random.default_rng(seed)

    def sample(self, view1_idx, view2_candidates):
        # self.rng.choice(view2_candidates[view2_candidates != view1_idx])
        # add a window to the uniform sampling
        c = len(view2_candidates)

        # get the position of view1_idx in the candidates list, that is, the
        # center of the window
        view1_pos = np.where(view2_candidates == view1_idx)[0][0]

        # get the normalized window start and end positions
        window_start = max(0, view1_pos - self.window) if self.window else 0
        window_end = min(c, view1_pos + self.window) if self.window else c

        # get the candidate subset
        view2_subset = view2_candidates[window_start:window_end]

        # remove view1_idx itself from the subset
        view2_subset = view2_subset[view2_subset != view1_idx]

        # sample a view2 index from the subset
        view2_idx = self.rng.choice(view2_subset)

        return view2_idx
