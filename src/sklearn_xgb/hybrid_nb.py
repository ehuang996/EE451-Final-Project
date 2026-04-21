"""
HybridGaussianBernoulliNB: mirrors nb/nb.cpp per-feature likelihood selection.

sklearn has no native mixed-likelihood Naive Bayes. This wrapper composes
GaussianNB (for continuous features) and BernoulliNB (for binary features)
and sums their per-class log-posterior contributions, matching the C++
log_prior + sum_j log P(x_j | c) formulation exactly.

Feature split comes from the binary_mask produced by loader.load_dataset,
which uses the same rules as nb.cpp detect_binary (nb.cpp:298-316):
  - map_* prefix
  - bomb_planted
  - auto-detected 0/1 columns in training data
"""

from __future__ import annotations

import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB


class HybridGaussianBernoulliNB:
    """
    GaussianNB(var_smoothing=1e-9) on continuous features.
    BernoulliNB(alpha=1.0, binarize=0.5) on binary features.
    Log-posterior at predict is log_prior + gaussian_ll + bernoulli_ll - (prior
    term is counted once; we subtract the duplicate class_log_prior_ from one
    component so it isn't added twice).
    """

    def __init__(self, binary_mask: np.ndarray):
        self.binary_mask = np.asarray(binary_mask, dtype=bool)
        self.gaussian = GaussianNB(var_smoothing=1e-9)
        self.bernoulli = BernoulliNB(alpha=1.0, binarize=0.5)
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        cont = X[:, ~self.binary_mask]
        binc = X[:, self.binary_mask]
        if cont.shape[1] > 0:
            self.gaussian.fit(cont, y)
            self.classes_ = self.gaussian.classes_
        if binc.shape[1] > 0:
            self.bernoulli.fit(binc, y)
            if self.classes_ is None:
                self.classes_ = self.bernoulli.classes_
        return self

    def _joint_log_likelihood(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        n_classes = len(self.classes_)
        cont = X[:, ~self.binary_mask]
        binc = X[:, self.binary_mask]

        has_cont = cont.shape[1] > 0
        has_bin = binc.shape[1] > 0

        if has_cont and has_bin:
            # Both estimators carry a class_log_prior_ term. Subtract one copy
            # so the total log-posterior has exactly one prior per class.
            jll_g = self.gaussian._joint_log_likelihood(cont)
            jll_b = self.bernoulli._joint_log_likelihood(binc)
            # BernoulliNB exposes class_log_prior_; GaussianNB stores priors in
            # class_prior_ (linear). Subtract the Bernoulli prior contribution
            # so only the Gaussian prior is counted.
            jll = jll_g + (jll_b - self.bernoulli.class_log_prior_)
        elif has_cont:
            jll = self.gaussian._joint_log_likelihood(cont)
        elif has_bin:
            jll = self.bernoulli._joint_log_likelihood(binc)
        else:
            # No features at all — degenerate. Return zero posteriors.
            jll = np.zeros((n, n_classes))
        return jll

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
