import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ot_benchmark"))

from algorithm_selector import (
    FEATURE_NAMES,
    SelectorTrainingRecord,
    extract_lightweight_features,
    extract_selector_features,
    predict_algorithm,
    train_decision_tree,
)
from data_generator import create_benchmark_dataset


def test_feature_extraction_is_finite():
    dataset = create_benchmark_dataset(
        n_samples=100, sparsity=0.7, noise_level=0.03,
        source_dist_type="zipf", target_dist_type="pareto", random_state=5,
    )
    features = extract_selector_features(dataset)
    assert list(features) == FEATURE_NAMES
    assert np.isfinite(list(features.values())).all()


def test_tree_training_and_prediction():
    records = []
    for index in range(12):
        features = [0.0] * len(FEATURE_NAMES)
        features[0] = 2.0 + index / 10
        label = "Fast OT" if index < 6 else "Large OT"
        records.append(SelectorTrainingRecord(str(index), features, label, 0.9))
    model, evaluation = train_decision_tree(records, max_depth=2, min_samples_leaf=1)
    assert evaluation.accuracy >= 0.5
    assert set(model.classes_) == {"Fast OT", "Large OT"}


def test_high_dimensional_feature_extraction_does_not_build_cost_matrix():
    features = extract_lightweight_features(
        dimension=50_000,
        sparsity=0.98,
        noise_level=0.02,
        source_dist_type="zipf",
        target_dist_type="pareto",
        random_state=9,
    )
    assert list(features) == FEATURE_NAMES
    assert np.isfinite(list(features.values())).all()
    assert features["log_dimension"] == np.log10(50_000)
