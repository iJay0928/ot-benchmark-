import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ot_benchmark"))

from downstream_knn import generate_domain_shift_dataset, run_knn_downstream_validation


def test_knn_downstream_validation_returns_baseline_and_ot_results():
    dataset = generate_domain_shift_dataset(
        n_samples=60, n_features=4, n_classes=3, shift_intensity=0.6,
        noise=0.25, random_state=3,
    )
    results = run_knn_downstream_validation(
        dataset, algorithms=["Sinkhorn (reg=0.1)", "Low-Rank OT"], neighbors=3
    )
    assert [result.algorithm for result in results] == [
        "No OT baseline", "Sinkhorn (reg=0.1)", "Low-Rank OT"
    ]
    assert all(result.success for result in results)
    assert all(0.0 <= result.accuracy <= 1.0 for result in results)
