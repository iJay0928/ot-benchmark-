import numpy as np
import pytest
import ot
from ot_benchmark.data_generator import compute_ot_reference, validate_transport_plan
from ot_benchmark.benchmark_engine import (
    AlgorithmResult, BenchmarkReport, WeightConfig,
    apply_practical_scoring_reference, compute_ranked_scores,
)
from ot_benchmark.algorithms.registry import ALGORITHMS
from ot_benchmark.algorithm_selector import (
    FEATURE_NAMES, SelectorTrainingRecord, train_decision_tree,
    save_training_records, load_training_records,
)


def result(name, distance, group='balanced'):
    return AlgorithmResult(name, distance, 1., 1., 0., 0, True, ranking_group=group)


def test_consensus_cannot_replace_exact_reference():
    report = BenchmarkReport('case', '', 1., ground_truth_is_exact=True,
                             results=[result('correct', 1.), result('wrong-a', 9.), result('wrong-b', 10.)])
    apply_practical_scoring_reference(report)
    assert report.scoring_reference == 1.
    assert report.results[0].distance_error == 0
    assert compute_ranked_scores(report.results, WeightConfig(1, 0, 0, 0))[0][0] == 'correct'


def test_invalid_reference_never_creates_accuracy_ranking():
    report = BenchmarkReport('case', '', 1., ground_truth_method='L1 fallback', results=[result('a', 1.)])
    apply_practical_scoring_reference(report)
    assert np.isnan(report.results[0].distance_error)
    assert compute_ranked_scores(report.results) == []


def test_statistics_path_rejects_invalid_reference(monkeypatch):
    from types import SimpleNamespace
    import ot_benchmark.benchmark_engine as engine
    from ot_benchmark.data_generator import create_benchmark_dataset
    dataset = create_benchmark_dataset(5, .2)
    dataset.ground_truth_distance = 1.
    dataset.ground_truth_is_exact = False
    dataset.ground_truth_method = 'L1 fallback'
    stats = SimpleNamespace(mean_distance=1., mean_runtime=1., mean_memory=1., success_rate=1.)
    monkeypatch.setattr(engine, 'run_with_stats', lambda *args, **kwargs: stats)
    report, _ = engine.benchmark_with_stats(dataset, {'a': {}}, verbose=False)
    assert report.overall_ranking == []
    assert np.isnan(report.results[0].distance_error)


def test_excluded_algorithms_do_not_reenter_empty_ranking():
    assert compute_ranked_scores([result('sliced', 1., 'sliced')]) == []
    for name in ['Sliced OT (200)', 'Sliced OT (500)', 'Low-Rank OT', 'Factored OT']:
        assert ALGORITHMS[name]['ranking_group'] != 'balanced'


def test_bad_plan_rejected():
    with pytest.raises(ValueError, match='residual'):
        validate_transport_plan(np.zeros((2, 2)), np.ones(2)/2, np.ones(2)/2)


def test_greenkhorn_calls_real_solver(monkeypatch):
    from ot_benchmark.algorithms.greenkhorn import run_greenkhorn
    calls = []
    def solver(a, b, *args, **kwargs):
        calls.append(True)
        return np.outer(a, b)
    monkeypatch.setattr(ot.bregman, 'greenkhorn', solver)
    assert run_greenkhorn(np.ones(2), np.ones(2), np.eye(2)) == .5
    assert calls == [True]


def test_sparse_dense_fallback_checks_feasibility(monkeypatch):
    from ot_benchmark.algorithms.sparse_ot import run_sparse_sinkhorn
    monkeypatch.setattr(ot, 'sinkhorn', lambda *a, **kw: np.zeros((2, 2)))
    with pytest.raises(ValueError, match='residual'):
        run_sparse_sinkhorn(np.ones(2), np.ones(2), np.eye(2), None)


def test_nonconverged_reference_is_unavailable(monkeypatch):
    monkeypatch.setattr(ot, 'sinkhorn', lambda *a, **kw: np.zeros((2, 2)))
    value, method, exact, algorithm = compute_ot_reference(
        np.ones(2)/2, np.ones(2)/2, np.eye(2), allow_exact=False)
    assert np.isnan(value) and not exact and not algorithm
    assert 'unavailable' in method


def test_invalid_inputs_do_not_produce_reference():
    value, _, exact, _ = compute_ot_reference(np.zeros(2), np.ones(2), np.eye(2))
    assert np.isnan(value) and not exact


def test_selector_split_is_group_disjoint_and_roundtrips(tmp_path):
    records = []
    for i in range(12):
        for profile in range(2):
            scores = {'a': .9 if i < 6 else .4, 'b': .5 if i < 6 else .95}
            winner = max(scores, key=scores.get)
            features = [float(i)] * len(FEATURE_NAMES)
            records.append(SelectorTrainingRecord(f'{i} [{profile}]', features, winner,
                           scores[winner], dataset_group=str(i), algorithm_scores=scores))
    path = save_training_records(tmp_path/'records.csv', records)
    loaded = load_training_records(path)
    _, evaluation = train_decision_tree(loaded, min_samples_leaf=1)
    assert set(evaluation.train_groups).isdisjoint(evaluation.test_groups)
    assert len(evaluation.train_groups) + len(evaluation.test_groups) == 12
    assert evaluation.selector_mean_score_regret is not None
    assert evaluation.fixed_baseline in ('a', 'b')
