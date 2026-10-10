import gzip
import numpy as np
from ot_benchmark import real_data


def test_runner_forwards_dataset_seed(monkeypatch):
    import run_benchmark_v1 as runner
    import real_data as legacy_real
    from types import SimpleNamespace
    observed = {}
    def load(name, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(legacy_real, 'load_real_dataset', load)
    runner.generate_dataset_from_scenario(dict(id='real', name='real', n_source=20,
        sparsity=.9, source_type='real', real_dataset='movielens', real_params={'random_state':42}), 123)
    assert observed['random_state'] == 123


def test_residual_failure_is_numerical():
    import run_benchmark_v1 as runner
    from types import SimpleNamespace
    result = SimpleNamespace(success=False, not_applicable=False, error_message='transport marginal residual 0.1 exceeds 1e-7')
    assert runner.classify_status(None, result) == 'numerical_error'


def test_movielens_seed_and_identity(tmp_path, monkeypatch):
    root = tmp_path/'ml-100k'
    root.mkdir()
    rows = [[u, i, 1 + (u * i) % 5, 0] for u in range(1, 21) for i in range(1, 7)]
    np.savetxt(root/'u.data', rows, fmt='%d', delimiter='\t')
    (root/'u.item').write_text('\n'.join('|'.join([str(i), 'title', '', '', 'url'] +
                                     [str(int(j == i % 19)) for j in range(19)])
                                     for i in range(1, 7)), encoding='latin-1')
    monkeypatch.setattr(real_data, 'CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(real_data, '_download', lambda *args: 'unused.zip')
    a = real_data.load_movielens_100k(20, 6, random_state=10)
    b = real_data.load_movielens_100k(20, 6, random_state=10)
    c = real_data.load_movielens_100k(20, 6, random_state=11)
    assert (a.source_observation_id, a.target_observation_id) == (b.source_observation_id, b.target_observation_id)
    assert (a.source_observation_id, a.target_observation_id) != (c.source_observation_id, c.target_observation_id)
    np.testing.assert_array_equal(a.source_dist, b.source_dist)


def test_bow_sampling_does_not_change_full_geometry(tmp_path, monkeypatch):
    path = tmp_path/'docword.kos.txt.gz'
    with gzip.open(path, 'wt') as handle:
        handle.write('12\n8\n96\n')
        for doc in range(1, 13):
            for term in range(1, 9):
                handle.write(f'{doc} {term} {1+(doc*term)%7}\n')
    monkeypatch.setattr(real_data, '_download', lambda *args: str(path))
    a = real_data.load_uci_bag_of_words(n_terms=8, max_docs=12, min_terms_per_doc=3,
                                      random_state=10, compress_active=False)
    b = real_data.load_uci_bag_of_words(n_terms=8, max_docs=12, min_terms_per_doc=3,
                                      random_state=11, compress_active=False)
    assert (a.source_observation_id, a.target_observation_id) != (b.source_observation_id, b.target_observation_id)
    np.testing.assert_allclose(a.cost_matrix, b.cost_matrix, atol=1e-12)
    assert a.dataset_seed == 10 and b.dataset_seed == 11
