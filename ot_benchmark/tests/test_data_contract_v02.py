"""Scientific data contract regression checks; runnable with unittest or pytest."""
import unittest
import numpy as np
from ot_benchmark.data_generator import (
    create_benchmark_dataset, generate_cost_matrix, generate_structured_sparsity,
)


class DataContractTests(unittest.TestCase):
    def test_lossless_cost_roundtrip(self):
        a = np.ones(20) / 20
        dense, sparse = generate_cost_matrix(a, a)
        np.testing.assert_array_equal(sparse.toarray(), dense)
        self.assertGreater(dense[0, -1], 0.9)

    def test_structure_changes_layout_without_changing_mass_spectrum(self):
        base = create_benchmark_dataset(100, .7, noise_level=.04, random_state=19)
        for structure in ('block', 'group', 'hierarchical'):
            ds = create_benchmark_dataset(100, .7, noise_level=.04,
                                          random_state=19, structure=structure)
            np.testing.assert_array_equal(np.sort(ds.source_dist), np.sort(base.source_dist))
            self.assertFalse(np.array_equal(ds.source_dist, base.source_dist))
            self.assertEqual(ds.sparsity, base.sparsity)

    def test_structured_counts_and_small_sizes(self):
        for n in (1, 7, 50, 201):
            for ratio in (0, .5, .97, 1):
                for structure in ('random', 'block', 'group', 'hierarchical'):
                    x = generate_structured_sparsity(n, ratio, structure)
                    self.assertEqual(np.count_nonzero(x), max(1, round(n * (1 - ratio))))
                    self.assertAlmostEqual(x.sum(), 1)
                    np.testing.assert_array_equal(x, generate_structured_sparsity(n, ratio, structure))

    def test_block_contiguity(self):
        x = generate_structured_sparsity(1000, .9, 'block') > 0
        starts = np.count_nonzero(x & ~np.r_[False, x[:-1]])
        self.assertLessEqual(starts, 10)  # at most two runs per wrapped block

    def test_ratio_semantics_and_dimensions(self):
        a = create_benchmark_dataset(20, .75, geometry_dim=5)
        b = create_benchmark_dataset(20, .25, geometry_dim=5, sparsity_semantics='nonzero_ratio')
        np.testing.assert_array_equal(a.source_dist, b.source_dist)
        self.assertEqual(a.cost_matrix.shape, (20, 20))
        self.assertEqual(a.support_points.shape, (20, 5))
        self.assertEqual(b.sparsity_semantics, 'zero_ratio')


if __name__ == '__main__':
    unittest.main()
