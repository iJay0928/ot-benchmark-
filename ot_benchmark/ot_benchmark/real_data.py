#!/usr/bin/env python3
"""
真实数据加载与预处理模块
========================
支持从真实世界高维稀疏数据源加载并转换为 SparseDataset 格式：
  - MovieLens 100K / 1M: 用户-电影评分矩阵
  - Reuters-21578: 文档-词频矩阵
  - scRNA-seq 模拟: 单细胞基因表达数据模拟

用于补充纯合成数据集的不足，增强评测的生态效度。
"""

import numpy as np
import os
import urllib.request
import zipfile
import shutil
import gzip
import bz2
import lzma
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import svds
from typing import Optional

try:
    from .data_generator import (
        SparseDataset, compute_ot_reference, generate_cost_matrix, normalize_support_points,
    )
except ImportError:
    from data_generator import (
        SparseDataset, compute_ot_reference, generate_cost_matrix, normalize_support_points,
    )

# 缓存目录
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")

UCI_BOW_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words"
UCI_BOW_DATASETS = {
    "kos": {
        "docword": "docword.kos.txt.gz",
        "description": "KOS blog entries",
        "docs": 3430,
        "words": 6906,
    },
    "nips": {
        "docword": "docword.nips.txt.gz",
        "description": "NIPS full papers",
        "docs": 1500,
        "words": 12419,
    },
    "enron": {
        "docword": "docword.enron.txt.gz",
        "description": "Enron emails",
        "docs": 39861,
        "words": 28102,
    },
    "nytimes": {
        "docword": "docword.nytimes.txt.gz",
        "description": "New York Times articles",
        "docs": 300000,
        "words": 102660,
    },
    "pubmed": {
        "docword": "docword.pubmed.txt.gz",
        "description": "PubMed abstracts",
        "docs": 8200000,
        "words": 141043,
    },
}

LIBSVM_BASE_URL = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
LIBSVM_SPARSE_DATASETS = {
    "colon-cancer": {
        "filename": "colon-cancer.bz2",
        "description": "Colon cancer gene expression",
        "samples": 62,
        "features": 2000,
    },
    "duke": {
        "filename": "duke.bz2",
        "description": "Duke breast cancer gene expression",
        "samples": 44,
        "features": 7129,
    },
    "leu": {
        "filename": "leu.bz2",
        "description": "Leukemia gene expression",
        "samples": 38,
        "features": 7129,
    },
    "rcv1": {
        "filename": "rcv1_train.binary.bz2",
        "description": "RCV1 newswire text classification",
        "samples": 20242,
        "features": 47236,
    },
    "news20": {
        "filename": "news20.binary.bz2",
        "description": "20 Newsgroups binary text classification",
        "samples": 19996,
        "features": 1355191,
    },
    "real-sim": {
        "filename": "real-sim.bz2",
        "description": "Real-sim text classification",
        "samples": 72309,
        "features": 20958,
    },
}


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _download(url: str, filename: str) -> str:
    """下载文件到缓存目录 (带进度条)"""
    _ensure_cache_dir()
    filepath = os.path.join(CACHE_DIR, filename)
    if os.path.exists(filepath):
        print(f"    使用缓存: {filename}")
        return filepath

    print(f"    下载: {filename} ...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"    下载完成: {filename}")
    except (OSError, urllib.error.URLError) as e:
        print(f"    下载失败: {e}")
        raise
    return filepath


def _open_text_auto(filepath: str):
    """Open plain/gzip/bzip2/xz text files with UTF-8 decoding."""
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rt", encoding="utf-8", errors="replace")
    if filepath.endswith(".bz2"):
        return bz2.open(filepath, "rt", encoding="utf-8", errors="replace")
    if filepath.endswith(".xz"):
        return lzma.open(filepath, "rt", encoding="utf-8", errors="replace")
    return open(filepath, "r", encoding="utf-8", errors="replace")


def _expand_dataset_selection(selection: str, available: dict) -> list[str]:
    """Expand comma-separated dataset selection strings."""
    value = (selection or "").strip().lower()
    if value in ("", "none"):
        return []
    if value == "all":
        return list(available.keys())

    names = []
    for part in value.split(","):
        name = part.strip()
        if not name:
            continue
        if name not in available:
            raise ValueError(f"未知数据集: {name}. 可用: {list(available)} 或 all")
        names.append(name)
    return names


def _compute_ot_reference(source: np.ndarray, target: np.ndarray, cost_dense: np.ndarray) -> tuple[float, str, bool, str]:
    """Compute exact or pseudo-reference OT distance for a loaded real dataset."""
    return compute_ot_reference(source, target, cost_dense)


def _learn_feature_geometry(
    sample_feature_matrix,
    latent_dim: int = 16,
    random_state: int = 42,
) -> np.ndarray:
    """Learn feature support coordinates from sparse co-occurrence patterns."""
    matrix = csr_matrix(sample_feature_matrix, dtype=np.float64)
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return np.linspace(0.0, 1.0, matrix.shape[1]).reshape(-1, 1)

    binary = matrix.copy()
    binary.data = np.ones_like(binary.data)
    document_frequency = np.asarray(binary.sum(axis=0)).ravel()
    idf = np.log((1.0 + matrix.shape[0]) / (1.0 + document_frequency)) + 1.0
    profiles = matrix.multiply(idf).T.tocsr()

    k = min(latent_dim, min(profiles.shape) - 1)
    if k < 1:
        return np.column_stack((document_frequency, np.arange(matrix.shape[1])))

    try:
        u, singular_values, _ = svds(
            profiles, k=k, which="LM", random_state=random_state,
        )
        order = np.argsort(singular_values)[::-1]
        embedding = u[:, order] * singular_values[order]
    except (ValueError, np.linalg.LinAlgError):
        embedding = np.column_stack((document_frequency, np.arange(matrix.shape[1])))

    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding / np.maximum(row_norms, 1e-12)


def _compress_to_active_union(
    source: np.ndarray,
    target: np.ndarray,
    support_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize only coordinates carrying mass in either distribution."""
    active = np.flatnonzero((source > 1e-15) | (target > 1e-15))
    if active.size == 0:
        raise ValueError("source and target have no active support")
    return source[active], target[active], support_points[active]


# ============================================================================
#  MovieLens 100K 加载器
# ============================================================================

def load_movielens_100k(
    n_users: Optional[int] = None,
    n_items: Optional[int] = None,
    min_ratings_per_user: int = 5,
    random_state: int = 42,
) -> SparseDataset:
    """
    加载 MovieLens 100K 数据集并转换为 OT 评测格式。

    MovieLens 100K 包含 943 个用户对 1682 部电影的 100,000 条评分。
    稀疏度约 93.7%。

    返回:
        SparseDataset，其中 source_dist 为某用户的评分分布,
        target_dist 为另一用户的评分分布 (随机配对)
    """
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    _ensure_cache_dir()

    # 下载并解压
    zip_path = _download(url, "ml-100k.zip")
    extract_dir = os.path.join(CACHE_DIR, "ml-100k")

    if not os.path.exists(extract_dir):
        print("    解压中...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(CACHE_DIR)
        # 解压后的实际目录名
        actual_dir = os.path.join(CACHE_DIR, "ml-100k")
        if actual_dir != extract_dir:
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)

    # 读取 u.data
    data_path = os.path.join(extract_dir, "u.data")
    if not os.path.exists(data_path):
        data_path = os.path.join(CACHE_DIR, "ml-100k", "u.data")

    print(f"    读取数据: {data_path}")
    data = np.loadtxt(data_path, delimiter="\t")

    user_ids = data[:, 0].astype(int) - 1
    item_ids = data[:, 1].astype(int) - 1
    ratings = data[:, 2].astype(float)

    n_total_users = int(user_ids.max()) + 1
    n_total_items = int(item_ids.max()) + 1

    n_users = n_users or min(500, n_total_users)
    n_items = n_items or min(600, n_total_items)

    # 构建稀疏评分矩阵
    print(f"    构建 {n_users}{n_items} 评分矩阵...")
    mat = lil_matrix((n_users, n_items))
    for u, i, r in zip(user_ids, item_ids, ratings):
        if u < n_users and i < n_items:
            mat[u, i] = r

    # 筛选有足够评分的用户
    row_counts = np.array((mat > 0).sum(axis=1)).flatten()
    valid_users = np.where(row_counts >= min_ratings_per_user)[0]

    if len(valid_users) < 2:
        raise ValueError(f"仅有 {len(valid_users)} 个有效用户 (要求 2)")

    # 随机选择两个用户作为 source / target
    rng = np.random.RandomState(random_state)
    u1, u2 = rng.choice(valid_users, 2, replace=False)

    source_raw = mat[u1].toarray().flatten().astype(np.float64)
    target_raw = mat[u2].toarray().flatten().astype(np.float64)

    # 归一化
    source = source_raw / (source_raw.sum() + 1e-10)
    target = target_raw / (target_raw.sum() + 1e-10)

    sparsity = 1.0 - (np.count_nonzero(source) + np.count_nonzero(target)) / (2 * n_items)
    support_points = np.zeros((n_items, 19), dtype=np.float64)
    item_path = os.path.join(extract_dir, "u.item")
    if os.path.exists(item_path):
        with open(item_path, "r", encoding="latin-1", errors="replace") as item_file:
            for line in item_file:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 24:
                    continue
                item_id = int(parts[0]) - 1
                if 0 <= item_id < n_items:
                    support_points[item_id] = np.asarray(parts[5:24], dtype=np.float64)

    support_points = normalize_support_points(support_points)
    cost_dense, cost_csr = generate_cost_matrix(
        source, target, "euclidean", support_points=support_points,
    )
    gt, ground_truth_method, ground_truth_is_exact, reference_algorithm = (
        _compute_ot_reference(source, target, cost_dense)
    )

    return SparseDataset(
        name=f"MovieLens-100K (u{u1+1}u{u2+1})",
        source_dist=source,
        target_dist=target,
        cost_matrix=cost_dense,
        cost_matrix_csr=cost_csr,
        ground_truth_distance=gt,
        dim=n_items,
        sparsity=sparsity,
        noise_level=0.0,
        description=f"dim={n_items}, sparsity={sparsity:.2f}, "
                     f"users={len(valid_users)}, source=movie_ratings",
        ground_truth_method=ground_truth_method,
        ground_truth_is_exact=ground_truth_is_exact,
        reference_algorithm=reference_algorithm,
        data_origin="online: MovieLens 100K",
        source_url=url,
        support_points=support_points,
        geometry_description="Movie genre vectors (19D)",
        dataset_seed=random_state,
        source_observation_id=f"movielens:user:{u1+1}",
        target_observation_id=f"movielens:user:{u2+1}",
    )


# ============================================================================
#  scRNA-seq 模拟数据生成器
# ============================================================================

def load_simulated_scrnaseq(
    n_cells: int = 1000,
    n_genes: int = 500,
    sparsity: float = 0.88,
    n_cell_types: int = 5,
    random_state: int = 42,
) -> SparseDataset:
    """
    生成模拟的单细胞 RNA 测序数据。

    模拟真实 scRNA-seq 的关键特征:
      - 极高的稀疏度 (通常 85-95% 为零)
      - 基因表达服从负二项分布
      - 不同细胞类型有不同的基因表达模式 (块稀疏结构)

    参数:
        n_cells: 细胞数
        n_genes: 基因数
        sparsity: 目标稀疏度
        n_cell_types: 细胞类型数
        random_state: 随机种子

    返回:
        SparseDataset，source_dist = 某类型细胞的平均表达,
        target_dist = 另一类型细胞的平均表达
    """
    rng = np.random.RandomState(random_state)
    cells_per_type = n_cells // n_cell_types

    # 每种细胞类型有特定的高表达基因集
    gene_sets = []
    genes_per_type = n_genes // n_cell_types
    all_genes = set(range(n_genes))

    for ct in range(n_cell_types):
        # 每种类型有 30-60% 的基因来自其专属集合
        n_specific = int(genes_per_type * (0.3 + 0.3 * rng.random()))
        specific = set(rng.choice(n_genes, n_specific, replace=False))
        gene_sets.append(specific)

    # 构建基因表达矩阵
    mat = lil_matrix((n_cells, n_genes))

    for ct in range(n_cell_types):
        start = ct * cells_per_type
        end = start + cells_per_type if ct < n_cell_types - 1 else n_cells

        for cell in range(start, end):
            # 专属基因高表达
            for g in gene_sets[ct]:
                if rng.random() < (1 - sparsity * 0.7):
                    mat[cell, g] = rng.exponential(2.0) + 0.5

            # 背景基因低表达
            n_background = int(n_genes * (1 - sparsity) * 0.2)
            bg_genes = rng.choice(n_genes, n_background, replace=False)
            for g in bg_genes:
                mat[cell, g] = rng.exponential(0.3) + 0.1

    # 选取两个不同细胞类型的细胞
    cell_a = rng.randint(0, cells_per_type)
    cell_b = rng.randint(cells_per_type, 2 * cells_per_type)

    source_raw = mat[cell_a].toarray().flatten().astype(np.float64)
    target_raw = mat[cell_b].toarray().flatten().astype(np.float64)

    source = source_raw / (source_raw.sum() + 1e-10)
    target = target_raw / (target_raw.sum() + 1e-10)

    actual_sparsity = 1.0 - (source > 1e-10).sum() / n_genes
    support_points = normalize_support_points(
        _learn_feature_geometry(mat.tocsr(), random_state=random_state)
    )
    cost_dense, cost_csr = generate_cost_matrix(
        source, target, "euclidean", support_points=support_points,
    )
    gt, ground_truth_method, ground_truth_is_exact, reference_algorithm = (
        _compute_ot_reference(source, target, cost_dense)
    )

    return SparseDataset(
        name=f"scRNA-seq-Sim (ct{1}ct{2})",
        source_dist=source,
        target_dist=target,
        cost_matrix=cost_dense,
        cost_matrix_csr=cost_csr,
        ground_truth_distance=gt,
        dim=n_genes,
        sparsity=actual_sparsity,
        noise_level=0.05,
        description=f"dim={n_genes}, sparsity={actual_sparsity:.2f}, "
                     f"cells={n_cells}, cell_types={n_cell_types}, source=scRNA_sim",
        ground_truth_method=ground_truth_method,
        ground_truth_is_exact=ground_truth_is_exact,
        reference_algorithm=reference_algorithm,
        data_origin="simulated: scRNA-seq",
        source_url="",
        support_points=support_points,
        geometry_description="Gene co-expression latent geometry",
    )


# ============================================================================
#  UCI Bag of Words 在线真实高维稀疏数据
# ============================================================================

def load_uci_bag_of_words(
    dataset_name: str = "kos",
    n_terms: int = 1000,
    max_docs: int = 2000,
    min_terms_per_doc: int = 20,
    random_state: int = 42,
    dimension_tier: str = "custom",
    geometry_seed: int = 42,
    compress_active: bool = True,
) -> SparseDataset:
    """
    从 UCI Bag of Words 数据集下载真实高维稀疏文本数据。

    数据格式为 doc_id term_id count 的稀疏词袋矩阵。为了控制 OT
    代价矩阵的内存占用，默认在前 max_docs 个文档中选择出现频率最高的
    n_terms 个词作为评测维度，然后随机挑选两个足够稀疏但非零项充足的真实
    文档作为 source / target 分布。

    可用数据集:
      - kos: 约 3,430 篇博客, 6,906 词
      - nips: 约 1,500 篇论文, 12,419 词
      - enron: 约 39,861 封邮件, 28,102 词
    """
    key = dataset_name.lower()
    if key not in UCI_BOW_DATASETS:
        raise ValueError(f"未知 UCI BOW 数据集: {dataset_name}. 可用: {list(UCI_BOW_DATASETS)}")

    meta = UCI_BOW_DATASETS[key]
    filename = meta["docword"]
    url = f"{UCI_BOW_BASE_URL}/{filename}"
    gz_path = _download(url, filename)

    docs: dict[int, dict[int, float]] = {}
    term_freq: dict[int, float] = {}

    print(f"    读取 UCI BOW: {filename} (前 {max_docs} 篇文档, top {n_terms} 词)")
    with _open_text_auto(gz_path) as f:
        try:
            total_docs = int(next(f).strip())
            total_terms = int(next(f).strip())
            _ = int(next(f).strip())  # total non-zero entries
        except (StopIteration, ValueError) as e:
            raise ValueError(f"UCI BOW 文件头解析失败: {filename}") from e

        doc_limit = min(max_docs, total_docs)
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            doc_id, term_id, count = int(parts[0]), int(parts[1]), float(parts[2])
            if doc_id > doc_limit:
                break
            docs.setdefault(doc_id, {})[term_id] = count
            term_freq[term_id] = term_freq.get(term_id, 0.0) + count

    if len(docs) < 2 or not term_freq:
        raise ValueError(f"UCI BOW 有效文档不足: {filename}")

    selected_terms = sorted(
        term_freq.keys(),
        key=lambda t: term_freq[t],
        reverse=True,
    )[:min(n_terms, len(term_freq))]
    term_to_idx = {term_id: idx for idx, term_id in enumerate(selected_terms)}
    dim = len(selected_terms)

    candidates: list[tuple[int, dict[int, float], float]] = []
    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_values: list[float] = []
    for doc_id, counts in docs.items():
        selected: dict[int, float] = {}
        for term_id, count in counts.items():
            idx = term_to_idx.get(term_id)
            if idx is not None:
                selected[idx] = count

        total = float(sum(selected.values()))
        if len(selected) >= min_terms_per_doc and total > 0:
            row = len(candidates)
            candidates.append((doc_id, selected, total))
            for idx, value in selected.items():
                matrix_rows.append(row)
                matrix_cols.append(idx)
                matrix_values.append(value / total)

    if len(candidates) < 2:
        raise ValueError(
            f"UCI BOW 可用文档不足: {filename}, "
            f"min_terms_per_doc={min_terms_per_doc}, candidates={len(candidates)}"
        )

    rng = np.random.RandomState(random_state)
    i, j = rng.choice(len(candidates), 2, replace=False)
    source_doc, source_values, source_total = candidates[i]
    target_doc, target_values, target_total = candidates[j]
    source_full = np.zeros(dim, dtype=np.float64)
    target_full = np.zeros(dim, dtype=np.float64)
    for idx, value in source_values.items():
        source_full[idx] = value / source_total
    for idx, value in target_values.items():
        target_full[idx] = value / target_total

    document_matrix = csr_matrix(
        (matrix_values, (matrix_rows, matrix_cols)),
        shape=(len(candidates), dim),
    )
    full_support_points = normalize_support_points(
        _learn_feature_geometry(document_matrix, random_state=geometry_seed)
    )
    if compress_active:
        source, target, support_points = _compress_to_active_union(
            source_full, target_full, full_support_points,
        )
    else:
        source, target, support_points = source_full, target_full, full_support_points
    cost_dense, cost_csr = generate_cost_matrix(
        source, target, "euclidean", support_points=support_points,
    )
    gt, gt_method, gt_exact, reference_algorithm = _compute_ot_reference(source, target, cost_dense)

    sparsity = float(1.0 - ((source > 1e-10).sum() + (target > 1e-10).sum()) / (2 * dim))

    return SparseDataset(
        name=f"UCI-BOW-{key.upper()}-{dimension_tier.upper()}-{dim}D (d{source_doc}-d{target_doc})",
        source_dist=source,
        target_dist=target,
        cost_matrix=cost_dense,
        cost_matrix_csr=cost_csr,
        ground_truth_distance=gt,
        dim=dim,
        sparsity=sparsity,
        noise_level=0.0,
        description=f"eval_dim={dim}, original_dim={total_terms}, active_union={len(source)}, sparsity={sparsity:.2f}, "
                    f"docs={len(docs)}/{total_docs}, dataset={key}, source=uci_bow",
        ground_truth_method=gt_method,
        ground_truth_is_exact=gt_exact,
        reference_algorithm=reference_algorithm,
        data_origin=f"online: UCI Bag of Words/{key}",
        source_url=url,
        support_points=support_points,
        geometry_description="TF-IDF term co-occurrence latent geometry",
        dataset_seed=random_state,
        source_observation_id=f"uci:{key}:document:{source_doc}",
        target_observation_id=f"uci:{key}:document:{target_doc}",
        original_dim=total_terms,
        dimension_tier=dimension_tier,
    )


# ============================================================================
#  LIBSVM 在线真实高维稀疏数据
# ============================================================================

def load_libsvm_sparse_dataset(
    dataset_name: str = "rcv1",
    n_features: int = 1000,
    max_samples: int = 3000,
    min_features_per_sample: int = 10,
    random_state: int = 42,
    dimension_tier: str = "custom",
) -> SparseDataset:
    """
    从 LIBSVM 官方数据仓库下载高维稀疏真实数据。

    LIBSVM 格式为: label index:value index:value ...。这里把每个样本的
    非负特征值归一化为概率分布，并选择两个真实样本作为 OT 的 source/target。
    对超高维文本数据会先按全局特征频率选取前 n_features 个特征，以控制
    代价矩阵规模。
    """
    key = dataset_name.lower()
    if key not in LIBSVM_SPARSE_DATASETS:
        raise ValueError(f"未知 LIBSVM 数据集: {dataset_name}. 可用: {list(LIBSVM_SPARSE_DATASETS)}")

    meta = LIBSVM_SPARSE_DATASETS[key]
    filename = meta["filename"]
    url = f"{LIBSVM_BASE_URL}/{filename}"
    data_path = _download(url, filename)

    samples: list[tuple[str, dict[int, float]]] = []
    feature_freq: dict[int, float] = {}

    print(f"    读取 LIBSVM: {filename} (前 {max_samples} 个样本, top {n_features} 特征)")
    with _open_text_auto(data_path) as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            label = parts[0]
            values: dict[int, float] = {}
            for item in parts[1:]:
                if ":" not in item:
                    continue
                idx_s, value_s = item.split(":", 1)
                try:
                    idx = int(idx_s)
                    value = abs(float(value_s))
                except ValueError:
                    continue
                if value <= 0:
                    continue
                values[idx] = value
                feature_freq[idx] = feature_freq.get(idx, 0.0) + value

            if values:
                samples.append((label, values))

    if len(samples) < 2 or not feature_freq:
        raise ValueError(f"LIBSVM 有效样本不足: {filename}")

    selected_features = sorted(
        feature_freq.keys(),
        key=lambda feature_id: feature_freq[feature_id],
        reverse=True,
    )[:min(n_features, len(feature_freq))]
    feature_to_idx = {feature_id: idx for idx, feature_id in enumerate(selected_features)}
    dim = len(selected_features)

    candidates: list[tuple[int, str, dict[int, float], float]] = []
    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_values: list[float] = []
    for sample_idx, (label, values) in enumerate(samples, 1):
        selected: dict[int, float] = {}
        for feature_id, value in values.items():
            idx = feature_to_idx.get(feature_id)
            if idx is not None:
                selected[idx] = value

        total = float(sum(selected.values()))
        if len(selected) >= min_features_per_sample and total > 0:
            row = len(candidates)
            candidates.append((sample_idx, label, selected, total))
            for idx, value in selected.items():
                matrix_rows.append(row)
                matrix_cols.append(idx)
                matrix_values.append(value / total)

    if len(candidates) < 2:
        raise ValueError(
            f"LIBSVM 可用样本不足: {filename}, "
            f"min_features_per_sample={min_features_per_sample}, candidates={len(candidates)}"
        )

    rng = np.random.RandomState(random_state)
    by_label: dict[str, list[tuple[int, str, dict[int, float], float]]] = {}
    for candidate in candidates:
        by_label.setdefault(candidate[1], []).append(candidate)

    labels = [label for label, group in by_label.items() if group]
    if len(labels) >= 2:
        label_a, label_b = rng.choice(labels, 2, replace=False)
        source_sample, source_label, source_values, source_total = by_label[label_a][rng.randint(len(by_label[label_a]))]
        target_sample, target_label, target_values, target_total = by_label[label_b][rng.randint(len(by_label[label_b]))]
    else:
        i, j = rng.choice(len(candidates), 2, replace=False)
        source_sample, source_label, source_values, source_total = candidates[i]
        target_sample, target_label, target_values, target_total = candidates[j]

    source_full = np.zeros(dim, dtype=np.float64)
    target_full = np.zeros(dim, dtype=np.float64)
    for idx, value in source_values.items():
        source_full[idx] = value / source_total
    for idx, value in target_values.items():
        target_full[idx] = value / target_total

    sample_matrix = csr_matrix(
        (matrix_values, (matrix_rows, matrix_cols)),
        shape=(len(candidates), dim),
    )
    full_support_points = normalize_support_points(
        _learn_feature_geometry(sample_matrix, random_state=random_state)
    )
    source, target, support_points = _compress_to_active_union(
        source_full, target_full, full_support_points,
    )
    cost_dense, cost_csr = generate_cost_matrix(
        source, target, "euclidean", support_points=support_points,
    )
    gt, gt_method, gt_exact, reference_algorithm = _compute_ot_reference(source, target, cost_dense)

    sparsity = float(1.0 - ((source > 1e-10).sum() + (target > 1e-10).sum()) / (2 * dim))

    return SparseDataset(
        name=f"LIBSVM-{key}-{dimension_tier.upper()}-{dim}D (s{source_sample}-s{target_sample})",
        source_dist=source,
        target_dist=target,
        cost_matrix=cost_dense,
        cost_matrix_csr=cost_csr,
        ground_truth_distance=gt,
        dim=dim,
        sparsity=sparsity,
        noise_level=0.0,
        description=f"eval_dim={dim}, original_dim={meta['features']}, active_union={len(source)}, sparsity={sparsity:.2f}, "
                    f"samples={len(samples)}/{meta['samples']}, labels={source_label}->{target_label}, "
                    f"dataset={key}, source=libsvm",
        ground_truth_method=gt_method,
        ground_truth_is_exact=gt_exact,
        reference_algorithm=reference_algorithm,
        data_origin=f"online: LIBSVM/{key}",
        source_url=url,
        support_points=support_points,
        geometry_description="TF-IDF feature co-occurrence latent geometry",
        original_dim=meta["features"],
        dimension_tier=dimension_tier,
    )


# ============================================================================
#  统一加载接口
# ============================================================================

AVAILABLE_REAL_DATASETS = {
    "movielens": load_movielens_100k,
    "scrnaseq": load_simulated_scrnaseq,
    "uci_bow": load_uci_bag_of_words,
    "libsvm": load_libsvm_sparse_dataset,
}


def load_real_dataset(name: str, **kwargs) -> SparseDataset:
    """
    统一接口：按名称加载真实/模拟数据集。

    参数:
        name: 数据集名称  "movielens", "scrnaseq"
        **kwargs: 传递给具体加载器的参数

    返回:
        SparseDataset
    """
    if name not in AVAILABLE_REAL_DATASETS:
        raise ValueError(
            f"未知数据集: {name}\n可用: {list(AVAILABLE_REAL_DATASETS.keys())}"
        )
    return AVAILABLE_REAL_DATASETS[name](**kwargs)


def create_real_dataset_suite(
    include_online_bow: bool = True,
    bow_dataset: str = "kos,nips,enron",
    bow_terms: int = 1000,
    bow_max_docs: int = 2000,
    include_online_libsvm: bool = True,
    libsvm_datasets: str = "colon-cancer,duke,leu,rcv1,news20,real-sim",
    libsvm_features: int = 1000,
    libsvm_max_samples: int = 3000,
    dimension_tier: str = "custom",
) -> list[SparseDataset]:
    """创建真实数据 + 模拟生物数据的数据集套件"""
    datasets = []

    print("\n 加载真实/模拟数据集...")

    try:
        ds_ml = load_movielens_100k(n_users=300, n_items=400)
        datasets.append(ds_ml)
        print(f"    MovieLens 100K 加载完成")
    except (OSError, ValueError, ImportError, np.linalg.LinAlgError) as e:
        print(f"    MovieLens 加载失败 ({e}), 跳过")

    try:
        ds_sc = load_simulated_scrnaseq(n_cells=800, n_genes=400)
        datasets.append(ds_sc)
        print(f"    scRNA-seq 模拟数据生成完成")
    except (OSError, ValueError, ImportError, np.linalg.LinAlgError) as e:
        print(f"    scRNA-seq 生成失败 ({e}), 跳过")

    if include_online_bow:
        try:
            bow_names = _expand_dataset_selection(bow_dataset, UCI_BOW_DATASETS)
        except ValueError as e:
            print(f"    UCI Bag of Words 选择无效 ({e}), 跳过")
            bow_names = []

        for bow_name in bow_names:
            try:
                ds_bow = load_uci_bag_of_words(
                    dataset_name=bow_name,
                    n_terms=bow_terms,
                    max_docs=bow_max_docs,
                    dimension_tier=dimension_tier,
                )
                datasets.append(ds_bow)
                print(f"    UCI Bag of Words ({bow_name}) 加载完成")
            except (OSError, ValueError, ImportError, np.linalg.LinAlgError) as e:
                print(f"    UCI Bag of Words ({bow_name}) 加载失败 ({e}), 跳过")

    if include_online_libsvm:
        try:
            libsvm_names = _expand_dataset_selection(libsvm_datasets, LIBSVM_SPARSE_DATASETS)
        except ValueError as e:
            print(f"    LIBSVM 选择无效 ({e}), 跳过")
            libsvm_names = []

        for libsvm_name in libsvm_names:
            try:
                ds_libsvm = load_libsvm_sparse_dataset(
                    dataset_name=libsvm_name,
                    n_features=libsvm_features,
                    max_samples=libsvm_max_samples,
                    dimension_tier=dimension_tier,
                )
                datasets.append(ds_libsvm)
                print(f"    LIBSVM ({libsvm_name}) 加载完成")
            except (OSError, ValueError, ImportError, np.linalg.LinAlgError) as e:
                print(f"    LIBSVM ({libsvm_name}) 加载失败 ({e}), 跳过")

    return datasets


if __name__ == "__main__":
    print("=" * 60)
    print("  真实数据加载模块 - 测试")
    print("=" * 60)

    suite = create_real_dataset_suite()
    for ds in suite:
        try:
            from .data_generator import print_dataset_diagnosis
        except ImportError:
            from data_generator import print_dataset_diagnosis
        print_dataset_diagnosis(ds)
