from gzip import GzipFile
import numpy as np
import os
import os.path as op
from sklearn.neighbors import LSHForest, NearestNeighbors
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from time import time
from copy import copy
import pandas as pd
from itertools import product
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

from joblib import Memory

GLOVE_6B_300D_URL = "http://www-nlp.stanford.edu/data/glove.6B.300d.txt.gz"

m = Memory(cachedir='.')


@m.cache
def load_glove_embedding(filepath):
    n_features = None
    vectors = []
    words = []
    with GzipFile(filepath, 'rb') as f:
        i = 0
        for line in f:
            i += 1
            components = line.strip().split()
            words.append(components[0].decode('utf-8'))
            vectors.append(np.array([float(x) for x in components[1:]]))
    print("loaded %d vectors" % i)
    return words, np.vstack(vectors)


@m.cache
def build_index(data, n_estimators=20, n_candidates=100, n_neighbors=10, seed=0):
    lshf = LSHForest(n_estimators=n_estimators, n_candidates=n_candidates,
                     n_neighbors=n_neighbors, random_state=seed)
    t0 = time()
    lshf.fit(data)
    duration = time() - t0
    return lshf, duration


@m.cache
def query_exact(data, query, n_neighbors=10, metric='cosine',
                algorithm='brute'):
    nn = NearestNeighbors(metric=metric, algorithm=algorithm,
                          n_neighbors=n_neighbors)
    t0 = time()
    nn.fit(data)
    build_duration = time() - t0

    t0 = time()
    neighbors = nn.kneighbors(query)
    query_duration = time() - t0
    return neighbors, build_duration, query_duration


@m.cache
def explore_lshf_forest(lshf, queries, exact_nn, n_neighbors=None):
    lshf = copy(lshf)  # shallow copy to modify top level attributes
    all_n_estimators, n_estimators = [], lshf.n_estimators
    while n_estimators > 1:
        all_n_estimators.append(n_estimators)
        n_estimators //= 2

    all_n_candidates, n_candidates = [], lshf.n_candidates
    while n_candidates > 1:
        all_n_candidates.append(n_candidates)
        n_candidates //= 2

    results = []
    iter_grid = product(all_n_estimators, all_n_candidates)
    for n_estimators, n_candidates in iter_grid:
        lshf.n_estimators = n_estimators
        lshf.n_candidates = n_candidates
        durations = []
        precisions = []
        for query in queries:
            t0 = time()
            nn = lshf.kneighbors(query, return_distance=False,
                                 n_neighbors=n_neighbors)
            durations.append(time() - t0)
            precisions.append(np.in1d(nn, exact_nn).mean())

        results.append(dict(
            n_estimators=n_estimators,
            n_candidates=n_candidates,
            query_durations_mean=np.mean(durations),
            query_durations_std=np.std(durations),
            query_precision_mean=np.mean(precisions),
            query_precision_std=np.std(precisions),
        ))
    return pd.DataFrame(results)



if __name__ == '__main__':
    import sys
    n_queries = 10
    n_neighbors = 10
    n_estimators = 100
    n_candidates = 10000
    if len(sys.argv) > 1:
        filepath = os.path.abspath(sys.argv[1])
    else:
        data_folder = op.expanduser('~/data')
        if not op.exists(data_folder):
            os.makedirs(data_folder)
        filename = op.basename(GLOVE_6B_300D_URL)
        filepath = op.join(data_folder, filename)
        if not op.exists(filepath):
            print('Downloading %s' % GLOVE_6B_300D_URL)
            urlretrieve(GLOVE_6B_300D_URL, filepath)
    words, vectors = load_glove_embedding(filepath)
    words = np.array(words, dtype='object')

    vectors_index, vectors_query, words_index, words_query = train_test_split(
        vectors, words, test_size=n_queries, random_state=0)

    # Perform exact knn queries with brute force as a reference
    exact_nn, _, exact_duration = query_exact(
        vectors_index, vectors_query, n_neighbors=n_neighbors)
    print("Performing %d exact queries on data with shape=%r took %0.3fs"
          % (n_queries, vectors_index.shape, exact_duration))

    # Benchmark LSHF model
    lshf, lshf_build_duration = build_index(
        vectors_index, n_estimators=n_estimators, n_candidates=n_candidates)
    print("Building LSHF(n_estimators=%d) on data with shape=%r took %0.3fs"
          % (n_estimators, vectors_index.shape, lshf_build_duration))

    t0 = time()
    lshf_nn = lshf.kneighbors(vectors_query, n_neighbors=n_neighbors)
    lshf_duration = time() - t0
    print("Performing %d LSHF queries on data with shape=%r took %0.3fs"
          % (n_queries, vectors_index.shape, lshf_duration))

    print("LSHF precision: %0.3f" % np.in1d(lshf_nn[1], exact_nn[1]).mean())

    results = explore_lshf_forest(lshf, vectors_query, exact_nn[1],
                                  n_neighbors=n_neighbors)

    # Benchmark LSHF model on normalized data
    # vectors_index_normed = normalize(vectors_index)
    # vectors_query_normed = normalize(vectors_query)
    #
    # lshf_normed, lshf_build_duration = build_index(
    #     vectors_index_normed, n_estimators=n_estimators,
    #     n_candidates=n_candidates)
    # print("Building LSHF(n_estimators=%d) on data with shape=%r took %0.3fs"
    #       % (n_estimators, vectors_index_normed.shape, lshf_build_duration))
    #
    # t0 = time()
    # lshf_normed_nn = lshf_normed.kneighbors(vectors_query_normed,
    #                                         n_neighbors=n_neighbors)
    # lshf_duration = time() - t0
    # print("Performing %d LSHF queries on data with shape=%r took %0.3fs"
    #       % (n_queries, vectors_index_normed.shape, lshf_duration))
    #
    # print("LSHF (normed) precision: %0.3f"
    #       % np.in1d(lshf_normed_nn[1], exact_nn[1]).mean())

    # Benchmark Ball Tree with euclidean distance as cosine is not available
    # bt_nn, bt_build_duration, bt_duration = query_exact(
    #     vectors_index, vectors_query, n_neighbors=n_neighbors,
    #     algorithm='ball_tree', metric='euclidean')
    # print("Build BT index on data with shape=%r took %0.3fs"
    #       % (vectors_index.shape, bt_build_duration))
    # print("Performing %d BT queries on data with shape=%r took %0.3fs"
    #       % (n_queries, vectors_index.shape, bt_duration))
    # print("BT precision: %0.3f" % np.in1d(bt_nn[1], exact_nn[1]).mean())
    #
    # print("LSHF / BT precision: %0.3f" % np.in1d(bt_nn[1], lshf_nn[1]).mean())
