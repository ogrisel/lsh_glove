from gzip import GzipFile
import numpy as np
import os
import os.path as op
from sklearn.neighbors import LSHForest, NearestNeighbors
from sklearn.cross_validation import train_test_split
from time import time
from urllib.request import urlretrieve

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
            words.append(components[0])
            vectors.append(np.array([float(x) for x in components[1:]]))
    print("loaded %d vectors" % i)
    return words, np.vstack(vectors)


@m.cache
def build_index(data, n_estimators=20, n_candidates=100, n_neighbors=10, seed=0):
    print("Building LSHF(n_estimators=%dn_candidates) on data with shape=%r."
          % (n_estimators, data.shape))
    lshf = LSHForest(n_estimators=n_estimators, n_candidates=n_candidates,
                     n_neighbors=n_neighbors, random_state=seed)
    t0 = time()
    lshf.fit(data)
    duration = time() - t0
    print("Built LSHF(n_estimators=%d) on data with shape=%r in %0.3fs"
          % (n_estimators, data.shape, duration))
    return lshf, duration


@m.cache
def query_exact(data, query, n_neighbors=10):
    nn = NearestNeighbors(metric='cosine', algorithm='brute')
    t0 = time()
    neighbors = nn.fit(data).kneighbors(query, n_neighbors=10)
    duration = time() - t0
    print("Perform %d exact queries on data with shape=%r in %0.3fs"
          % (n_neighbors, data.shape, duration))
    return neighbors, duration


if __name__ == '__main__':
    import sys
    n_queries = 100
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
    lshf_20_100 = build_index(vectors, n_estimators=20, n_candidates=200)
