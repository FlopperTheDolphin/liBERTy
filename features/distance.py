import os
# import tensorflow as tf
from numpy import linalg as LA
import numpy as np
# import tensorflow as tf
from sklearn.cluster import KMeans
from features.utiliy import get_sentence, get_bert_tokens
from fun.loader import load_matrix, save_in_json, load_from_json
from fun.comp_att import get_all_matrix
from fun.view import view_dist, view_matrix


def distance(name, out_dir):
    sentence = get_sentence(out_dir, name)
    dic_head, l_index = get_all_matrix(out_dir, name)
    dist = dict()
    l = list()
    for i in range(len(l_index), 0, -1):
        for j in range(i):
            if i != j:
                l.append(i + j)
                dist[l_index[i] + '/' + l_index[j]] = diff_norm(dic_head[l_index[i]].to_numpy(),
                                                                dic_head[l_index[j]].to_numpy())

    print(len(l))
    view_dist(dist, l_index)


# def diff_norm(mtx1,mtx2):
# return abs(LA.norm(mtx1) - LA.norm(mtx2))

def diff_norm(mtx1, mtx2):
    return LA.norm(abs(mtx1 - mtx2), ord='fro')


def cluster(name, out_dir, model_dir, c):
    dist_path = os.path.join(os.path.join(out_dir, name), "distance.json")

    if not os.path.exists(dist_path):

        dic_head, l_index = get_all_matrix(out_dir, name)
        sentence = get_sentence(out_dir, name)
        l_head = dic_head.items()
        X = list()
        bert_tokens = get_bert_tokens(os.path.join(out_dir, name), model_dir, sentence)

        A = np.zeros((144, 144))

        for i in range(len(l_index)):
            l1 = l_index[i]
            v = np.zeros(len(l_index))
            for j in range(len(l_index)):
                l2 = l_index[j]
                v.flat[j] = diff_norm(dic_head[l1], dic_head[l2])

            X.append(v.tolist())
            A[i, :] = v

        print('Finito il calcolo')
        save_in_json(X, dist_path)
    else:
        A = np.zeros((144, 144))
        X = load_from_json(dist_path)
        for i in range(len(X)):
            A[i, :] = X[i]
        l_index = list()
        for i in range(12):
            for j in range(12):
                l_index.append((i + 1, j + 1))

    kmeans = KMeans(n_clusters=int(c), random_state=42).fit(X)

    print(kmeans.labels_)

    B = np.zeros((12, 12))

    for i in range(len(l_index)):
        layer = l_index[i][0]
        head = l_index[i][1]
        B[layer - 1, head - 1] = kmeans.labels_[i]

    view_matrix(B)
