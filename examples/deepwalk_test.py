# -*- coding: utf-8 -*-#

# Name:     deepwalk_test
# Author:   Jasper.X
# Date:     2021/4/7
# Description:


import networkx as nx

from ge.models.deepwalk import DeepWalk


def get_graph_from_csv(file_name):
    """
    直接根据[cid1,cid2,weight]格式的csv文件构建
    :return:
    """
    G = nx.read_edgelist(file_name,
                         create_using=nx.DiGraph(),
                         delimiter=',',
                         nodetype=None,
                         data=[('weight', int)])
    print(len(G.nodes()))
    return G


def make_embedding():
    # 读取带权有向图
    G = get_graph_from_csv('../data/c_c_network.csv')

    from time import time
    start = time()

    model = DeepWalk(G, walk_length=10, num_walks=80, workers=8)

    end1time = time()
    model_init_time = end1time - start
    print("Model Init Waste time: {}ms".format(round(model_init_time * 1000, 2)))

    model.train(window_size=5, iter=5)
    model.make_embeddings()

    end2time = time() - start
    print("Model training Waste time: {}ms".format(round(end2time * 1000, 2)))

    model_path = "../data/deepwalk_model.pkl"
    model.save_model(model_path)


if __name__ == '__main__':
    make_embedding()