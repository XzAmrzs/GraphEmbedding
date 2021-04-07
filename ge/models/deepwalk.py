# -*- coding:utf-8 -*-

"""



Author:

    Weichen Shen,wcshen1994@163.com



Reference:

    [1] Perozzi B, Al-Rfou R, Skiena S. Deepwalk: Online learning of social representations[C]//Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2014: 701-710.(http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)



"""
import pickle

from ..walker import RandomWalker
from gensim.models import Word2Vec
import pandas as pd


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):
        """

        :param graph:  networkx中的graph对象
        :param walk_length: 每次随机游走的步长
        :param num_walks: 每个worker走的次数
        :param workers: worker的个数
        """
        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}

        self.walker = RandomWalker(graph, p=1, q=1, )
        self.sentences = self.walker.simulate_walks_multi_process(num_walks=num_walks, walk_length=walk_length,
                                                                  num_workers=workers)

        # self.sentences = self.walker.simulate_walks(
        #     num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")
        self.w2v_model = model

        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings

    def make_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            try:
                self._embeddings[word] = self.w2v_model.wv[word]
            except Exception as e:
                print(e)

    def save_embeddings(self, file_names):
        """save embedding to file"""
        if self._embeddings:
            print(len(self._embeddings))
            with open(file_names, 'wb') as f:
                pickle.dump(self._embeddings, f)
        else:
            print("you must get embeddings first")
            print("try get_embeddings function first")
            return None

    def save_model(self, file_names):
        """save model to file"""
        with open(file_names, 'wb') as f:
            pickle.dump(self, f)
            print("save model success")

    def save_embedding2redis(self, redis_conn):
        import json

        # 存储每个item对应的向量
        for item, vector in self._embeddings.items():
            try:
                # course_uuid = cid_lbe.inverse_transform(np.array([int(item)]))[0]
                course_uuid = item
                course_vector = vector.tolist()
                redis_conn.set('edu:c:embed:{0}'.format(course_uuid), course_vector)
            except Exception as e:
                print(e)

        # 存储向量矩阵
        redis_conn.set("edu:c:embed", json.dumps(self.w2v_model.wv.vectors.tolist()))
        redis_conn.set("edu:c:mapping", json.dumps(self.w2v_model.wv.index2word))

    def save_embedding2redis_group(self, redis_conn):
        import json

        # 存储每个item对应的向量
        for item, vector in self._embeddings.items():
            try:
                # course_uuid = cid_lbe.inverse_transform(np.array([int(item)]))[0]
                course_uuid = item
                course_vector = vector.tolist()
                redis_conn.set('edu:cg:embed:{0}'.format(course_uuid), course_vector)
            except Exception as e:
                print(e)

        # 存储向量矩阵
        redis_conn.set("edu:cg:embed", json.dumps(self.w2v_model.wv.vectors.tolist()))
        # 存储向量矩阵索引
        redis_conn.set("edu:cg:mapping", json.dumps(self.w2v_model.wv.index2word))
