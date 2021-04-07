# -*- coding: utf-8 -*-#

# Name:     walker
# Author:   Jasper.X
# Date:     2021/4/7
# Description:

import itertools
import random

import numpy as np

from time import time

from joblib import Parallel, delayed

# 多进程相关包
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

from ge.utils import partition_num


class RandomWalker:
    def __init__(self, G, p=1, q=1):
        """
        :param G:
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        """
        self.G = G
        self.p = p
        self.q = q

    def random_walk(self, walk_length, start_node):
        """
        随机从相邻节点点中选取一个节点进行生成，是random_walk的无权图版本（是否有方向要看穿的图是否为有向图）
        :param walk_length: 游走的步长
        :param start_node: 起始节点
        :return: 此次遍历生成的序列
        """
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def random_walk_with_weight(self, walk_length, start_node):
        """
        加权随机游走
        """
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                unnormalized_probs = [self.G[cur][nbr].get('weight', 1.0) for nbr in self.G.neighbors(cur)]
                norm_const = sum(unnormalized_probs)
                # 获得该node到出点的跳转概率（归一化）
                normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
                walk.append(np.random.choice(cur_nbrs, size=1, p=normalized_probs)[0])
            else:
                break

        return walk

    def random_walk_in_batch(self, args):
        """
        随机游走算法--批量

        :param args: list 参数 [(node, walk_length)]
        :return:
        """
        start = time()

        # 1,随机游走
        walks = []
        for arg_tuple in args:
            start_node, walk_length = arg_tuple
            walk = self.random_walk_with_weight(start_node, walk_length)
            if len(walk) >= 5:  # 过短的句子无法在w2v中有效训练，舍弃
                walks.append(walk)

        consume_time = time() - start
        print("Waste time: {}s".format(consume_time))
        return walks

    def simulate_walks_multi_process(self, num_walks, walk_length=[8, 9, 10], num_workers=0):
        """
        多进程进行随机游走，先生成所有待遍历的句子，随后对数据进行切分，分发到不同的worker进行遍历
        generate sentences for training(假设num_walks=80,nodes个数为6000,将会产生80*6000=480000个句子)
        :param num_walks: <int> num_walks * len(nodes)=total_sentence
        :param walk_length: <int> length of sentence
        :param workers: <int> number of workers
        :return: <list>:[ <list>[node_1,node_2,xxx,node_n],,,], n=walk_length
        """
        print("using multiprocessing ")
        # 1.初始化
        nodes = list(self.G.nodes())
        if num_workers == 0:
            NUM_CPU = cpu_count() // 2 if len(nodes) > cpu_count() else 1
        else:
            NUM_CPU = num_workers
        if type(walk_length) == int:
            walk_length = [walk_length]

        # 2.数据生成
        arg_list = []
        for _ in range(num_walks):  # 对图进行遍历的次数
            random.shuffle(nodes)  # 将图中的nodes打乱
            depth = random.choice(walk_length)  # 从wak_length中随机选一个作为DFS遍历的深度
            # 生成每个起始点对应的要遍历的深度
            for node in nodes:
                arg_list.append((depth, node))

        # 3.数据分片（根据可用的CPU个数进行CPU分片）
        batch_arg_list = []
        batch_size = len(nodes) * num_walks // NUM_CPU
        for idx in range(0, len(arg_list), batch_size):
            batch_arg_list.append(arg_list[idx:idx + batch_size])

        # 4.多进程随机游走
        total_sentences = []
        with ProcessPoolExecutor(max_workers=NUM_CPU) as executor:
            for sentence in executor.map(self.random_walk_in_batch, batch_arg_list):
                print('a walker has generated {} sentence'.format(len(sentence)))
                total_sentences = total_sentences + sentence

        return total_sentences

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):

        G = self.G

        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, ):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == 1 and self.q == 1:
                    walks.append(self.random_walk(
                        walk_length=walk_length, start_node=v))
        return walks


if __name__ == '__main__':
    # 当总节点个数为奇数，CPU个数为偶数时，会有余数，会有一部分数据丢失
    arg_list = [1,2,3,4,5,6,7,8,9,10]
    NUM_CPU = 3
    batch_size = len(arg_list)//NUM_CPU
    for p in range(NUM_CPU):
        print(arg_list[p * batch_size:p * batch_size + batch_size])
    #
    print("-----改进后-------")
    for idx in range(0, len(arg_list), batch_size):
        print(arg_list[idx:idx + batch_size])



