# -*- coding: UTF-8 -*-
'''
Test the embeddings on word similarity task
Select 240.txt 297.txt 
'''
import numpy as np
import pdb
import sys,getopt
from scipy.stats import spearmanr
from utils import logging_set
import logging
import argparse

def build_dictionary(word_list):
        dictionary = dict()
        cnt = 0
        for w in word_list:
                dictionary[w] = cnt
                cnt += 1
        return dictionary


def read_wordpair(sim_file):
        f1 = open(sim_file, 'r')
        pairs = []
        for line in f1:
                pair = line.split()
                pair[2] = float(pair[2])
                pairs.append(pair)
        f1.close()
        return pairs


def read_vectors(vec_file):
        # input:  the file of word2vectors
        # output: word dictionay, embedding matrix -- np ndarray
        f = open(vec_file,'r')
        cnt = 0
        word_list = []
        embeddings = []
        word_size = 0
        embed_dim = 0
        for line in f:
                data = line.split()
                if cnt == 0:
                        word_size = int(data[0])
                        embed_dim = int(data[1])
                else:
                        word_list.append(data[0])
                        tmpVec = [float(x) for x in data[1:(embed_dim+1)]]
                        embeddings.append(tmpVec)
                cnt = cnt + 1
        f.close()
        embeddings = np.array(embeddings)
        for i in range(int(word_size)):
                embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
        dict_word = build_dictionary(word_list)
        return word_size, embed_dim, dict_word, embeddings


def get_harmonic_mean(list):
        sum = 0.0
        for i in list:
            sum += 1.0 / (i + 1)
        return len(list) / sum


def frange(start, stop, step):
        i = start
        while i < stop:
            yield i
            i += step


def calc_sim(word_size, embed_dim, dict_word, embeddings, similarity_file):
        pairs = read_wordpair(similarity_file)

        human_sim = []
        vec_sim = []
        cnt = 0
        total = len(pairs)
        for pair in pairs:
                w1 = pair[0]
                w2 = pair[1]
                # w1 = w1.decode('utf-8')
                # w2 = w2.decode('utf-8')
                if w1 in dict_word and w2 in dict_word:
                        cnt += 1
                        id1 = dict_word[w1]
                        id2 = dict_word[w2]
                        vsim = embeddings[id1].dot(embeddings[id2].T) / (
                        np.linalg.norm(embeddings[id1]) * np.linalg.norm(embeddings[id2]))
                        human_sim.append(pair[2])
                        vec_sim.append(vsim)
        score = spearmanr(human_sim, vec_sim)
        logging.info('%d word pairs appeared in the training dictionary , total word pairs %d' % (cnt, total))
        return cnt, total, score


if  __name__ == '__main__':
        parser = argparse.ArgumentParser(
                formatter_class=argparse.RawDescriptionHelpFormatter,
                description="Philly arguments parser")

        parser.add_argument('emb_file_name', type=str)
        parser.add_argument('similarity_test_data', type=str)
        parser.add_argument('--log_path', type=str, default='train.log')
        args, _ = parser.parse_known_args()

        logging_set(args.log_path)
        word_size, embed_dim, dict_word, embeddings = read_vectors(args.emb_file_name)
        cnt, total, score = calc_sim(word_size, embed_dim, dict_word, embeddings, args.similarity_test_data)
        logging.info('test score: %0.6f' % score.correlation)
