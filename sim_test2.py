import codecs
import gensim
import logging
import argparse
from utils import logging_set

def read_synset(path):
    synset = dict()
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for lines in f:
            lines = lines.split()
            if lines[1] not in synset.keys():
                synset[lines[1]] = set()
                synset[lines[1]].add(lines[0])
            else:
                synset[lines[1]].add(lines[0])
    return synset

def synset_test(synset, emb):
    vocab = emb.index2word
    vocab_set = set([w for w in vocab])
    emb_score = 0.0
    std = 0.0
    for word in vocab:
        synset_w = set()
        emb_simi_w = set()
        for key in synset.keys():
            if word in synset[key]:
                synset_w = synset_w | synset[key]
        synset_w = synset_w & vocab_set
        if len(synset_w) <= 1:
            continue
        else:
            for index, (w, sim) in enumerate(emb.most_similar(positive = [word], topn = len(synset_w)-1)):
                emb_simi_w.add(w)
            emb_score_w = len(synset_w & emb_simi_w)
            std_w = len(synset_w)-1
            emb_score += emb_score_w
            std += std_w
    emb_finalscore = emb_score/std
    return emb_finalscore


if  __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Philly arguments parser")

    parser.add_argument('emb_file_name', type=str)
    parser.add_argument('synset_data', type=str)
    parser.add_argument('--log_path', type=str, default='train.log')
    args, _ = parser.parse_known_args()

    logging_set(args.log_path)
    synset = read_synset(args.synset_data)

    emb = gensim.models.KeyedVectors.load_word2vec_format(args.emb_file_name, binary=False, unicode_errors='ignore')
    score = synset_test(synset, emb)
    logging.info('emb score: %0.6f' % score)
