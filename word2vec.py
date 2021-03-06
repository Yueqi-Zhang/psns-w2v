from input_data import InputData
from input_data import InputVector
import numpy
from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import load_from_pkl, convert_word_to_id
import sys
import random
import string
import platform

import sys
import argparse
import codecs
import logging
import os
import tempfile
import random
import string
import shutil
import debugger
from evaluation import evaluation
from input_data import InputData
from input_data import InputVector

from utils import logging_set

class Word2Vec:
    def __init__(self,
                 input_file_name,
                 input_wvectors,
                 input_cvectors,
                 input_ps,
                 input_ns,
                 output_file_name,
                 emb_dimension=100,
                 batch_size=50,
                 window_size=5,
                 kn = 20,
                 iteration=1,
                 initial_lr=0.001,
                 clip=1.0,
                 min_count=30,
                 batch_num_to_valid=100000,
                 ):
        """Initilize class parameters.

        Args:
            input_file_name: Name of a text data from file. Each line is a sentence splited with space.
            input_vectors: Pretrained vector
            input_psns: Pretrained positive sample & negative sample
            output_file_name: Name of the final embedding file.
            emb_dimention: Embedding dimention, typically from 50 to 500.
            batch_size: The count of word pairs for one forward.
            window_size: Max skip length between words.
            kn: k neighbors.
            iteration: Control the multiple training iterations.
            initial_lr: Initial learning rate.
            min_count: The minimal word frequency, words with lower frequency will be filtered.

        Returns:
            None.
        """
        self.data = InputData(input_file_name, min_count)
        self.pre_wvectors = InputVector(input_wvectors)
        self.pre_cvectors = InputVector(input_cvectors)
        self.ps_w = load_from_pkl(input_ps)
        self.ns_w = load_from_pkl(input_ns)
        self.ps = convert_word_to_id(self.ps_w, self.data.word2id)
        self.ns = convert_word_to_id(self.ns_w, self.data.word2id)
        self.output_file_name = output_file_name
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.kn = kn
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.clip = clip
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension, self.pre_wvectors, self.pre_cvectors)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.skip_gram_model.cuda()
        self.optimizer = optim.SGD(
            self.skip_gram_model.parameters(), lr=self.initial_lr)
        self.batch_num_to_valid = batch_num_to_valid

    def train(self, similarity_test_paths, synset_paths, analogy_paths):
        """Multiple training.

        Returns:
            None.
        """
        pair_count = self.data.evaluate_pair_count(self.window_size)
        batch_count = self.iteration * pair_count / self.batch_size
        process_bar = tqdm(range(int(batch_count)))
        # self.skip_gram_model.save_embedding(
        #     self.data.id2word, 'begin_embedding.txt', self.use_cuda)

        best_scores = dict()
        tmp_emb_dir = os.path.join(tempfile.gettempdir(), 'embedding')
        tmp_emb_path = os.path.join(tmp_emb_dir, ''.join(random.sample(string.ascii_letters + string.digits, 16)))

        for epoch in range(self.iteration):
            for i in process_bar:
                pos_pairs = self.data.get_batch_pairs(self.batch_size,
                                                      self.window_size)
                pos_u, mask_pos_u = self.data.get_ps_batch(pos_pairs, self.ps, self.kn)
                neg_u, mask_neg_u = self.data.get_ns_batch(pos_pairs, self.ns, self.kn)
                pair_u = [pair[0] for pair in pos_pairs]
                pair_v = [pair[1] for pair in pos_pairs]

                pair_u = Variable(torch.LongTensor(pair_u))
                pair_v = Variable(torch.LongTensor(pair_v))
                pos_u = Variable(torch.LongTensor(pos_u))
                mask_pos_u = Variable(torch.FloatTensor(mask_pos_u))
                neg_u = Variable(torch.LongTensor(neg_u))
                mask_neg_u = Variable(torch.FloatTensor(mask_neg_u))
                if self.use_cuda:
                    pair_u = pair_u.cuda()
                    pair_v = pair_v.cuda()
                    pos_u = pos_u.cuda()
                    mask_pos_u = mask_pos_u.cuda()
                    neg_u = neg_u.cuda()
                    mask_neg_u = mask_neg_u.cuda()

                self.optimizer.zero_grad()
                '''
                param = self.skip_gram_model.parameters()
                tmp = []
                try:
                    while True:
                        tmp.append(param.__next__())
                except:
                    pass
                '''
                loss = self.skip_gram_model.forward(pair_u, pair_v, pos_u, mask_pos_u, neg_u, mask_neg_u)
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.skip_gram_model.parameters(), self.clip)
                self.optimizer.step()

                process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                            (loss.data[0],
                                             self.optimizer.param_groups[0]['lr']))
                if i * self.batch_size % 100000 == 0:
                    lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

                if i % self.batch_num_to_valid == 0:
                    logging.info('epoch%d_batch%d, evaluating...' % (epoch, i))
                    self.save_embedding(self.data.id2word, tmp_emb_path, self.use_cuda)

                    best_scores, save_flag = evaluation(tmp_emb_path, similarity_test_paths, synset_paths, analogy_paths,
                                                        best_scores)
                    if save_flag == True:
                        emb_save_path = self.output_file_name + "_epoch%d_batch%d" % (epoch, i)
                        shutil.move(tmp_emb_path, emb_save_path)
                        logging.info('Save current embedding to %s' % emb_save_path)


            self.skip_gram_model.save_embedding(
                self.data.id2word, self.output_file_name, self.use_cuda)
            logging.info('final evaluating...')
            self.save_embedding(self.data.id2word, tmp_emb_path, self.use_cuda)
            best_scores, save_flag = evaluation(tmp_emb_path, similarity_test_paths, synset_paths, analogy_paths, best_scores)
            if save_flag == True:
                emb_save_path = self.output_file_name + "_epoch%d" % epoch
                shutil.move(tmp_emb_path, emb_save_path)
                logging.info('Save current embedding to %s' % emb_save_path)

    def save_embedding(self, id2word, file_name, use_cuda):
        """Save all embeddings to file.

        As this class only record word id, so the map from id to word has to be transfered from outside.

        Args:
            id2word: map from word id to word.
            file_name: file name.
        Returns:
            None.
        """
        if use_cuda:
            embedding = self.skip_gram_model.u_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.skip_gram_model.u_embeddings.weight.data.numpy()
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))



if __name__ == '__main__':
    #w2v = Word2Vec(input_file_name=sys.argv[1], input_wvectors=sys.argv[2], input_cvectors=sys.argv[3], input_ps=sys.argv[4], input_ns=sys.argv[5], output_file_name=sys.argv[6])
    #w2v.train()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Philly arguments parser")

    parser.add_argument('input_file_name', type=str)
    parser.add_argument('input_wvectors', type=str)
    parser.add_argument('input_cvectors', type=str)
    parser.add_argument('input_ps', type=str)
    parser.add_argument('input_ns', type=str)
    parser.add_argument('output_file_name', type=str)
    parser.add_argument('--similarity_test_paths', type=str, default='data/240.txt|data/297.txt')
    parser.add_argument('--synset_paths', type=str, default='data/nsem3-adjusted.txt')
    parser.add_argument('--analogy_test_paths', type=str, default='data/analogy.txt')
    parser.add_argument('--emb_dimension', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--kn', type=int, default=20)
    parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--min_count', type=int, default=30)
    parser.add_argument('--initial_lr', type=float, default=0.01)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--batch_num_to_valid', type=int, default=100000)
    parser.add_argument('--log_path', type=str, default='train.log')
    args, _ = parser.parse_known_args()
    if args.similarity_test_paths == 'None':
        args.similarity_test_paths = None
    if args.synset_paths == 'None':
        args.synset_paths = None
    if args.analogy_test_paths == 'None':
        args.analogy_test_paths = None

    #如果是mac系统，就不做analogy test
    if platform.system() == "Darwin":
        args.analogy_test_paths=None

    logging_set(args.log_path)
    #w2v = Word2Vec(input_file_name=sys.argv[1], input_wvectors = sys.argv[2], input_cvectors = sys.argv[3], output_file_name=sys.argv[4])
    w2v = Word2Vec(input_file_name=args.input_file_name, input_wvectors=args.input_wvectors, input_cvectors = args.input_cvectors,
        input_ps = args.input_ps, input_ns = args.input_ns, output_file_name=args.output_file_name, emb_dimension = args.emb_dimension,
        batch_size=args.batch_size, window_size=args.window_size, kn = args.kn, iteration=args.iteration, min_count=args.min_count,
        initial_lr=args.initial_lr, clip=args.clip, batch_num_to_valid=args.batch_num_to_valid)
    w2v.train(similarity_test_paths=args.similarity_test_paths, synset_paths=args.synset_paths, analogy_paths=args.analogy_test_paths)