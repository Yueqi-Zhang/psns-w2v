import numpy
import codecs
from collections import deque
numpy.random.seed(12345)


class InputData:
    """Store data for word2vec, such as word map, sampling table and so on.

    Attributes:
        word_frequency: Count of each word, used for filtering low-frequency words and sampling table
        word2id: Map from word to word id, without low-frequency words.
        id2word: Map from word id to word, without low-frequency words.
        sentence_count: Sentence count in files.
        word_count: Word count in files, without low-frequency words.
    """

    def __init__(self, file_name, min_count):
        self.input_file_name = file_name
        self.get_words(min_count)
        self.word_pair_catch = deque()
        self.init_sample_table()
        print('Word Count: %d' % len(self.word2id))
        print('Sentence Length: %d' % (self.sentence_length))

    def get_words(self, min_count):
        self.input_file = open(self.input_file_name)
        self.sentence_length = 0
        self.sentence_count = 0
        word_frequency = dict()
        for line in self.input_file:
            self.sentence_count += 1
            line = line.strip().split(' ')
            self.sentence_length += len(line)
            for w in line:
                try:
                    word_frequency[w] += 1
                except:
                    word_frequency[w] = 1
        sort_word_frequency = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
        self.word2id = dict()
        self.id2word = dict()
        wid = 0
        self.word_frequency = dict()
        for w, c in sort_word_frequency:
            if c < min_count:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        self.word_count = len(self.word2id)

    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = numpy.array(list(self.word_frequency.values()))**0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = numpy.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy.array(self.sample_table)

    # @profile
    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pair_catch) < batch_size:
            sentence = self.input_file.readline()
            if sentence is None or sentence == '':
                self.input_file = open(self.input_file_name)
                sentence = self.input_file.readline()
            word_ids = []
            for word in sentence.strip().split(' '):
                try:
                    word_ids.append(self.word2id[word])
                except:
                    continue
            for i, u in enumerate(word_ids):
                for j, v in enumerate(
                        word_ids[max(i - window_size, 0):i + window_size + 1]):
                    assert u < self.word_count
                    assert v < self.word_count
                    if i == j:
                        continue
                    self.word_pair_catch.append((u, v))
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        return numpy.random.shuffle(batch_pairs)

    # @profile
    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(
            self.sample_table, size=(len(pos_word_pair), count)).tolist()
        return neg_v

    def get_ns_batch(self, pos_word_pair, ns, kn):
        neg_u = []
        mask_neg_u = []
        for pair in pos_word_pair:
            cand = ns[pair[0]] + [0]*(kn-len(ns[pair[0]]))
            mask = [1]*(len(ns[pair[0]])) + [0]*(kn-len(ns[pair[0]]))
            neg_u.append(cand)
            mask_neg_u.append(mask)
        return neg_u, mask_neg_u

    def get_ps_batch(self, pos_word_pair, ps, kn):
        pos_u = []
        mask_pos_u = []
        for pair in pos_word_pair:
            cand = ps[pair[0]] + [0]*(kn-len(ps[pair[0]]))
            mask = [1]*(len(ps[pair[0]])) + [0]*(kn-len(ps[pair[0]]))
            pos_u.append(cand)
            mask_pos_u.append(mask)
        return pos_u, mask_pos_u

    def evaluate_pair_count(self, window_size):
        return self.sentence_length * (2 * window_size - 1) - (
            self.sentence_count - 1) * (1 + window_size) * window_size

def InputVector(file_name):
    vectors = []
    i = 0
    with codecs.open(file_name, 'r', encoding='utf-8') as f:
        for lines in f:
            if i != 0:
                vector = lines.strip().split()[1:]
                vector = [float(x) for x in vector]
                vectors.append(vector)
            i += 1
    return vectors

def test():
    a = InputData('./zhihu.txt')


if __name__ == '__main__':
    test()