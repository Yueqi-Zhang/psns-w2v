import pickle as pkl

def load_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as fin:
        obj = pkl.load(fin)
    return obj

def convert_word_to_id(ps_w, word2id):
    ps = dict()
    for cword in ps_w.keys():
        ps[word2id[cword]] = [word2id[w] for w in ps_w[cword]]
    return ps
