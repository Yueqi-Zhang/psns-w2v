import pickle as pkl
import logging

def load_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as fin:
        obj = pkl.load(fin)
    return obj

def convert_word_to_id(ps_w, word2id):
    ps = dict()
    for cword in ps_w.keys():
        ps[word2id[cword]] = [word2id[w] for w in ps_w[cword]]
    return ps

def logging_set(log_path):
    """
    Note: if you invoke logging.info or something before basicConfig, some problems may appear because
    the logging module has fabricate a default configuration

    Args:
    log_path:

    Returns:

    """

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(filename=log_path, filemode='w',
        format='%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s',
        level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s'))
    logging.getLogger().addHandler(console)

