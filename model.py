import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    """Skip gram model of word2vec.

    Attributes:
        emb_size: Embedding size.
        emb_dimention: Embedding dimention, typically from 50 to 500.
        u_embedding: Embedding for center word.
        v_embedding: Embedding for neibor words.
    """

    def __init__(self, emb_size, emb_dimension, wvectors, cvectors):
        """Initialize model parameters.

        Apply for two embedding layers.
        Initialize layer weight

        Args:
            emb_size: Embedding size.
            emb_dimention: Embedding dimention, typically from 50 to 500.

        Returns:
            None
        """
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension)
        #self.init_emb()

        # to init emb
        self.u_embeddings.weight = nn.Parameter(torch.Tensor(wvectors))
        self.v_embeddings.weight = nn.Parameter(torch.Tensor(cvectors))

    def forward(self, pair_u, pair_v, pos_u, mask_pos_u, neg_u, mask_neg_u):
        """Forward process.

        As pytorch designed, all variables must be batch format, so all input of this method is a list of word id.

        Args:
            pair_u: list of center word ids for word pairs. [bs]
            pair_v: list of neibor word ids for word pairs. [bs]
            pos_u: list of center word ids for positive samples. [bs, kn]
            neg_u: list of center word ids for negative samples. [bs, kn]
            mask_pos_u:
            mask_neg_u:


        Returns:
            Loss of this process, a pytorch variable.
        """
        emb_u = self.u_embeddings(pair_u)
        emb_v = self.v_embeddings(pair_v)
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1) # [bs]
        score = F.logsigmoid(score)
        neg_emb_u = self.u_embeddings(neg_u)
        mask_neg_u = mask_neg_u.unsqueeze(2).expand(mask_neg_u.size()[0], mask_neg_u.size()[1],
                                                               self.emb_dimension)
        neg_emb_u = torch.mul(neg_emb_u, mask_neg_u) # [bs, kn, emb_dim]
        neg_score = torch.bmm(neg_emb_u, emb_v.unsqueeze(2)).squeeze() # [bs, kn]
        neg_score = F.logsigmoid(-1 * neg_score)
        pos_emb_u = self.u_embeddings(pos_u)
        mask_pos_u = mask_pos_u.unsqueeze(2).expand(mask_pos_u.size()[0], mask_pos_u.size()[1],
                                                    self.emb_dimension)
        pos_emb_u = torch.mul(pos_emb_u, mask_pos_u)  # [bs, kn, emb_dim]
        pos_score_t = torch.bmm(pos_emb_u, emb_v.unsqueeze(2)).squeeze()  # [bs, kn]
        pos_score_t = F.logsigmoid(pos_score_t)
        pos_score = score.unsqueeze(1).expand(pos_score_t.size()[0], pos_score_t.size()[1])-pos_score_t

        return -1 * (torch.sum(score)+torch.sum(neg_score)+torch.sum(pos_score))

def test():
    model = SkipGramModel(100, 100)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    model.save_embedding(id2word)


if __name__ == '__main__':
    test()
