from torch import tensor, bernoulli, randint, ones, rand, cat
from torch.utils.data import DataLoader

from torchkge.utils import get_bernoulli_probs, get_possible_heads_tails
from torchkge.exceptions import NotYetImplementedError

from torchkge.sampling import BernoulliNegativeSampler


class ModifiedPositionalNegativeSampler(BernoulliNegativeSampler):
    def __init__(self, kg, kg_val=None, kg_test=None):
        super().__init__(kg, kg_val, kg_test)
        self.possible_heads, self.possible_tails, self.n_poss_heads, self.n_poss_tails = self.find_possibilities()

    def find_possibilities(self):
        possible_heads, possible_tails = get_possible_heads_tails(self.kg)

        if self.n_facts_val > 0:
            possible_heads, possible_tails = get_possible_heads_tails(self.kg_val, possible_heads, possible_tails)

        n_poss_heads = []
        n_poss_tails = []

        assert possible_heads.keys() == possible_tails.keys()

        for r in range(self.kg.n_rel):
            if r in possible_heads.keys():
                n_poss_heads.append(len(possible_heads[r]))
                n_poss_tails.append(len(possible_tails[r]))
                possible_heads[r] = list(possible_heads[r])
                possible_tails[r] = list(possible_tails[r])
            else:
                n_poss_heads.append(0)
                n_poss_tails.append(0)
                possible_heads[r] = list()
                possible_tails[r] = list()

        n_poss_heads = tensor(n_poss_heads)
        n_poss_tails = tensor(n_poss_tails)

        return possible_heads, possible_tails, n_poss_heads, n_poss_tails

    def corrupt_batch(self, heads, tails, relations):
        use_cuda = heads.is_cuda
        assert (use_cuda == tails.is_cuda)
        if use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        batch_size = heads.shape[0] * 10
        neg_heads, neg_tails = heads.clone(), tails.clone()

        # Randomly choose which samples will have head/tail corrupted
        mask = bernoulli(self.bern_probs[relations]).double()
        n_heads_corrupted = int(mask.sum().item())

        # Get the number of possible entities for head and tail
        n_poss_heads = self.n_poss_heads[relations[mask == 1]]
        n_poss_tails = self.n_poss_tails[relations[mask == 0]]

        # assert n_poss_heads.shape[0] == n_heads_corrupted
        # assert n_poss_tails.shape[0] == batch_size - n_heads_corrupted

        # Choose a rank of an entity in the list of possible entities
        choice_heads = (n_poss_heads.float() * rand((n_heads_corrupted,))).floor().long()
        choice_tails = (
                n_poss_tails.float() * rand((batch_size - n_heads_corrupted,))).floor().long()

        corr = []
        rels = relations[mask == 1]
        for i in range(n_heads_corrupted):
            r = rels[i].item()
            choices = self.possible_heads[r]
            if len(choices) == 0:
                # in this case the relation r has never been used with any head
                # choose one entity at random
                corr.append(randint(low=0, high=self.n_ent, size=(1,)).item())
            else:
                corr.append(choices[choice_heads[i].item()])
        neg_heads[mask == 1] = tensor(corr, device=device).long()

        corr = []
        rels = relations[mask == 0]
        for i in range(batch_size - n_heads_corrupted):
            r = rels[i].item()
            choices = self.possible_tails[r]
            if len(choices) == 0:
                # in this case the relation r has never been used with any tail
                # choose one entity at random
                corr.append(randint(low=0, high=self.n_ent, size=(1,)).item())
            else:
                corr.append(choices[choice_tails[i].item()])
        neg_tails[mask == 0] = tensor(corr, device=device).long()

        return neg_heads.long(), neg_tails.long()
