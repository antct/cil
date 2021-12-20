from google.protobuf.descriptor import Error
from torch import nn
import torch
import torch.nn.functional as F


class BagAttention(nn.Module):
    def __init__(self, sentence_encoder, num_class, rel2id, mil='att', hparams=None):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.domain_fc = nn.Linear(self.sentence_encoder.hidden_size, 2)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        self.criterion = nn.CrossEntropyLoss()
        self.mil = mil
        if hparams is None:
            self.hparams = {
                'temperature': 0.05
            }
        else:
            self.hparams = hparams
        for rel, id in rel2id.items():
            self.id2rel[id] = rel


    def infer(self, bag):
        pass


    def cl(self, rep, aug_rep, bag_rep):
        # rep: (B, bag, H)
        # aug_rep: (B, bag, H)
        # bag_rep: (B, H)
        temperature = self.hparams.temperature
        # (B, bag, H)
        batch_size, bag_size, hidden_size = rep.size()
        aug_rep = aug_rep.view(batch_size, bag_size, hidden_size)
        # positive pairs
        # instance ~ augmented instance
        # (B, bag, H) ~ (B, bag, H) - (B, bag)
        pos_sim = F.cosine_similarity(rep, aug_rep, dim=-1)
        pos_sim = torch.exp(pos_sim / temperature)
        # negative pairs
        # instance ~ other bag representation
        # (B, H) - (B, bag, H)
        tmp_bag_rep = bag_rep.unsqueeze(1).repeat(1, bag_size, 1)
        # each instance ~ its own bag representation
        axis_sim = F.cosine_similarity(rep, tmp_bag_rep, dim=-1) # (B, bag)
        # axis_sim = axis_sim.unsqueeze(-1) # (B, bag, 1)
        tmp_bag_rep = bag_rep.unsqueeze(0).repeat(batch_size, 1, 1) # (B, B, H)
        # (B, bag, H) ~ (B, B, H) - (B, bag, B)
        tmp_rep = rep.permute((1, 2, 0)) # (bag, H, B)
        tmp_bag_rep = tmp_bag_rep.permute((1, 2, 0)) # (B, H, B)
        tmp_bag_rep = tmp_bag_rep.unsqueeze(1) # (B, 1, H, B)
        # (bag, H, B) ~ (B, 1, H, B) - (B, bag, B)
        pair_sim = F.cosine_similarity(tmp_rep, tmp_bag_rep, dim=-2) # (B, bag, B)
        # bug sum(2) ? any effect ?
        neg_sim = torch.exp(pair_sim / temperature).sum(2) - torch.exp(axis_sim / temperature)
        pos_sim = pos_sim.view(-1)
        neg_sim = neg_sim.view(-1)
        loss = -1.0 * torch.log(pos_sim / (pos_sim + neg_sim))
        loss = loss.mean()
        return loss

    
    def forward(self, label, scope, arg1, arg2, arg3, arg4, arg5=None, arg6=None, arg7=None, arg8=None, train=True, bag_size=0):
        if bag_size > 0:
            flat = lambda x: x.view(-1, x.size(-1))
            arg1, arg2, arg3, arg4 = flat(arg1), flat(arg2), flat(arg3), flat(arg4)
        else:
            begin, end = scope[0][0], scope[-1][1]
            flat = lambda x: x[:, begin:end, :].view(-1, x.size(-1))
            arg1, arg2, arg3, arg4 = flat(arg1), flat(arg2), flat(arg3), flat(arg4)
            scope = torch.sub(scope, torch.zeros_like(scope).fill_(begin))

        rep, mlm_loss = self.sentence_encoder(arg1, arg2, arg3, arg4, mlm=train)

        items = []
        if train:
            items.append(mlm_loss)

            assert arg5 is not None
            flat = lambda x: x.view(-1, x.size(-1))
            arg5, arg6, arg7, arg8 = flat(arg5), flat(arg6), flat(arg7), flat(arg8)
            aug_rep, _ = self.sentence_encoder(arg5, arg6, arg7, arg8, mlm=False)

            if bag_size == 0:
                bag_rep = []
                query = torch.zeros((rep.size(0))).long()
                if torch.cuda.is_available():
                    query = query.cuda()
                for i in range(len(scope)):
                    query[scope[i][0]:scope[i][1]] = label[i]
                att_mat = self.fc.weight.data[query] # (nsum, H)
                att_score = (rep * att_mat).sum(-1) # (nsum)
                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
                    softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]]) # (n)
                    if self.mil == 'att':
                        pass
                    elif self.mil == 'one':
                        one_index = softmax_att_score.argmax(1).cpu()
                        one_att_score = torch.zeros(softmax_att_score.shape).scatter(1, one_index.unsqueeze(1), 1.0).cuda()
                        softmax_att_score = one_att_score
                    bag_rep.append((softmax_att_score.unsqueeze(-1) * bag_mat).sum(0)) # (n, 1) * (n, H) -> (n, H) -> (H)
                bag_rep = torch.stack(bag_rep, 0) # (B, H)
                bag_logits = self.fc(self.drop(bag_rep))
                # when bag size is zero, different bags have different size
                # and it's hard to implement cl in parallel
                raise NotImplementedError('cl for bag size 0 is not implemented!')
            else:
                batch_size = label.size(0)
                query = label.unsqueeze(1) # (B, 1)
                att_mat = self.fc.weight.data[query] # (B, 1, H)
                rep = rep.view(batch_size, bag_size, -1)
                att_score = (rep * att_mat).sum(-1) # (B, bag)
                softmax_att_score = self.softmax(att_score) # (B, bag)
                if self.mil == 'att':
                    pass
                elif self.mil == 'one':
                    one_index = softmax_att_score.argmax(1).cpu()
                    one_att_score = torch.zeros(softmax_att_score.shape).scatter(1, one_index.unsqueeze(1), 1.0).cuda()
                    softmax_att_score = one_att_score
                bag_rep = (softmax_att_score.unsqueeze(-1) * rep).sum(1) # (B, bag, 1) * (B, bag, H) -> (B, bag, H) -> (B, H)
                bag_logits = self.fc(self.drop(bag_rep)) # (B, N)

                cl_loss = self.cl(rep, aug_rep, bag_rep)
                items.append(cl_loss)
        else:
            if bag_size == 0:
                bag_logits = []
                att_score = torch.matmul(rep, self.fc.weight.data.transpose(0, 1)) # (nsum, H) * (H, N) -> (nsum, N)
                for i in range(len(scope)):
                    bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
                    softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]].transpose(0, 1)) # (N, n) num_labels
                    rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat) # (N, n) * (n, H) -> (N, H)
                    logit_for_each_rel = self.softmax(self.fc(rep_for_each_rel)) # ((each rel)N, (logit)N)
                    logit_for_each_rel = logit_for_each_rel.diag() # (N)
                    bag_logits.append(logit_for_each_rel)
                bag_logits = torch.stack(bag_logits, 0) # after **softmax**
            else:
                batch_size = rep.size(0) // bag_size
                att_score = torch.matmul(rep, self.fc.weight.data.transpose(0, 1)) # (nsum, H) * (H, N) -> (nsum, N)
                att_score = att_score.view(batch_size, bag_size, -1) # (B, bag, N)
                rep = rep.view(batch_size, bag_size, -1) # (B, bag, H)
                softmax_att_score = self.softmax(att_score.transpose(1, 2)) # (B, N, (softmax)bag)
                rep_for_each_rel = torch.matmul(softmax_att_score, rep) # (B, N, bag) * (B, bag, H) -> (B, N, H)
                bag_logits = self.softmax(self.fc(rep_for_each_rel)).diagonal(dim1=1, dim2=2) # (B, (each rel)N)

        items.append(bag_logits)

        items = items if len(items) > 1 else items[0]
        return items