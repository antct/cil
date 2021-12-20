import torch
import torch.nn as nn

from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import CrossEntropyLoss


class BertForMaskedLMCustom(BertForMaskedLM):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        output = (sequence_output, prediction_scores,) + outputs[2:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output


class BERTEntityLMEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False, freeze=True, dropout_prob=0.1):
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        self.bert = BertForMaskedLMCustom.from_pretrained(
            pretrained_model_name_or_path=pretrain_path,
            attention_probs_dropout_prob=dropout_prob,
            hidden_dropout_prob=dropout_prob
        )
        self.config = self.bert.config
        self.hidden_size = self.config.hidden_size * 2
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrain_path,
            do_lower_case=True
        )
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    # modify from https://github.com/thunlp/RE-Context-or-Names
    def mask_tokens(self, inputs, not_mask_pos=None):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        if not_mask_pos is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()
        else:
            # do not mask entity words
            masked_indices = torch.bernoulli(probability_matrix).bool() & (~(not_mask_pos.bool()))
        labels[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # the rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs.cuda(), labels.cuda()

    def forward(self, token, att_mask, pos1, pos2, mlm=False):
        # masked language model loss
        mlm_loss = None

        if mlm:
            indice = torch.arange(0, token.size()[0])
            not_mask_pos = torch.zeros((token.size()[0], token.size()[1]), dtype=int)
            not_mask_pos[indice, pos1] = 1
            not_mask_pos[indice, pos2] = 1
            mask_token, mask_labels = self.mask_tokens(token.cpu(), not_mask_pos)
            mlm_loss, _, _ = self.bert(input_ids=mask_token, attention_mask=att_mask, labels=mask_labels)

        # we re-calculate the hidden states on clean tokens
        # hidden states on mask tokens may be ok
        hidden, _ = self.bert(input_ids=token, attention_mask=att_mask)

        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        x = torch.cat([head_hidden, tail_hidden], 1)  # (B, 2H)
        x = self.linear(x)
        return x, mlm_loss

    def tokenize(self, item):
        # sentence -> token
        sentence = item['text']
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
        ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
        sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
        ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
        sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        if self.mask_entity:
            ent0 = ['[unused4]']
            ent1 = ['[unused5]']
            if rev:
                ent0 = ['[unused5]']
                ent1 = ['[unused4]']
        pos_head = [len(sent0), len(sent0) + len(ent0)]
        pos_tail = [
            len(sent0) + len(ent0) + len(sent1),
            len(sent0) + len(ent0) + len(sent1) + len(ent1)
        ]
        if rev:
            pos_tail = [len(sent0), len(sent0) + len(ent0)]
            pos_head = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        tokens = sent0 + ent0 + sent1 + ent1 + sent2
        # token -> index
        re_tokens = ['[CLS]']
        cur_pos = 0
        pos1 = 0
        pos2 = 0
        for token in tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                pos1 = len(re_tokens)
                re_tokens.append('[unused0]')
            if cur_pos == pos_tail[0]:
                pos2 = len(re_tokens)
                re_tokens.append('[unused1]')
            re_tokens.append(token)
            if cur_pos == pos_head[1] - 1:
                re_tokens.append('[unused2]')
            if cur_pos == pos_tail[1] - 1:
                re_tokens.append('[unused3]')
            cur_pos += 1
        re_tokens.append('[SEP]')
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)
        # position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()
        # padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)
        # attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2

    def tokenize_pair(self, item):
        # original sentence
        ori = {}
        ori['text'] = item['text']
        ori['h'] = item['h']
        ori['t'] = item['t']
        # augmented sentence
        aug = {}
        aug['text'] = item['aug_text']
        aug['h'] = item['aug_h']
        aug['t'] = item['aug_t']
        return self.tokenize(ori) + self.tokenize(aug)
