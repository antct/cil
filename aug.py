import re
import os
import json

import nlpaug.augmenter.word as naw
import nlpaug.model.word_stats as nmw

from tqdm import tqdm
from multiprocessing import Pool


def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)


# --- meta info ---
dataset = 'nyt10m'
train_ori_path = './benchmark/{}/{}_train.txt'.format(dataset, dataset)
model_path = './tfidf/{}'.format(dataset)
train_aug_path = './benchmark/{}/{}_train_aug.txt'.format(dataset, dataset)

# --- load data ---
print('load data from {} ...'.format(train_ori_path))
train_data = open(train_ori_path).read().splitlines()
train_data = [eval(i) for i in train_data]
print('get data lines: {}'.format(len(train_data)))

# --- train tf-idf model
if not os.path.exists(model_path):
    print('tfidf model not exists, start training ...')
    train_x = [i['text'] for i in train_data]
    train_x_tokens = [_tokenizer(x) for x in train_x]
    tfidf_model = nmw.TfIdf()
    os.makedirs(model_path)
    tfidf_model.train(train_x_tokens)
    tfidf_model.save(model_path)
    print('tfidf model training done!')
else:
    print('tfidf model exists, skip!')

# --- define task ---
def sub_task(data):
    aug = naw.TfIdfAug(model_path=model_path, tokenizer=_tokenizer)
    for i in tqdm(data):
        text = i['text']
        e1 = i['h']['name']
        e1_l, e1_r = i['h']['pos']
        e2 = i['t']['name']
        e2_l, e2_r = i['t']['pos']
        assert text[e1_l:e1_r] == e1
        assert text[e2_l:e2_r] == e2

        if e1_r <= e2_l:
            left = text[:e1_l]
            middle = text[e1_r:e2_l]
            right = text[e2_r:]
            aug_left = '{} '.format(aug.augment(left))
            aug_middle = ' {} '.format(aug.augment(middle))
            aug_right = ' {}'.format(aug.augment(right))

            aug_text = aug_left + e1 + aug_middle + e2 + aug_right

            aug_e1_l = len(aug_left)
            aug_e1_r = len(aug_left) + len(e1)

            aug_e2_l = len(aug_left) + len(e1) + len(aug_middle)
            aug_e2_r = len(aug_left) + len(e1) + len(aug_middle) + len(e2)
            assert aug_text[aug_e1_l:aug_e1_r] == e1
            assert aug_text[aug_e2_l:aug_e2_r] == e2
        else:
            left = text[:e2_l]
            middle = text[e2_r:e1_l]
            right = text[e1_r:]
            aug_left = '{} '.format(aug.augment(left))
            aug_middle = ' {} '.format(aug.augment(middle))
            aug_right = ' {}'.format(aug.augment(right))
            aug_text = aug_left + e2 + aug_middle + e1 + aug_right

            aug_e2_l = len(aug_left)
            aug_e2_r = len(aug_left) + len(e2)

            aug_e1_l = len(aug_left) + len(e2) + len(aug_middle)
            aug_e1_r = len(aug_left) + len(e2) + len(aug_middle) + len(e1)
            assert aug_text[aug_e1_l:aug_e1_r] == e1
            assert aug_text[aug_e2_l:aug_e2_r] == e2

        i['aug_text'] = aug_text
        i['aug_h'] = {}
        i['aug_h']['id'] = i['h']['id']
        i['aug_h']['name'] = i['h']['name']
        i['aug_h']['pos'] = [aug_e1_l, aug_e1_r]
        i['aug_t'] = {}
        i['aug_t']['id'] = i['t']['id']
        i['aug_t']['name'] = i['t']['name']
        i['aug_t']['pos'] = [aug_e2_l, aug_e2_r]

    return data

# --- multi-processing augmentation ---
process_num = 48
print('augmentate training data with {} threads ...'.format(process_num))
pool = Pool(processes=process_num)

results = []
thread_idx = [int(len(train_data)/(process_num/i))
              if i else 0 for i in range(0, process_num+1)]
parts = [train_data[thread_idx[i]:thread_idx[i+1]] for i in range(process_num)]
results = pool.map(sub_task, parts)

pool.close()
pool.join()
print('augmentate done!')

data = []
for x in results:
    data.extend(x)

# --- write to file ---
print('write augmentated data into {} ...'.format(train_aug_path))
with open(train_aug_path, 'w+') as wf:
    for i in data:
        wf.write(json.dumps(i))
        wf.write('\n')
print('write augmentated data done!'.format(train_aug_path))
