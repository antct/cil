import torch
import json
import os
import random
import numpy as np
import cil

from torch.utils.tensorboard import SummaryWriter
from config import args

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PL_GLOBAL_SEED"] = str(args.seed)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.device_count() > 0:
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic=True

# create dirs
for d in ['ckpt', 'result', 'summary']:
    if os.path.exists(d): continue
    os.makedirs(d)

dataset = args.dataset
model_name = args.model_name
ckpt_path = 'ckpt/{}_{}.pt'.format(dataset, model_name)
p_path = 'result/{}_{}_p.npy'.format(dataset, model_name)
r_path = 'result/{}_{}_r.npy'.format(dataset, model_name)

rel2id = json.load(open('benchmark/{}/{}_rel2id.json'.format(dataset, dataset)))

sentence_encoder = cil.encoder.BERTEntityLMEncoder(
    max_length=128, 
    pretrain_path=args.pretrain_path,
    mask_entity=False,
    freeze=False,
    dropout_prob=args.dropout_prob
)

model = cil.model.BagAttention(
    sentence_encoder=sentence_encoder,
    num_class=len(rel2id),
    rel2id=rel2id,
    mil=args.mil,
    hparams=args
)

writer = SummaryWriter('summary/{}_{}'.format(dataset, model_name)) if args.writer else None

framework = cil.framework.BagRE(
    model=model,
    writer=writer,
    train_path='benchmark/{}/{}_train_aug.txt'.format(dataset, dataset),
    val_path='benchmark/{}/{}_val.txt'.format(dataset, dataset),
    test_path='benchmark/{}/{}_test.txt'.format(dataset, dataset),
    ckpt=ckpt_path,
    args=args
)

framework.print('model: {}'.format(model_name))
framework.print('encoder: {}'.format(framework.model.module.sentence_encoder))
framework.print('framework: {}'.format(framework.__class__.__name__))
framework.print('args: {}'.format(vars(args)))

if args.mode == 'train':
    framework.train_model()

if args.mode == 'test':
    framework.load_model(ckpt_path)
    result = framework.eval_model(framework.test_loader)

    np.save(p_path, result['prec'])
    np.save(r_path, result['rec'])

    p = result['prec']
    framework.print('auc: {:.4f}'.format(result['auc']))
    framework.print('f1: {:.4f}'.format(result['f1']))
    framework.print("p@100: %.4f p@200: %.4f p@300: %.4f p@500: %.4f p@1000: %.4f p@2000: %.4f" % (p[100], p[200], p[300], p[500], p[1000], p[2000]))
    framework.print('P@m: {:.4f}'.format((p[100] + p[200] + p[300]) / 3))
