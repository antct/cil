
from torch import nn
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from accelerate import Accelerator, DistributedDataParallelKwargs

from .data_loader import CILDataLoader
from .utils import AverageMeter

import torch
import time
import numpy as np


class BagRE(nn.Module):

    def __init__(self,
                 model,
                 writer,
                 train_path,
                 val_path,
                 test_path,
                 ckpt,
                 args):

        super().__init__()
        self.args = args
        self.model = model

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.device = self.accelerator.device

        # data loader
        # max_bag_size: avoid oom
        self.train_loader = CILDataLoader(
            path=train_path,
            rel2id=model.rel2id,
            tokenizer=model.sentence_encoder.tokenize_pair,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            bag_size=self.args.bag_size,
            max_bag_size=self.args.max_bag_size,
            entpair_as_bag=False
        )
        self.eval_loader = CILDataLoader(
            path=val_path,
            rel2id=model.rel2id,
            tokenizer=model.sentence_encoder.tokenize,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            bag_size=0,
            max_bag_size=self.args.max_bag_size,
            entpair_as_bag=True
        )
        self.test_loader = CILDataLoader(
            path=test_path,
            rel2id=model.rel2id,
            tokenizer=model.sentence_encoder.tokenize,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            bag_size=0,
            max_bag_size=0,
            entpair_as_bag=True
        )

        # tensorboard writer
        self.writer = writer

        # criterion
        if self.args.loss_weight:
            self.criterion = nn.CrossEntropyLoss(weight=self.train_loader.dataset.weight)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # params and optimizer
        params = list(self.model.named_parameters())

        # bert adam
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in params if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.eps, correct_bias=False)
        total_steps = len(self.train_loader) * self.args.max_epoch
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=total_steps)
        self.model, self.optimizer, self.train_loader, self.eval_loader, self.test_loader = self.accelerator.prepare(self.model, self.optimizer, self.train_loader, self.eval_loader, self.test_loader)

        self.ckpt = ckpt
        self.global_steps = 0

    def train_model(self):
        best_auc, best_f1 = 0., 0.
        best_p100, best_p200, best_p300 = 0., 0., 0.
        run_time = 0.
        for epoch in range(1, self.args.max_epoch+1):
            self.model.train()
            self.accelerator.print("=== epoch {} train start ===".format(epoch))
        
            avg_loss = AverageMeter()
            avg_mlm_loss = AverageMeter()
            avg_cl_loss = AverageMeter()
            avg_acc = AverageMeter()
            avg_pos_acc = AverageMeter()

            t = tqdm(self.train_loader, disable=not self.accelerator.is_local_main_process)

            start_time = time.time()
            for _, data in enumerate(t):
                for i in range(len(data)):
                    try:
                        data[i] = data[i].to(self.device)
                    except:
                        pass

                self.global_steps += 1

                p = float(self.global_steps) / (self.args.warmup_steps * 2.0)
                alpha = 2. / (1. + np.exp(-2. * p)) - 1

                label, bag_name, scope, bag, args = data[0], data[1], data[2], data[3], data[4:]

                mlm_loss, cl_loss, logits = self.model(label, scope, *args, bag_size=self.args.bag_size)

                cl_loss = cl_loss.mean()
                mlm_loss = mlm_loss.mean()

                cl_loss = 10.0 * cl_loss
                mlm_loss = 0.1 * mlm_loss

                loss = self.criterion(logits, label)

                score, pred = logits.max(-1)
                acc = float((pred == label).long().sum()) / label.size(0)
                pos_total = (label != 0).long().sum()
                pos_correct = ((pred == label).long() * (label != 0).long()).sum()
                if pos_total > 0:
                    pos_acc = float(pos_correct) / float(pos_total)
                else:
                    pos_acc = 0

                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                avg_mlm_loss.update(mlm_loss.item(), 1)
                avg_cl_loss.update(cl_loss.item(), 1)
                avg_pos_acc.update(pos_acc, 1)

                t.set_postfix(mil_loss=avg_loss.avg, mlm_loss=avg_mlm_loss.avg, cl_loss=avg_cl_loss.avg, acc=avg_acc.avg, pos_acc=avg_pos_acc.avg, alpha=alpha)

                loss = loss + alpha * cl_loss + mlm_loss
                loss = loss / self.args.grad_acc_steps
                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)

                if self.global_steps % self.args.grad_acc_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.global_steps % self.args.save_steps == 0:
                    self.accelerator.print("=== steps {} eval start ===".format(self.global_steps))
                    result = self.eval_model(self.eval_loader)
                    p = result['prec']
                    self.accelerator.print("auc: %.4f f1: %.4f p@100: %.4f p@200: %.4f p@300: %.4f p@500: %.4f p@1000: %.4f p@2000: %.4f" % (
                        result['auc'], result['f1'], p[100], p[200], p[300], p[500], p[1000], p[2000]))

                    if result['auc'] > best_auc:
                        self.accelerator.print("auc@best: %.4f auc: %.4f best ckpt and save to %s" % (best_auc, result['auc'], self.ckpt))
                        self.accelerator.wait_for_everyone()
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        self.accelerator.save(unwrapped_model.state_dict(), self.ckpt)
                        best_auc = result['auc']
                        best_f1 = result['f1']
                        best_p100 = p[100]
                        best_p200 = p[200]
                        best_p300 = p[300]
                    else:
                        self.accelerator.print("auc@best: %.4f auc: %.4f no improvement and skip" % (best_auc, result['auc']))

                    self.model.train()
                    self.accelerator.print("=== steps {} eval end ===".format(self.global_steps))

                if self.global_steps == self.args.max_steps:
                    break

            # runtime
            end_time = time.time()
            epoch_time = end_time - start_time
            run_time += epoch_time

            # writer
            if self.accelerator.is_local_main_process and self.writer is not None:
                self.writer.add_scalar('train/loss', avg_loss.avg, epoch)
                self.writer.add_scalar('train/acc', avg_acc.avg, epoch)
                self.writer.add_scalar('train/pos_acc', avg_pos_acc.avg, epoch)
                self.writer.add_scalar('train/run_time', run_time, epoch)

            self.accelerator.print("=== epoch %d train end time: %ds avg epoch time: %ds run time: %ds ===" % (epoch, epoch_time, run_time / epoch, run_time))

            if self.global_steps == self.args.max_steps:
                break

        self.accelerator.print("auc@best on eval set: %f" % (best_auc))

    def eval_model(self, eval_loader):
        self.model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            t = tqdm(eval_loader, disable= not self.accelerator.is_main_process)
            pred_result = []
            for _, data in enumerate(t):
                for i in range(len(data)):
                    try:
                        data[i] = data[i].to(self.device)
                    except:
                        pass

                label, bag_name, scope, bag, args = data[0], data[1], data[2], data[3], data[4:]
                logits = self.model(None, scope, *args, train=False, bag_size=0)

                bag_name = torch.LongTensor(bag_name).to(self.device)

                logits = self.accelerator.gather(logits)
                label = self.accelerator.gather(label)
                bag_name = self.accelerator.gather(bag_name)

                logits = logits.cpu().numpy()
                label = label.cpu().numpy()
                bag_name = bag_name.cpu().numpy()

                for i in range(len(logits)):
                    for relid in range(self.model.module.num_class):
                        if relid != 0:
                            pred_result.append({
                                'entpair': bag_name[i][:2],
                                'relation': self.model.module.id2rel[relid],
                                'score': logits[i][relid]
                            })
            result = eval_loader.dataset.eval(pred_result)
        return result

    def load_model(self, ckpt_path):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(torch.load(ckpt_path))

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)
