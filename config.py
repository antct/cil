import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--dataset", type=str, default='nyt10d')
parser.add_argument("--model_name", type=str, default='')
parser.add_argument("--pretrain_path", type=str, default='pretrain/bert-base-uncased')
parser.add_argument("--seed", type=int, default=36)

parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--eval_batch_size", type=int, default=32)

parser.add_argument("--max_epoch", type=int, default=5)

parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--warmup_steps", type=int, default=500)

parser.add_argument("--max_steps", type=int, default=10000)
parser.add_argument("--max_grad_norm", type=float, default=5.0)
parser.add_argument("--grad_acc_steps", type=int, default=1)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--dropout_prob", type=float, default=0.1)
parser.add_argument("--temperature", type=float, default=0.1)

parser.add_argument("--weight_decay", type=float, default=1e-5)

parser.add_argument("--bag_size", type=int, default=0)
parser.add_argument("--max_bag_size", type=int, default=0)
parser.add_argument("--loss_weight", action='store_true')
parser.add_argument("--writer", action='store_true')

parser.add_argument("--mil", type=str, default='att')

args, unknown = parser.parse_known_args()
