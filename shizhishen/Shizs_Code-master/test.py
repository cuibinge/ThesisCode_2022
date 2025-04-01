from argparse import ArgumentParser
import torch

from src.config.base_config import get_config
from src.models.evaluator import *
from src.utils import *
print(torch.cuda.is_available())


"""
eval the CD model
"""

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='train218', type=str)
    parser.add_argument('--print_models', default=False, type=bool, help='print models')
    parser.add_argument('--is_val', default=True, type=bool, help='print models')

    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='Glacier', type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="vis", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='VPGCD-Net', type=str,
                        help='VPGCD-Net| EACD-Net')

    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)
    config = get_config()
    #  checkpoints dir
    args.checkpoint_dir = os.path.join('weights', args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join('result', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False, is_Val=args.is_val,
                                  split=args.split)
    # model = CDEvaluator(args=args, dataloader=dataloader, config=config)   #train03
    model = CDEvaluator(args=args, dataloader=dataloader)  # train01
    model.eval_models(checkpoint_name=args.checkpoint_name)


if __name__ == '__main__':
    main()

