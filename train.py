
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import pprint
import yaml

from data import *
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from data.custom import CustomDetection

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

######## ###################
# 
# argument parser
# 
#  ############################
parser = argparse.ArgumentParser( description='Single Shot MultiBox Detector Training With Pytorch')

train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--cfg', default="./config.yml", type=str,
                    help='cfg specify network architecture and training parameter')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,        
                    help='Resume training at this iter')
parser.add_argument('--cuda', default=False, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

######## ###################
# 
# train
# 
#  ############################
def train():
    ######## config ########
    with open(args.cfg, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        print('\n=========\nconfig \n==========\n')
        pprint.pprint(cfg)

    ######## cuda ########
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    ######## save directory ########
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    ######## load dataset ########
    dataset = CustomDetection(  cfg = cfg)
    ######## logging by visdom instead of tensorboard ########
    if args.visdom:
        print("visdom")
        import visdom
        viz = visdom.Visdom()

    ######## form network ########
    ssd_net = build_ssd(cfg, 'train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net
    net.train()

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    ######## resume network or not ########
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(cfg['basenet_checkpt_path'])
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    ############## load loss function and optimizer ####################
    optimizer = optim.SGD(  net.parameters(), 
                            lr=cfg['initial_lr'],
                            momentum=cfg['momentum'],
                            weight_decay=cfg['weight_decay'])

    criterion = MultiBoxLoss(cfg, cfg['num_classes'],
                             0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    # loss counters
    epoch = 0
    loc_loss = 0
    conf_loss = 0

    print('Loading the dataset...')

    epoch_size = len(dataset) // cfg['batch_size']
    print('Training SSD on:', dataset.name)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot(viz,'Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot(viz,'Epoch', 'Loss', vis_title, vis_legend)

    ######## dataloader ########
    data_loader = data.DataLoader(dataset, cfg['batch_size'],
                                  num_workers=cfg['num_workers'],
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)


    ######## training loop ########
    # create batch iterator
    batch_iterator = iter(data_loader)
    print(" epoch_size = " , epoch_size)
    print(" Dataset size = " , len(batch_iterator))

    checkpt_save_iter = cfg['checkpt_save_iter']
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(viz, epoch,  loc_loss, conf_loss,epoch_plot, 'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(cfg['initial_lr'], optimizer, cfg['gamma'], step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            print("No data left. Repeat from start")
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = [t.cuda() for t in targets] # from a list of tensors to a list of tensor cuda 

        if cfg['isDetailLog']:
           # print("images = {}".format(images.type()))
           # print("targets[0] = {}".format(targets[0].type()))

            img = images[0].cpu().numpy().transpose(1, 2, 0)
           # print("img = {}".format(img.shape))
            path = './log/srcimg_{}.bmp'.format(iteration)
            cv2.imwrite(path,img)

            path = './log/target_{}.txt'.format(iteration)
            with open(path,'w') as f:
                for t in targets:
                    savetarget = t.cpu().detach().numpy()
                    np.savetxt(f, savetarget, fmt='%.2f')

        # forward
        t0 = time.time()
        out = net(images)
        #print("out = {} {} {}".format(out[0].size(), out[1].size(), out[2].size()))

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()

        loc_loss += loss_l
        conf_loss += loss_c

        print('epoch {} || iter {}  || Loss: {} || loss_l = {}, loss_c = {} || timer {} sec'.format(epoch, iteration,loss,loss_l,loss_c,(t1 - t0)) )
        #raise NameError('raise error to debug first iteration')
        if args.visdom:
            update_vis_plot(viz,iteration, loss_l, loss_c, iter_plot, 'append')

        if iteration != 0 and iteration % checkpt_save_iter == 0:
            print('Saving state, iter:', iteration)
            path = '{}/{}_{}.pth'.format(args.save_folder,cfg['name'],repr(iteration))
            torch.save(ssd_net.state_dict(), path)

    path = '{}/{}_end.pth'.format(args.save_folder,cfg['name'])
    torch.save(ssd_net.state_dict(),  path)

######## ###################
# 
# util
# 
#  ############################
def adjust_learning_rate(initial_lr, optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = initial_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(viz, _xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

def update_vis_plot(viz, iteration, loc, conf, window, update_type, average_size=1):
    if window is not None:
        viz.line(
            X=torch.ones((1, 3)).cpu() * iteration,
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / average_size,
            win=window,
            update=update_type
        )
        
######## ###################
# 
# main
# 
#  ############################
if __name__ == '__main__':
    train()
