'''
Ḿisc utility functions
'''
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from PIL import Image
import json
import pickle
import matplotlib as mpl
mpl.use('Agg')


'''
 o8o               o8o      .
 `"'               `"'    .o8
oooo  ooo. .oo.   oooo  .o888oo
`888  `888P"Y88b  `888    888
 888   888   888   888    888
 888   888   888   888    888 .
o888o o888o o888o o888o   "888"
'''


def setup_env(args):
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True


# create necessary folders and config files
def init_output_env(args):
    check_dir(os.path.join(args.data_dir, 'runs'))
    check_dir(args.log_dir)
    check_dir(os.path.join(args.log_dir, 'pics'))
    check_dir(os.path.join(args.log_dir, 'tensorboard'))
    # check_dir(os.path.join(args.log_dir, 'watch'))
    check_dir(args.res_dir)
    with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f)


'''
 o8o
 `"'
oooo  ooo. .oo.  .oo.    .oooo.    .oooooooo  .ooooo.   .oooo.o
`888  `888P"Y88bP"Y88b  `P  )88b  888' `88b  d88' `88b d88(  "8
 888   888   888   888   .oP"888  888   888  888ooo888 `"Y88b.
 888   888   888   888  d8(  888  `88bod8P'  888    .o o.  )88b
o888o o888o o888o o888o `Y888""8o `8oooooo.  `Y8bod8P' 8""888P'
                                  d"     YD
                                  "Y88888P'
'''


def to_img(tensor):
    toPIL = torchvision.transforms.ToPILImage()
    img = toPIL(tensor[0, :, :, :])
    img = np.array(img)
    return img


def save_res_grid(input, val_loader, pred, target, out_fn, max_images=6):

    if max_images < input.size(0):
        input = input[:max_images, :, :, :]
        pred = pred[:max_images]
        target = target[:max_images]

    raw_batch = val_loader.dataset.unprocess_batch(input)

    imlist = []
    for i in range(raw_batch.size(0)):
        imlist.append(raw_batch[i, :, :, :].squeeze().mul(255).clamp(
            0, 255).byte().permute(1, 2, 0).to('cpu').numpy())

    plt.figure(figsize=(24, 24))
    for idx, _ in enumerate(imlist):
        plt.subplot(1, 6, idx+1)
        plt.imshow(imlist[idx])
        plt.xlabel(
            f'pred: {val_loader.dataset.classnames[pred[idx]]}/\ngt: {val_loader.dataset.classnames[target[idx]]}')
        plt.ylabel('valid images')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    plt.savefig(out_fn, bbox_inches='tight', dpi=200)
    plt.gcf().clear()
    plt.close()


def save_input_output_img(target, output, input, val_loader, out_fn, max_images=6):

    if max_images < output.size(0):
        input = input[:max_images, :, :, :]
        output = output[:max_images, :, :, :]
        target = target[:max_images, :, :, :]

    # raw_batch = val_loader.dataset.unprocess_batch(input)

    imlist = []
    for i in range(target.size(0)):
        imlist.append(to_img(input))
        imlist.append(to_img(target))
        imlist.append(to_img(output))

# .permute(1, 2, 0)
    plt.figure(figsize=(24, 24))
    for idx, _ in enumerate(imlist):
        plt.subplot(1, 6, idx+1)
        plt.imshow(imlist[idx])
        if idx % 3 == 0:
            plt.ylabel('input image')
        elif idx % 3 == 1:
            plt.ylabel('groundtruth image')
        else:
            plt.ylabel('output image')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    plt.savefig(out_fn, bbox_inches='tight', dpi=200)
    plt.gcf().clear()
    plt.close()


def save_res_img(res, out_fn):

    imlist = []
    for test in res:
        input, target, output = test
        imlist.append(np.transpose(input[0], (1, 2, 0)))
        imlist.append(np.squeeze(target[0]))
        imlist.append(np.squeeze(output[0]))

# .permute(1, 2, 0)
    plt.figure(figsize=(24, 48))
    for idx, _ in enumerate(imlist):
        plt.subplot(1+(len(imlist)//3), 3, idx+1)
        plt.imshow(imlist[idx])
        if idx % 3 == 0:
            plt.ylabel('input image')
        elif idx % 3 == 1:
            plt.ylabel('groundtruth image')
        else:
            plt.ylabel('output image')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    plt.savefig(out_fn, bbox_inches='tight', dpi=200)
    plt.gcf().clear()
    plt.close()


'''
                   o8o
                   `"'
ooo. .oo.  .oo.   oooo   .oooo.o  .ooooo.
`888P"Y88bP"Y88b  `888  d88(  "8 d88' `"Y8
 888   888   888   888  `"Y88b.  888
 888   888   888   888  o.  )88b 888   .o8
o888o o888o o888o o888o 8""888P' `Y8bod8P'
'''

# check if folder exists, otherwise create it


def check_dir(dir_path):
    dir_path = dir_path.replace('//', '/')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_res_list(res_list, fn):
    with open(fn, 'wb') as f:
        pickle.dump(res_list, f)


def count_params(model):
    return sum([p.data.nelement() for p in model.parameters()])
