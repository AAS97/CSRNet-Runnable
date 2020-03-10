"""
This file is used to run the model
"""
import time
import os
import numpy as np
import torch
from PIL import Image
import imageio

from toolbox import utils, metrics

'''
    .                       o8o
  .o8                       `"'
.o888oo oooo d8b  .oooo.   oooo  ooo. .oo.
  888   `888""8P `P  )88b  `888  `888P"Y88b
  888    888      .oP"888   888   888   888
  888 .  888     d8(  888   888   888   888
  "888" d888b    `Y888""8o o888o o888o o888o
'''


def train(args, train_loader, model, criterion, optimizer, logger, epoch,
          eval_score=None, print_freq=10, tb_writer=None):

    # switch to train mode
    model.train()
    meters = logger.reset_meters('train')
    meters_params = logger.reset_meters('hyperparams')

    # TODO
    # Pas sur de l'implémentation du lr scheduler
    meters_params['learning_rate'].update(optimizer.param_groups[0]['lr'])
    end = time.time()

    # From CSRNet train
    for i, (input, target) in enumerate(train_loader):
        batch_size = input.size(0)
        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)

        input, target = input.to(args.device).requires_grad_(), target.type(
            torch.FloatTensor).unsqueeze(0).to(args.device)
        output = model(input)

        loss = criterion(output, target)

        # Update loss loger
        meters['loss'].update(loss.data.item(), n=batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        # TODO
        # eval_score is supposed to come from metric.accuracy ---> no equivalent in CSRNet
        # measure accuracy and record loss
        if eval_score is not None:
            acc1 = eval_score(output, target)
            meters['mae'].update(acc1['mae'], n=batch_size)
            meters['mse'].update(acc1['mse'], n=batch_size)

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LR {lr.val:.2e}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE@1 {mae1.val:.3f} ({mae1.avg:.3f})\t'
                  'MSE@1 {mse1.val:.3f} ({mse1.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=meters['batch_time'],
                      data_time=meters['data_time'], lr=meters_params['learning_rate'], loss=meters['loss'], mae1=meters['mae'], mse1=meters['mse']))

        if True == args.short_run:
            if 12 == i:
                print(' --- running in short-run mode: leaving epoch earlier ---')
                break

    if args.tensorboard:
        tb_writer.add_scalar('mae/train', meters['mae'].avg, epoch)
        tb_writer.add_scalar('mse/train', meters['mse'].avg, epoch)
        tb_writer.add_scalar('loss/train', meters['loss'].avg, epoch)
        tb_writer.add_scalar(
            'learning rate', meters_params['learning_rate'].val, epoch)

    logger.log_meters('train', n=epoch)
    logger.log_meters('hyperparams', n=epoch)


'''
                      oooo
                      `888
oooo    ooo  .oooo.    888
 `88.  .8'  `P  )88b   888
  `88..8'    .oP"888   888
   `888'    d8(  888   888
    `8'     `Y888""8o o888o
'''


def validate(args, val_loader, model, criterion, logger, epoch, eval_score=None, print_freq=10, tb_writer=None):

    # switch to evaluate mode
    model.eval()
    meters = logger.reset_meters('val')
    end = time.time()
    # TODO
    # no histogram because no classif
    # hist = np.zeros((args.num_classes, args.num_classes))

    # no sure if this should stay
    res_list = {}
    grid_pred = None

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            batch_size = input.size(0)

            meters['data_time'].update(time.time()-end, n=batch_size)

            input, target = input.to(
                args.device).requires_grad_(), target.to(args.device)

            output = model(input)

            # TODO
            # why is the loss calculated in validation ?
            loss = criterion(output, target)
            meters['loss'].update(loss.data.item(), n=batch_size)

            # measure accuracy and record loss
            if eval_score is not None:
                acc1 = eval_score(output, target)
                meters['mse'].update(acc1['mse'], n=batch_size)
                meters['mae'].update(acc1['mae'], n=batch_size)

            # measure elapsed time
            end = time.time()
            meters['batch_time'].update(time.time() - end, n=batch_size)

            # save samples from first mini-batch for qualitative visualization
            if i == 0:
                #    utils.save_res_grid(input.detach().to('cpu').clone(), val_loader, pred,
                #             target.to('cpu').clone(),
                #             out_fn=os.path.join(args.log_dir,'pics', '{}_watch_mosaic_pred_labels.jpg'.format(args.name)))

                utils.save_input_output_img(target.detach().to('cpu').clone(),
                                            output.to('cpu').clone(), input.detach().to(
                                                'cpu'), val_loader,
                                            out_fn=os.path.join(args.log_dir, 'pics', '{}_watch_mosaic_pred_labels.png'.format(args.name)))

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE@1 {score_mae.val:.3f} ({score_mae.avg:.3f})\t'
                      'MSE@1 {score_mse.val:.3f} ({score_mse.avg:.3f})'.format(
                          i, len(val_loader), batch_time=meters['batch_time'], loss=meters['loss'],
                          score_mae=meters['mae'], score_mse=meters['mse']), flush=True)

            if True == args.short_run:
                if 12 == i:
                    print(' --- running in short-run mode: leaving epoch earlier ---')
                    break

    print(' * Validation set: Average loss {:.4f}, MAE {:.3f}, MSE {:.3f} \n'.format(
        meters['loss'].avg, meters['mae'].avg, meters['mse'].val))

    logger.log_meters('val', n=epoch)

    if args.tensorboard:
        tb_writer.add_scalar('mae/val', meters['mae'].avg, epoch)
        tb_writer.add_scalar('mse/val', meters['mse'].avg, epoch)
        tb_writer.add_scalar('loss/train', meters['loss'].avg, epoch)
        im = imageio.imread('{}'.format(os.path.join(
            args.log_dir, 'pics', '{}_watch_mosaic_pred_labels.png'.format(args.name))))
        tb_writer.add_image('Image', im.transpose((2, 0, 1)), epoch)
    return meters['mae'].val, meters['mse'].val, meters['loss'].val


'''
    .                          .
  .o8                        .o8
888oo  .ooooo.   .oooo.o .o888oo
  888   d88' `88b d88(  "8   888
  888   888ooo888 `"Y88b.    888
  888 . 888    .o o.  )88b   888 .
  "888" `Y8bod8P' 8""888P'   "888"
'''
# Not used for the moment ---> use validate instead


def test(args, eval_data_loader, model, criterion, epoch, eval_score=None,
         output_dir='pred', has_gt=True, print_freq=10, tb_writer=None):

    model.eval()
    meters = metrics.make_meters()
    end = time.time()
    res_list = []
    with torch.no_grad():
        for i, (input, target) in enumerate(eval_data_loader):
            # print(input.size())
            batch_size = input.size(0)
            meters['data_time'].update(time.time()-end, n=batch_size)

            # label = target_class.numpy()
            input, target = input.to(
                args.device).requires_grad_(), target.to(args.device)

            output = model(input)

            loss = criterion(output, target)

            meters['loss'].update(loss.data.item(), n=batch_size)

            # measure accuracy and record loss
            if eval_score is not None:
                acc1 = eval_score(output, target)
                meters['mae'].update(acc1['mae'], n=batch_size)
                meters['mse'].update(acc1['mse'], n=batch_size)

            res_list.append([input.cpu().data.numpy(), target.cpu(
            ).data.numpy(), output.cpu().data.numpy()])

            end = time.time()
            meters['batch_time'].update(time.time() - end, n=batch_size)

            end = time.time()
            print('Testing: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE@1 {score_mae.val:.3f} ({score_mae.avg:.3f})\t'
                  'MSE@1 {score_mse.val:.3f} ({score_mse.avg:.3f})'.format(
                      i, len(eval_data_loader), batch_time=meters['batch_time'], loss=meters['loss'],
                      score_mae=meters['mae'], score_mse=meters['mse']), flush=True)

            if True == args.short_run:
                if 12 == i:
                    print(' --- running in short-run mode: leaving epoch earlier ---')
                    break

    utils.save_res_img(res_list, out_fn=os.path.join(
        args.log_dir, 'pics', '{}_results_pic.png'.format(args.name)))
    metrics.save_meters(meters, os.path.join(
        args.log_dir, 'test_results_ep{}.json'.format(epoch)), epoch)
    utils.save_res_list(res_list, os.path.join(
        args.res_dir, 'test_results_list_ep{}.json'.format(epoch)))

    if args.tensorboard:
        tb_writer.add_scalar('mae/test', meters['mae'].avg, epoch)
        tb_writer.add_scalar('mse/test', meters['mse'].avg, epoch)
        tb_writer.add_scalar('loss/test', meters['loss'].avg, epoch)
        im = imageio.imread('{}'.format(os.path.join(
            args.log_dir, 'pics', '{}_results_pic.png'.format(args.name))))
        tb_writer.add_image('Image/test', im, epoch)
