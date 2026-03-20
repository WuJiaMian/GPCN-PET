import os
import sys
import math
import argparse
import random
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
from utils import utils_early_stopping
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from tensorboardX import SummaryWriter
from collections import OrderedDict
import time

def main(json_path=''):
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='train_wdpcnet_CCnpi_G1D30.json', help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    os.environ['TORCH_HOME'] = 'path'
    opt = option.parse(parser.parse_args().opt, is_train=True)
    os.environ['RANK'] = str(0)
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()


    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],  net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)
    if opt['rank'] == 0:
        option.save(opt)
    opt = option.dict_to_nonedict(opt)
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))
        logger_tensorboard = SummaryWriter(os.path.join(opt['path']['log']))
    seed = opt['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    for phase, dataset_opt in opt['datasets'].items():

        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle= False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],

                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)
    model = define_Model(opt)
    model.init_train()
    if opt['train']['is_early_stopping']:
        early_stopping = utils_early_stopping.EarlyStopping(patience=opt['train']['early_stopping_num'])

    # record
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    # Initialize best PSNR and SSIM for saving best checkpoint
    best_psnr = -1.0
    best_ssim = -1.0

    for epoch in range(3000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):

            current_step += 1
            model.update_learning_rate(current_step)
           # time0 = time.time()
            model.feed_data(train_data)

            model.optimize_parameters(current_step)
           # time1 = time.time()
            #print(time1 - time0)
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                #print(time1 - time0)
                for k, v in logs.items():
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

                logger_tensorboard.add_scalar('Learning Rate', model.current_learning_rate(), global_step=current_step)
                logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss', logs['G_loss'], global_step=current_step)

                if 'G_loss_image' in logs.keys():
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss_image', logs['G_loss_image'], global_step=current_step)
                if 'G_loss_frequency' in logs.keys():
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss_frequency', logs['G_loss_frequency'], global_step=current_step)
                if 'G_loss_preceptual' in logs.keys():
                    logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_loss_preceptual', logs['G_loss_preceptual'], global_step=current_step)
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                img_dir_tmp_H = os.path.join(opt['path']['images'], 'tempH')
                util.mkdir(img_dir_tmp_H)
                img_dir_tmp_E = os.path.join(opt['path']['images'], 'tempE')
                util.mkdir(img_dir_tmp_E)
                img_dir_tmp_L = os.path.join(opt['path']['images'], 'tempL')
                util.mkdir(img_dir_tmp_L)
                test_results = OrderedDict()
                test_results['psnr'] = []
                test_results['ssim'] = []

                test_results['G_loss'] = []
                test_results['G_loss_image'] = []
                test_results['G_loss_frequency'] = []
                for idx, test_data in enumerate(test_loader):
                    with torch.no_grad():
                        model.feed_data(test_data)
                        model.check_windowsize()
                        model.test()
                        model.recover_windowsize()
                        results = model.current_results_gpu()

                        E_img = results['E']

                       # E_img = (torch.sqrt(results['E'][:,0,:,:]**2+results['E'][:,1,:,:]**2)).unsqueeze(1)
                        H_img = results['H']
                        L_img = results['L']
                        E_img = (torch.sqrt(results['E'][:, 0, :, :] ** 2 + results['E'][:, 1, :, :] ** 2)).unsqueeze(1)
                        L_img = util.tensor2float(L_img)
                        E_img = util.tensor2float(E_img)
                        H_img = util.tensor2float(H_img)

                        current_psnr = util.calculate_psnr_single(H_img, E_img, border=0)
                        current_ssim = util.calculate_ssim_single(H_img, E_img, border=0)
                        test_results['psnr'].append(current_psnr)
                        test_results['ssim'].append(current_ssim)

                #print(np.sum(np.abs(E_img - H_img)))
                ave_psnr = np.mean(test_results['psnr'])
                ave_ssim = np.mean(test_results['ssim'])
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}; Average Average SSIM : {:<.4f};'
                            .format(epoch, current_step, ave_psnr, ave_ssim))

                logger_tensorboard.add_scalar('VALIDATION PSNR', ave_psnr, global_step=current_step)
                logger_tensorboard.add_scalar('VALIDATION SSIM', ave_ssim, global_step=current_step)

                # Save best checkpoint if PSNR and SSIM are both better
                if ave_psnr > best_psnr and ave_ssim > best_ssim:
                    best_psnr = ave_psnr
                    best_ssim = ave_ssim
                    logger.info('Saving best checkpoint with PSNR: {:<.2f}, SSIM: {:<.4f}'.format(best_psnr, best_ssim))
                    model.save('best')
    print("Training Stop")
if __name__ == '__main__':
    main()
