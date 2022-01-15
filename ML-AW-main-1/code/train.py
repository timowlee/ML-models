import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from trainer.trainer_ae import Trainer_AE
from trainer.trainer_ae_encoder import Trainer_AE_Encoder
from trainer.trainer_diae import Trainer_DIAE
from utils import prepare_device

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):

    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    if "Encoding" in config['arch']['type']:
        model_1 = config.init_obj('arch', module_arch)
    else: 
        model = config.init_obj('arch', module_arch)
        logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    if "Encoding" in config['arch']['type']:
        model_1 = model_1.to(device)
        if len(device_ids) > 1:
            model_1 = torch.nn.DataParallel(model_1, device_ids=device_ids)
    else:
        model = model.to(device)
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = [getattr(module_loss, los) for los in config['loss']]
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    if "Encoding" in config['arch']['type']:
        trainable_params = filter(lambda p: p.requires_grad, model_1.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    else:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
        
    if "Encoding" in config['arch']['type']:
        trainer = Trainer_AE_Encoder(model_1, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            data_loader=data_loader,
                            valid_data_loader=valid_data_loader,
                            lr_scheduler=lr_scheduler)
    elif "DAE" in config['arch']['type']:
        trainer = Trainer_AE(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            data_loader=data_loader,
                            valid_data_loader=valid_data_loader,
                            lr_scheduler=lr_scheduler)

    elif "DIAE" in config['arch']['type']:
        trainer = Trainer_DIAE(model, criterion[0], metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    else:
        trainer = Trainer(model, criterion[0], metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
