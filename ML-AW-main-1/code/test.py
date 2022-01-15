import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import json


def main(config):
    
    print(json.dumps(config['data_loader']['args']['test_selected']))

    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        train_selected = config['data_loader']['args']['test_selected'],
        data_balance=0,
        data_normalization = config['data_loader']['args']['data_normalization'],
        batch_size=config['test_batch_size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    #logger.info(model)

    # get function handles of loss and metrics
    loss_fn = [getattr(module_loss, los) for los in config['loss']]
    loss_fn = loss_fn[0]

    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            # 如果是DAE，取第二个为output，忽略第一个，第一个是decoder的结果，在test阶段不会用
            if "DAE" in config['arch']['type'] or "Encoding" in config['arch']['type']:
                _, output = model(data)
            # 如果不是DAE，网络直接输出output
            elif "DIAE" in config['arch']['type']:
                data_pred, output, w_loss, l_loss = model(data)
            else:
                output = model(data)

            if "DIAE" in config['arch']['type']:
                loss = loss_fn(x_act=data, x_pred=data_pred, 
                                      y_act=target, y_pred=output, 
                                      l_loss=l_loss, w_loss=w_loss)
            else:
                loss = loss_fn(output, target)
            
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
    test_selected = config['data_loader']['args']['test_selected']
    for i in test_selected:
        config['data_loader']['args']['test_selected'] = [i]
        main(config)
