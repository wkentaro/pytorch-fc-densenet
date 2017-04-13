#!/usr/bin/env python

import os
import os.path as osp

import click
import torch
import torchfcn
import yaml

import torchfcdense


here = osp.dirname(osp.abspath(__file__))


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
def main(config_file):
    config = yaml.load(open(config_file))

    prefix = osp.splitext(osp.basename(config_file))[0]
    out = '%s' % prefix
    for key, value in config.items():
        out += '_%s-%s' % (key.upper(), str(value))
    out = osp.join(here, 'logs', out)
    if not osp.exists(out):
        os.makedirs(out)
    config['out'] = out[len(here) + 1:]
    config['config_file'] = config_file
    yaml.dump(config, open(osp.join(out, 'config.yaml'), 'w'))

    cuda = torch.cuda.is_available()

    seed = 1
    max_iter = 1000000

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # 1. dataset

    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SBDClassSeg(root, split='train', transform=True),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.VOC2011ClassSeg(
            root, split='seg11valid', transform=True),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    model = torchfcdense.models.FCDense(
        depths=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4],
        growth_rates=16,
        n_classes=21,
        drop_rate=0.2,
    )
    start_epoch = 0
    if config.get('resume'):
        checkpoint = torch.load(config['resume'])
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = getattr(torch.optim, config['optimizer'])(
        model.parameters(), lr=config['lr'],
        weight_decay=config['weight_decay'])
    if config.get('resume'):
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=config['out'],
        max_iter=max_iter,
        size_average=True,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_epoch * len(train_loader)
    trainer.train()


if __name__ == '__main__':
    main()
