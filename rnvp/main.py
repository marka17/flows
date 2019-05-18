import argparse
import logging
from time import monotonic

import torch
from torch import nn
from torch import optim

from model import RealNVP

from sklearn.datasets import make_circles
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)

def _parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--num-epoch', type=int, default=5000)
    parser.add_argument('--num-layers', type=int, default=20)
    parser.add_argument('--hidden-size', type=int, default=25)
    parser.add_argument('--hidden-dim', type=int, default=2)
    parser.add_argument('--log-train-step', type=int, default=500)

    return parser.parse_args()

def test(flow: nn.Module):
    fix, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=150)

    with torch.no_grad():
        axes[1].scatter(*zip(*flow.sample(2000).tolist()))

    axes[0].scatter(*zip(*make_circles(n_samples=1000, factor=0.5, noise=0.05)[0]))

    plt.savefig('result.pdf')


def run_epoch(flow: nn.Module,
              optimizer: optim.Optimizer,
              batch_size: int = 100) -> float:

    x, _ = make_circles(batch_size, factor=0.5, noise=.05)
    x = torch.from_numpy(x).float()

    # print(flow.log_prob(x))
    loss = -torch.mean(flow.log_prob(x))
    # print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def main(args):
    scale_network = nn.Sequential(
        nn.Linear(args.hidden_dim, args.hidden_size),
        nn.LeakyReLU(),
        nn.Linear(args.hidden_size, args.hidden_size),
        nn.LeakyReLU(),
        nn.Linear(args.hidden_size, args.hidden_dim),
        nn.Tanh())

    translation_network = nn.Sequential(
        nn.Linear(args.hidden_dim, args.hidden_size),
        nn.LeakyReLU(),
        nn.Linear(args.hidden_size, args.hidden_size),
        nn.LeakyReLU(),
        nn.Linear(args.hidden_size, args.hidden_dim))

    masks = torch.tensor([1, 0] * args.num_layers, dtype=torch.float)
    masks = torch.stack((masks, 1 - masks), dim=1)

    prior = torch.distributions.MultivariateNormal(
        torch.zeros(args.hidden_dim),
        torch.eye(args.hidden_dim))

    flow = RealNVP(scale_network, translation_network, masks, prior)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad == True, flow.parameters()))


    for epoch in range(args.num_epoch):

        loss = run_epoch(flow, optimizer, args.batch_size)

        if epoch % args.log_train_step == 0:
            logging.info(" Epoch: {} | Loss: {}".format(epoch, loss))

    test(flow)

if __name__ == '__main__':
    args = _parser()
    main(args)
