from mil_pytorch.model import eval, data_utils
import mil_pytorch.model.mil_pytorch as mil
from mil_pytorch.train import train
from mil_pytorch.create_dataset import create_bags

import numpy

import torch
from torch.utils.data import DataLoader

from sklearn.datasets import make_classification

import time

