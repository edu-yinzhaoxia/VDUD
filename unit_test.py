import unittest
import numpy as np
import torch


class ModelTest(unittest.TestCase):
    def test_gpu(self):
        import torch
        from pprint import pprint
        pprint(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True))
