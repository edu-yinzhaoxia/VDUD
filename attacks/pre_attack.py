import os
import torchattacks
from foolbox.attacks import HopSkipJump
from torchvision.transforms import transforms

import foolbox as fb


os.environ["GIT_PYTHON_REFRESH"] = "quiet"



def FGSM(model, images, labels):
    attack = torchattacks.FGSM(model, eps=16 / 255)
    perturbed_images = attack(images, labels)
    return perturbed_images

def autoattack(model, images, labels):
    attack = torchattacks.AutoAttack( model, norm='Linf', eps=8/255, version='plus', n_classes=10)
    perturbed_images = attack(images, labels)
    return perturbed_images

def BIM(model, images, labels):
    attack = torchattacks.BIM(model, eps=16/255, alpha=2 / 255, steps=10)
    perturbed_images = attack(images, labels)
    return perturbed_images

# def f_CW_l2(model, images, labels):
#     fmodel = fb.PyTorchModel(model, bounds=(0, 1))
#     attack = fb.attacks.L2CarliniWagnerAttack()
#     _, perturbed_images, success = attack(fmodel, images, labels, epsilons=[0.1])
#     return perturbed_images

def CW(model, images, labels):
    attack = torchattacks.CW(model, c=1, kappa=0, steps=200, lr=0.01)
    perturbed_images = attack(images, labels)
    return perturbed_images


def f_PGD_l2(model, images, labels):
    attack = torchattacks.PGDL2(model, eps=1, alpha=0.01, steps=100, random_start=True)
    perturbed_images = attack(images, labels)
    return perturbed_images


def f_PGD_linf(model, images, labels):
    attack = torchattacks.PGD(model, eps=64 / 255, alpha=0.01, steps=100, random_start=True)
    perturbed_images = attack(images, labels)
    return perturbed_images





def JSMA(model, images, labels):
    attack = torchattacks.JSMA(model, theta=1.0, gamma=0.1)
    perturbed_images = attack(images, labels)
    return perturbed_images

# 黑盒对抗攻击
def PA(model, images, labels):
    attack = torchattacks.OnePixel(model)
    perturbed_images = attack(images, labels)
    return perturbed_images

def SA(model, images, labels):
    attack = torchattacks.Square(model, eps=0.30, n_queries=200)
    perturbed_images = attack(images, labels)
    return perturbed_images








# 可迁移攻击

def MIM(model, images, labels):
    attack = torchattacks.MIFGSM(model, eps=16 / 255, steps=10, decay=1.0)
    perturbed_images = attack(images, labels)
    return perturbed_images



def DIM(model, images, labels):
    attack = torchattacks.DIFGSM(model, eps=8 / 255, alpha=1/255, steps=20, decay=1.0, diversity_prob=0.7)
    perturbed_images = attack(images, labels)
    return perturbed_images
#
#
def TIM(model, images, labels):
    attack = torchattacks.TIFGSM(model, eps=4 / 255, alpha=1/255, steps=20, decay=1.0)
    perturbed_images = attack(images, labels)
    return perturbed_images
#
def SINIFGSM(model, images, labels):
    attack = torchattacks.SINIFGSM(model, eps=2 / 255,  alpha=1/255, steps=40, decay=1.0, m=5)
    perturbed_images = attack(images, labels)
    return perturbed_images
#
def VNI(model, images, labels):
    attack = torchattacks.VNIFGSM(model, eps=4 / 255, alpha=1/255, steps=10, decay=1.0, N=20)
    perturbed_images = attack(images, labels)
    return perturbed_images
#
#
def VMI(model, images, labels):
    attack = torchattacks.VMIFGSM(model, eps=8 / 255, alpha=1/255, steps=10, decay=1.0, N=20)
    perturbed_images = attack(images, labels)
    return perturbed_images


