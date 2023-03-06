'''
reference : https://github.com/mseitzer/pytorch-fid
https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

separate calculating FID and statistics
'''

import os
import pathlib

import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

from inception import InceptionV3


def calculate_statistics_for_given_paths(dataloader_dict, batch_size, device, dims):

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    for key in dataloader_dict.keys():
        print(f'calculating statistics for {key}')
        m, s = calculate_activation_statistics(dataloader_dict[key], model, dims, device)
        os.makedirs(f'./np_saves', exist_ok = True)
        np.save(f'./np_saves/{key}_m.npy', m)
        np.save(f'./np_saves/{key}_s.npy', s)
    
    print(f'saving statistics done for {len(list(dataloader_dict.keys()))}')

    
def calculate_activation_statistics(dataloader, model, dims, device):
    act = get_activations(dataloader, model, dims, device)
    print(act.shape)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def get_activations(dataloader, model, dims, device):
    model.eval()
    total = []
    print('total images : ', dataloader.dataset.__len__())

    #pred_arr = np.empty((dataloader.dataset.__len__(), dims))
    start_idx = 0
    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.cpu().data.numpy().reshape(pred.size(0), -1)
        total.append(pred)


    return np.vstack(total)




def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):


    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)







def main():
    args = munch.Munch()
    args.batch_size = 128
    args.num_workers = 4
    args.device = 'cuda'
    args.dims = 2048

    # path : Dict
    # first element is reference dir
    # [1, 2, 3, 4] then (1,2), (1,3), (1,4) so that compare 2,3,4 datasets
    # also statistics of 1(reference) will be saved
    args.path = {
        'test1' : './cifar10',
        'test2' : './cifar10_blur1',
        'test3' : './cifar10_blur3',
        'test4' : './cifar10_blur5',
    }

    # statistics for FID

    calculate_statistics_for_given_paths(args.path,
                                        args.batch_size,
                                        torch.device(args.device),
                                        args.dims,
                                        args.num_workers)


    # Calculate FID by statistics
    for idx, key in enumerate(args.path.keys()):
        if idx == 0:
            print(f'loading statistics {key}')
            ref_key = key
            ref_m = np.load(f'./np_saves/{key}_m.npy')
            ref_s = np.load(f'./np_saves/{key}_s.npy')
            continue
        else:
            print(f'loading statistics {key}')
            m = np.load(f'./np_saves/{key}_m.npy')
            s = np.load(f'./np_saves/{key}_s.npy')
        print(f'calculating FID for {key}')
        FID = calculate_frechet_distance(ref_m, ref_s, m, s, eps=1e-6)
        print(f'FID : {FID}   ({ref_key}, {key})')
if __name__ == '__main__':
    main()