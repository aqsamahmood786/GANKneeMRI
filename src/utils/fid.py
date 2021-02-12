# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:46:08 2020

@author: adds0
"""    
  
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from multiprocessing import cpu_count

import numpy as np
import torch
import torchvision.transforms as TF
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from pytorch_fid.inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))


class ImagesPathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        if path.endswith('.npy'):
            img = np.load(path).astype(np.float32)
        else:
            img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

class FidCalculator:
    def __init__(self, model,dims):
        self.model = model
        self.dims = dims
        
    def calculate_fid_given_paths(self,paths, batch_size, device):
        """Calculates the FID of two paths"""
        for p in paths:
            if not os.path.exists(p):
                raise RuntimeError('Invalid path: %s' % p)
    
        m1, s1 = self._compute_statistics_of_path(paths[0], self.model, batch_size,
                                             self.dims, device)
        m2, s2 = self._compute_statistics_of_path(paths[1], self.model, batch_size,
                                             self.dims, device)
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
    
        return fid_value
        
    def calculat_activations(self,files, model, batch_size=1, dims=2048, device='cpu'):
        """Calculates the activations of the pool_3 layer for all images.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : Batch size of images for the model to process at once.
                         Make sure that the number of samples is a multiple of
                         the batch size, otherwise some samples are ignored. This
                         behavior is retained to match the original FID score
                         implementation.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
           activations of the given tensor when feeding inception with the
           query tensor.
        """
        model.eval()
    
        if batch_size > len(files):
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = len(files)
    
        ds = ImagesPathDataset(files, transforms=TF.Compose([TF.ToPILImage(),TF.ToTensor()]))
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         drop_last=False, num_workers=cpu_count())
    
        pred_arr = np.empty((len(files), dims))
    
        start_idx = 0
    
        for batch in tqdm(dl):
            batch = batch.to(device)
    
            with torch.no_grad():
                pred = model(batch)[0]
    
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
    
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
    
            start_idx = start_idx + pred.shape[0]
    
        return pred_arr
    
    
    def calculate_frechet_distance(self,mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """
    
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
    
        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
    
    
    def calculate_activation_statistics(self,files, model, batch_size=1, dims=2048, device='cpu'):
        """Calculation of the statistics used by the FID.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : The images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size
                         depends on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                   the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                   the inception model.
        """
        act = self.calculat_activations(files, model, batch_size, dims, device)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma
    
    
    def _compute_statistics_of_path(self,path, model, batch_size, dims, device):
        if path.endswith('.npz'):
            f = np.load(path)
            m, s = f['mu'][:], f['sigma'][:]
            f.close()
        else:
            path = pathlib.Path(path)
            files = list(path.glob('*.jpg')) + list(path.glob('*.png'))+ list(path.glob('*.npy'))
            m, s = self.calculate_activation_statistics(files, model, batch_size,
                                                   dims, device)
    
        return m, s
    
    


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)
        
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[arg.dims]
    
    model = InceptionV3([block_idx]).to(device)
    
    fid_calculator = FidCalculator(model,arg.dims)

    fid_value = fid_calculator.calculate_fid_given_paths(args.path,
                                          args.batch_size,
                                          device)
    print('FID: ', fid_value)


if __name__ == '__main__':
    main()
