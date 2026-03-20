import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
from utils.utils_swinmr import *
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import os


def generate_mask(img_height, img_width, radius, center_x, center_y):
    y, x = np.ogrid[0:img_height, 0:img_width]
    # circle mask
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    return mask


def imgtofrequency(img):  # input:[320,320,1]
    zero = torch.zeros_like(img)
    img = torch.cat((img, zero), dim=-1)
    img = torch.view_as_complex(img)
    ksp = torch.fft.fftshift(torch.fft.fft2(img, dim=(0, 1)), dim=(0, 1))
    ksp = torch.view_as_real(ksp)
    return ksp


class DatasetCCsagnpi(data.Dataset):

    def __init__(self, opt):
        super(DatasetCCsagnpi, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt = opt
        self.n_channels = self.opt['n_channels']
        self.patch_size = self.opt['H_size']
        self.is_noise = self.opt['is_noise']
        self.noise_level = self.opt['noise_level']
        self.noise_var = self.opt['noise_var']
        self.is_mini_dataset =  1
        self.mini_dataset_prec = 0.2
        # Get lists of all .mat file paths
        paths_H_files = sorted(
            [os.path.join(opt['dataroot_H'], f) for f in os.listdir(opt['dataroot_H']) if f.endswith('.mat')])
        paths_L_files = sorted(
            [os.path.join(opt['dataroot_L'], f) for f in os.listdir(opt['dataroot_L']) if f.endswith('.mat')])

        assert paths_H_files, 'Error: High-resolution data path is empty.'
        assert paths_L_files, 'Error: Low-resolution data path is empty.'
        assert len(paths_H_files) == len(paths_L_files), 'Error: Mismatch in number of H and L data files.'

        self.mask = generate_mask(128, 128, 64, 64, 64)
        self.mask = np.expand_dims(self.mask, axis=-1)
        self.target_size = (200, 200)  # <--- 定义目标尺寸
        # Create a manifest of valid slices instead of loading them into memory
        self.slice_info = []
        print("Scanning dataset to build slice manifest...")
        for h_path, l_path in zip(paths_H_files, paths_L_files):
            try:
                # Load headers to get dimensions without loading all data
                h_mat = sio.loadmat(h_path, variable_names=['output'])['output']
                l_mat = sio.loadmat(l_path, variable_names=['output'])['output']
                if abs(h_mat.shape[2]-l_mat.shape[2])<10 and (h_mat.shape[2] != l_mat.shape[2]):
                    sub_slice = abs(h_mat.shape[2]-l_mat.shape[2])
                    if (h_mat.shape[2]>l_mat.shape[2]):
                        h_mat = h_mat[sub_slice:]
                    else:
                        l_mat = l_mat[sub_slice:]
                #print(h_mat.shape[0],l_mat.shape[0])
                # Assert that the number of slices in paired files is the same
                assert h_mat.shape[2] == l_mat.shape[2], f"Slice count mismatch in {h_path} and {l_path}"

                for i in range(h_mat.shape[2]):
                    # Check if max value equals min value for both H and L slices
                    h_slice = h_mat[:, : , i]
                    l_slice = l_mat[:, : , i]
                    if h_slice.max() == h_slice.min() or l_slice.max() == l_slice.min():
                        continue  # Skip this slice

                    # If valid, add its info to the manifest
                    self.slice_info.append({
                        'H_path': h_path,
                        'L_path': l_path,
                        'slice_index': i
                    })
            except Exception as e:
                print(f"Could not process file pair: {h_path}, {l_path}. Error: {e}")

        print(f"Found {len(self.slice_info)} valid slices.")
        assert len(self.slice_info) > 0, "Error: No valid slices found in the dataset."
        

    def __getitem__(self, index):
        time0 = time.time()

        # Get the information for the requested slice from the manifest
        slice_meta = self.slice_info[index]
        h_path = slice_meta['H_path']
        l_path = slice_meta['L_path']
        slice_idx = slice_meta['slice_index']

        # --- On-demand loading and processing ---
        # Load High-Resolution slice

        gt_slice = sio.loadmat(h_path)['output'][:, : ,slice_idx].astype(np.float32)
        gt_slice_resized = cv2.resize(gt_slice, self.target_size, interpolation=cv2.INTER_CUBIC)
        gt = np.reshape(gt_slice_resized, (self.target_size[0],self.target_size[1], 1))
        gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-15)
        patch_H = torch.from_numpy(gt)
        # Load Low-Resolution slice
        l_slice = sio.loadmat(l_path)['output'][:,:,slice_idx].astype(np.float32)
        l_slice_resized = cv2.resize(l_slice, self.target_size, interpolation=cv2.INTER_CUBIC)
        l = np.reshape(l_slice_resized, (self.target_size[0],self.target_size[1], 1))
        l = (l - l.min()) / (l.max() - l.min() + 1e-15)
        patch_L = torch.from_numpy(l)
        # --- End of on-demand loading ---

        b = imgtofrequency(patch_L)
        img_L, img_H, b = util.float2tensor3(patch_L), util.float2tensor3(patch_H), util.float2tensor3(b)

        descriptive_path = f"{h_path}_slice_{slice_idx}"
        time1 = time.time()
        # print(time1-time0)
        return {'L': img_L,  'H': img_H, 'H_path': descriptive_path, 'img_info': str(index)}

    def __len__(self):
        # The length of the dataset is the number of valid slices found
        return len(self.slice_info)

    def undersample_kspace(self, x, mask, is_noise, noise_level, noise_var):
        fft = fft2(x)
        mask = mask.unsqueeze(-1)
        fft = fft * mask
        x = ifft2(fft)
        return x, fft

    def generate_gaussian_noise(self, x, noise_level, noise_var):
        spower = np.sum(x ** 2) / x.size
        npower = noise_level / (1 - noise_level) * spower
        noise = np.random.normal(0, noise_var ** 0.5, x.shape) * np.sqrt(npower)
        return noise
