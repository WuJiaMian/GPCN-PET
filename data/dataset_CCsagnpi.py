import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
from utils.utils_swinmr import *
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import os
import cv2


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

        # --- 小数据集配置 ---
        self.is_mini_dataset = 1  # 1: 开启, 0: 关闭
        self.mini_dataset_prec = 300  # 这里现在代表【文件数量】（例如取50个文件）

        # --- 标签映射字典 ---
        self.manufacturer_map = {'SIEMENS': 0, 'UIH': 1, 'SINOUNION': 2}
        self.tracer_map = {'FDG': 0, 'DOTATATE': 1, 'PSMA': 2, 'DOPA': 3, 'MFBG': 4, 'FAPI': 5}
        self.metadata_cache = {}

        # 1. 获取所有文件路径
        paths_H_files = sorted(
            [os.path.join(opt['dataroot_H'], f) for f in os.listdir(opt['dataroot_H']) if f.endswith('.mat')])
        paths_L_files = sorted(
            [os.path.join(opt['dataroot_L'], f) for f in os.listdir(opt['dataroot_L']) if f.endswith('.mat')])

        assert paths_H_files, 'Error: High-resolution data path is empty.'
        assert paths_L_files, 'Error: Low-resolution data path is empty.'
        assert len(paths_H_files) == len(paths_L_files), 'Error: Mismatch in number of H and L data files.'

        self.mask = generate_mask(128, 128, 64, 64, 64)
        self.mask = np.expand_dims(self.mask, axis=-1)
        self.target_size = (200, 200)

        # 2. 打包成对
        file_pairs = list(zip(paths_H_files, paths_L_files))
        total_files = len(file_pairs)

        # ==========================================================
        # 3. 极速模式核心修改：在读文件前直接对【文件列表】进行采样
        # ==========================================================
        if self.is_mini_dataset:
            target_file_count =  200  # 默认值

            # 解析 mini_dataset_prec 是数量还是比例
            if self.mini_dataset_prec > 1:
                target_file_count = int(self.mini_dataset_prec)
            elif 0 < self.mini_dataset_prec <= 1:
                target_file_count = int(total_files * self.mini_dataset_prec)

            # 确保不超过总文件数
            if target_file_count > total_files:
                target_file_count = total_files

            print(f"\n[Mini-Dataset Mode] ON 🟢. Selecting {target_file_count} FILES from {total_files} total files.")

            random.seed(42)  # 固定种子，保证每次选的文件一样
            # 直接截取文件列表！这是最快的方法，不用读任何数据
            file_pairs = random.sample(file_pairs, target_file_count)

            # ==========================================================

        # 4. 扫描数据 (现在只扫描被选中的那几十个文件)
        self.slice_info = []
        print(f"Scanning {len(file_pairs)} files to build slice manifest...")

        for h_path, l_path in file_pairs:
            try:
                # 加载 mat
                h_mat_struct = sio.loadmat(h_path, squeeze_me=True, struct_as_record=False)
                l_mat_struct = sio.loadmat(l_path, squeeze_me=True, struct_as_record=False)

                h_mat = h_mat_struct['output']
                l_mat = l_mat_struct['output']

                # 提取 Metadata
                dicom_key = 'dicom_metadata' if hasattr(h_mat_struct, 'dicom_metadata') else 'dicom_data'
                if hasattr(h_mat_struct, dicom_key):
                    meta = getattr(h_mat_struct, dicom_key)
                    meta_vec, cat_dict = self._extract_metadata_from_struct(meta)
                    self.metadata_cache[h_path] = {'vec': meta_vec, 'cat': cat_dict}
                else:
                    self.metadata_cache[h_path] = {
                        'vec': np.zeros(6, dtype=np.float32),
                        'cat': {'manu': 2, 'tracer': 0}
                    }

                # 切片对齐逻辑
                if abs(h_mat.shape[2] - l_mat.shape[2]) < 10 and (h_mat.shape[2] != l_mat.shape[2]):
                    sub_slice = abs(h_mat.shape[2] - l_mat.shape[2])
                    if (h_mat.shape[2] > l_mat.shape[2]):
                        h_mat = h_mat[sub_slice:]
                    else:
                        l_mat = l_mat[sub_slice:]

                if h_mat.shape[2] != l_mat.shape[2]: continue

                # 遍历文件内的切片
                for i in range(h_mat.shape[2]):
                    h_slice = h_mat[:, :, i]
                    l_slice = l_mat[:, :, i]
                    if h_slice.max() == h_slice.min() or l_slice.max() == l_slice.min():
                        continue

                    self.slice_info.append({
                        'H_path': h_path,
                        'L_path': l_path,
                        'slice_index': i
                    })
            except Exception as e:
                print(f"Error processing {os.path.basename(h_path)}: {e}")
                continue

        print(f"Dataset ready. Total slices loaded: {len(self.slice_info)}")
        assert len(self.slice_info) > 0, "Error: No valid slices found."

    def _extract_metadata_from_struct(self, meta):
        """辅助函数：将 8 个标签提取并转化为数值特征"""
        try:
            age = float(getattr(meta, 'PatientAge', 0))
            weight = float(getattr(meta, 'PatientWeight', 0))
            size = float(getattr(meta, 'PatientSize', 0))

            spacing = getattr(meta, 'PixelSpacing', [1.0, 1.0])
            spacing = spacing[0] if isinstance(spacing, (list, np.ndarray)) else spacing

            thickness = float(getattr(meta, 'SliceThickness', 0))
            dose = float(getattr(meta, 'RadionuclideTotalDose', 1e8))
            log_dose = np.log10(dose + 1e-15)
            # Weight: 0~150kg -> 0~1 (这是影响衰减最重要的参数)
            age = age / 100.0
            weight = weight / 250.0
            # Size: 0~200cm -> 0~1
            size = size / 2
            # Spacing: 0~5mm -> 0~1
            spacing = spacing / 5
            thickness = thickness / 5.0
            log_dose = (log_dose - 6.0) / 3.0
            cont_vec = np.array([age, weight, size, spacing, thickness, log_dose], dtype=np.float32)
        except:
            cont_vec = np.zeros(6, dtype=np.float32)



        manu_str = str(getattr(meta, 'Manufacturer', 'OTHER')).upper()
        tracer_str = str(getattr(meta, 'Radiopharmaceutical', 'OTHER')).upper()

        manu_idx = self.manufacturer_map.get(manu_str, 2)
        tracer_idx = self.tracer_map.get(tracer_str, 0)

        cat_dict = {'manu': manu_idx, 'tracer': tracer_idx}
        return cont_vec, cat_dict

    def __getitem__(self, index):
        time0 = time.time()

        slice_meta = self.slice_info[index]
        h_path = slice_meta['H_path']
        l_path = slice_meta['L_path']
        slice_idx = slice_meta['slice_index']

        # Load High-Resolution slice
        gt_slice = sio.loadmat(h_path)['output'][:, :, slice_idx].astype(np.float32)
        gt_slice_resized = cv2.resize(gt_slice, self.target_size, interpolation=cv2.INTER_CUBIC)
        gt = np.reshape(gt_slice_resized, (self.target_size[0], self.target_size[1], 1))
        gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-15)
        patch_H = torch.from_numpy(gt)

        # Load Low-Resolution slice
        l_slice = sio.loadmat(l_path)['output'][:, :, slice_idx].astype(np.float32)
        l_slice_resized = cv2.resize(l_slice, self.target_size, interpolation=cv2.INTER_CUBIC)
        l = np.reshape(l_slice_resized, (self.target_size[0], self.target_size[1], 1))
        l = (l - l.min()) / (l.max() - l.min() + 1e-15)
        patch_L = torch.from_numpy(l)

        b = imgtofrequency(patch_L)
        img_L, img_H, b = util.float2tensor3(patch_L), util.float2tensor3(patch_H), util.float2tensor3(b)

        # 获取 Prompt Metadata
        meta_data = self.metadata_cache[h_path]
        prompt_vec = torch.from_numpy(meta_data['vec'])
        prompt_manu = torch.tensor(meta_data['cat']['manu'], dtype=torch.long)
        prompt_tracer = torch.tensor(meta_data['cat']['tracer'], dtype=torch.long)

        descriptive_path = f"{h_path}_slice_{slice_idx}"

        return {
            'L': img_L,
            'H': img_H,
            'H_path': descriptive_path,
            'img_info': str(index),
            'prompt_vec': prompt_vec,
            'prompt_manu': prompt_manu,
            'prompt_tracer': prompt_tracer
        }

    def __len__(self):
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