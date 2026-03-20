'''
This is the testing code of our WDPC-Net.
Our training procedure adheres to the standard training protocol of SwinMR.
The complete training code for WDPCNet will be made publicly available upon the acceptance of this paper.
'''
import numpy as np
import os
import torch
from models.restormer_prompt import Restormer as net
import time
from scipy.io import savemat,loadmat
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import random
np.random.seed(5)
random.seed(5)
torch.manual_seed(5)
torch.cuda.manual_seed_all(5)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.cuda.set_device(0)
def normalize(img):
    mean = img.mean()
    std = img.std()
    return (img - mean) / (std + 1e-11), mean, std
def compare_psnr_norm(image, target):

    return compare_psnr(image, target,data_range=target.max()), compare_ssim(image, target,data_range=target.max()), image
def quality(tensor1, tensor2):  # tensor2: data_gt
    torch.cuda.empty_cache()
    tensor1 = np.array(tensor1.cpu())
    tensor2 = np.array(tensor2.cpu())
    n_slices = tensor1.shape[2]
    psnr_list = []
    ssim_list = []
    for i in range(n_slices):
        img1 = tensor1[:, :, i]
        img2 = tensor2[:, :, i]
        # 检查最大值等于最小值或存在NaN
        if (
            np.isnan(img1).any() or np.isnan(img2).any() or
            np.max(img1) == np.min(img1) or np.max(img2) == np.min(img2)
        ):
            continue  # 跳过该片
        psnr, ssim, _ = compare_psnr_norm(img1, img2)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
    avg_psnr = np.mean(psnr_list) if psnr_list else float('nan')
    avg_ssim = np.mean(ssim_list) if ssim_list else float('nan')
    return avg_psnr, avg_ssim
def myfftshift(x, axes=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    assert torch.is_tensor(x) is True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)


def myifftshift(x, axes=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    assert torch.is_tensor(x) is True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)


def myfft2(data):
    assert data.shape[-1] == 2
    data = torch.view_as_complex(data)
    data = torch.fft.fftn(data, dim=(-3, -2), norm='ortho')
    data = torch.view_as_real(data)
    data = myfftshift(data, axes=(-4, -3))
    return data

def myifft2(data):
    assert data.shape[-1] == 2
    data = myifftshift(data, axes=(-4, -3))
    data = torch.view_as_complex(data)
    torch.manual_seed(5)
    # savemat('data2.mat',{'data2':data.numpy()})
    data = torch.fft.ifftn(data, dim=(-3, -2), norm='ortho')
    data = torch.view_as_real(data)
    return data
def sample_in_kspace(data, mask):   #fullsampled image to undersampled iamge
    data = data.unsqueeze(-1)
    assert data.shape[-1] == 1
    torch.manual_seed(5)
    mask = torch.as_tensor(mask)
    data = torch.cat([data, torch.zeros_like(data)], dim=-1)
    data = myfft2(data)
    f = data
    mask = mask.unsqueeze(2)
    data = data * mask
    b = data
    data = myifft2(data)
    data = data[..., 0]
    return data,b,f
def DC(x,b,mask):
    x1 = torch.cat((x,torch.zeros_like(x)),dim=1)
    x1 = x1.permute(0, 2, 3, 1).contiguous()
    x1 = torch.view_as_complex(x1)
    k = torch.fft.fftshift(torch.fft.fftn(x1, dim=(-2, -1), norm='ortho'), dim=(-2, -1))
    k = torch.view_as_real(k)
    k = k.permute(0, 3, 1, 2).contiguous()
    k = k * (1 - mask) + b * mask
    k = k.permute(0, 2, 3, 1).contiguous()
    k = torch.view_as_complex(k)
    x = torch.fft.ifftn(torch.fft.ifftshift(k, dim=(-2, -1)), norm='ortho', dim=(-2, -1))
    x = torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()
    return x[:,0:1,:,:]
def maxminnormal(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-15),x.min(),x.max()
def count_trainable_parameters(model):
       return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net(
        inp_channels=1,
        out_channels=1,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False
    )
    param_key_g = 'params'
    root = 'E:\PET2025/testset_allin'

    # datalistH = os.listdir('E:\PET2025/testset_allin/AC')
    # datalistL = os.listdir('E:\PET2025/testset_allin/NAC')
    datalistH = os.listdir(root+'/AC')
    datalistL = os.listdir(root+'/NAC')
    datalistH.sort()
    datalistL.sort()

    print("count_trainable_parameters:", count_trainable_parameters(model))

    # ===== 初始化总PSNR、SSIM和文件计数器 =====
    total_psnr = 0
    total_ssim = 0
    file_count = 0
    for acc in [10]:  ###Example for Cartesian 1D mask at the acceleration factor of 10 $\times$.
        # pretrained_model = torch.load('checkpoints/'+'220000_E.pth')petpacm
        #pretrained_model = torch.load('checkpoints/'+'150000_E.pth') #petmsm
        pretrained_model = torch.load('checkpoints/' + '1500000_E.pth')  # petmsm

        model.load_state_dict(
            pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
            strict=True)

        model = model.to(device)

        for fileL,fileH in zip(datalistL,datalistH):
            # ===== 添加了Try-Except块 =====
            try:
                avepsnr = 0
                avessim = 0
                zfpsnr = 0
                zfssim = 0
                data  = loadmat(os.path.join(root,'AC',fileH))['output']
                H = torch.from_numpy(data).float()

                data = loadmat(os.path.join(root,'NAC',fileL))['output']
                L = torch.from_numpy(data).float()
                out = np.zeros_like(data)
                gt = np.zeros_like(data)
                time1 = time.time()
                for i in range(data.shape[2]):#224
                    # i=5
                    with torch.no_grad():
                        time_start = time.time()
                        img_H = H[:,:,i]
                        maxH = img_H.max().item()
                        minH = img_H.min().item()
                        #img_H = img_H /maxH
                        img_H = (img_H - img_H.min()) / (img_H.max() - img_H.min() + 1e-15)
                        img_L = L[:,:,i].unsqueeze(0).unsqueeze(0).to(device)
                        maxL = img_L.max().item()
                        minL = img_L.min().item()
                        #img_L = img_L/maxL
                        img_L = (img_L - img_L.min()) / (img_L.max() - img_L.min() + 1e-15)

                        img_gen = model(img_L)
                        if(img_gen.shape[1]==2):
                            img_gen = torch.sqrt(img_gen[:,0,...]**2+img_gen[:,1,...]**2)
                        output = img_gen.detach().cpu().numpy()
                        out[:, :, i] = output
                        gt[:, :, i] = img_H
                        out[:, :, i] = (output-1e-15)*(maxH-minH)+minH
                        gt[:, :, i] = (img_H-1e-15)*(maxH-minH)+minH

                        img_gen = img_gen.squeeze().float().cpu().numpy()
                        i += 1
                        #print(maxH)
                print(time.time()-time1)
                savemat('results/Restormerp_'+fileH[0:-4]+'_output.mat',{'output':out})
                #savemat('results/'+fileH[0:-4]+'_t.mat',{'output':gt})

                psnr,ssim = quality(torch.as_tensor(out),torch.as_tensor(gt))

                # ===== 累加PSNR和SSIM，并增加文件计数 =====
                if not np.isnan(psnr) and not np.isnan(ssim):
                    total_psnr += psnr
                    total_ssim += ssim
                    file_count += 1

                zfpsnr /= data.shape[2]
                zfssim /= data.shape[2]
                print(fileL)
                print("completepsnr:{:.3f}, completessim{:.4f}".format(psnr, ssim))
            except Exception as e:
                print(f"无法处理文件 {fileH} 或 {fileL}，错误: {e}。跳过此文件对。")
                continue # 跳到下一个循环

    # ===== 计算并打印所有文件的平均PSNR和SSIM =====
    if file_count > 0:
        avg_psnr = total_psnr / file_count
        avg_ssim = total_ssim / file_count
        print("\n" + "="*40)
        print("           Overall Average Metrics")
        print("="*40)
        print(f"Average PSNR for all {file_count} files: {avg_psnr:.3f}")
        print(f"Average SSIM for all {file_count} files: {avg_ssim:.4f}")
        print("="*40)
    else:
        print("\nNo files were processed successfully to calculate the average.")

    print(1)