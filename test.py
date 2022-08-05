import os
import glob
import torch
import numpy as np
import torchvision
import torch.optim
from PIL import Image
from measure import get_SSIM_PSNR
from Networks import LEPFNet as net
# 优先选择空闲的gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['CUDA_CACHE_PATH'] = '~/.cudacache'


real_file_path = "/home/jqyan/YiJianWang/LEPF-Net/dataset/clear/"


UIEB_test_path = "/home/jqyan/YiJianWang/LEPF-Net/dataset/test/UIEB/*"
UWCNN_test_path = "/home/jqyan/YiJianWang/LEPF-Net/dataset/test/UWCNN/*"


UIEB_result_path = "/home/jqyan/YiJianWang/LEPF-Net/results/UIEB/"
UWCNN_result_path = "/home/jqyan/YiJianWang/LEPF-Net/results/UWCNN/"


modle_path = "/home/jqyan/YiJianWang/LEPF-Net/snapshots/epoch_16.pth"


def dehaze_image(images_path, dehaze_net, result_path):

    haze_images = sorted(glob.glob(images_path))
    for h_i in haze_images:

        data_hazy = Image.open(h_i)
        
        fileName = result_path + h_i.split("/")[-1]

        data_hazy = data_hazy.resize((350, 350), Image.ANTIALIAS)
        data_hazy = (np.asarray(data_hazy)/255.0)
        data_hazy = torch.from_numpy(data_hazy).float()
        data_hazy = data_hazy.permute(2, 0, 1)
        data_hazy = data_hazy.cuda().unsqueeze(0)

        clean_image = dehaze_net(data_hazy, False)

        torchvision.utils.save_image(clean_image, fileName)


def getPSNR(test_UIEB_dir, test_NUY_dir, clear_dir, result_UIEB_dir, result_NUY_dir, modle_path):

    if not os.path.exists(result_UIEB_dir):
        os.makedirs(result_UIEB_dir)
    
    if not os.path.exists(result_NUY_dir):
        os.makedirs(result_NUY_dir)

    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.load_state_dict(torch.load(modle_path))

    dehaze_image(test_UIEB_dir, dehaze_net, result_UIEB_dir)
    dehaze_image(test_NUY_dir, dehaze_net, result_NUY_dir)

    return get_SSIM_PSNR(clear_dir, result_UIEB_dir, result_NUY_dir)



if __name__ == '__main__':
    UIEB_PSNR, UIEB_SSIM, NUY_PSNR, NUY_SSIM = getPSNR(UIEB_test_path,
                                                       UWCNN_test_path,
                                                       real_file_path,
                                                       UIEB_result_path,
                                                       UWCNN_result_path,
                                                       modle_path
                                                       )
    print("UIEB_PSNR : %.4f UIEB_SSIM : %.4f NUY_PSNR : %.4f NUY_SSIM : %.4f \n"
          % (UIEB_PSNR, UIEB_SSIM, NUY_PSNR, NUY_SSIM))
    
