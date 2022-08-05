import dataloader
import numpy as np
from PIL import Image
from imqual_utils import getSSIM, getPSNR

def SSIMs_PSNR(gtr_dir, gen_dir, im_res=(350, 350)):

    ssims, psnrs = [], []
    test_dataset = dataloader.loader(gtr_dir, gen_dir, 1)

    for i, (gtr_path, gen_path) in enumerate(test_dataset):

        r_im = Image.open(gtr_path).resize(im_res)
        g_im = Image.open(gen_path).resize(im_res)

        ssim = getSSIM(np.array(r_im), np.array(g_im))
        ssims.append(ssim)

        r_im = r_im.convert("L")
        g_im = g_im.convert("L")
        psnr = getPSNR(np.array(r_im), np.array(g_im))
        psnrs.append(psnr)

    return np.array(psnrs), np.array(ssims)

def get_SSIM_PSNR(gtr_dir, result_UIEB_dir, result_NUY_dir):

    UIEB_PSNRs, UIEB_SSIMs = SSIMs_PSNR(gtr_dir, result_UIEB_dir)
    NUY_PSNRs, NUY_SSIMs = SSIMs_PSNR(gtr_dir, result_NUY_dir)

    return np.mean(UIEB_PSNRs), np.mean(UIEB_SSIMs), np.mean(NUY_PSNRs), np.mean(NUY_SSIMs)
          
