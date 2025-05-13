import os, time, shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.image as mpimg

import numpy as np
import time
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
from utils.dataset_utils import PromptTrainDataset5D,DenoiseTestDataset, DerainDehazeDataset,DeblurTestDataset,LOLTestDataset
from net.model import ChannelShuffle_skip_textguaid
import subprocess
from torch.utils.data import DataLoader
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
import clip


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu', type=str, default="0,1", # -----------GPU
                    help='GPUs') 
parser.add_argument('--cuda', type=int, default=0) # -----------GPU
parser.add_argument('--pretrained_1', type=str, default=
        './',
        help='training loss')
parser.add_argument('--denoise_path', type=str, default="/mnt/d/DL_module/2-DFPIR/test/denoise/", help='save path of test noisy images')
parser.add_argument('--derain_path', type=str, default="/mnt/d/DL_module/2-DFPIR/test/derain/", help='save path of test raining images')
parser.add_argument('--dehaze_path', type=str, default="/mnt/d/DL_module/2-DFPIR/test/dehaze/", help='save path of test hazy images')
parser.add_argument('--deblur_path', type=str, default="/mnt/f/datasets/GOPRO/GOPRO_Large/test/", help='test dataset blur images') 
parser.add_argument('--lowlight_path', type=str, default="/mnt/f/datasets/LOL/LOLdataset/test15/", help='test dataset blur images') 
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
args = parser.parse_args()

psnr_max = 10

clip_model, _ = clip.load("ViT-B/32", device=args.cuda)
for param in clip_model.parameters():
    param.requires_grad = False  


inputext = ["Gaussian noise with a standard deviation of 15","Gaussian noise with a standard deviation of 25"
            ,"Gaussian noise with a standard deviation of 50","Rain degradation with rain lines"
            ,"Hazy degradation with normal haze", "Blur degradation with motion blur","Lowlight degradation"] # Degradation prompt text

denoise_splits = ["bsd68/"]
derain_splits = ["Rain100L/"]
denoise_tests = []
derain_tests = []
base_path = args.denoise_path

derain_base_path = args.derain_path

args.derain_path = args.derain_path+"Rain100L/" 

for i in denoise_splits:
    args.denoise_path = os.path.join(base_path,i)
    denoise_testset = DenoiseTestDataset(args)
    denoise_tests.append(denoise_testset)
# ------------------------------------------------------------------------  

def test(model, criterion):
    model.eval()
# ------------------------
    for testset,name in zip(denoise_tests,denoise_splits) :
        # print('Start {} testing Sigma=15...'.format(name))
        # psnr_g15,ssim_g15 = test_Denoise(model, testset, sigma=15,text_prompt=inputext[0])
        # print('{}test ok psnr_g15:{:.4f} ssim_g15:{:.4f},'.format(name,psnr_g15,ssim_g15))

        print('Start {} testing Sigma=25...'.format(name))
        psnr_g25,ssim_g25 = test_Denoise(model, testset, sigma=25,text_prompt=inputext[1])
        print('{}test ok psnr_g25:{:.4f} ssim_g25:{:.4f},'.format(name,psnr_g25,ssim_g25))

        # print('Start {} testing Sigma=50...'.format(name))
        # psnr_g50,ssim_g50 = test_Denoise(model, testset, sigma=50,text_prompt=inputext[2])
        # print('{}test ok psnr_g50:{:.4f} ssim_g50:{:.4f},'.format(name,psnr_g50,ssim_g50))  
# -----------------------------------  
    print('Start testing Rain100L rain streak removal...') # 
    derain_set = DerainDehazeDataset(args,addnoise=False,sigma=15)
    psnr_rain,ssim_rain = test_Derain_Dehaze(model, derain_set, task="derain",text_prompt=inputext[3])
    print('Rain100L test ok psnr_rain:{:.4f} ssim_rain:{:.4f},'.format(psnr_rain,ssim_rain))
# ---------------------------------
    print('Start testing SOTS...')
    psnr_haze,ssim_haze = test_Derain_Dehaze(model, derain_set, task="dehaze",text_prompt=inputext[4]) 
    print('dehaze test ok psnr_haze:{:.4f} ssim_haze:{:.4f},'.format(psnr_haze,ssim_haze))
# ------------------------------
    print('Start testing blurry removal...') # 
    deblur_set = DeblurTestDataset(args,addnoise=False,sigma=15)
    psnr_blur,ssim_blur = test_Deblur(model, deblur_set,text_prompt=inputext[5])
    print('gopro test ok psnr_blur:{:.4f} ssim_blur:{:.4f},'.format(psnr_blur,ssim_blur))   
# ---------------------------------
    print('Start testing lowlight enhancement...') # 
    loltest_set = LOLTestDataset(args,addnoise=False,sigma=15)
    psnr_lol,ssim_lol = test_lowlight(model, loltest_set,text_prompt=inputext[6])
    print('LOL test ok psnr_lol:{:.4f} ssim_lol:{:.4f},'.format(psnr_lol,ssim_lol))   
    return psnr_g25,ssim_g25,psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_blur,ssim_blur,psnr_lol,ssim_lol


def test_Denoise(net, dataset, sigma=15, text_prompt=""):
    output_path = args.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    
    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
  
    psnr = AverageMeter()
    ssim = AverageMeter()
    text_token = clip.tokenize(text_prompt).to(args.cuda) 
    text_code = clip_model.encode_text(text_token).to(dtype=torch.float32) 
    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            restored = net(degrad_patch,text_code)          
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            psnr_value_formatted = "{:.2f}".format(temp_psnr)  
            filename = f"_{psnr_value_formatted}"
            save_image_tensor(restored, output_path + clean_name[0] + filename + '.png')
        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))
    return psnr.avg,ssim.avg

def test_Derain_Dehaze(net, dataset, task="derain",text_prompt=""):
    output_path = args.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    text_token = clip.tokenize(text_prompt).to(args.cuda) 
    text_code = clip_model.encode_text(text_token).to(dtype=torch.float32) 
    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch,text_code)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            psnr_value_formatted = "{:.2f}".format(temp_psnr)  
            filename = f"_{psnr_value_formatted}"
            save_image_tensor(restored, output_path + degraded_name[0] + filename + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
    return psnr.avg,ssim.avg

def test_Deblur(net, dataset,text_prompt=""):
    output_path = args.output_path + 'deblur/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    text_token = clip.tokenize(text_prompt).to(args.cuda) 
    text_code = clip_model.encode_text(text_token).to(dtype=torch.float32)  
    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()           

            restored = net(degrad_patch,text_code)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            psnr_value_formatted = "{:.2f}".format(temp_psnr)  
            filename = f"_{psnr_value_formatted}"
            save_image_tensor(restored, output_path + degraded_name[0] + filename + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
    return psnr.avg,ssim.avg

def test_lowlight(net, dataset,text_prompt=""):
    output_path = args.output_path + 'lowlight/'
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    text_token = clip.tokenize(text_prompt).to(args.cuda) 
    text_code = clip_model.encode_text(text_token).to(dtype=torch.float32)  
    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            restored = net(degrad_patch,text_code)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            psnr_value_formatted = "{:.2f}".format(temp_psnr)  
            filename = f"_{psnr_value_formatted}"
            save_image_tensor(restored, output_path + degraded_name[0] + filename + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
    return psnr.avg,ssim.avg



if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# -------------------------------------------------------
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(args.cuda) 

    model = ChannelShuffle_skip_textguaid(device=args.cuda)
    criterionL1 = nn.L1Loss()
    model.cuda()

    if args.pretrained_1:
        if os.path.isfile(args.pretrained_1):
            print("=> loading model '{}'".format(args.pretrained_1))
            model_pretrained = torch.load(args.pretrained_1,map_location=torch.device('cuda:0'))
            pretrained_dict = model_pretrained['state_dict']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print("=> no model found at '{}'".format(args.pretrained_1))
    psnr_noise,ssim_noise,psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_blur,ssim_blur,psnr_lol,ssim_lol =test( model, criterionL1) 
    psnr_avr = (psnr_noise+psnr_rain+psnr_haze+psnr_blur+psnr_lol)/5
    ssim_avr = (ssim_noise+ssim_rain+ssim_haze+ssim_blur+ssim_lol)/5

    print('test ok! pn:{:.2f}-{:.4f},--pr:{:.2f}-{:.4f},--ph:{:.2f}-{:.4f},pb:{:.2f}-{:.4f},pl:{:.2f}-{:.4f}, avr:{:.2f}-{:.4f}'
          .format(psnr_noise,ssim_noise,psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_blur,ssim_blur,psnr_lol,ssim_lol,psnr_avr,ssim_avr))

            