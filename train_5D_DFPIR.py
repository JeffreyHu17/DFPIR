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


parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--save_epoch', type=int, default=1,
                    help='save model per every N epochs')
parser.add_argument('--save_item', type=int, default=3000, #-----------------
                    help='save model per every N item')
parser.add_argument('--init_epoch', type=int, default=1, # -------------------
                    help='if finetune model, set the initial epoch')
parser.add_argument('--save_dir', type=str, default='./', # ----------------
                     help='save parameter dir')

parser.add_argument('--gpu', type=str, default="0,1", # -----------GPU
                    help='GPUs') 
parser.add_argument('--cuda', type=int, default=1) # -----------GPU
parser.add_argument('--pretrained_1', type=str, default=
        './', 
        help='training loss')
 
parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=5,help="Batch size to use per GPU") #------
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of encoder.') # -----

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze','deblur','lowlight'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

# train datasets path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='/mnt/d/DL_module/PromptIR-main/data/Train/Denoise2/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='/mnt/d/DL_module/PromptIR-main/data/Train/Derain2/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='/mnt/d/DL_module/PromptIR-main/data/Train/Dehaze2/', 
                    help='where training images of dehazing saves.')
parser.add_argument('--deblur_dir', type=str, default='/mnt/f/datasets/GOPRO/GOPRO_Large/train/', # train sharp->gt blur
                    help='where training images of dehblur saves.')
parser.add_argument('--lowlight_dir', type=str, default='/mnt/f/datasets/LOL/LOLdataset/train485/', # 
                    help='where training images of lowlight saves.')
# ------------------------------
parser.add_argument('--denoise_path', type=str, default="/mnt/d/DL_module/2-DFPIR/test/denoise/", help='save path of test noisy images')
parser.add_argument('--derain_path', type=str, default="/mnt/d/DL_module/2-DFPIR/test/derain/", help='save path of test raining images')
parser.add_argument('--dehaze_path', type=str, default="/mnt/d/DL_module/2-DFPIR/test/dehaze/", help='save path of test hazy images')
parser.add_argument('--deblur_path', type=str, default="/mnt/f/datasets/GOPRO/GOPRO_Large/test/", help='test dataset blur images') # test下面的sharp为gt，blur为退化
parser.add_argument('--lowlight_path', type=str, default="/mnt/f/datasets/LOL/LOLdataset/test15/", help='test dataset blur images') # test15下面的high为gt，low为退化

parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_name', type=str, default="model.ckpt", help='checkpoint save path')

args = parser.parse_args()

psnr_max = 10

clip_model, _ = clip.load("ViT-B/32", device=args.cuda)
for param in clip_model.parameters():
    param.requires_grad = False  


inputext = ["Gaussian noise with a standard deviation of 15","Gaussian noise with a standard deviation of 25"
            ,"Gaussian noise with a standard deviation of 50","Rain degradation with rain lines"
            ,"Hazy degradation with normal haze", "Blur degradation with motion blur","Lowlight degradation"] 

# -----------------------这里配置测试数据集-全局变量-------------
# denoise_splits = ["urban100/","bsd68/"] 
denoise_splits = ["bsd68/"]
derain_splits = ["Rain100L/"]
denoise_tests = []
derain_tests = []
base_path = args.denoise_path

derain_base_path = args.derain_path

args.derain_path = args.derain_path+"Rain100L/" # os.path.join(derain_base_path,derain_splits)

for i in denoise_splits:
    args.denoise_path = os.path.join(base_path,i)
    denoise_testset = DenoiseTestDataset(args)
    denoise_tests.append(denoise_testset)
# ------------------------------------------------------------------------  

def train(train_loader, model, optimizer, epoch, epoch_total,criterionL1):
    loss_sum = 0
    losses = AverageMeter()
    
# -----------------------------------------------------------

    # 添加tensorboard
    writer = SummaryWriter("./logs_train")
    psnr_tqdm = 10
    ssim_tqdm = 0.009
    loss_tqdm = 0.0

    model.train()
    start_time = time.time()
    global psnr_max

    loop_train = tqdm((train_loader), total = len(train_loader),leave=False) 
    for i, ([clean_name, de_id], degrad_patch, clean_patch) in enumerate(loop_train):

        input_var = Variable(degrad_patch.cuda())
        target_var = Variable(clean_patch.cuda())

        result = [clean_name, de_id]
        img_id = result[1] 
        img_id = img_id.tolist() 
        text_prompt_list = [inputext[idx] for idx in img_id]
# ------------------------------------------
        text_token = clip.tokenize(text_prompt_list).to(args.cuda) 
        text_code = clip_model.encode_text(text_token).to(dtype=torch.float32)  
# --------------------------------------------------
        output = model(input_var,text_code) #     
        loss = criterionL1(output,target_var)
        loss_sum+=loss.item()     
        losses.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i % 10 == 0) and (i != 0):
            loss_avg = loss_sum / 10
            loss_sum = 0.0
            loss_tqdm = loss_avg                                                   
            writer.add_scalar("train_loss", loss.item(), i)
            start_time = time.time()
        if (i % args.save_item == 0) and (i != 0): 
            psnr_noise,ssim_noise,psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_blur,ssim_blur,psnr_lol,ssim_lol =test( model, criterionL1)
            psnr_avr = (psnr_noise+psnr_rain+psnr_haze+psnr_blur+psnr_lol)/5
            ssim_avr = (ssim_noise+ssim_rain+ssim_haze+ssim_lol+ssim_blur)/5
            psnr_tqdm = psnr_avr
            ssim_tqdm  = ssim_avr
            if psnr_avr > psnr_max - 0.00001:
                psnr_max = max(psnr_avr, psnr_max)
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    os.path.join(args.save_dir,
                                    'FT-checkpoint_epoch_V1_{:0>4}_{}_pn{:.2f}-{:.4f}_pr{:.2f}-{:.4f}_ph{:.2f}-{:.4f}_pb{:.2f}-{:.4f}_pl{:.2f}-{:.4f}_avr{:.2f}-{:.4f}.pth.tar'.format(
                                        epoch, i//args.save_item, psnr_noise,ssim_noise,psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_blur,ssim_blur,psnr_lol,ssim_lol,psnr_avr,ssim_avr)))
            else:
                torch.save({}, os.path.join(args.save_dir,
                                        'FT-checkpoint_epoch_V1_{:0>4}_{}_pn{:.2f}-{:.4f}_pr{:.2f}-{:.4f}_ph{:.2f}-{:.4f}_pb{:.2f}-{:.4f}_pl{:.2f}-{:.4f}_avr{:.2f}-{:.4f}.pth.tar'.format(
                                        epoch, i//args.save_item, psnr_noise,ssim_noise,psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_blur,ssim_blur,psnr_lol,ssim_lol,psnr_avr,ssim_avr)))
                
        loop_train.set_description(f'trainning->epoch:[{epoch}/{args.epochs}],item:[{i}/{len(train_loader)}]') 
        loop_train.set_postfix(loss = loss_tqdm,psnr = f'{psnr_tqdm:.4f}', ssim = f'{ssim_tqdm:.4f}')       
    writer.close()
    return losses.avg


def test(model, criterion):
    model.eval()

# -------------------------
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
      
# ---------------------------------  
    print('Start testing Rain100L rain streak removal...') # 
    derain_set = DerainDehazeDataset(args,addnoise=False,sigma=15)
    psnr_rain,ssim_rain = test_Derain_Dehaze(model, derain_set, task="derain",text_prompt=inputext[3])
    print('Rain100L test ok psnr_rain:{:.4f} ssim_rain:{:.4f},'.format(psnr_rain,ssim_rain))
# --------------------------------
    print('Start testing SOTS...')
    psnr_haze,ssim_haze = test_Derain_Dehaze(model, derain_set, task="dehaze",text_prompt=inputext[4])
    print('dehaze test ok psnr_haze:{:.4f} ssim_haze:{:.4f},'.format(psnr_haze,ssim_haze))
# ---------------------------------
    print('Start testing gopro removal...') # 
    deblur_set = DeblurTestDataset(args,addnoise=False,sigma=15)
    psnr_blur,ssim_blur = test_Deblur(model, deblur_set,text_prompt=inputext[5])
    print('gopro test ok psnr_blur:{:.4f} ssim_blur:{:.4f},'.format(psnr_blur,ssim_blur))   

# -------------------------------
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
            # save_image_tensor(restored, output_path + degraded_name[0] + filename + '.png')
            # save_image_tensor(restored, output_path + degraded_name[0] + '.png')
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
            # save_image_tensor(restored, output_path + degraded_name[0] + filename + '.png')
            # save_image_tensor(restored, output_path + degraded_name[0] + '.png')
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
            # save_image_tensor(restored, output_path + degraded_name[0] + filename + '.png')
            # save_image_tensor(restored, output_path + degraded_name[0] + '.png')
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
            # save_image_tensor(restored, output_path + degraded_name[0] + filename + '.png')
            # save_image_tensor(restored, output_path + degraded_name[0] + '.png')
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

# ------------------------------------------------------
    if os.path.exists(os.path.join(args.save_dir, 'checkpoint_{:0>4}.pth.tar'.format(args.init_epoch))):
        # load existing model
        model_info = torch.load(os.path.join(args.save_dir, 'checkpoint_{:0>4}.pth.tar'.format(args.init_epoch)))
        print('==> loading existing model:',
              os.path.join(args.save_dir, 'checkpoint_{:0>4}.pth.tar'.format(args.init_epoch)))
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scheduler.load_state_dict(model_info['scheduler'])
        cur_epoch = model_info['epoch']
    else:
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        # create model
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        cur_epoch = args.init_epoch

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
# -----------------------------------------------------------
    train_dataset = PromptTrainDataset5D(args)                
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    
    print('load dataset ok')
    psnr_noise,ssim_noise,psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_blur,ssim_blur,psnr_lol,ssim_lol =test( model, criterionL1) # 训练前先测试一下 自己加的
    print('test ok! pn:{:.2f}-{:.4f},--pr:{:.2f}-{:.4f},--ph:{:.2f}-{:.4f},pb:{:.2f}-{:.4f},pl:{:.2f}-{:.4f}'
          .format(psnr_noise,ssim_noise,psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_blur,ssim_blur,psnr_lol,ssim_lol))

    for epoch in range(cur_epoch, args.epochs + 1):
        loss = train(train_loader, model, optimizer, epoch, args.epochs + 1,criterionL1)
        scheduler.step()
        if epoch % args.save_epoch == 0:
            psnr_noise,ssim_noise,psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_blur,ssim_blur,psnr_lol,ssim_lol =test( model, criterionL1)
            psnr_avr = (psnr_noise+psnr_rain+psnr_haze+psnr_blur+psnr_lol)/5
            ssim_avr = (ssim_noise+ssim_rain+ssim_haze+ssim_blur+ssim_lol)/5
            if psnr_avr > psnr_max - 0.001:
                psnr_max = max(psnr_avr, psnr_max)
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    os.path.join(args.save_dir,
                                 'FT-checkpoint_epoch_V1_{:0>4}__pn{:.2f}-{:.4f}_pr{:.2f}-{:.4f}_ph{:.2f}-{:.4f}_pb{:.2f}-{:.4f}_pl{:.2f}-{:.4f}_avr{:.2f}-{:.4f}.pth.tar'.format(
                                     epoch, psnr_noise,ssim_noise,psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_blur,ssim_blur,psnr_lol,ssim_lol,psnr_avr,ssim_avr)))
            else:
                torch.save({}, os.path.join(args.save_dir,
                                            'FT-checkpoint_epoch_V1_{:0>4}__pn{:.2f}-{:.4f}_pr{:.2f}-{:.4f}_ph{:.2f}-{:.4f}_pb{:.2f}-{:.4f}_pl{:.2f}-{:.4f}_avr{:.2f}-{:.4f}.pth.tar'.format(
                                                    epoch, psnr_noise,ssim_noise,psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_blur,ssim_blur,psnr_lol,ssim_lol,psnr_avr,ssim_avr)))

        print('Epoch [{0}]\t'
              'lr: {lr:.6f}\t'
              'Loss: {loss:.5f}'
            .format(
            epoch,
            lr=optimizer.param_groups[-1]['lr'],
            loss=loss))

            