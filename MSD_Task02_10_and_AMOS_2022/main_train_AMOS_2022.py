                      
                       
from torch.cuda.amp import autocast, GradScaler

from monai.utils import set_determinism
from monai.transforms import AsDiscrete


from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceCELoss
from monai_utils.inferers.utils import sliding_window_inference_1out
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.apps import DecathlonDataset
from monai.transforms import (
    Compose,
    Activations,
    )

import torch
from load_datasets_transforms import data_loader, data_transforms
from monai.networks.blocks import UnetOutBlock
import torch.nn as nn
import torch.nn.functional as F

from ptflops import get_model_complexity_info

import csv
import os
import numpy as np
import scipy.ndimage as ndimage
from medpy import metric
from tqdm import tqdm
import argparse

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

parser = argparse.ArgumentParser(description='Hyperparameters for medical image segmentation')
                             
parser.add_argument('--root', type=str, default='./data/amos22', help='Root folder of all your images and labels')
parser.add_argument('--output_parameter', type=str, default='./data/3D_parameter/SegRWKV_AMOS.pth', help='Output folder for the best model')
parser.add_argument('--dataset', type=str, default='amos', help='Datasets: {feta, flare, amos}, Fyi: You can add your dataset here')
parser.add_argument('--img_size', type=int, nargs='+', default=[128,128,128], help='3D ROI size, e.g., [128,128,128]')     
parser.add_argument('--n_channels', type=int, default=1, help='number of channels in input image')     

                                         
parser.add_argument('--network', type=str, default='SegRWKV', help='Network models')

parser.add_argument('--mode', type=str, default='train', help='Training or testing mode')
                                                                                                
parser.add_argument('--ds', default=False, help='Use of deep supervision (ds) or not')
parser.add_argument('--pretrain', default=False, help='Have pretrained weights or not')
parser.add_argument('--pretrained_weights', default='', help='Path of pretrained weights')
parser.add_argument('--pretrain_classes', default='', help='Number of classes output from pretrained model')
parser.add_argument('--batch_size', type=int, default='1', help='Batch size for subject input')
parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--optim', type=str, default='AdamW', help='Optimizer types: Adam / AdamW')
parser.add_argument('--max_iter', type=int, default=40000, help='Maximum iteration steps for training')
parser.add_argument('--eval_step', type=int, default=200, help='Per steps to perform validation')     
parser.add_argument('--val_batch', type=int, default=1, help='Validation batch size')     
parser.add_argument('--overlap', type=float, default=0.5, help='Amount of overlap between scans')     
parser.add_argument('--overlap_mode', type=str, default='constant', help='overlap mode')     

                             
parser.add_argument('--gpu', type=str, default='0,1', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print('Used GPU: {}'.format(args.gpu))

train_samples, valid_samples, out_classes = data_loader(args)

train_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_samples['images'], train_samples['labels'])
]

val_files = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(valid_samples['images'], valid_samples['labels'])
]

print(train_files)
print(val_files)

set_determinism(seed=0)

train_transforms, val_transforms = data_transforms(args)

                                        
print('Start caching datasets!')
                                        
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

                                        
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
    
val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)


if args.ds == 'True':
    args.ds = True
    
                
device = torch.device("cuda:0")

if args.network == '3DUXNET':
    from networks.UXNet_3D.network_backbone import UXNET
    model = UXNET(
        in_chans=args.n_channels,   
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    ).to(device)
elif args.network == 'SegFormer3D':
    from networks.SegFormer3D.segformer3d import SegFormer3D 
    model = SegFormer3D(
        in_channels = args.n_channels,
        sr_ratios = [4, 2, 1, 1],
        embed_dims = [32, 64, 160, 256],
        patch_kernel_size = [7, 3, 3, 3],
        patch_stride = [4, 2, 2, 2],
        patch_padding = [3, 1, 1, 1],
        mlp_ratios = [4, 4, 4, 4],
        num_heads = [1, 2, 5, 8],
        depths = [2, 2, 2, 2],
        decoder_head_embedding_dim = 256,
        num_classes = out_classes,
        decoder_dropout = 0.0,
    ).to(device)
elif args.network == 'SegMamba':
    from networks.segmamba import SegMamba
    model = SegMamba(args.n_channels, out_classes).to(device)
elif args.network == 'Zig_RiR3d':
    from networks.Zig_RiR3d import Z_RiR
    model = Z_RiR(in_channels=args.n_channels, out_channels=out_classes).to(device)
elif args.network == 'EMNet':
    from networks.em_net_model import EMNet
    model = EMNet(in_chans=args.n_channels,
        out_chans=out_classes,
        depths=[3,3,3,3],
                                        
        feat_size=[48, 96, 192, 384],
        hidden_size=384,
                                       
                          
        fft_nums=[0,0,0,0],
        conv_decoder=False,
        in_shpae=[128, 128, 128],
        ).to(device)
elif args.network == 'UKAN_EP':
    from networks.ukanep import UKAN
    model = UKAN(input_channnel=args.n_channels, num_classes=out_classes).to(device)
elif args.network == 'UNETVL':
    from networks.UNETVL import UNETR_LSTM
    model = UNETR_LSTM(input_dim=args.n_channels, output_dim=out_classes).to(device)
elif args.network == 'SuperLightNet':
    from networks.superlightnet import NormalU_Net
    model = NormalU_Net(init_channels=args.n_channels,class_nums=out_classes,depths_unidirectional='small').to(device)
elif args.network == 'WNet':
    from networks.nnWNet import WNet3D
    model = WNet3D(in_channel=args.n_channels, num_classes=out_classes, deep_supervised=False).to(device)
elif args.network == 'SwinSMT':
    from networks.SwinSMT import SwinSMT
    model = SwinSMT(
            in_channels=args.n_channels,
            out_channels=out_classes,
            img_size=(128, 128, 128),
            spatial_dims=3,
            use_v2=False,
            feature_size=48,
            use_moe=True,
            num_experts=4,
            num_layers_with_moe=2
            ).to(device)
elif args.network == 'PHNet':
    from networks.phnet import PHNet
    model = PHNet(
            res_ratio=5 / 0.74,
            layers=(15, 4),
            in_channels=args.n_channels,  
            out_channels=out_classes,  
            embed_dims=(42, 84, 168, 168, 336),
            segment_dim=(8, 8),
            mlp_ratio=4.0,
            dropout_rate=0.2
        ).to(device)
elif args.network == 'SegRWKV':
    from networks.SegRWKV import SegRWKV
    model = SegRWKV(args.n_channels, out_classes, kn=(28, 36, 48, 64, 80), FMU='sub').to(device)

                                          
                                          
print('Chosen Network Architecture: {}'.format(args.network))

                                                                                                                                                                                    
                                                                   
                                                                 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
   print("Let's use", torch.cuda.device_count(), "GPUs!")
   model = nn.DataParallel(model)
model.to(device)
                                            
                                     
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
print('Loss for training: {}'.format('DiceCELoss'))

if args.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
print('Optimizer for training: {}, learning rate: {}'.format(args.optim, args.lr))
                                                                                                     
scaler = GradScaler()


global_step = 0

def forward_for_infer(x):
    p = model(x)

                             
    if isinstance(p, torch.Tensor):
        return p
    elif isinstance(p, (list, tuple)):
                               
        return p[0]
    elif isinstance(p, dict):
                                             
        return p["out"]
    else:
        raise TypeError(f"Unexpected model output type: {type(p)}")


def validation(epoch_iterator_val, max_batches=8):
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            if step >= max_batches:
                break
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            if args.network == 'SegRWKV':
                val_inputs = val_inputs.permute(0, 1, 4, 2, 3)
                val_labels = val_labels.permute(0, 1, 4, 2, 3)
            if out_classes == 9:
                val_labels[val_labels==5] = 0
                val_labels[val_labels==9] = 0
                val_labels[val_labels==10] = 0
                val_labels[val_labels==12] = 0
                val_labels[val_labels==13] = 0
                val_labels[val_labels==11] = 5
            
            with autocast(enabled=False):
                val_outputs = sliding_window_inference_1out(val_inputs, (args.img_size[0], args.img_size[1], args.img_size[2]), args.val_batch, forward_for_infer, overlap=args.overlap)

                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]

                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice = dice_metric.aggregate().item()
                dice_vals.append(dice)
                epoch_iterator_val.set_description("Validate (%d / %d Steps) (dice=%2.5f mean_dice=%2.5f)" % (global_step, 10.0, dice, np.mean(dice_vals)))
            dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    return mean_dice_val

def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        if args.network == 'SegRWKV':
                x = x.permute(0, 1, 4, 2, 3)
                y = y.permute(0, 1, 4, 2, 3)
        with autocast(enabled=False):
            p = model(x)
            P = []
            if isinstance(p, torch.Tensor):
                P = [p]
            elif isinstance(p, (list, tuple)):
                P = list(p)                                    
            else:
                raise TypeError(f"Unexpected model output type: {type(p)}")
            if out_classes == 9:       
                y[y==5] = 0
                y[y==9] = 0
                y[y==10] = 0
                y[y==12] = 0
                y[y==13] = 0
                y[y==11] = 5
            if args.ds == True:
                ss = [[0],[1],[2],[3]]
            else:
                ss = [[0]]

                      
            loss = 0.0
            
            for s in ss:
                iout = 0.0
                if(s==[]):
                    continue
                for idx in range(len(s)):
                    iout += F.interpolate(P[s[idx]], (y.shape[-3],y.shape[-2],y.shape[-1]), mode='trilinear')
                loss += loss_function(iout, y)
                
        scaler.scale(loss).backward()
                                                   
        scaler.step(optimizer)
                                             
        scaler.update()
        epoch_loss += loss.item()
        optimizer.zero_grad()
        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))

        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(args.output_parameter))
                print("###########  Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val))
            else:
                print("Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val))
        global_step += 1
    return global_step, dice_val_best, global_step_best

max_iterations = args.max_iter
print('Maximum Iterations for training: {}'.format(str(args.max_iter)))
eval_num = args.eval_step
post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
global_step = 1 
dice_val_best = 0.0
global_step_best = 1
epoch_loss_values = []
metric_values = []

                   
                   
args.overlap_mode = 'gaussian'

if args.mode == 'train':
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)

model.load_state_dict(torch.load(os.path.join(args.output_parameter)))
model.eval()

epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)

def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
                                             
                                           
        return dice, hd95               
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0       
    else:
        return 0, 0       

def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    else:
        return 0

def validation_last(epoch_iterator_val):
    model.eval()
    dice_vals = []
    hd_vals = []
    
    per_class_dice = []
    per_class_hd = []
    
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            if args.network == 'SegRWKV':
                val_inputs = val_inputs.permute(0, 1, 4, 2, 3)
                val_labels = val_labels.permute(0, 1, 4, 2, 3)
            
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)

            if out_classes == 9:
                val_labels[val_labels == 5] = 0
                val_labels[val_labels == 9] = 0
                val_labels[val_labels == 10] = 0
                val_labels[val_labels == 12] = 0
                val_labels[val_labels == 13] = 0
                val_labels[val_labels == 11] = 5
            
            with autocast(enabled=False):
                val_outputs = sliding_window_inference_1out(
                    val_inputs, 
                    (args.img_size[0], args.img_size[1], args.img_size[2]), 
                    args.val_batch, 
                    forward_for_infer, 
                    overlap=args.overlap,
                    mode=args.overlap_mode
                )
                val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
                val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
                val_outputs = resample_3d(val_outputs, target_shape)
                dice_list_sub = []
                hd_list_sub = []
                for i in range(1, out_classes):
                    organ_Dice, organ_hd = calculate_metric_percase(val_outputs == i, val_labels == i)                                         
                    dice_list_sub.append(organ_Dice)
                    hd_list_sub.append(organ_hd)
                mean_dice = np.mean(dice_list_sub)
                mean_hd = np.mean(hd_list_sub)
                          
                                                                                                                             
                  

                                              
                per_class_dice.append(dice_list_sub)
                                      
                per_class_hd.append(hd_list_sub)
                                    

                dice_vals.append(mean_dice)
                hd_vals.append(mean_hd)
                
                epoch_iterator_val.set_description(
                    "Validate (%d Steps) (mean_dice=%2.5f mean_hd=%2.5f)" % (step, np.mean(dice_vals), np.mean(hd_vals))
                )
        
                              
                                                                                   
    mean_dice_val = np.mean(dice_vals)
    mean_hd_val = np.mean(hd_vals)

                                                                 
    return np.array(per_class_dice), np.array(per_class_hd), mean_dice_val, mean_hd_val


import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from matplotlib import cm

def generate_colormap(num_classes):
    """"""
                                          
    color_list = [
        (0, 0, 0),                              
        (255, 0, 0),             
        (0, 255, 0),               
        (135, 206, 250),             
        (255, 165, 0),              
        (255, 0, 255),               
        (128, 0, 128),              
        (255, 255, 0),              
        (205, 133, 63),           
        (128, 128, 0),             
        (75, 0, 130),               
        (0, 255, 128),            
        (0, 0, 255),              
        (255, 20, 147)                
    ]
    
                                                            
    assert num_classes <= len(color_list), "Not enough colors specified for the number of classes."
    
                                                         
    colormap = {cls: color_list[cls] for cls in range(num_classes)}
    return colormap


def overlay_segmentation(image, segmentation, colormap):
    """"""
    overlay = np.stack([image] * 3, axis=-1)                            
    for cls, color in colormap.items():
        if cls == 0:                                                
            continue
        overlay[segmentation == cls] = np.array(color)
    return overlay

def validation_save(epoch_iterator_val): 
    model.eval()
    dice_vals = []
    hd_vals = []
    per_class_dice = []
    per_class_hd = []

                                                     
                                               

    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            if args.network == 'SegRWKV':
                val_inputs = val_inputs.permute(0, 1, 4, 2, 3)
                val_labels = val_labels.permute(0, 1, 4, 2, 3)
            
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
                                                                      
                                                       

            if out_classes == 9:
                val_labels[val_labels == 5] = 0
                val_labels[val_labels == 9] = 0
                val_labels[val_labels == 10] = 0
                val_labels[val_labels == 12] = 0
                val_labels[val_labels == 13] = 0
                val_labels[val_labels == 11] = 5
            
            with autocast(enabled=False):
                val_outputs = sliding_window_inference_1out(val_inputs, (args.img_size[0], args.img_size[1], args.img_size[2]), args.val_batch, forward_for_infer, 
                    overlap=args.overlap,
                    mode=args.overlap_mode
                )
                val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
                val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
                val_outputs = resample_3d(val_outputs, target_shape)
                
                                                               
                                     
                                                                                   

                                                      
                                                                                                                    
                                                                              
                                                                                                
                                                                                                                 
                                                                             
                    
                                                                                     
                                                               
                                                            
                                                                                                 
                                                                                             

                                   
                                                                              
                                                                              
                                                                        
                    
                                                                   
                                                                  
                                                                

                                                                                                       
                                                                                                     
                                                                                                 

                                        
                                                                                  
                                     
                                                                                 
                                                                             

                dice_list_sub = []
                hd_list_sub = []
                for i in range(1, out_classes):                                   
                    organ_Dice, organ_hd = calculate_metric_percase(val_outputs == i, val_labels == i)
                    dice_list_sub.append(organ_Dice)
                    hd_list_sub.append(organ_hd)
                mean_dice = np.mean(dice_list_sub)
                mean_hd = np.mean(hd_list_sub)

                                              
                per_class_dice.append(dice_list_sub)
                per_class_hd.append(hd_list_sub)
                dice_vals.append(mean_dice)
                hd_vals.append(mean_hd)
                
                epoch_iterator_val.set_description(
                    "Validate (%d Steps) (mean_dice=%2.5f mean_hd=%2.5f)" % (step, np.mean(dice_vals), np.mean(hd_vals))
                )
        
    mean_dice_val = np.mean(dice_vals)
    mean_hd_val = np.mean(hd_vals)

    return np.array(per_class_dice), np.array(per_class_hd), mean_dice_val, mean_hd_val

                                

def save_metrics_to_csv(trained_weights, dataset_name, network_name, overlap, overlap_mode, class_labels, per_class_dice, per_class_hd, mean_dice_val, mean_hd_val, csv_filename):
                                                        
    file_exists = os.path.isfile(csv_filename)
    
                             
    header = ["Trained_Weights", "Dataset", "Network", "Overlap", "OverlapdMode"]
    
                                                             
    header += [f"Dice_{class_labels[str(i)]}" for i in range(len(class_labels))] + ["Mean_Dice"]
    
                                                           
    header += [f"HD_{class_labels[str(i)]}" for i in range(len(class_labels))] + ["Mean_HD"]
    
                          
    row = [
        trained_weights,                             
        dataset_name,                   
        network_name,                    
        overlap,
        overlap_mode
    ]
    
                                   
    row += [f"{dice:.4f}" for dice in per_class_dice.mean(axis=0)]                                                    
    
                            
    row.append(f"{mean_dice_val:.4f}")
    
                                 
    row += [f"{hd:.4f}" for hd in per_class_hd.mean(axis=0)]                                                  
    
                          
    row.append(f"{mean_hd_val:.4f}")

def print_metrics_to_terminal(trained_weights, dataset_name, network_name, overlap, overlap_mode,
                              class_labels, per_class_dice, per_class_hd, mean_dice_val, mean_hd_val):
    print("\n==================== Validation Metrics ====================")
    print(f"Trained Weights: {trained_weights}")
    print(f"Dataset       : {dataset_name}")
    print(f"Network       : {network_name}")
    print(f"Overlap       : {overlap}")
    print(f"Overlap Mode  : {overlap_mode}")
    print("-------------------------------------------------------------")
    
             
    class_names = [class_labels[str(i)] for i in range(len(class_labels))]
    class_names_no_first = class_names[1:]
    header_dice = " | ".join([f"{name:^10}" for name in class_names_no_first])
    header_hd = header_dice
    
    print("\nDice per Class:")
    print(header_dice)
    print("-" * len(header_dice))
    
    avg_dice_per_class = per_class_dice.mean(axis=0)
    dice_row = " | ".join([f"{dice:.4f}".center(10) for dice in avg_dice_per_class])
    print(dice_row)
    print(f"Mean Dice: {mean_dice_val:.4f}")

    print("\nHD per Class:")
    print(header_hd)
    print("-" * len(header_hd))
    
    avg_hd_per_class = per_class_hd.mean(axis=0)
    hd_row = " | ".join([f"{hd:.4f}".center(10) for hd in avg_hd_per_class])
    print(hd_row)
    print(f"Mean HD: {mean_hd_val:.4f}")
    
    print("=============================================================\n")



                                                         
per_class_dice, per_class_hd, mean_dice_val, mean_hd_val = validation_save(epoch_iterator_val)


if out_classes == 9:
    class_labels = {
        "0": "background",
        "1": "spleen",
        "2": "rkid",                 
        "3": "lkid",                
        "4": "gall",                
        "5": "pancreas",
        "6": "liver",
        "7": "sto",             
        "8": "aorta",
    }
else:
    class_labels = {
        "0": "BG",
        "1": "Spl.",
        "2": "R.kid",   
        "3": "L.kid",   
        "4": "Gall.",   
        "5": "Eso.",    
        "6": "Liver",
        "7": "Stom.",    
        "8": "Aorta",
        "9": "IVC",   
        "10": "Panc.",
        "11": "RAG",
        "12": "LAG",   
        "13": "Duo.",    
        "14": "Blad.",    
        "15": "Pros.",       
    }

                                                                                                                                                 
                                                                                                                                                                      

print_metrics_to_terminal(args.output_parameter, args.dataset, args.network, args.overlap, args.overlap_mode, class_labels, per_class_dice, per_class_hd, mean_dice_val, mean_hd_val)