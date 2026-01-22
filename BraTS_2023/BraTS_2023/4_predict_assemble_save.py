from light_training.dataloading.dataset import get_test_loader_from_test_list
import torch 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.evaluation.metric import dice
set_determinism(123)
import os
from light_training.prediction import Predictor
import SimpleITK as sitk
from medpy import metric
import os, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_dir = "./data/train_fullres_process"
env = "pytorch"
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"

patch_size = [128, 128, 128]

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        self.patch_size = patch_size
        self.augmentation = False

    def convert_labels(self, labels):
                        
        result = [labels == 0, (labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
        
        return torch.cat(result, dim=1).float()
    
    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
        properties = batch["properties"]
                                                                   
                                                   
        raw_label = label
        label = self.convert_labels(label)
        
        return image, label, properties, raw_label
    
    def define_model(self):

        from models.SegRWKV import SegRWKV
        model = SegRWKV(4, 4, kn=(28, 36, 48, 64, 80), FMU='sub')

        model_path = "./data/3D_parameter/SegRWKV_BraTS_2023.pth"
        new_sd = self.filte_state_dict(torch.load(model_path, map_location="cpu"))
        model.load_state_dict(new_sd)
        model.eval()
        window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.5,
                                        progress=True,
                                        mode="gaussian")

        predictor = Predictor(window_infer=window_infer,
                              mirror_axes=[0,1,2])

        file_name = os.path.splitext(os.path.basename(model_path))[0]
        save_path = "./data/3D_results/brats23"+"/"+file_name
        os.makedirs(save_path, exist_ok=True)
        save_visual_path = "./data/3D_visual_results/brats23"+"/"+file_name
        os.makedirs(save_visual_path, exist_ok=True)

        return model, predictor, save_path, save_visual_path

    def convert_labels_dim0(self, labels):
                        
        result = [labels == 0, (labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
        
        return torch.cat(result, dim=0).float()
    
    def convert_labels_unsqueeze(self, labels):
                        
        labels = labels.unsqueeze(dim=0)

        result = [labels == 0, (labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
        
        return torch.cat(result, dim=0).float()

    def validation_step(self, batch):
        image, label, properties, raw_label = self.get_input(batch)
    
        model, predictor, save_path, save_visual_path = self.define_model()
       
        model_output = predictor.maybe_mirror_and_predict(image, model, device=device)

                                                                           
        model_output = predictor.predict_raw_probability(model_output, properties=properties)            
        
        model_output = model_output.argmax(dim=0)

        model_output = predictor.predict_noncrop_probability(model_output, properties)
        predictor.save_to_nii(model_output, 
                              raw_spacing=[1,1,1],
                              case_name = properties['name'][0],
                              save_dir=save_path)
        
        voxel_spacing = [1, 1, 1]
        case_name = properties['name'][0]
        raw_data_dir = "./data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
        t1_itk = os.path.join(raw_data_dir, case_name, f"t1c.nii.gz")
        t1_itk = sitk.ReadImage(t1_itk)
        t1_array = sitk.GetArrayFromImage(t1_itk).astype(np.int32)

        gt_itk = os.path.join(raw_data_dir, case_name, f"seg.nii.gz")
        gt_itk = sitk.ReadImage(gt_itk)
        gt_array = sitk.GetArrayFromImage(gt_itk).astype(np.int32)

        gt_array = torch.from_numpy(gt_array)
        gt_array = self.convert_labels_unsqueeze(gt_array)
        model_final_output = torch.from_numpy(model_output)
        model_output = self.convert_labels_unsqueeze(model_final_output)
        print(model_output.shape)
        print(gt_array.shape)
        save_topk_slices(model_output, gt_array, case_name, t1_array, save_dir=save_visual_path, top_k=4, step=10, dpi=400)

        label = gt_array
        c = 4
        dices = []
        hd95s = []
        for i in range(1, c):
            output_i = model_output[i].cpu().numpy()
                                   
            label_i = label[i].cpu().numpy()
                                  
                                         
            dice, hd96 = cal_metric(output_i, label_i, voxel_spacing)
            dices.append(dice)
            hd95s.append(hd96)
        print(dices)
        print(hd95s)

        return dices, hd95s


    def filte_state_dict(self, sd):
        if "module" in sd :
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v 
        del sd 
        return new_sd

def cal_metric(pred, gt, voxel_spacing=[1,1,1]):
                       
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        return np.array([dice, hd95])
    else:
        return np.array([0.0, 50])

def save_topk_slices(pred, label, case_name, T1_bg, top_k=4, step=10, save_dir="./data", dpi=400):
    os.makedirs(save_dir, exist_ok=True)

    def to_np(x):
        if hasattr(x, "cpu"): 
            x = x.cpu()
        return np.array(x)

    pred, label, T1_bg = to_np(pred), to_np(label), to_np(T1_bg)                         
    C, Z, H, W = pred.shape

                       
    def collapse_labels(multi_hot):
        """multi-hot (C,Z,H,W) -> 单通道(Z,H,W)"""
        bg, tc, wt, et = multi_hot
        final = np.zeros_like(et, dtype=np.int32)
                                           
        final[et > 0] = 3
        final[(tc > 0) & (final == 0)] = 1
        final[(wt > 0) & (final == 0)] = 2
        return final

    pred_labels = collapse_labels(pred)            
    gt_labels   = collapse_labels(label)           

    fg_counts = np.array([np.sum(gt_labels[z] > 0) for z in range(Z)])
    sorted_idx = np.argsort(-fg_counts)
    idx = sorted_idx[::step][:top_k]

    cmaps = mcolors.CSS4_COLORS
    my_colors = ['red','blue','purple','darkorange','yellow','forestgreen','magenta','cyan','deeppink']
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:C-1]}
    cmap_list = list(cmap.values())
    alpha = 0.45

    def overlay(image2d, mask2d, colors):
        bg = (image2d - image2d.min()) / (image2d.max() - image2d.min() + 1e-8)
        img = np.stack([bg]*3, axis=-1)
        for c, col in enumerate(colors, start=1):
            m = (mask2d == c).astype(np.uint8)
            if m.sum() == 0: 
                continue
            img = img*(1 - m[...,None]*alpha) + m[...,None]*alpha*np.array(mcolors.to_rgb(col))
        return img

    def get_bbox(image2d, pad=2):
        coords = np.argwhere(image2d > 0)
        if coords.shape[0] == 0:
            return (0, image2d.shape[0], 0, image2d.shape[1])
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        y0 = max(y0 - pad, 0)
        y1 = min(y1 + pad, image2d.shape[0]-1)
        x0 = max(x0 - pad, 0)
        x1 = min(x1 + pad, image2d.shape[1]-1)
        return (y0, y1, x0, x1)

    out_files = []
    for rank, z in enumerate(idx):
        slice_img = T1_bg[z]

        img_gt   = overlay(slice_img, gt_labels[z], cmap_list)
        img_pred = overlay(slice_img, pred_labels[z], cmap_list)

                        
        y0, y1, x0, x1 = get_bbox(slice_img, pad=2)
        slice_img_crop = slice_img[y0:y1+1, x0:x1+1]
        img_gt   = img_gt[y0:y1+1, x0:x1+1, :]
        img_pred = img_pred[y0:y1+1, x0:x1+1, :]

        f_gt       = os.path.join(save_dir, f"{case_name}_z{z:03d}_rank{rank+1}_gt.png")
        f_pred     = os.path.join(save_dir, f"{case_name}_z{z:03d}_rank{rank+1}_pred.png")
        f_t1_crop  = os.path.join(save_dir, f"{case_name}_z{z:03d}_rank{rank+1}_T1_crop.png")

            
                                            
                            
                         
                                                              
                     

               
        plt.figure(figsize=(4,4), dpi=dpi)
        plt.imshow(img_pred)
        plt.axis("off")
        plt.savefig(f_pred, bbox_inches="tight", pad_inches=0)
        plt.close()

              
                                            
                                                 
                         
                                                                   
                     

        out_files.extend([f_gt, f_pred, f_t1_crop])

    return out_files

if __name__ == "__main__":

    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir="",
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=16667,
                            training_script=__file__)

    from data.test_list import test_list
    
    test_ds = get_test_loader_from_test_list(data_dir=data_dir, test_list=test_list)

    trainer.validation_single_gpu_two_metrics(test_ds)



