import numpy as np
from light_training.dataloading.dataset import get_train_test_loader_from_test_list
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from light_training.evaluation.metric import dice
import os
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class Dice_loss(nn.Module):
    def __init__(self, n_classes):
        super(Dice_loss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        inputs = inputs.float()
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def filte_state_dict(sd):
        if "module" in sd :
            sd = sd["module"]
        new_sd = {}
        for k, v in sd.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k 
            new_sd[new_k] = v 
        del sd 
        return new_sd

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=patch_size,
                                        sw_batch_size=2,
                                        overlap=0.5)
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.train_process = 12

        from models.SegRWKV import SegRWKV
        self.model = SegRWKV(4, 4, kn=(28, 36, 48, 64, 80), FMU='sub')

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                    momentum=0.99, nesterov=True)
        self.scheduler_type = "poly"
        
        self.loss_func = nn.CrossEntropyLoss()
        
        self.loss_dice = Dice_loss(n_classes = 4)
      
    def convert_labels(self, labels):
                        
                                  
                          
                      
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
        
        return torch.cat(result, dim=1).float()

    def training_step(self, batch):
        import time 
        image, label = self.get_input(batch)

        pred1,pred2,pred3,pred4 = self.model(image)
        loss = self.loss_func(pred1, label) + self.loss_dice(pred1, label, softmax=True)
        
        label2 = F.interpolate(label.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze(1).long()
        label3 = F.interpolate(label.unsqueeze(1).float(), scale_factor=0.25, mode='nearest').squeeze(1).long()
        label4 = F.interpolate(label.unsqueeze(1).float(), scale_factor=0.125, mode='nearest').squeeze(1).long()

        loss += (self.loss_func(pred2, label2) + self.loss_dice(pred2, label2, softmax=True))
        loss += (self.loss_func(pred3, label3) + self.loss_dice(pred3, label3, softmax=True))
        loss += (self.loss_func(pred4, label4) + self.loss_dice(pred4, label4, softmax=True))
        loss = loss / 4.0

        self.log("train_loss", loss, step=self.global_step)
        return loss 

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]

        label = label[:, 0].long()

        return image, label 

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])
        
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        
        else:
            return np.array([0.0, 50])
    
    def validation_step(self, batch):
        image, label = self.get_input(batch)
       
        output = self.model(image)[0].argmax(dim=1)
        output = output.cpu().numpy()
        target = label.cpu().numpy()
        
        dices = []

        c = 4
        for i in range(1, c):
            pred_c = output == i
            target_c = target == i

            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)
        
        return dices
    
    def validation_end(self, val_outputs):
        dices = val_outputs

        dices_mean = []
        c = 3
        for i in range(0, c):
            dices_mean.append(dices[i].mean())

        mean_dice = sum(dices_mean) / len(dices_mean)
        
        print("*" * 50)
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            path = os.path.join(model_save_path, parameter_name)
            save_new_model_and_delete_last(self.model, 
                                            path, 
                                            delete_symbol="best_model")
            print("Save best model!!!  Path: {}".format(path))

        print(f"epoch {self.epoch} mean_dice is {mean_dice}, best_mean_dice is {self.best_mean_dice}")


parameter_save_path = "./data/3D_parameter"
parameter_name = "SegRWKV_BraTS_2023.pth"
env = "DDP"
model_save_path = os.path.join(parameter_save_path)
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 2
device = "cuda:0"
patch_size = [128, 128, 128]

augmentation = True 


if __name__ == "__main__":
    data_dir = "./data/train_fullres_process"
    
    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            
                            logdir=parameter_save_path,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17745,
                            training_script=__file__)

    from data.test_list import test_list
    
    train_ds, test_ds = get_train_test_loader_from_test_list(data_dir=data_dir, test_list=test_list)

    trainer.train(train_dataset=train_ds, val_dataset=test_ds)
                         
                          


