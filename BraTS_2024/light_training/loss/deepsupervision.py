import torch 
import torch.nn as nn 
import numpy as np 

class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, *args):
        for i in args:
            assert isinstance(i, (tuple, list)), "all args must be either tuple or list, got %s" % type(i)
                                                                                                                 
                                                   

        if self.weight_factors is None:
            weights = [1] * len(args[0])
        else:
            weights = self.weight_factors

                                                                                                                   
                          
        l = weights[0] * self.loss(*[j[0] for j in args])
        for i, inputs in enumerate(zip(*args)):
            if i == 0:
                continue
            l += weights[i] * self.loss(*inputs)
        return l
    


class AutoDeepSupervision(nn.Module):
    def __init__(self, loss, label_scale) -> None:
        super().__init__()

        weights = np.array([1 / (2 ** i) for i in range(len(label_scale))])
        weights[-1] = 0
                                                                                    
        weights = weights / weights.sum()
        print(f"loss weights is {weights}")

        self.warpper = DeepSupervisionWrapper(loss, weights)
        self.label_scale = label_scale
    
    def forward(self, preds, label):
        pred_len = len(preds)
        assert pred_len == len(self.label_scale)
        labels = []
        for scale in self.label_scale:
            labels.append(torch.nn.functional.interpolate(label, scale_factor=scale, mode="nearest"))
                                                                                                              
                                                                                                              
                                                                                                              
                                                                                                               
                                                                       

        return self.warpper(preds, labels)