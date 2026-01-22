import numpy as np 
from typing import Union, Tuple
import time 

class DataLoaderMultiProcess:
    def __init__(self, dataset, 
                 patch_size,
                 batch_size=2,
                 oversample_foreground_percent=0.33,
                 probabilistic_oversampling=False,
                 print_time=False):
        pass
        self.dataset = dataset
        self.patch_size = patch_size
                                                                            
        self.batch_size = batch_size
        self.keys = [i for i in range(len(dataset))]
        self.thread_id = 0
        self.oversample_foreground_percent = oversample_foreground_percent
        self.need_to_pad = (np.array([0, 0, 0])).astype(int)

        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling\
            else self._probabilistic_oversampling
        self.data_shape = None 
        self.seg_shape = None
        self.print_time = print_time

    def determine_shapes(self):
                       
        item = self.dataset.__getitem__(0)
        data, seg, properties = item["data"], item["seg"], item["properties"]
        num_color_channels = data.shape[0]
        num_output_channels = seg.shape[0]
        patch_size = self.patch_size
        data_shape = (self.batch_size, num_color_channels, patch_size[0], patch_size[1], patch_size[2])
        seg_shape = (self.batch_size, num_output_channels, patch_size[0], patch_size[1], patch_size[2])
        return data_shape, seg_shape
    
    def generate_train_batch(self):
        
        selected_keys = np.random.choice(self.keys, self.batch_size, True, None)
        if self.data_shape is None:
            self.data_shape, self.seg_shape = self.determine_shapes()

        data_all = np.zeros(self.data_shape, dtype=np.float32)
        data_all_global = np.zeros(self.data_shape, dtype=np.float32)
        data_global = None
        seg_all = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []

        index = 0
        for j, key in enumerate(selected_keys):

            force_fg = self.get_do_oversample(j)
            s = time.time()
            item = self.dataset.__getitem__(key)
            e = time.time()
            if self.print_time:
                print(f"read single data time is {e - s}")
                                                 
            data, seg, properties = item["data"], item["seg"], item["properties"]
            
            if "data_global" in item:
                data_global = item["data_global"]

            case_properties.append(properties)
                                                                                                                        
                                                                   
            shape = data.shape[1:]
            dim = len(shape)
            
            s = time.time()
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            e = time.time()
            if self.print_time:
                print(f"get bbox time is {e - s}")
                                                                                                                      
                                                                                                                       
                                                                                                                    
                   
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

                                                                                                                   
                                                                                                                    
                                                                                                                  
                                                                                           
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]


            s = time.time()
            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
                                                                         
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=0)

            if data_global is not None :
                data_all_global[j] = data_global

            e = time.time()
            if self.print_time:
                print(f"box is {bbox_lbs, bbox_ubs}, padding is {padding}")
                print(f"setting data value time is {e - s}")
                
        
        if data_global is None:
            return {'data': data_all,
                    'seg': seg_all, 'properties': case_properties, 
                    'keys': selected_keys}
    
        return {'data': data_all, "data_global": data_all_global, 
                    'seg': seg_all, 'properties': case_properties, 
                    'keys': selected_keys}

    def __next__(self):
    
        return self.generate_train_batch() 
    
    def set_thread_id(self, thread_id):
        self.thread_id = thread_id
    
                                                                     
             
                                                                                               
             
                                                                                                   
    
    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        if self.batch_size == 1:
            return True
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
                                
        return np.random.uniform() < self.oversample_foreground_percent
    
    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
                                                                                                                     
                                       
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
                                                                                                                  
                    
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

                                                                                                             
                                                                                                      
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

                                                                                                                    
                                                             
        if not force_fg:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
                                               
        else:
            assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
            if overwrite_class is not None:
                assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not '\
                                                                    'have class_locations (missing key)'
                                                                                            
                                                    
            eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                                                                                                                                      
                                                      
                                                                                                                     
                                                                                                                             
                          
                                                          
                                                                          

            if len(eligible_classes_or_regions) == 0:
                                                                                           
                selected_class = None
                if verbose:
                    print('case does not contain any foreground classes')
            else:
                                                                           
                                                                  
                selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if\
                    (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                                                                                   
          
            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                                                                                                       
                                                                
                                                          
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                                                                                                       
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs               