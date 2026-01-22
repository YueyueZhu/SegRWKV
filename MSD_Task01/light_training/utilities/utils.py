                                                                                                                       
                                
 
                                                                    
                                                                     
                                            
 
                                                   
 
                                                                        
                                                                      
                                                                             
                                                                        
                                   
from typing import Union

from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import re

def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False) -> None:
    try:
        a = np.load(npz_file)                                                                      
        if overwrite_existing or not isfile(npz_file[:-3] + "npy"):
            np.save(npz_file[:-3] + "npy", a['data'])
        if unpack_segmentation and (overwrite_existing or not isfile(npz_file[:-4] + "_seg.npy")):
            np.save(npz_file[:-4] + "_seg.npy", a['seg'])
    except KeyboardInterrupt:
        if isfile(npz_file[:-3] + "npy"):
            os.remove(npz_file[:-3] + "npy")
        if isfile(npz_file[:-4] + "_seg.npy"):
            os.remove(npz_file[:-4] + "_seg.npy")
        raise KeyboardInterrupt
    
def get_identifiers_from_splitted_dataset_folder(folder: str, file_ending: str):
    files = subfiles(folder, suffix=file_ending, join=False)
                                                              
    crop = len(file_ending) + 5
    files = [i[:-crop] for i in files]
                           
    files = np.unique(files)
    return files


def create_lists_from_splitted_dataset_folder(folder: str, file_ending: str, identifiers: List[str] = None) -> List[List[str]]:
    """
    does not rely on dataset.json
    """
    if identifiers is None:
        identifiers = get_identifiers_from_splitted_dataset_folder(folder, file_ending)
    files = subfiles(folder, suffix=file_ending, join=False, sort=True)
    list_of_lists = []
    for f in identifiers:
        p = re.compile(re.escape(f) + r"_\d\d\d\d" + re.escape(file_ending))
        list_of_lists.append([join(folder, i) for i in files if p.fullmatch(i)])
    return list_of_lists
