import os
import pathlib
import re

from .runnerbase import Utility

def find_min_mse(best_path,):
    if not os.path.isdir(best_path):
        return None
    
    pth_list = os.listdir(best_path)
    if pth_list == []: return None
    
    mse_list = [re.findall(r's\d+_test_(\d+\.\d+).pth', pth) for pth in pth_list]
    mse_list = [float(mse[0]) for mse in mse_list if mse]
    min_mse = min(mse_list) if mse_list else None      
    
    return min_mse

def is_min_mse(pth, min_mse):
    if min_mse is None:
        return False
    return re.match(rf's\d+_test_{min_mse:.4f}+.pth', pth)

def process_value(value):
    if value is None:
        return "-"
    
    return f"{value:.3f}"
def base(path= ''):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    
def get_main_model(path):
    return path.split('/')[1].split('-')[0]

def get_student_model(path):  
    path =  path.split('/')[1].split('-')[1]
    if path == 'placeholder': path = None
    return path

def get_full_trainer(path):
    # try:
    #     path = path.split('/')[2].split('--')[2]
    # except: 
    #     import ipdb; ipdb.set_trace()
    path = path.split('/')[2].split('--')[2]
    return path

def get_full_main_model(path):
    return path.split('/')[2].split('--')[0]

def get_full_student_model(path):
    return path.split('/')[2].split('--')[1]

###### filter bad paths ######
def filter_bad_paths(path_stored_file = base("./optuna/paths.txt")):
    # print(base())
    with open(path_stored_file) as f:
        paths = f.readlines()
        paths = [x.strip() for x in paths]

    paths = sorted(list(set(paths)))
    paths = [x for x in paths if pathlib.Path(base(x)).exists()]

    with open(path_stored_file, 'w') as f:
        for p in paths:
            f.write(p + '\n')

    return paths        


def get_output_path(sett):
    
    path = f"{Utility.get_header(sett['main_model'])}"
    path += f"-{Utility.get_header(sett['student_model'])}"
    path += f"-{Utility.get_header(sett['trainer'])}"
    path += f"/{Utility.replace_slash(sett['main_model'])}"
    path += f"--{Utility.replace_slash(sett['student_model'])}"
    path += f"--{Utility.replace_slash(sett['trainer'])}"
    
    return path