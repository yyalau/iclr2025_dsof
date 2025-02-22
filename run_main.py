import subprocess
import os
import time

from misc.runnerbase import Utility, BaseCLI

def base(path= ''):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


class MainCLI(BaseCLI):
    def __init__(self, main_models, student_models, opts, 
                 trainers, test_run, mode = 'zip',):
        super().__init__(main_models, student_models, opts, trainers, 
                         test_run, mode)
        
    def create_cli(self, sett, dataset, pred_len, gpu_to_use, itr = 1):    

        command = f"CUDA_VISIBLE_DEVICES={gpu_to_use} "            
        
        if not self.test_run: command += 'nohup '
        command += f"python -u main.py --itr {itr} --pred_len {pred_len} --data {dataset} --num_workers {num_workers} "
        command +=f"--y_model_main {sett['main_model']} "
        command += f"--y_model_student {sett['student_model']} "
        command += f"--y_opt {self.get_opt_sett(sett['main_model'], sett['student_model'], sett['opt'], dataset) } "
        command += f"--y_trainer {sett['trainer']} --comments {sett['comment']} "
        
        # TODO: create variable "settings" and reuse in exp_paths
        if self.test_run: 
            command += '--test_run'
        else:
            logger_file = self.get_nohup_filename(sett, dataset, pred_len)
            command += f'> {logger_file} 2>&1 &'
            os.makedirs(os.path.dirname(logger_file), exist_ok=True)
            
        # print(command)
        return command
    def get_nohup_filename(self, sett, dataset, pred_len):
        
        folder = f"./logs/nohup/exp/{Utility.get_header(sett['main_model'])}"
        folder += f"-{Utility.get_header(sett['student_model'])}"
        folder += f"-{Utility.get_header(sett['trainer'])}"

        return f"{folder}/{dataset}-{pred_len}.out"


    def save_outputpath_file(self, exp_paths):
        super().save_outputpath_file(exp_paths, 'exps')
        
    def get_comment(self, main_model, student_model, trainer):
        
        cm = f"exp--{Utility.replace_slash(main_model)}" 
        cm += f"--{Utility.replace_slash(student_model)}--{Utility.replace_slash(trainer)}"
        
        return cm
    
    def get_exp_path(self, sett):
        path = self.get_output_path(sett)
        return "exps/" + path


gpus = {
    'Electricity': 3,
    'ETTh2': 3,
    'ETTm1': 4,
    'WTH': 4,
    'Traffic': 4,
    'Exchange': 4,
    'ILI': 4,
    'Weather': 4,
    'others': 4,
}


pred_lens = [  48 ]
# pred_lens = [  24 ]

mode = 'product'                                    

itr = 1
test_run = True

datasets = [   'ETTh2',  ]
# datasets = ['ETTh2', 'Exchange', 'Electricity']

main_models = [  'DLinear' ]
# main_models = ['DLinear', 'FITS', 'FSNet', 'iTransformer', 'NSTransformer', 'OneNet', 'PatchTST',  ]


student_models = ['MLP']
opts = ['w_student',]
trainers = ['w_student/residual/dsof', 'w_student/residual/td', 'w_student/residual/er', 'w_student/residual/baseline', 'w_student/non_residual/ertd' ]



utility = Utility(process_threshold=12)
mainCLI = MainCLI(main_models, student_models, opts, 
                  trainers, test_run, mode, )
num_workers = 8 if test_run else 0


######### completed



pids = []
exp_paths = []
# subprocess.run(['cd', base()], shell=True)
os.chdir(base())
os.makedirs('./logs/nohup', exist_ok=True)
subprocess.run(['pwd'], shell=True)


utility.pid_gpu_cmd_mapping()
print(len(mainCLI.get_settings()))

for sett in mainCLI.get_settings():
    for dataset in datasets:                
        for pred_len in pred_lens:
            
            gpu_to_use = gpus[dataset]
                
                
            main_model = mainCLI.get_model_name(sett['main_model'], None)
            student_model = mainCLI.get_model_name(sett['student_model'], dataset)
                        
            # utility.cpu_available()
            if not test_run:
                while not utility.cpu_available():
                    time.sleep(60)
                
                while True:
                    gpu_to_use = utility.allocate_gpu(gpu_to_use, dataset, 
                                                      main_model, student_model, 
                                                      gpus)                
                    if gpu_to_use is not None: break
                    
                    time.sleep(60)
                    utility.update_estimated_free_gpu()
                    
            
            print(f'Running {dataset} with {sett["comment"]} for pred_len {pred_len} in gpu {gpu_to_use}')
            command = mainCLI.create_cli(sett, dataset, pred_len, 
                                         gpu_to_use, itr)
            subprocess.run(command, shell=True)

            if test_run: break
            time.sleep(1)
            
        if test_run: break
        time.sleep(1)
        
    if not test_run:
        exp_paths.append(mainCLI.get_exp_path(sett))        
        time.sleep(2)

if not test_run:
    mainCLI.save_outputpath_file(exp_paths)




