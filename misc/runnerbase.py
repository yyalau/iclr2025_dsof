import subprocess
import os
import yaml
# import time
import csv


def base(path= ''):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)


class Utility:
    
    def __init__(self, process_threshold = 15):
        self.process_threshold = process_threshold
        with open(base('./gpu_capacity.yaml')) as f:
            self.capacity = yaml.load(f, Loader=yaml.FullLoader)

        self.estimated_free = self.get_gpu_memory().copy()

    def cpu_available(self):
        condition =  self.num_processes() < self.process_threshold
        if not condition: print('server too busy...')
        return condition

    def update_estimated_free_gpu(self):
        self.estimated_free = self.get_gpu_memory().copy()

    @staticmethod
    def num_processes(username = 'yyalau'):
        python_processes = subprocess.Popen(['pgrep', '-u', username, 'python', '-a'], stdout=subprocess.PIPE)
        tuning = subprocess.Popen(['grep', '-c', 'python -u tuning.py'], stdin=python_processes.stdout, stdout=subprocess.PIPE,)
        tuning_count, _ = tuning.communicate()
        tuning.stdout.close()
        python_processes.stdout.close()
                
        python_processes = subprocess.Popen(['pgrep', '-u', username, 'python', '-a'], stdout=subprocess.PIPE)
        main = subprocess.Popen(['grep', '-c', 'python -u main.py'], stdin=python_processes.stdout, stdout=subprocess.PIPE,)
        main_count, _ = main.communicate()
        main.stdout.close()
        python_processes.stdout.close()
        
        y = lambda x: int(x.decode('utf-8'))
        tuning_count, main_count = y(tuning_count), y(main_count)
        
        print(f'[number of processes] tuning: {tuning_count}, main: {main_count}')

        return tuning_count + main_count

    @staticmethod
    def get_gpu_memory():
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        gpu5_free = memory_free_values.pop(-1)
        memory_free_values.insert(5, gpu5_free)
        return memory_free_values
    
    @staticmethod
    def get_gpu_usage():
        command = "nvidia-smi --query-gpu=memory.used --format=csv"
        memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        gpu5_free = memory_free_values.pop(-1)
        memory_free_values.insert(5, gpu5_free)
        return memory_free_values
    
    def gpumem_available(self, gpu_no, dataset, main_model, student_model):
        student_cap = self.capacity[dataset][student_model] if student_model is not None else 0
        
        return self.estimated_free[gpu_no] > self.capacity[dataset][main_model] + student_cap
    
    @staticmethod
    def get_gpu_names(gpu_distribution):
        # command = "nvidia-smi --query-gpu=name --format=csv"
        # gpu_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[1:-1]
        # return gpu_info
        x = []
        
        x = [ v for v in gpu_distribution.values()]
        return set(x)
    
    @staticmethod
    def get_header(x):
        if x is None: return 
        return x.split('/')[0]
            
    @staticmethod
    def replace_slash(x):
        if x is None: return
        return x.replace('/', '_')
    
    def allocate_gpu(self, gpu_to_use, dataset, main_model, student_model, gpus_default_dist):
        if gpu_to_use is not None and self.gpumem_available(gpu_to_use, dataset, main_model, student_model): 
            self.estimated_free[gpu_to_use] -= self.capacity[dataset][main_model]
            
            if student_model is not None:
                self.estimated_free[gpu_to_use] -= self.capacity[dataset][student_model]
                
            print(f'gpu {gpu_to_use} is available; estimated free memory: {self.estimated_free[gpu_to_use]}')
            return gpu_to_use
        
        for k in self.get_gpu_names(gpus_default_dist):
            if self.gpumem_available(k, dataset, main_model, student_model):
                self.estimated_free[k] -= self.capacity[dataset][main_model]
                if student_model is not None:
                    self.estimated_free[k] -= self.capacity[dataset][student_model]
                print(f'gpu {k} is available; estimated free memory: {self.estimated_free[k]}')
                return k
            print(f'gpu {k} is not available; estimated free memory: {self.estimated_free[k]}')
            
        print('none of the gpus are available...')
        return None

    @staticmethod
    def pid_gpu_cmd_mapping( username = 'yyalau'):
        hostname = subprocess.check_output("hostname").decode('ascii')[:-1].split('.')[0]

        hex2dec_busmap = { 
            'dycpu3': {
                '1A': 0,
                '1B': 1,
                '1D': 2,
                '1E': 3,
                '3D': 4,
                '3F': 5,
                '40': 6,
                '41': 7,
            },
            'dycpu4': {
                '1B': 0,
                '1C': 1,
                '1D': 2,
                '1E': 3,
                '3D': 4,
                '41': 5,
                '3F': 6,
                '40': 7,
            },
            'dycpu5': {
                '01': 0,
                '25': 1,
                '41': 2,
                '61': 3,
                
            },
            'dycpu6': {
                '81': 0,
                'A1': 1,
                'C1': 2,
                'E1': 3
            }
            
        }
        
        
        gpu_command = "nvidia-smi --query-compute-apps=pid,used_memory,gpu_bus_id --format=csv,noheader,nounits"
        gpu_processes = subprocess.check_output(gpu_command.split()).decode('ascii').split('\n')[1:-1]
        
        gpu_info = {}
        for process in gpu_processes:
            pid, used_memory, bus_id = process.split(', ')
            gpu_info[pid] = {
                'used_memory': used_memory,
                'bus_id': hex2dec_busmap[hostname][bus_id.split(':')[1]]
            }
        # import ipdb; ipdb.set_trace()
        
        python_command = f"pgrep python -u {username}"
        python_processes = subprocess.check_output(python_command.split()).decode('ascii').split('\n')[:-1]


        unrelated_processes = set([x for x in gpu_info.keys()]) | set([x for x in python_processes]) 
        unrelated_processes -= set([x for x in gpu_info.keys()]) & set( [x for x in python_processes])
        
        for pid in unrelated_processes:
            gpu_info.pop(pid, None)

        # import ipdb; ipdb.set_trace()

        for pid, info in gpu_info.items():
            pid_cmd = subprocess.check_output(f"ps -p {pid} -o cmd=".split()).decode('ascii')[:-1].split()        
            pid_cmd = pid_cmd[6::2]
            info.update({'dataset': pid_cmd[1],
                        'pred_len': pid_cmd[0],
                        'main_model': pid_cmd[3],
                        'student_model': pid_cmd[4],
                        'trainer': pid_cmd[6],
                        'opt': pid_cmd[5],
                        'comment': pid_cmd[7]})
        
        with open(base(f'./logs/{hostname}_processes.csv'), 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['pid', 'used_memory', 'gpu_id', 'dataset','pred_len',  'main_model', 'student_model', 'opt', 'trainer', 'comment'])
            for pid, info in gpu_info.items():
                csv_writer.writerow([pid, info['used_memory'], info['bus_id'],  info['dataset'], info['pred_len'],
                                     info['main_model'], info['student_model'], info['opt'],
                                     info['trainer'], info['comment']])
        
        print(f'Processes written to {base("./logs/"+hostname +"_processes.csv")}')
        return gpu_info


    
class BaseCLI:
    
    def __init__(self, main_models, student_models, opts, trainers, 
                 test_run, mode = 'zip'):
        self.main_models = main_models
        self.student_models = student_models
        self.opts = opts
        self.trainers = trainers
        self.mode = mode
        self.test_run = test_run

    # def stuDataset_inp(self, sett_model):
    #     if sett_model is None or sett_model == 'placeholder': return True
    #     if sett_model.startswith('MLP') and sett_model != 'MLP/org': return False
    #     return True

    def get_settings(self):
        
        # import ipdb; ipdb.set_trace()
        get_sett  = lambda main_model, student_model, opt,  trainer: {
            'main_model': main_model,
            'student_model': student_model if student_model is not None else "placeholder",
            'opt': opt,
            'trainer': trainer,
            'comment': self.get_comment(main_model, student_model, trainer)
        }
        
        haha = []
        
        if self.mode == 'zip':
            for main_model, student_model, opt,  trainer in zip(self.main_models, self.student_models, 
                                                                self.opts,  self.trainers):
                                    
                    haha.append(get_sett(main_model, student_model, opt, trainer))
            return haha

        for main_model in self.main_models:
            for student_model in self.student_models:
                for opt in self.opts:
                    for trainer in self.trainers:
                            
                        haha.append(get_sett(main_model, student_model, opt, trainer))
        return haha
    
    def get_comment(self, main_model, student_model, trainer):
        raise NotImplementedError
    
    def get_model_name(self, sett_model, dataset ):
        if sett_model is None: return None
        
        path = f'config/model/{sett_model}.yaml' 
        with open(path) as f:
            model_name = yaml.load(f, Loader=yaml.FullLoader)['model']
        return model_name

    def get_opt_sett(self, y_main_model, y_student, y_opt_series, dataset):
        # if y_model is None: return "placeholder"
        # if y_opt_series is not None: return y_opt_series
        
        
        main_model = self.get_model_name(y_main_model, None)
        y_opt = f"{y_opt_series}/{main_model}"

        if y_student != "placeholder":
            student_model = self.get_model_name(y_student, dataset)
            y_opt += f"_{student_model}"
        
        y_opt += f"/{dataset}"
        
        return y_opt
        
    def get_output_path(self, sett):
        
        path = f"{Utility.get_header(sett['main_model'])}"
        path += f"-{Utility.get_header(sett['student_model'])}"
        path += f"-{Utility.get_header(sett['trainer'])}"
        path += f"/{Utility.replace_slash(sett['main_model'])}"
        path += f"--{Utility.replace_slash(sett['student_model'])}"
        path += f"--{Utility.replace_slash(sett['trainer'])}"
        
        return path

    def save_outputpath_file(self, exp_paths, exp_type = 'exp'):
        os.makedirs(base(f'./{exp_type}'), exist_ok=True)
        with open(base(f'./{exp_type}/paths.txt'), 'a+') as f:
            f.write("\n".join(exp_paths))
            f.write("\n")
            f.close()
            
        print(base(f'./{exp_type}/paths.txt'))
        
if __name__ == "__main__":
    Utility.pid_gpu_cmd_mapping()