import argparse
import os
import torch
import random
import numpy as np
import datetime
import importlib
import yaml
from src.utils.misc import bcolors
from src.utils.tools import DataLogger
from src.data.data_loader import Dataset_Custom 
from torch.utils.data import DataLoader

class MergeArguments:
    def __init__(self,  args, parser):
        # self.args = args
        self.parser = parser
        self.specified_program_options = args.__dict__.copy()

        self.sanity_check(args)
        
    def sanity_check(self, args):
        
        assert args.data is not None, 'Data is not defined'
        assert args.y_model_main is not None, 'Main model is not defined'
        assert args.y_opt is not None, 'optimizer is not defined'
        # student model is optional
        assert args.y_trainer is not None, 'Trainer is not defined'
        
        if args.use_gpu: assert torch.cuda.is_available(), 'GPU is not available'

    def get_subargs(self, path):
        with open(path) as f:
            subargs = yaml.load(f, Loader=yaml.FullLoader)    
        return subargs
        
    def add_subargs(self, path, header = None):
        if not os.path.exists(path):
            print(f"{bcolors.WARNING}Warning: File {path} does not exist {bcolors.ENDC}")
            return 
        
        subargs = self.get_subargs(path)
        if header: subargs = {header: subargs}
        self.parser.set_defaults(**subargs)    
    
    def parse_args(self):
        self.parser.set_defaults(**self.specified_program_options)
        return self.parser.parse_args()


class PostprocessArguments:
    
    def set_multi_gpu(self, args):
        # TODO: implement this
        # if args.use_gpu and args.use_multi_gpu:
        #     args.devices = args.devices.replace(' ','')
        #     device_ids = args.devices.split(',')
        #     args.device_ids = [int(id_) for id_ in device_ids]
        #     args.gpu = args.device_ids[0]
        
        return args

    def set_dataset_specs(self, args):
        args.enc_in, args.dec_in, args.c_out = map(lambda dd: dd[args.features], [args.enc_in, args.dec_in, args.c_out])
        return args
    
    def set_settings(self, args, exp_starttime, itr):
        if args.test_run: 
            args.setting = f'logs/test_run/itr{itr}'
            return args
    
        get_header = lambda x: x.split('/')[0]
        replace_slash = lambda x: x.replace('/', '_')
        
        y_model_student = replace_slash(args.y_model_student) 
        
        if len(y_model_student.split('_') ) > 2:
            y_model_student = '_'.join(y_model_student.split('_')[:2])
            
        try:
            int(args.y_opt.split('/')[0].split('_')[-1])
            lr = "_"+args.y_opt.split('/')[0].split('_')[-1]
            
        except ValueError:
            lr = ""
        
        args.setting = f"exps/{get_header(args.y_model_main)}-{get_header(args.y_model_student)}-{get_header(args.y_trainer)}/"
        args.setting += f"{replace_slash(args.y_model_main)}--{y_model_student}--{replace_slash(args.y_trainer)}{lr}/"
        args.setting += f"{args.data}_pl{args.pred_len}/{exp_starttime}/itr{itr}" 
        
        # import ipdb; ipdb.set_trace()
        return args

    def set_device(self, args):
        args.device = DeviceSettings().get_cuda_device(args.gpu_id)
        return args

class DeviceSettings:
    def threading(self, max_threads = None):
        if max_threads is None: return 
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
    
    def seeding(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def get_cuda_device(self, gpu_id):
        return torch.device(f'cuda:{gpu_id}')
        
    def cuda_backends(self, use_cudnn = True, deterministic = False, benchmark = False, use_tf32 = False):
        torch.backends.cudnn.enabled = use_cudnn
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = use_tf32
            torch.backends.cuda.matmul.allow_tf32 = use_tf32

class LoadData:
    @staticmethod
    def _get_data( args, flag):
        # args = self.args
        # import ipdb; ipdb.set_trace()
        # TODO: where is args.timeenc? in which config file
        timeenc = args.timeenc

        if flag  == 'test':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.test_bsz #if args.online_learning != 'none' else args.batch_size;
            freq = args.freq[-1:]
        elif flag == 'val':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.batch_size;
            freq = args.freq
        elif flag == 'pred':
            raise NotImplementedError('prediction data loader not implemented')
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            # Data = dl.Dataset_Pred
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq[-1:]

        data_set = Dataset_Custom(
            root_path=args.root_path,
            data= args.data,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len if not hasattr(args, 'ar_pred_len') else args.ar_pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
            online='full',
        )
        print(flag, len(data_set))
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader
    
    
def get_total_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def save_hparams(args):
    hparams = filter(lambda x: x[0] not in  ['gpu', 'use_multi_gpu', 'devices', 'use_gpu'], 
                args.__dict__.items())
    with open(os.path.join(args.setting, '../hparams.yaml'), 'w+') as file:
        yaml.dump(dict(hparams), file)

    
parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

# MAIN settings
parser.add_argument('--data', type=str, default='ECL', help='data of experiment')
parser.add_argument('--y_model_main', type=str, default='FSNet/org', help='main model of experiment')
parser.add_argument('--y_model_student', type=str, default=None, help='student model of experiment')
parser.add_argument('--y_opt', type=str, default='FSNet/org', help='main optimizer of experiment')
# parser.add_argument('--y_opt_student', type=str, default=None, help='student optimizer of experiment')
parser.add_argument('--y_trainer', type=str, default='FSNet', help='trainer of experiment')
parser.add_argument('--comments', type=str, default='test',help='exp description')

# general settings
# parser.add_argument('--k', type=int, default=10, help='tensorboard top/bottom k entries')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--test_run', action='store_true', default=False, help='test run the code')
parser.add_argument('--timing', action='store_true', default=False, help='test run the code')
parser.add_argument('--resume', type=str, default=None, help='specify the model path and resume its training')

# common dataset and dataloader settings
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# devices
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
# parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
# parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')


###########################################################################################

args = parser.parse_args()
merger = MergeArguments(args, parser)
subargs = {
    f'./config/data/{args.data}.yaml': None,
    f'./config/model/{args.y_model_main}.yaml': 'main_model',
    f'./config/optimizer/{args.y_opt}.yaml': None,
    f'./config/trainer/{args.y_trainer}.yaml': None
}

for path, header in subargs.items():
    merger.add_subargs(path, header)

subargs2 = {
    f'./config/model/{args.y_model_student}.yaml': 'student_model',
}
for path, header in subargs2.items():
    merger.add_subargs(path, header)

args = merger.parse_args()

###########################################################################################

exp_starttime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
args = PostprocessArguments().set_dataset_specs(args)
args = PostprocessArguments().set_device(args)

DeviceSettings().threading()
DeviceSettings().cuda_backends()

###########################################################################################
# import ipdb; ipdb.set_trace()
Trainer = getattr(importlib.import_module('src.trainers.{}'.format(args.trainer)), 'Exp_TS2VecSupervised')
mainModel = getattr(importlib.import_module('src.models.{}'.format(args.main_model['model'])), 'Model')
studentModel = getattr(importlib.import_module('src.models.{}'.format(args.student_model['model'])), 'Model') \
    if args.student_model['model'] is not None else None
###########################################################################################
train_data, train_loader = LoadData._get_data(args, 'train')
val_data, val_loader = LoadData._get_data(args, 'val')
test_data, test_loader = LoadData._get_data(args, 'test')

###########################################################################################

datalogger = DataLogger(['metrics', 'preds', 'trues', 'mae', 'mse'])
for ii in range(args.itr):
    metric, mae, mse, pred, true = None, None, None, None, None

    print('\n ====== Run {} ====='.format(ii))
    args = PostprocessArguments().set_settings(args, exp_starttime, ii)
    DeviceSettings().seeding(args.seed + ii)

    trainer = Trainer(args, mainModel, studentModel)
    print("total params for main model: ", get_total_params(trainer.main_model))
    
    if trainer.student_model is not None:
        print("total params for student model: ", get_total_params(trainer.student_model))
    
    print('\n>>>>>>> start training : {} >>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.setting))
    trainer.train(train_data, train_loader, val_data, val_loader )

    print('\n>>>>>>> testing : {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.setting))
    metric, mae, mse, pred, true, = trainer.test(test_data, test_loader)
    

    datalogger.update({'metrics': metric, 'preds': pred, 'trues': true, 'mae': mae, 'mse': mse})
    
    if args.timing: continue
    
    os.makedirs(f'{args.setting}/results', exist_ok=True)
    for key in datalogger.keys:
        np.save(f'{args.setting}/results/{key}.npy', datalogger[key])
    
save_hparams(args)