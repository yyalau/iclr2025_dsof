import numpy as np
import torch
import os

def save_model(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(content, path)
    print(f'ckpt saved at {path}')

# def load_model(model, path, device='cpu'):
#     model.load_state_dict(torch.load(os.path.join(path), map_location=device))
#     model.to(device)
    

class Struct:
    def __init__(self, args):
        

        self.__dict__.update(args.__dict__)
                
    def dict2attr(self, entry_name):
        if not isinstance(self.__dict__[entry_name], dict ): return 
        entries = self.__dict__.pop(entry_name)
        self.__dict__.update(**entries)
        return self
    
    def __repr__(self): 
        return '<%s>' % str('\n '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        # self.best_path = None
            
    def __call__(self, val_loss, ckpt_content, path, name='checkpoint.pth'):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,ckpt_content, path, name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, ckpt_content,  path, name)
            self.counter = 0

    def save_checkpoint(self, val_loss, ckpt_content,  path, name='checkpoint.pth'):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        self.best_path = path+'/'+name
        
        # save_model(ckpt_content, self.best_path)
        save_model(ckpt_content, path+'/checkpoint.pth')
        

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class DataLogger:
    def __init__(self, keys):
        self.keys = keys
        self.data = {key: [] for key in keys}
    
    def update(self, values):
        keys = set(values.keys()) & set(self.keys)
        for key in keys:
            if values[key] is None: continue
            if torch.is_tensor(values[key]): values[key] = values[key].detach().cpu().numpy()
            
            self.data[key].append(values[key])
    
    def __getitem__(self, key):
        
        '''
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        '''
        if self.data[key] == []: return None
        
        if isinstance(self.data[key][0], np.ndarray): return np.concatenate(self.data[key], axis = 0)        
        return np.array(self.data[key])