import torch
from src.trainers.trainerBase import TrainerBase
import warnings
warnings.filterwarnings('ignore')


class Exp_TS2VecSupervised(TrainerBase):
    def __init__(self, args, main_model, student_model):
        
        super().__init__(args, main_model, student_model)
        assert args.student_model['model'] is None, 'Student model exists. This framework is for training without student model.'
        # assert args.pred_len == 1, 'This framework can only be used for training with prediction length 1. Please set `ar_pred_len` for running in AR mode.'
        self.laststeps = []

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):
    
        outputs = self.mainFFN.ffn(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode)
        return outputs
    
    def train_step(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):

        outputs = self._process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train')
        # import ipdb; ipdb.set_trace()
        loss = self.criterion(outputs['pred'], self.mainFFN.gt4update(outputs))
        loss.backward()
        
        for opt in self.opt['train']: opt.step()
        self.store_grad()
        self.update_ocp(outputs, outputs)
        
        return {'loss':loss.item()}


    def test_step(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, model = None):        

        if len(self.laststeps) >= self.args.pred_len:
            px, py, px_mark, py_mark = self.laststeps.pop(0)
            prev = self._process_one_batch(dataset_object, px, py, px_mark, py_mark, mode='test')
            loss = self.criterion(prev['pred'], self.mainFFN.gt4update(prev))
            loss.backward()
        
            for opt in self.opt['test']['online']: opt.step()
            self.store_grad()
            self.update_ocp(prev, prev)
            for opt in self.opt['test']['online']: opt.zero_grad()
            
        
        self.laststeps.append((batch_x, batch_y, batch_x_mark, batch_y_mark))
        
        with torch.no_grad():
            curr = self._process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark,  mode='test')            
        curr = self.clean(curr)            
        
        return curr