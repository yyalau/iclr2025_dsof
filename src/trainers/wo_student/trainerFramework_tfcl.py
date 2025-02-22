import torch
from src.trainers.wo_student.trainerFramework_plain import Exp_TS2VecSupervised as TrainerPlain
from src.utils.buffer import BufferFIFO as Buffer
from collections import deque
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class Exp_TS2VecSupervised(TrainerPlain):
    def __init__(self, args, main_model, student_model):
        
        super().__init__(args, main_model, student_model)
        assert args.student_model['model'] is None, 'Student model exists. This framework is for training without student model.'
        # assert args.pred_len == 1, 'This framework can only be used for training with prediction length 1. Please set `ar_pred_len` for running in AR mode.'
        self.laststeps = []
        self.recentBuffer = Buffer(args.recent_buffer_size)
        self.hardBuffer = Buffer(args.hard_buffer_size)
        self.mas_weight = args.mas_weight
        self.gradient_steps = args.gradient_steps
        self.lossBuffer = deque(maxlen= args.loss_buffer_size)

        self.old_mu = 0; self.old_std = 0
        self.star_variables = None
        self.omegas = None
        self.count_update =0
        
        self.mse_NR = torch.nn.MSELoss(reduction='none')




    def test_step(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, model = None):        
        recent_loss = None
        hard_loss = None
        hx, hy, hx_mark, hy_mark = None, None, None, None

        if len(self.laststeps) >= self.args.pred_len:
            px, py, px_mark, py_mark = self.laststeps.pop(0)
            prev = self._process_one_batch(dataset_object, px, py, px_mark, py_mark, mode='test')
            loss = self.criterion(prev['pred'], self.mainFFN.gt4update(prev))
            
            for _ in range(self.gradient_steps):
            
                if self.recentBuffer.is_full():
                    bx, by, bx_mark, by_mark, = self.recentBuffer.get_all_data()
                    b_new = self._process_one_batch(dataset_object, bx, by, bx_mark, by_mark, mode='test')
                    recent_loss = self.mse_NR(b_new['pred'], self.mainFFN.gt4update(b_new))
                    loss += recent_loss.sum()
                
                if self.hardBuffer.is_full():
                    hx, hy, hx_mark, hy_mark, = self.hardBuffer.get_all_data()
                    h_new = self._process_one_batch(dataset_object, hx, hy, hx_mark, hy_mark, mode='test')  
                    hard_loss = self.mse_NR(h_new['pred'], self.mainFFN.gt4update(h_new))
                    loss += hard_loss.sum()
                
                
                mas_loss = 0
                if self.omegas is not None and self.star_variables is not None:
                    for omega, star_var, param in zip(self.omegas, self.star_variables, self.mainFFN.model.parameters()):
                        mas_loss += self.mas_weight/2. * torch.sum(omega) * (param - star_var).pow(2).sum()
                    # todo
                loss += mas_loss
                
                loss.backward(); self.lossBuffer.append(loss.item())
            
                for opt in self.opt['test']['online']: opt.step()
                self.store_grad()
                self.update_ocp(prev, prev)
                for opt in self.opt['test']['online']: opt.zero_grad()
            

            
            
            
            
            if self.hardBuffer.is_full() and len(self.lossBuffer) >= self.args.loss_buffer_size:
                new_mu = np.mean(self.lossBuffer); new_std = np.std(self.lossBuffer)

                # import ipdb; ipdb.set_trace()
                
                if new_mu > self.old_mu + self.old_std:
                    self.count_update +=1
                    self.old_mu = new_mu; self.old_std = new_std
                    
                    self.mainFFN.model.zero_grad()
                    h_new = self._process_one_batch(dataset_object, hx, hy, hx_mark, hy_mark, mode='test')
                    torch.norm(h_new['pred'], p=2).backward() # todo   
                    grad = [p.grad.data for p in self.mainFFN.model.parameters() if p.grad is not None]
                    
                    self.omegas = [1/ self.count_update * g + (1- 1/self.count_update) * omega  \
                        for omega, g in zip(self.omegas, grad)] if self.omegas is not None else grad
                    self.star_variables  = [p.data for p in self.mainFFN.model.parameters()]
            
            
            self.recentBuffer.add_data(batch_x = px, batch_y= py, batch_x_mark= px_mark, batch_y_mark=py_mark)

            if recent_loss is not None:     
                
                cat = lambda x, y: torch.cat([x, y], dim = 0) if y is not None else x
                hx, hy, hx_mark, hy_mark, = map(cat, [bx, by, bx_mark, by_mark], [hx, hy, hx_mark, hy_mark], )
                

                individual_loss = cat(recent_loss, hard_loss) if hard_loss is not None else recent_loss
                individual_loss = torch.mean(individual_loss.data, dim = [1,2])
                
                indices= torch.Tensor([ x for x,_  in sorted(enumerate(individual_loss), 
                                        key=lambda a: a[1], reverse = True)],
                                    ).int()
                indices = indices[:self.args.hard_buffer_size]
                self.hardBuffer.add_data(batch_x = hx[indices], batch_y= hy[indices], batch_x_mark= hx_mark[indices], batch_y_mark=hy_mark[indices])    

        
        self.laststeps.append((batch_x, batch_y, batch_x_mark, batch_y_mark))
        
        with torch.no_grad():
            curr = self._process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark,  mode='test')            
        curr = self.clean(curr)            
        
        return curr