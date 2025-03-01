
import torch
from trainers.w_student.trainerBaseFramework import TrainerBaseFramework
from utils.tools import Struct
import importlib
import warnings
warnings.filterwarnings('ignore')


class Exp_TS2VecSupervised(TrainerBaseFramework):
    def __init__(self, args, main_model, student_model):

        assert student_model is not None, 'Student model is required for this framework'
       
        super().__init__(args, main_model, student_model)
        self.student_model = student_model(Struct(args).dict2attr('student_model'), args.seq_len + args.pred_len).to(self.device)   
        self.studentFFN = getattr(importlib.import_module(f'trainers.forward.trainerForward_{args.student_model["model"]}'), 
                                  'TrainerForward')(args, self.student_model, self.device)
    
    
    def get_final_output(self, main_pred, student_pred, batch_xy):
        match_dim = main_pred.shape == student_pred.shape

        if match_dim:
            return main_pred + student_pred
        
        if student_pred.shape[1] > main_pred.shape[1]:
            return batch_xy + student_pred
        
        return self.mainFFN.get_future(main_pred) + student_pred
        
    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):
        
        main_outputs = self.mainFFN.ffn(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode)
        outputs = main_outputs
        outputs.update({
            'predT': main_outputs['pred'],
            'pred': torch.clone(main_outputs['pred']).detach(),
        })
        
        batch_xy = torch.cat([batch_x.to(self.device), self.mainFFN.get_future(outputs['pred'])], dim=1) #.permute(0,2,1)
        batch_xy_mark = torch.cat([batch_x_mark.to(self.device), batch_y_mark.to(self.device)], dim=1) #.permute(0,2,1)
        student_outputs = self.studentFFN.ffn(dataset_object, batch_xy, batch_y, batch_xy_mark, batch_y_mark, mode)
        
        outputs.update({ 
                        'predS': student_outputs['pred'],
                        'pred': self.get_final_output(outputs['pred'], student_outputs['pred'], batch_xy),
                        })
        return outputs    
    
    def train_step(self, train_data, batch_x, batch_y, batch_x_mark, batch_y_mark):
        outputs = self._process_one_batch(
            train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode = 'train')
        
        lossT = self.backward_lossT_train(outputs)
        lossFT = self.backward_lossFT(outputs, outputs)
        

        for opt in self.opt['train']: opt.step()    
        self.store_grad()
        self.update_ocp(outputs, outputs)
        for opt in self.opt['train']: opt.zero_grad()    
                
        return {'loss':lossT.item()}