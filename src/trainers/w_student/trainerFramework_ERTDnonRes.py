import importlib
import torch
from src.trainers.w_student.trainerBaseFramework import TrainerBaseFramework
from src.utils.tools import Struct

import warnings
warnings.filterwarnings('ignore')


class Exp_TS2VecSupervised(TrainerBaseFramework):
    def __init__(self, args,  main_model, student_model):
        
        assert student_model is not None, 'Student model is required for this framework'
        
        super().__init__(args,  main_model, student_model)
        
        self.same_arch = True if args.main_model == args.student_model else False
        self.student_model = student_model(Struct(args).dict2attr('student_model'), args.seq_len).to(self.device)         
        self.studentFFN = getattr(importlib.import_module(f'src.trainers.forward.trainerForward_{args.student_model["model"]}'), 
                                  'TrainerForward')(args, self.student_model, self.device)

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y[:,-self.args.pred_len:,self.f_dim:].float().to(self.device)
        
        main_outputs = self.mainFFN.ffn(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode)
        
        student_model_exists = mode == 'test' or not self.same_arch        
        if student_model_exists:
            student_outputs = self.studentFFN.ffn(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode)
        
        outputs = main_outputs
        outputs.update({
            'predT': main_outputs['pred'],
            'predS':  student_outputs['pred'] if student_model_exists else main_outputs['pred'],
            'pred': student_outputs['pred'] if student_model_exists else main_outputs['pred'],
        })

        outputs.update({'true':  batch_y, })
        return outputs
    
    
    
    def train_step(self, train_data, batch_x, batch_y, batch_x_mark, batch_y_mark):
        
        outputs = self._process_one_batch(
            train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode = 'train')
        # lossT = self.criterion(outputs['predT'], outputs['true'])
        # lossT.backward()
        lossT = self.backward_lossT_train(outputs)
        
        if not self.same_arch:
            # lossS = self.criterion(outputs['pred'], outputs['true'])
            # lossS.backward()
            self.backward_lossFT(outputs, outputs)
                    
        for opt in self.opt['train']: opt.step()                                
        self.store_grad()
        self.update_ocp(outputs, outputs)
        
        return {'loss':lossT.item()}
    

    def test(self, test_data, test_loader,):

        if self.same_arch:
            self.student_model.load_state_dict(self.main_model.state_dict())
            
        return super().test(test_data, test_loader,)


