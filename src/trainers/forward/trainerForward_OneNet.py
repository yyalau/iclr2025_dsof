# from einops import rearrange
import torch
from src.trainers.forward.trainerBaseForward import TrainerBaseForward
import torch.nn.functional as F

from torch import optim
from src.models.MLP import Model as MLP
from torch import nn

import warnings
warnings.filterwarnings('ignore')


class TrainerForward(TrainerBaseForward):
    def __init__(self, args, model, device,):
        super().__init__(args, model, device)
        # self.init_ocp_wb()
        
        self.individual = self.args.main_model['individual']
        # import ipdb; ipdb.set_trace()
        self.decision = MLP(args, n_inputs=self.args.pred_len * 3, n_outputs=1, mlp_width=32, mlp_depth=3, 
                            mlp_dropout=0.1, act_str='Tanh').to(self.device) if self.individual \
                    else MLP(args, n_inputs=(self.args.c_out * self.args.pred_len) * 3, n_outputs=1, mlp_width=32, 
                            mlp_depth=3, mlp_dropout=0.1, act_str='Tanh').to(self.device)

        if self.individual:
            self.weight = torch.zeros((1,1,self.args.enc_in), device = self.device, requires_grad=True)
            self.bias = torch.zeros((1,1,self.args.enc_in), device = self.device)
            # self.bias = None
        else:
            self.weight = torch.zeros(1, device = self.device, requires_grad=True)
            self.bias = torch.zeros(1, device = self.device)
            # self.bias = None
            
        #  self.individual = True

        # TODO: use _select_optimizer!!
        self.opt_w = optim.Adam([self.weight], lr=self.args.learning_rate_w)
        self.opt_bias =  optim.Adam(self.decision.parameters(), lr=self.args.learning_rate_bias)


    
    def store_grad(self):
        self.model.store_grad()
        
    def update_ocp(self, prev, curr):
        # import ipdb; ipdb.set_trace()
        if 'predT' not in curr.keys():
            curr['predT'] = curr['pred']
        
        y1 = prev['y1']
        y2 = prev['y2']
        true_y = torch.cat([prev['true'][:, :1], 
                    curr['predT'][:, :-1]], 
                    dim=1) if prev is curr else curr['true']
        
        b, t, d = y1.shape
        y1, y2, true_y = map(lambda x: x.data.detach(), [y1, y2, true_y])

        ################################### update bias ###################################
        
        if self.individual:
            
            weight1_v2 = F.sigmoid(self.weight).view(1, 1, -1); weight1_v2 = weight1_v2.repeat(b, t, 1)
            inputs_decision = torch.cat([weight1_v2*y1, (1-weight1_v2)*y2, true_y], dim=1)
            # inputs_decision = torch.cat([y1, y2, true_y], dim=1)

            self.bias = self.decision(inputs_decision)['pred']
            # import ipdb; ipdb.set_trace()
            # self.bias = self.decision(inputs_decision)
            bias = self.bias.view(b, 1, -1); bias = bias.repeat(1, t, 1)
            weight = self.weight.view(1, 1, -1); weight = weight.repeat(b, t, 1)
            weight1_v3 = F.sigmoid(weight + bias)

        else:
            weight1_v2 = F.sigmoid(self.weight)  
            inputs_decision = torch.cat([weight1_v2*y1, (1-weight1_v2)*y2, true_y], dim=-1)
            self.bias = self.decision(inputs_decision)
            weight1_v3 = F.sigmoid(self.weight + self.bias)
        
        
        y1, y2, true_y = map(lambda x: x.data.detach(), [y1, y2, true_y])
        loss_bias = F.mse_loss(y1 *  weight1_v3 + y2 * (1- weight1_v3), true_y)

        loss_bias.backward()
        self.opt_bias.step()   
        self.opt_bias.zero_grad()
        
        self.bias = self.bias.detach()
        
        # import ipdb; ipdb.set_trace()
        
        ################################### update weight ###################################
        # if self.individual:
        #     weight1_v4 = F.sigmoid(self.weight).view(1, 1, -1).repeat(b, t, 1)
        # else:
        #     weight1_v4 = F.sigmoid(self.weight)
        
        # loss_w = F.mse_loss(weight1_v4 * y1 + (1 - weight1_v4) * y2, true_y)
        # loss_w.backward()        
        # self.opt_w.step()
        # self.opt_w.zero_grad()
        
        # import ipdb; ipdb.set_trace()
        # return weight1_v3
    
    def feedforward_m(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, weight1):

        x = batch_x.float().to(self.device) #torch.cat([batch_x.float(), batch_x_mark.float()], dim=-1).to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        
        outputs = self.model.forward_weight(x, batch_x_mark, weight1, 1 - weight1)
        batch_y = batch_y[:,-self.args.pred_len:,self.f_dim:].to(self.device)

        outputs.update({'true': batch_y, })
        return outputs
    
    def ffn(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):

        batch_x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        
        with torch.cuda.amp.autocast(enabled = self.args.use_amp):
            outputs = self.model.forward_weight(batch_x, batch_x_mark)
            # weight1 = self.feedforward_ocp(outputs["y1"], outputs["y2"])
        
        # x = torch.cat([batch_x, batch_x_mark], dim=-1)
        batch_y = batch_y[:,-self.args.pred_len:,self.f_dim:].float().to(self.device)
        b, t, d = batch_y.shape

        '''
        if self.individual:
            loss1 = F.sigmoid(self.weight).view(1, 1, -1)
            loss1 = loss1.repeat(b, t, 1)
            loss1 = rearrange(loss1, 'b t d -> b (t d)')
        
        
        if self.individual:
            weight = self.weight.view(1, 1, -1).repeat(b, t, 1)
            bias = self.bias.view(-1, 1, d).repeat(1, t, 1)
            loss1 = F.sigmoid(weight + bias).view(b, t, d)
            loss1 = rearrange(loss1, 'b t d -> b (t d)')
        else:
            loss1 = F.sigmoid(self.weight + self.bias)
        '''

        # CONTINUE HERE!!!
        
        if (b != self.bias.shape[0]):
            self.bias = self.bias.mean(dim=0, keepdim=True)
            
        
        if b == 1:            
            weight1 = self.weight.view(1,1,-1).repeat(b,t,1) + self.bias.view(-1,1,d).repeat(1, t, 1)
            weight1 = F.sigmoid(weight1).view(b, t, d)
            
        else:
            weight1 = F.sigmoid(self.weight).view(1,1,-1).repeat(b,t,1) 

        outputs.update({'pred': weight1 * outputs["y1"] + (1-weight1) * outputs["y2"] ,
                        'true':  batch_y, })
        return outputs
    
