from torch import nn

class Model(nn.Module):
    """Just  an MLP"""
    def __init__(self, configs, n_inputs, n_outputs = None, mlp_width = None, 
                 mlp_dropout = None, mlp_depth = None, act_str = None):
        super(Model, self).__init__()
        
        self.n_outputs = n_outputs if n_outputs is not None else configs.pred_len
        self.mlp_width = mlp_width if mlp_width is not None else configs.mlp_width
        self.mlp_dropout = mlp_dropout if mlp_dropout is not None else configs.mlp_dropout
        self.mlp_depth = mlp_depth if mlp_depth is not None else configs.mlp_depth
        self.act_str = act_str if act_str is not None else configs.act_str
        
        
        self.input = nn.Linear(n_inputs, self.mlp_width)
        self.dropout = nn.Dropout(self.mlp_dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(self.mlp_width, self.mlp_width)
            for _ in range(self.mlp_depth-2)])
        self.output = nn.Linear(self.mlp_width, self.n_outputs)
        # self.n_outputs = configs.pred_len
        
        self.act = getattr(nn, self.act_str)()
        self.pred_len = configs.pred_len

    def forward(self, x, train=True):
        x = x.permute(0,2,1)
        x = self.input(x)
        if train:
            x = self.dropout(x)
        x = self.act(x)
        for hidden in self.hiddens:
            x = hidden(x)
            if train:
                x = self.dropout(x)
            x = self.act(x)
        x = self.output(x)
        # x = F.sigmoid(x)
        return {'pred': x.permute(0,2,1)}