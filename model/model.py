import torch
import torch.nn as nn
import numpy as np
from model.modules import *
from abc import abstractmethod

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class BayesianFilter(BaseModel):
    def __init__(self,
                 num_channel,
                 latent_dim,
                 obs_dim,
                 ode_func_type,
                 ode_num_layers,
                 ode_method,
                 rnn_type):
        super().__init__()
        self.nf = num_channel
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.ode_func_type = ode_func_type
        self.ode_num_layers = ode_num_layers
        self.ode_method = ode_method
        self.rnn_type = rnn_type

        # encoder
        self.conv1 = Spatial_Block(self.nf[0], self.nf[1], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = Spatial_Block(self.nf[1], self.nf[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = Spatial_Block(self.nf[2], self.nf[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv4 = Spatial_Block(self.nf[3], self.nf[4], dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Conv2d(self.nf[4], self.nf[5], 1)
        self.fce2 = nn.Conv2d(self.nf[5], latent_dim, 1)

        # Domain model
        self.domain_embedding = RnnEncoder(latent_dim, latent_dim,
                                           dim=3,
                                           kernel_size=3,
                                           norm=False,
                                           n_layer=1,
                                           rnn_type=rnn_type,
                                           bd=False,
                                           reverse_input=False)
        self.domain = DomainEncoder(latent_dim, latent_dim, obs_dim, stochastic=False)
        
        self.propagation = Propagation(latent_dim, fxn_type=ode_func_type, num_layers=ode_num_layers, method=ode_method, rtol=1e-5, atol=1e-7)
        self.correction = Correction(latent_dim, rnn_type=rnn_type, dim=3, kernel_size=3, norm=False)

        # decoder
        self.fcd3 = nn.Conv2d(latent_dim, self.nf[5], 1)
        self.fcd4 = nn.Conv2d(self.nf[5], self.nf[4], 1)

        self.deconv4 = Spatial_Block(self.nf[4], self.nf[3], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv3 = Spatial_Block(self.nf[3], self.nf[2], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv2 = Spatial_Block(self.nf[2], self.nf[1], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv1 = Spatial_Block(self.nf[1], self.nf[0], dim=3, kernel_size=(3, 1), process='d', norm=False)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

        self.P10 = dict()
        self.P21 = dict()
        self.P32 = dict()
        self.P43 = dict()

        self.P01 = dict()
        self.P12 = dict()
        self.P23 = dict()
        self.P34 = dict()

    def setup(self, heart_name, data_path, batch_size, ecgi, graph_method):
        params = get_params(data_path, heart_name, batch_size, ecgi, graph_method)
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        self.bg4[heart_name] = params["bg4"]
        
        self.P10[heart_name] = params["P10"]
        self.P21[heart_name] = params["P21"]
        self.P32[heart_name] = params["P32"]
        self.P43[heart_name] = params["P43"]

        self.P01[heart_name] = params["P01"]
        self.P12[heart_name] = params["P12"]
        self.P23[heart_name] = params["P23"]
        self.P34[heart_name] = params["P34"]

    def embedding(self, data, heart_name):
        batch_size, seq_len = data.shape[0], data.shape[-1]
        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = \
            data.view(batch_size, -1, self.nf[0], seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[1] * seq_len)
        x = torch.matmul(self.P01[heart_name], x)
        
        # layer 2
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[1], seq_len), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.conv2(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[2] * seq_len)
        x = torch.matmul(self.P12[heart_name], x)
        
        # layer 3
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], seq_len), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.conv3(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[3] * seq_len)
        x = torch.matmul(self.P23[heart_name], x)

        # layer 4
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], seq_len), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.conv4(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[4] * seq_len)
        x = torch.matmul(self.P34[heart_name], x)

        # latent
        x = x.view(batch_size, -1, self.nf[4], seq_len)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = F.elu(self.fce1(x), inplace=True)
        x = torch.tanh(self.fce2(x))

        x = x.permute(0, 2, 1, 3).contiguous()
        return x
    
    def time_modeling(self, x, heart_name):
        N, V, C, T = x.shape
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr

        x = x.permute(3, 0, 1, 2).contiguous()
        last_h = x[0]

        # Domain
        _x = self.domain_embedding(x, edge_index, edge_attr)
        z_D = self.domain(_x)

        z = []
        z.append(last_h.view(1, N, V, C))

        x = x.view(T, N * V, C)
        for t in range(1, T):
            last_h = last_h.view(N, V, -1)

            # Propagation
            last_h = self.propagation(last_h, z_D, 1, steps=1)
            # Corrrection
            last_h = last_h.view(N * V, -1)
            h = self.correction(x[t], last_h, edge_index, edge_attr)

            last_h = h
            z.append(h.view(1, N, V, C))
        
        z = torch.cat(z, dim=0)
        z = z.permute(1, 2, 3, 0).contiguous()
        return z
    
    def decoder(self, x, heart_name):
        batch_size, seq_len = x.shape[0], x.shape[-1]
        x = x.permute(0, 2, 1, 3).contiguous()

        x = F.elu(self.fcd3(x), inplace=True)
        x = F.elu(self.fcd4(x), inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.nf[4] * seq_len)
        x = torch.matmul(self.P43[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[4], seq_len), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.deconv4(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[3] * seq_len)
        x = torch.matmul(self.P32[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], seq_len), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.deconv3(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[2] * seq_len)
        x = torch.matmul(self.P21[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], seq_len), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.deconv2(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[1] * seq_len)
        x = torch.matmul(self.P10[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[1], seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        x = self.deconv1(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, seq_len)
        return x

    def forward(self, x, heart_name):
        embed = self.embedding(x, heart_name)
        z = self.time_modeling(embed, heart_name)
        x = self.decoder(z, heart_name)
        return (x, None), (None, None, None, None)


class DomainInvariantDynamics(BaseModel):
    def __init__(self,
                 num_channel,
                 latent_dim,
                 obs_dim,
                 rnn_type):
        super().__init__()
        self.nf = num_channel
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.rnn_type = rnn_type

        # encoder
        self.signal_encoder = Encoder(num_channel, latent_dim, cond=True)
        self.condition_encoder = Encoder(num_channel, latent_dim)

        # Domain model
        self.domain_seq = RnnEncoder(latent_dim, latent_dim,
                                     dim=3,
                                     kernel_size=3,
                                     norm=False,
                                     n_layer=1,
                                     rnn_type=rnn_type,
                                     bd=False,
                                     reverse_input=False)
        self.domain = Aggregator(latent_dim, latent_dim, obs_dim, stochastic=False)
        self.mu_c = nn.Linear(latent_dim, latent_dim)
        self.var_c = nn.Linear(latent_dim, latent_dim)
        # self.mu_c.weight.data = torch.eye(latent_dim)
        # self.mu_c.bias.data = torch.zeros(latent_dim)

        # initialization
        self.initial = nn.Linear(latent_dim, latent_dim)

        # time modeling
        self.propagation = Transition(latent_dim, latent_dim, stochastic=False)

        # decoder
        self.decoder = Decoder(num_channel, latent_dim)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

    def setup(self, heart_name, data_path, batch_size, ecgi, graph_method):
        params = get_params(data_path, heart_name, batch_size, ecgi, graph_method)
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        self.bg4[heart_name] = params["bg4"]
        
        self.signal_encoder.setup(heart_name, params)
        self.condition_encoder.setup(heart_name, params)
        self.decoder.setup(heart_name, params)
    
    def get_latent_domain(self, xs, heart_name):
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr
        
        # Domain
        z_Ds = []
        for x in xs:
            _x = self.domain_seq(x, edge_index, edge_attr)
            z_Ds.append(self.domain(_x))

        z_c = sum(z_Ds) / len(z_Ds)
        mu_c = self.mu_c(z_c)
        logvar_c = self.var_c(z_c)
        mu_c = torch.clamp(mu_c, min=-100, max=85)
        logvar_c = torch.clamp(logvar_c, min=-100, max=85)

        return mu_c, logvar_c
    
    def get_latent_initial(self, y, heart_name):
        N, V, T = y.shape
        y = y[:, :, 0].view(N, V, 1)
        z_0 = self.condition_encoder(y, heart_name)
        z_0 = torch.squeeze(z_0)
        z_0 = self.initial(z_0)
        
        return z_0
    
    def time_modeling(self, T, z_0, z_c):
        N, V, C = z_0.shape

        z_prev = z_0
        z = []
        for i in range(1, T):
            z_t = self.propagation(z_prev, z_c)
            z_prev = z_t
            z_t = z_t.view(1, N, V, C)
            z.append(z_t)
        z = torch.cat(z, dim=0)
        z_0 = z_0.view(1, N, V, C)
        z = torch.cat([z_0, z], dim=0)
        z = z.permute(1, 2, 3, 0).contiguous()

        return z
    
    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, heart_name, label=None, D=None, D_label=None):
        # q(c | D)
        N, V, T = x.shape
        y = one_hot_label(label[:, 2] - 1, x)
        z_x = self.signal_encoder(x, heart_name, y)

        z_Ds = []
        if D is not None:
            K = D.shape[1]
            D_ys = []
            for i in range(K):
                D_yi = D_label[:, i, :]
                D_yi = one_hot_label(D_yi[:, 2] - 1, x)
                D_ys.append(D_yi)

                Di = D[:, i, :, :].view(N, V, T)
                z_Ds.append(self.signal_encoder(Di, heart_name, D_yi))
    
        z_Ds.append(z_x)
        mu_c, logvar_c = self.get_latent_domain(z_Ds, heart_name)
        z_c = self.reparameterization(mu_c, logvar_c)

        # q(z)
        z_0 = self.get_latent_initial(y, heart_name)

        # p(x | z, c)
        z = self.time_modeling(T, z_0, z_c)
        x = self.decoder(z, heart_name)

        # reconstruction of D
        D_ = []
        if D is not None:
            for i in range(K):
                Dz_0 = self.get_latent_initial(D_ys[i], heart_name)
                Dz = self.time_modeling(T, Dz_0, z_c)
                Dxi = self.decoder(Dz, heart_name)
                Dxi = Dxi.view(N, -1, V, T)
                D_.append(Dxi)
            D_ = torch.cat(D_, dim=1)
        
        return (x, D_), (mu_c, logvar_c, None, None)
    
    def personalization(self, eval_x, heart_name, label=None, eval_label=None, D=None, D_label=None):        
        # q(c | D)
        N, V, T = eval_x.shape
        eval_y = one_hot_label(eval_label[:, 2] - 1, eval_x)
        z_x = self.signal_encoder(eval_x, heart_name, eval_y)
        
        z_Ds = []
        if D is not None:
            K = D.shape[1]
            D_ys = []
            for i in range(K):
                D_yi = D_label[:, i, :]
                D_yi = one_hot_label(D_yi[:, 2] - 1, eval_x)
                D_ys.append(D_yi)

                Di = D[:, i, :, :].view(N, V, T)
                z_Ds.append(self.signal_encoder(Di, heart_name, D_yi))
        else:
            z_Ds.append(z_x)
        
        mu_c, logvar_c = self.get_latent_domain(z_Ds, heart_name)
        z_c = self.reparameterization(mu_c, logvar_c)

        # q(z)
        y = one_hot_label(label[:, 2] - 1, eval_x)
        z_0 = self.get_latent_initial(y, heart_name)

        # p(x | z, c)
        z = self.time_modeling(T, z_0, z_c)
        x = self.decoder(z, heart_name)
        
        return (x, None), (mu_c, logvar_c, None, None)


class MetaDynamics(BaseModel):
    def __init__(self,
                 num_channel,
                 latent_dim,
                 obs_dim,
                 rnn_type):
        super().__init__()
        self.nf = num_channel
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.rnn_type = rnn_type

        # encoder
        self.signal_encoder = Encoder(num_channel, latent_dim, cond=True)
        self.condition_encoder = Encoder(num_channel, latent_dim)

        # Domain model
        self.domain_seq = RnnEncoder(latent_dim, latent_dim,
                                     dim=3,
                                     kernel_size=3,
                                     norm=False,
                                     n_layer=1,
                                     rnn_type=rnn_type,
                                     bd=False,
                                     reverse_input=False)
        self.domain = Aggregator(latent_dim, latent_dim, obs_dim, stochastic=False)
        self.mu_c = nn.Linear(latent_dim, latent_dim)
        self.var_c = nn.Linear(latent_dim, latent_dim)
        # self.mu_c.weight.data = torch.eye(latent_dim)
        # self.mu_c.bias.data = torch.zeros(latent_dim)

        # initialization
        self.initial = nn.Linear(latent_dim, latent_dim)

        # time modeling
        self.propagation = Transition(latent_dim, latent_dim, stochastic=False)

        # decoder
        self.decoder = Decoder(num_channel, latent_dim)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

    def setup(self, heart_name, data_path, batch_size, ecgi, graph_method):
        params = get_params(data_path, heart_name, batch_size, ecgi, graph_method)
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        self.bg4[heart_name] = params["bg4"]
        
        self.signal_encoder.setup(heart_name, params)
        self.condition_encoder.setup(heart_name, params)
        self.decoder.setup(heart_name, params)
    
    def get_latent_domain(self, xs, heart_name):
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr
        
        # Domain
        z_Ds = []
        for x in xs:
            _x = self.domain_seq(x, edge_index, edge_attr)
            z_Ds.append(self.domain(_x))

        z_c = sum(z_Ds) / len(z_Ds)
        mu_c = self.mu_c(z_c)
        logvar_c = self.var_c(z_c)
        mu_c = torch.clamp(mu_c, min=-100, max=85)
        logvar_c = torch.clamp(logvar_c, min=-100, max=85)

        return mu_c, logvar_c
    
    def get_latent_initial(self, y, heart_name):
        N, V, T = y.shape
        y = y[:, :, 0].view(N, V, 1)
        z_0 = self.condition_encoder(y, heart_name)
        z_0 = torch.squeeze(z_0)
        z_0 = self.initial(z_0)
        
        return z_0
    
    def time_modeling(self, T, z_0, z_c):
        N, V, C = z_0.shape

        z_prev = z_0
        z = []
        for i in range(1, T):
            z_t = self.propagation(z_prev, z_c)
            z_prev = z_t
            z_t = z_t.view(1, N, V, C)
            z.append(z_t)
        z = torch.cat(z, dim=0)
        z_0 = z_0.view(1, N, V, C)
        z = torch.cat([z_0, z], dim=0)
        z = z.permute(1, 2, 3, 0).contiguous()

        return z
    
    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, heart_name, label, D, D_label):
        # q(c | D)
        N, V, T = x.shape
        y = one_hot_label(label[:, 2] - 1, x)
        z_x = self.signal_encoder(x, heart_name, y)

        z_Ds = []
        K = D.shape[1]
        D_ys = []
        for i in range(K):
            D_yi = D_label[:, i, :]
            D_yi = one_hot_label(D_yi[:, 2] - 1, x)
            D_ys.append(D_yi)

            Di = D[:, i, :, :].view(N, V, T)
            z_Ds.append(self.signal_encoder(Di, heart_name, D_yi))
        
        mu_c, logvar_c = self.get_latent_domain(z_Ds, heart_name)
        z_c = self.reparameterization(mu_c, logvar_c)

        # q(z)
        z_0 = self.get_latent_initial(y, heart_name)

        # p(x | z, c)
        z = self.time_modeling(T, z_0, z_c)
        x = self.decoder(z, heart_name)

        # KL on all data
        z_Ds.append(z_x)
        mu_c_full, logvar_c_full = self.get_latent_domain(z_Ds, heart_name)

        return (x, None), (mu_c, logvar_c, mu_c_full, logvar_c_full)
    
    def personalization(self, eval_x, heart_name, label, eval_label, D, D_label):        
        # q(c | D)
        N, V, T = eval_x.shape
        eval_y = one_hot_label(eval_label[:, 2] - 1, eval_x)
        z_x = self.signal_encoder(eval_x, heart_name, eval_y)
        
        z_Ds = []
        K = D.shape[1]
        D_ys = []
        for i in range(K):
            D_yi = D_label[:, i, :]
            D_yi = one_hot_label(D_yi[:, 2] - 1, eval_x)
            D_ys.append(D_yi)

            Di = D[:, i, :, :].view(N, V, T)
            z_Ds.append(self.signal_encoder(Di, heart_name, D_yi))
        
        mu_c, logvar_c = self.get_latent_domain(z_Ds, heart_name)
        z_c = self.reparameterization(mu_c, logvar_c)

        # q(z)
        y = one_hot_label(label[:, 2] - 1, eval_x)
        z_0 = self.get_latent_initial(y, heart_name)

        # p(x | z, c)
        z = self.time_modeling(T, z_0, z_c)
        x = self.decoder(z, heart_name)
        
        return (x, None), (mu_c, logvar_c, None, None)


class ODEDisentangledDynamics(BaseModel):
    def __init__(self,
                 num_channel,
                 latent_dim,
                 obs_dim,
                 initial_dim,
                 ode_func_type,
                 ode_num_layers,
                 ode_method,
                 rnn_type):
        super().__init__()
        self.nf = num_channel
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.initial_dim = initial_dim
        self.ode_func_type = ode_func_type
        self.ode_num_layers = ode_num_layers
        self.ode_method = ode_method
        self.rnn_type = rnn_type

        # encoder
        self.conv1 = Spatial_Block(self.nf[0], self.nf[1], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = Spatial_Block(self.nf[1], self.nf[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = Spatial_Block(self.nf[2], self.nf[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv4 = Spatial_Block(self.nf[3], self.nf[4], dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Conv2d(self.nf[4], self.nf[5], 1)
        self.fce2 = nn.Conv2d(self.nf[5], latent_dim, 1)

        # Domain model
        self.domain_seq = RnnEncoder(latent_dim, latent_dim,
                                           dim=3,
                                           kernel_size=3,
                                           norm=False,
                                           n_layer=1,
                                           rnn_type=rnn_type,
                                           bd=False,
                                           reverse_input=False)
        self.domain = Aggregator(latent_dim, latent_dim, obs_dim, stochastic=False)

        self.initial_seq = RnnEncoder(latent_dim, latent_dim,
                                           dim=3,
                                           kernel_size=3,
                                           norm=False,
                                           n_layer=1,
                                           rnn_type=rnn_type,
                                           bd=False,
                                           reverse_input=False)
        self.initial = Aggregator(latent_dim, latent_dim, initial_dim, stochastic=False)
        
        self.propagation = Propagation(latent_dim, fxn_type=ode_func_type, num_layers=ode_num_layers, method=ode_method, rtol=1e-5, atol=1e-7)
        # self.propagation = Transition(latent_dim, latent_dim, stochastic=False)

        # decoder
        self.fcd3 = nn.Conv2d(latent_dim, self.nf[5], 1)
        self.fcd4 = nn.Conv2d(self.nf[5], self.nf[4], 1)

        self.deconv4 = Spatial_Block(self.nf[4], self.nf[3], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv3 = Spatial_Block(self.nf[3], self.nf[2], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv2 = Spatial_Block(self.nf[2], self.nf[1], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv1 = Spatial_Block(self.nf[1], self.nf[0], dim=3, kernel_size=(3, 1), process='d', norm=False)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

        self.P10 = dict()
        self.P21 = dict()
        self.P32 = dict()
        self.P43 = dict()

        self.P01 = dict()
        self.P12 = dict()
        self.P23 = dict()
        self.P34 = dict()

    def setup(self, heart_name, data_path, batch_size, ecgi, graph_method):
        params = get_params(data_path, heart_name, batch_size, ecgi, graph_method)
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        self.bg4[heart_name] = params["bg4"]
        
        self.P10[heart_name] = params["P10"]
        self.P21[heart_name] = params["P21"]
        self.P32[heart_name] = params["P32"]
        self.P43[heart_name] = params["P43"]

        self.P01[heart_name] = params["P01"]
        self.P12[heart_name] = params["P12"]
        self.P23[heart_name] = params["P23"]
        self.P34[heart_name] = params["P34"]

    def embedding(self, data, heart_name):
        batch_size, seq_len = data.shape[0], data.shape[-1]
        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = \
            data.view(batch_size, -1, self.nf[0], seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[1] * seq_len)
        x = torch.matmul(self.P01[heart_name], x)
        
        # layer 2
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[1], seq_len), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.conv2(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[2] * seq_len)
        x = torch.matmul(self.P12[heart_name], x)
        
        # layer 3
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], seq_len), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.conv3(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[3] * seq_len)
        x = torch.matmul(self.P23[heart_name], x)

        # layer 4
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], seq_len), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.conv4(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[4] * seq_len)
        x = torch.matmul(self.P34[heart_name], x)

        # latent
        x = x.view(batch_size, -1, self.nf[4], seq_len)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = F.elu(self.fce1(x), inplace=True)
        x = torch.tanh(self.fce2(x))

        x = x.permute(0, 2, 1, 3).contiguous()
        return x
    
    def time_modeling(self, x, heart_name):
        N, V, C, T = x.shape
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr

        # Domain
        _x = self.domain_seq(x, edge_index, edge_attr)
        z_D = self.domain(_x)

        # Initial state
        _x = self.initial_seq(x[:, :, :, :self.initial_dim], edge_index, edge_attr)
        z_0 = self.initial(_x)

        z_prev = z_0
        z = []
        for i in range(T):
            z_t = self.propagation(z_prev, z_D, 1, steps=1)
            z_prev = z_t[0]
            # z_t = self.propagation(z_prev, z_D)
            # z_prev = z_t
            z_t = z_t.view(1, N, V, C)
            z.append(z_t)
        z = torch.cat(z, dim=0)
        z = z.permute(1, 2, 3, 0).contiguous()

        return z
    
    def decoder(self, x, heart_name):
        batch_size, seq_len = x.shape[0], x.shape[-1]
        x = x.permute(0, 2, 1, 3).contiguous()

        x = F.elu(self.fcd3(x), inplace=True)
        x = F.elu(self.fcd4(x), inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.nf[4] * seq_len)
        x = torch.matmul(self.P43[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[4], seq_len), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.deconv4(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[3] * seq_len)
        x = torch.matmul(self.P32[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], seq_len), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.deconv3(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[2] * seq_len)
        x = torch.matmul(self.P21[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], seq_len), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.deconv2(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[1] * seq_len)
        x = torch.matmul(self.P10[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[1], seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        x = self.deconv1(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, seq_len)
        
        return x

    def forward(self, x, heart_name, label=None):
        embed = self.embedding(x, heart_name)
        z = self.time_modeling(embed, heart_name)
        x = self.decoder(z, heart_name)
        return (x, None), (None, None, None, None)


def temp_figure(signal, name):
    import os
    import matplotlib.pyplot as plt
    plt.plot(signal.cpu().detach().numpy())
    plt.savefig('vtk/signals/{}.png'.format(name))
    plt.close()
