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


class DKF(BaseModel):
    def __init__(self,
                 num_channel,
                 obs_dim,
                 latent_feature,
                 rnn_type,
                 orthogonal_init,
                 rnn_bidirection,
                 reverse_input,
                 rnn_dim,
                 train_init,
                 z_dim,
                 transition_dim,
                 sample,
                 clip=True):
        super().__init__()
        self.nf = num_channel
        self.latent_feature = latent_feature
        self.rnn_type = rnn_type
        self.orthogonal_init = orthogonal_init
        self.rnn_bidirection = rnn_bidirection
        self.reverse_input = reverse_input
        self.rnn_dim = rnn_dim
        self.train_init = train_init
        self.z_dim = z_dim
        self.transition_dim = transition_dim
        self.sample = sample
        self.clip = clip

        # encoder + inverse
        self.conv1 = Spatial_Block(self.nf[0], self.nf[1], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = Spatial_Block(self.nf[1], self.nf[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = Spatial_Block(self.nf[2], self.nf[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv4 = Spatial_Block(self.nf[3], self.nf[4], dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Conv2d(self.nf[4], self.nf[5], 1)
        self.fce2 = nn.Conv2d(self.nf[5], latent_feature, 1)
        
        # Domain model
        self.domain = DomainEncoder(rnn_dim, z_dim, obs_dim)

        # RNN encoder
        self.rnn_encoder = RnnEncoder(latent_feature, rnn_dim, dim=3, kernel_size=3, norm=False,
                                      n_layer=1, rnn_type=rnn_type, bd=rnn_bidirection,
                                      reverse_input=reverse_input, orthogonal_init=orthogonal_init)
        # combiner
        self.combiner = Combiner(z_dim, rnn_dim, dim=3, kernel_size=3, norm=False, clip=clip)
        # transition
        self.transition = Transition(z_dim, transition_dim, identity_init=True, clip=clip)
        # initialize hidden states
        self.mu_p_0, self.var_p_0 = self.transition.init_z_0(trainable=train_init)
        self.mu_p_D, self.var_p_D = self.transition.init_z_0(trainable=train_init)
        self.z_q_0 = self.combiner.init_z_q_0(trainable=train_init)

        # decoder
        self.fcd3 = nn.Conv2d(z_dim, self.nf[5], 1)
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
    
    def reparameterization(self, mu, var):
        if not self.sample:
            return mu
        
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def time_modeling(self, x, heart_name):
        batch_size, V, T = x.shape[0], x.shape[1], x.shape[-1]
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr
        h_rnn = self.rnn_encoder(x, edge_index, edge_attr)

        # Domain
        mu_q_D, var_q_D = self.domain(h_rnn)
        z_q_D = self.reparameterization(mu_q_D, var_q_D)
        mu_p_D = self.mu_p_D.expand(batch_size, V, self.z_dim)
        var_p_D = self.var_p_D.expand(batch_size, V, self.z_dim)
        z_p_D = self.reparameterization(mu_p_D, var_p_D)

        # initial condition
        z_q_0 = self.z_q_0.expand(batch_size, V, self.z_dim)  # q(z_0)
        mu_p_0 = self.mu_p_0.expand(batch_size, V, self.z_dim)
        var_p_0 = self.var_p_0.expand(batch_size, V, self.z_dim)
        z_prev = z_q_0

        mu_q_seq = torch.zeros([batch_size, V, self.z_dim, T]).to(x.device)
        var_q_seq = torch.zeros([batch_size, V, self.z_dim, T]).to(x.device)
        mu_p_seq = torch.zeros([batch_size, V, self.z_dim, T]).to(x.device)
        var_p_seq = torch.zeros([batch_size, V, self.z_dim, T]).to(x.device)
        z_q_seq = torch.zeros([batch_size, V, self.z_dim, T]).to(x.device)
        z_p_seq = torch.zeros([batch_size, V, self.z_dim, T]).to(x.device)

        for t in range(T):
            # q(z_t | z_{t-1}, x_{t:T})
            mu_q, var_q = self.combiner(h_rnn[:, :, :, t], z_prev, edge_index, edge_attr)
            zt_q = self.reparameterization(mu_q, var_q)
            z_prev = zt_q

            # p(z_{t+1} | z_t)  # TODO: change p to separate the input from the observation
            mu_p, var_p = self.transition(z_prev, z_q_D)
            zt_p = self.reparameterization(mu_p, var_p)

            mu_q_seq[:, :, :, t] = mu_q
            var_q_seq[:, :, :, t] = var_q
            z_q_seq[:, :, :, t] = zt_q
            mu_p_seq[:, :, :, t] = mu_p
            var_p_seq[:, :, :, t] = var_p
            z_p_seq[:, :, :, t] = zt_p
        
        # move one step ahead for p(z_t | z_{t-1})
        mu_p_0 = mu_p_0.reshape(batch_size, V, self.z_dim, 1)
        mu_p_seq = torch.cat([mu_p_0, mu_p_seq[:, :, :, :-1]], dim=-1)
        var_p_0 = var_p_0.reshape(batch_size, V, self.z_dim, 1)
        var_p_seq = torch.cat([var_p_0, var_p_seq[:, :, :, :-1]], dim=-1)
        z_p_0 = self.reparameterization(mu_p_0, var_p_0)
        z_p_seq = torch.cat([z_p_0, z_p_seq[:, :, :, :-1]], dim=-1)

        # cat domain term
        mu_p_D = mu_p_D.view(batch_size, V, self.z_dim, 1)
        mu_p_seq = torch.cat([mu_p_seq, mu_p_D], dim=-1)
        var_p_D = var_p_D.view(batch_size, V, self.z_dim, 1)
        var_p_seq = torch.cat([var_p_seq, var_p_D], dim=-1)
        mu_q_D = mu_q_D.view(batch_size, V, self.z_dim, 1)
        mu_q_seq = torch.cat([mu_q_seq, mu_q_D], dim=-1)
        var_q_D = var_q_D.view(batch_size, V, self.z_dim, 1)
        var_q_seq = torch.cat([var_q_seq, var_q_D], dim=-1)

        return z_q_seq, z_p_seq, mu_q_seq, var_q_seq, mu_p_seq, var_p_seq

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
        z_q_seq, z_p_seq, mu_q_seq, var_q_seq, mu_p_seq, var_p_seq = self.time_modeling(embed, heart_name)
        x_q = self.decoder(z_q_seq, heart_name)

        x_p = self.decoder(z_p_seq, heart_name)
        return (x_q, x_p), (mu_q_seq, var_q_seq, mu_p_seq, var_p_seq)


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

        # u1 = data[0, 300, :]
        # u2 = data[0, 286, :]
        # temp_figure(u1, 'u1_en_l0_2')
        # temp_figure(u2, 'u2_en_l0_2')

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

        # u1 = x[0, 45, :]
        # u2 = x[0, 45, :]
        # temp_figure(u1, 'u1_latent_in_2')
        # temp_figure(u2, 'u2_latent_in_2')

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

        # u1 = z[0, 45, :]
        # u2 = z[0, 45, :]
        # temp_figure(u1, 'u1_latent_out_2')
        # temp_figure(u2, 'u2_latent_out_2')
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
        
        # u1 = x[0, 300, :]
        # u2 = x[0, 286, :]
        # temp_figure(u1, 'u1_de_l1_2')
        # temp_figure(u2, 'u2_de_l1_2')
        return x

    def forward(self, x, heart_name):
        embed = self.embedding(x, heart_name)
        z = self.time_modeling(embed, heart_name)
        x = self.decoder(z, heart_name)
        return (x, None), (None, None, None, None)


class BayesianFilterMask(BaseModel):
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

        # u1 = data[0, 300, :]
        # u2 = data[0, 286, :]
        # temp_figure(u1, 'u1_en_l0_2')
        # temp_figure(u2, 'u2_en_l0_2')

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
    
    def time_modeling(self, x, heart_name, label=None):
        N, V, C, T = x.shape
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr

        # u1 = x[0, 45, :]
        # u2 = x[0, 45, :]
        # temp_figure(u1, 'u1_latent_in_2')
        # temp_figure(u2, 'u2_latent_in_2')

        # Domain
        mask = torch.zeros_like(x[:, :, :, 0])
        if label is not None:
            mask[:, label[:, -1]] = 1
        _x = self.domain_embedding(x, edge_index, edge_attr)
        z_D = self.domain(_x)

        z = []
        # z.append(last_h.view(1, N, V, C))

        mask = mask.view(N, V, C, 1)
        x = torch.cat((mask, x), dim=-1)
        x = x.permute(3, 0, 1, 2).contiguous()
        x = x.view(T + 1, N * V, C)
        last_h = x[0]

        for t in range(T):
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

        # u1 = z[0, 45, :]
        # u2 = z[0, 45, :]
        # temp_figure(u1, 'u1_latent_out_2')
        # temp_figure(u2, 'u2_latent_out_2')
        return z
    
    def decoder(self, x, heart_name, label=None):
        batch_size, seq_len = x.shape[0], x.shape[-1]
        x = x.permute(0, 2, 1, 3).contiguous()

        x = F.elu(self.fcd3(x), inplace=True)
        x = F.elu(self.fcd4(x), inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.nf[4] * seq_len)
        x = torch.matmul(self.P43[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[4], seq_len), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        mask = torch.zeros_like(x[:, :, :, 0]).contiguous()
        if label is not None:
            mask[:, label[:, 3]] = 1
        mask = mask.view(batch_size, -1, self.nf[4], 1)
        x = torch.cat((mask, x), dim=-1)
        x = self.deconv4(x, edge_index, edge_attr)
        x = x[:, :, :, 1:].contiguous()

        x = x.view(batch_size, -1, self.nf[3] * seq_len)
        x = torch.matmul(self.P32[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], seq_len), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        mask = torch.zeros_like(x[:, :, :, 0]).contiguous()
        if label is not None:
            mask[:, label[:, 2]] = 1
        mask = mask.view(batch_size, -1, self.nf[3], 1)
        x = torch.cat((mask, x), dim=-1)
        x = self.deconv3(x, edge_index, edge_attr)
        x = x[:, :, :, 1:].contiguous()

        x = x.view(batch_size, -1, self.nf[2] * seq_len)
        x = torch.matmul(self.P21[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], seq_len), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        mask = torch.zeros_like(x[:, :, :, 0]).contiguous()
        if label is not None:
            mask[:, label[:, 1]] = 1
        mask = mask.view(batch_size, -1, self.nf[2], 1)
        x = torch.cat((mask, x), dim=-1)
        x = self.deconv2(x, edge_index, edge_attr)
        x = x[:, :, :, 1:].contiguous()

        x = x.view(batch_size, -1, self.nf[1] * seq_len)
        x = torch.matmul(self.P10[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[1], seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        mask = torch.zeros_like(x[:, :, :, 0]).contiguous()
        if label is not None:
            mask[:, label[:, 0]] = 1
        mask = mask.view(batch_size, -1, self.nf[1], 1)
        x = torch.cat((mask, x), dim=-1)
        x = self.deconv1(x, edge_index, edge_attr)
        x = x[:, :, :, 1:].contiguous()

        x = x.view(batch_size, -1, seq_len)
        
        # u1 = x[0, 300, :]
        # u2 = x[0, 286, :]
        # temp_figure(u1, 'u1_de_l1_2')
        # temp_figure(u2, 'u2_de_l1_2')
        return x

    def forward(self, x, heart_name, label=None):
        embed = self.embedding(x, heart_name)
        z = self.time_modeling(embed, heart_name, label)
        x = self.decoder(z, heart_name, label)
        return (x, None), (None, None, None, None)


def temp_figure(signal, name):
    import os
    import matplotlib.pyplot as plt
    plt.plot(signal.cpu().detach().numpy())
    plt.savefig('vtk/signals/{}.png'.format(name))
    plt.close()
