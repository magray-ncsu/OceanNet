import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
import netCDF4 as nc
import xarray as xr
import time

# custom functions
from data_loader_SSH import load_test_data
from data_loader_SSH import load_train_data
from count_trainable_params import count_parameters
from utilities3 import *
import mod_hausdorff as mh

DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
data_dir = '/path_to_data_directory/'

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(3, self.width) 
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, DEVICE):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(DEVICE)
    

def spectral_loss_val(output, target,wavenum_init,wavenum_init_ydir,lambda_reg,ocean_grid):
    '''
        Spectral regularizer + MSE
        wavenum_init:      float, cutoff wavenumber in zonal direction
        wavenum_init_ydir: float, cutoff wavenumber in meridional direction
        lambda_reg:        float, regularization coefficient (1-lambda_reg)*MSE + lambda_reg*spectral_loss
        ocean_grid:        float, number of ocean points
    '''
    # MSE
    loss1 = torch.sum((output-target)**2)/ocean_grid
    
    # Spectral
    out_fft = torch.mean(torch.abs(torch.fft.rfft(output[:,:,:,0],dim=2)),dim=1)
    target_fft = torch.mean(torch.abs(torch.fft.rfft(target[:,:,:,0],dim=2)),dim=1)

    out_fft_ydir = torch.mean(torch.abs(torch.fft.rfft(output[:,:,:,0],dim=1)),dim=2)
    target_fft_ydir = torch.mean(torch.abs(torch.fft.rfft(target[:,:,:,0],dim=1)),dim=2)

    loss2 = torch.mean(torch.abs(out_fft[:,wavenum_init:]-target_fft[:,wavenum_init:]))
    loss2_ydir = torch.mean(torch.abs(out_fft_ydir[:,wavenum_init_ydir:]-target_fft_ydir[:,wavenum_init_ydir:]))
    
    # Total
    loss = ((1-lambda_reg)*loss1 + 0.25*(lambda_reg)*loss2 + 0.25*(lambda_reg)*loss2_ydir)

    return loss


def PECstep(net,input_batch,delta_t):
    '''
        Numerical integration scheme: Pretictor-Evaluator-Corrector
        net:            Model object
        input_batch:    imput images
        delta_t:        hyperparameter
    '''
    output_1 = net(input_batch.cuda()) + input_batch.cuda() 
    return input_batch.cuda() + delta_t*0.5*(net(input_batch.cuda())+net(output_1))

def Eulerstep(net,input_batch, delta_t):
    '''
        Numerical integration scheme: 1st Order Euler
        net:            Model object
        input_batch:    imput images
        delta_t:        hyperparameter
    '''
    output_1 = net(input_batch.to(DEVICE))
    return input_batch.to(DEVICE) + delta_t*(output_1)

def directstep(net,input_batch,delta_t):
    '''
        No numerical integrator
    '''
    return net(input_batch.to(DEVICE))

# function for 1 epoch of training
def one_epoch(batch_size, optimizer, net, lambda_reg, lead, trainN, wavenum_init, wavenum_init_ydir, delta_t, timesteps = False):

    for k in range(1993, 2019): # loop through years of data
        
        # load data
        G = xr.open_dataset(data_dir + 'EnKF_surface_'+str(k)+'_5dmean_ec.nc')
        
        psi_train_input_Tr_torch, psi_train_label_Tr_torch, ocean_grid  = \
            load_train_data(G,lead,trainN)
        
        # Normalize
        psi_train_input_Tr_torch_norm_level1 = ((psi_train_input_Tr_torch[:,0,None,:,:].to(DEVICE)-M_test_level1.to(DEVICE))/STD_test_level1.to(DEVICE))
        psi_train_label_Tr_torch_norm_level1 = ((psi_train_label_Tr_torch[:,0,None,:,:].to(DEVICE)-M_test_level1.to(DEVICE))/STD_test_level1.to(DEVICE))
        
        # shuffle
        indices = torch.randperm(psi_train_label_Tr_torch.size(0))
        
        for step in range(0,trainN-batch_size,batch_size): # loop through batches
            
            # get batch
            idxs = indices[step:step+batch_size]
            input_batch = psi_train_input_Tr_torch_norm_level1[idxs,:,:,:]
            label_batch = psi_train_label_Tr_torch_norm_level1[idxs,:,:,:]

            # permute dimensions to match model
            input_batch = input_batch.to(DEVICE).permute(0,2,3,1)
            label_batch = label_batch[:,0,:,:,:]
            label_batch = label_batch.to(DEVICE).permute(0,2,3,1)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            output = PECstep(net,input_batch.to(DEVICE),delta_t)

            loss = spectral_loss_val(
                output.to(DEVICE),
                label_batch.to(DEVICE), 
                wavenum_init,wavenum_init_ydir,lambda_reg,
                (torch.tensor(ocean_grid)).to(DEVICE))

            loss.backward()
            optimizer.step()
                


def objective(trial):
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    # hyper parameters
    modes = 128             # for FNO
    width = 20              # for FNO
    wavenum_init = 10       # for spectral regularizer
    wavenum_init_ydir = 30  # for spectral regularizer

    # Generate the model
    net = FNO2d(modes, modes, width).to(DEVICE)
  
    # Generate the optimizer
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    # Other hyperparameters
    lead = 4
    trainN = 365 - lead
    lambda_reg = 0.9        # for loss function
    epochs_init = 180
    batch_size = 30
    delta_t=1.0             # for numerical integrator
    
    # load & normalize validation data
    FF=xr.open_dataset(data_dir + 'EnKF_surface_2020_5dmean_ec.nc')
    global M_test_level1
    global STD_test_level1
    M_test_level1 = xr.open_dataset(data_dir + 'EnKF_surface_1993to2020_avg.nc')    # mean SSH field
    STD_test_level1 = xr.open_dataset(data_dir + 'EnKF_surface_5day_std.nc')        # standard deviation SSH field
    
    lat_bounds = np.where(                                                          # ensure same grid as training data  
        (M_test_level1.lat_rho[:,0]>=FF.lat_rho.min()) & (M_test_level1.lat_rho[:,0]<=FF.lat_rho.max())
    )[0]
    lon_bounds = np.where(
        (M_test_level1.lon_rho[0,:]>=FF.lon_rho.min()) & (M_test_level1.lon_rho[0,:]<=FF.lon_rho.max())
    )[0]
    
    M_test_level1 = torch.tensor(
        M_test_level1.sel(eta_rho = lat_bounds, xi_rho = lon_bounds).SSH.values
    ).to(DEVICE)
    M_test_level1[torch.isnan(M_test_level1)] = 0.0
    
    STD_test_level1 = torch.tensor(
        STD_test_level1.sel(eta_rho = lat_bounds, xi_rho = lon_bounds).SSH_std.values
    ).to(DEVICE)
    STD_test_level1[torch.isnan(STD_test_level1)] = 1.0
    STD_test_level1[STD_test_level1==0] = 1.0
    
    lon = torch.tensor(FF.lon_rho.values)
    lat = torch.tensor(FF.lat_rho.values)
    
    psi_test_input_Tr_torch, psi_test_label_Tr_torch,ocean_grid  = load_test_data(FF,lead)
    psi_test_input_Tr_torch = \
        ((psi_test_input_Tr_torch[:,0,None,:,:].to(DEVICE)-M_test_level1)/STD_test_level1)
    psi_test_label_Tr_torch = \
        ((psi_test_label_Tr_torch[:,0,None,:,:].to(DEVICE)-M_test_level1)/STD_test_level1)
    
    Nlat = psi_test_label_Tr_torch.size(2)
    Nlon = psi_test_label_Tr_torch.size(3)

    # Loop: train single timestep model for one epoch, validate
    for epoch in range(0, epochs_init):
        net.train()
        one_epoch(batch_size, optimizer, net, lambda_reg, lead, trainN, wavenum_init, wavenum_init_ydir, delta_t)
        net.eval()
        with torch.no_grad():
            indices = torch.randperm(psi_test_label_Tr_torch.size(0)-120)[:20]
            output = PECstep(
                net,
                psi_test_input_Tr_torch[indices,:,0:Nlat,0:Nlon].reshape(
                    [len(indices),1,Nlat,Nlon]).permute(0,2,3,1).to(DEVICE), delta_t
            )
            for i in range(1,30):
                output =PECstep(net,output,delta_t)
            
            label = psi_test_label_Tr_torch[indices+i*lead,:,:,:].permute(0,2,3,1).to(DEVICE)
            val_loss = spectral_loss_val(
                output,label,wavenum_init,wavenum_init_ydir,lambda_reg,ocean_grid)

        print('[ {} , {} ]'.format(epoch, val_loss))
            
    torch.save(net.state_dict(),
               './Models/{}_trial{}.pt'.format(study_name, trial))
        
    return val_loss


if __name__ == "__main__":
    global study_name
    study_name = 'FNO_single'
    objective('PECstep')
