import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import netCDF4 as nc

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_test_data(FF,lead):

  SSH = np.asarray(FF['SSH'])
  ocean_grid_size = float(np.count_nonzero(~np.isnan(SSH[0,:,:])))
##### convert Nans to zero ####
  SSH[np.isnan(SSH)]=0.0
################################
  SSH=SSH[0:,:,:]
  
  uv = np.zeros([np.size(SSH,0),1,np.size(SSH,1),np.size(SSH,2)])

  uv [:,0,:,:] = SSH

  uv_test_input = uv[0:np.size(SSH,0)-lead,:,:,:]
  uv_test_label = uv[lead:np.size(SSH,0),:,:,:]
 


## convert to torch tensor
  uv_test_input_torch = torch.from_numpy(uv_test_input).float()
  uv_test_label_torch = torch.from_numpy(uv_test_label).float()

  return uv_test_input_torch, uv_test_label_torch,ocean_grid_size


def load_train_data(GG, lead,trainN, multistep = 1):
    
    try: multistep = int(multistep)
    except: raise ValueError('Arguement "multistep" must be a positive integer')
    
    if multistep < 1:
        raise ValueError('Argument "multistep" must be a positive integer')
        
    SSH=np.asarray(GG['SSH'])
  
    #### remove NANS###########
    ocean_grid_size = np.count_nonzero(~np.isnan(SSH[0,:,:]))
    
    SSH[np.isnan(SSH)]=0
    SSH=SSH[0:trainN,:,:]    

    if multistep > 1:
        ssh_train_input_torch = torch.from_numpy(SSH[0:np.size(SSH,0)-lead-multistep,:,:]).float()
        ssh_train_label_torch = torch.zeros([ssh_train_input_torch.size()[0], multistep, np.size(SSH,1), np.size(SSH,2)])
        ssh_train_label_torch[:,0,:,:] = torch.from_numpy(SSH[lead:np.size(SSH,0)-multistep,:,:])
        for k in range(1,multistep):
            ssh_train_label_torch[:,k,:,:] = torch.from_numpy(SSH[lead+k:np.size(SSH,0)-multistep+k,:,:])
            
    else:
        ssh_train_input_torch = torch.from_numpy(SSH[0:np.size(SSH,0)-lead])
        ssh_train_label_torch = torch.from_numpy(SSH[lead:np.size(SSH,0)])
        
    ssh_train_input_torch = ssh_train_input_torch.reshape(ssh_train_input_torch.size(0), 1, np.size(SSH,1), np.size(SSH,2))
    ssh_train_label_torch = ssh_train_label_torch.reshape(ssh_train_input_torch.size(0), multistep, 1, np.size(SSH,1), np.size(SSH,2))

    return ssh_train_input_torch, ssh_train_label_torch, ocean_grid_size
