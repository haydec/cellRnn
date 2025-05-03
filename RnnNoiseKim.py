
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import sys
import os
import re
from itertools import accumulate
import warnings
import gc
import random
import string
import sqlite3

import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.std import tqdm
import pandas as pd
import seaborn as sns

from collections import Counter, namedtuple
from itertools import combinations

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import locale
import datetime

import torch.nn.functional as F  # Import the functional API

# Set the locale to Japanese
locale.setlocale(locale.LC_TIME, 'ja_JP')

# Define the namedtuple only once, outside Class
ReinitData = namedtuple('ReinitData', ['Tepoch', 'Replace_Idx'])

class CustomDataset(Dataset):
    def __init__(self, u, z):
        self.u = u
        self.z = z
        

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):

        try:
            return self.u[idx], self.z[idx]
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            raise

class SettingsDict(dict):
    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(f"Cannot add new key: '{key}' is not allowed.")
        super().__setitem__(key, value)


class SurrogateSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sigmoid(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        surrogate_grad = SurrogateSigmoid.fast_sigmoid_surrogate(input, alpha=1.0)
        return grad_output * surrogate_grad
    
    @staticmethod
    def fast_sigmoid_surrogate(x, alpha=1.0):
        return 1 / (1 + alpha * x.abs())**2
    
surrogate_sigmoid = SurrogateSigmoid.apply

class SurrogateGeneralizedSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input,fmax,beta,c):
        ctx.save_for_backward(input, fmax, beta, c)
        output = fmax * torch.sigmoid(beta * (input - c))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, fmax, beta, c = ctx.saved_tensors
        sig = torch.sigmoid(beta * (input - c))
        grad_input = grad_output * fmax * beta * sig * (1 - sig) + 0.001
        grad_fmax = None
        grad_beta = None
        grad_c = None
        return grad_input, grad_fmax, grad_beta, grad_c

# Alias for Generalized edsigmoid
surrogate_generalized_sigmoid = SurrogateGeneralizedSigmoid.apply

class CustomRnnStruct(nn.Module):
    def __init__(self, N, Prop_PUL,Prop_TRN,apply_dale, prop_inh, prop_som,prob_rec,gain, input_size, output_size, noise_channels,tausRange, ppc_rate_String, pul_rate_String, trn_rate_String, device=None, NewModel=None, SubModelName = None, db_path = None):
        super(CustomRnnStruct, self).__init__()

        if NewModel is None:
            raise ValueError("Specify if Model is New")
        assert NewModel is True or NewModel is False, "The value must be True or False"

        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #||||||||| INITILIASE VALUES BASED ON SETTINGS |||||||||
        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||

        # All Neurons
        
        Neuron_idx = np.arange(N, dtype=np.uint16)
        

        N_TRN = int(np.round(Prop_TRN*N))
        N_PUL = int(np.round(Prop_PUL*N))



        #==============================
        #======== PPC Index============
        #==============================

        N_PPC = int(N - N_TRN - N_PUL)
        PPC_idx = Neuron_idx[0:N_PPC]
        PPC_N = len(PPC_idx)

        # Assign Inhibitory Neurons
        num_PPCinh = int(np.floor(N_PPC*prop_inh))
        PPCinh_idx = np.random.choice(PPC_idx,size=num_PPCinh,replace=False)
        PPCexh_idx = np.delete(PPC_idx, PPCinh_idx)

        # Calculate Number of Excititory and Inhibitory (PV and SomN) Neurons
        PPCexh_N = len(PPCexh_idx)
        PPCinh_N = len(PPCinh_idx)


        # Assign Som Neurons (Subset of Inhibitory Neurons) 
        num_som = int(np.floor(PPCinh_N *prop_som))
        PPCsomM_idx = np.random.choice(PPCinh_idx,size=num_som,replace=False)
        PPCsomM_N = len(PPCsomM_idx)

        #==============================
        #======== PUL Index============
        #==============================

        PULexh_idx  = Neuron_idx[N_PPC:N_PPC+N_PUL]
        PULexh_N = len(PULexh_idx)

        #==============================
        #======== TRN Index============
        #==============================

        TRNinh_idx = Neuron_idx[N_PPC+N_PUL:N_PPC+N_PUL+N_TRN]
        TRNinh_N = len(TRNinh_idx)

        assert(sum( PPCexh_idx >= 0)+  sum( PPCinh_idx >= 0 )+ sum( PULexh_idx >= 0 )+ sum( TRNinh_idx >= 0 ) == N )


        StructureMask = np.zeros((N,N))

        #---------------------------------
        #------PPC To PPC Connections-----
        # --------------------------------

        # PPC -> PPC
        for To in PPC_idx:
            for From in PPC_idx:
                StructureMask[To,From] = 1

        # PPC_Inh -> PPC_Som
        for To in PPCsomM_idx:
            for From in PPCinh_idx:
                StructureMask[To,From] = 0


        #---------------------------------
        #--PPC Connections To Other Areas-
        # --------------------------------

        #PPC_Exh - > PUL_Exh
        for To in PULexh_idx:
            for From in PPCexh_idx:
                StructureMask[To,From] = 1


        #PPC_Exh to Trn_Inh
        for To in TRNinh_idx:
            for From in PPCexh_idx:
                StructureMask[To,From] = 1

        #---------------------------------
        #------Thalmas Connections--------
        # --------------------------------


        #PUL_Exh - > PPC (ALL)
        for To in PPC_idx:
            for From in PULexh_idx:
                StructureMask[To,From] = 1

        #PUL_Exh - > TRN Inh
        for To in TRNinh_idx:
            for From in PULexh_idx:
                StructureMask[To,From] = 1



        # TRN_Inh -> PUL_Exh
        for To in PULexh_idx:
            for From in TRNinh_idx:
                StructureMask[To,From] = 1

        #*********************************
        # ****** Synaptic Type Mask ******
        #*********************************
        SynapticTypeMask = np.ones((N,N)) # Initialize every Neuron to Excitatory

        #---------------------------------
        #------PPC To PPC Connections-----
        # --------------------------------

        # PPC Inh -> All Neurons
        for To in Neuron_idx:
            for From in PPCinh_idx:
                SynapticTypeMask[To,From] = -1


        #---------------------------------
        #------Thalmas Connections--------
        # --------------------------------

        #----------- PUL -----------------

        # Only Pul Exh

        #------------TRN------------------

        # TRN_Inh -> All Neurons
        for To in Neuron_idx:
            for From in TRNinh_idx:
                SynapticTypeMask[To,From] = -1

        SynapseMask= np.multiply(SynapticTypeMask,StructureMask)
        
        #--------------------------------------------------------

        # Input Weights (Not Trained)
        Win_init = np.random.normal(loc=0.0, scale=1.0, size=(N, input_size))
        
        # Recurrent Matrix (Trained)
        Wr_init = np.zeros((N, N), dtype = np.float16)
        idx = np.where(np.random.rand(N, N) < prob_rec)
        Wr_init[idx[0], idx[1]] = np.random.normal(loc=0, scale=1.0, size=len(idx[0]))
        Wr_init = Wr_init/np.sqrt(N*prob_rec)*gain # scale by a gain to make it chaotic
        if apply_dale == True:
            Wr_init = np.abs(Wr_init)
        
        # Synaptic Time Constant (Trained)
        tauS_init = np.random.uniform(tausRange[0], tausRange[1], (N,1))

        # Readout Weights (Trained)
        Wout_init = np.random.normal(loc=0.0, scale=1.0, size=(output_size, N))/100

        # Internal Noise Weights
        Wnoise_init = np.random.normal(loc=0.0, scale=1.0, size=(N, noise_channels))

        #---------------------------------------------------------

        # Optogenetic Mask 
        WrSize = Wr_init.shape
        OptoMask = torch.ones(WrSize,device=device)
        WinSize = Win_init.shape
        OptoMaskIn = torch.ones(WinSize,device=device)


        PPC_Params_Dict = {"negative_slope":0.01,"max":20,"alpha":1, "centre":0,"fmax":20,"beta":1,"c":0}
        PUL_Params_Dict = {"negative_slope":0.01,"max":20,"alpha":1, "centre":0,"fmax":20,"beta":1,"c":0}
        TRN_Params_Dict = {"negative_slope":0.01,"max":20,"alpha":10,"centre":9,"fmax":20,"beta":1,"c":0}

        
        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #||||||||| CREATE CLASS DATA MEMBERS |||||||||||||||||||
        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||

        self.NumberOfFits = 0
        self.Tepoch = 0
        self.Session = 0



        if db_path == None:
            # Define the characters you want to include (letters and digits)
            characters = string.ascii_letters + string.digits
            if SubModelName == None:
                self.dataBasePath = os.path.join(os.getcwd(), "RNNmodelDB")
            else:
                self.dataBasePath = os.path.join(os.getcwd(), "RNNmodelDB",SubModelName)
            if not os.path.exists(self.dataBasePath):
                os.makedirs(self.dataBasePath)
            NumDB = str(len([name for name in os.listdir(self.dataBasePath) if name.endswith('.db') and os.path.isfile(os.path.join(self.dataBasePath, name))]))
            self.modelSigniture ='RateRNNstruct'+'NUM'+ NumDB 
            # Create a new SQLite database with the same name as the folder inside the new folder
            self.db_path = os.path.join(self.dataBasePath, f"{self.modelSigniture}.db")
        
        self.dType = torch.float32

        # Initialize device as None
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Device Selected: ", self.device)


        self.N = N
        self.apply_dale = apply_dale
        self.prop_inh = prop_inh
        self.prop_som = prop_som
        self.prob_rec = prob_rec
        self.Prop_PUL = Prop_PUL
        self.Prop_TRN = Prop_TRN
        self.gain = gain
        self.tausRange = tausRange
        self.noise_channels = noise_channels

        self.ppc_rate_String = ppc_rate_String
        self.pul_rate_String = pul_rate_String
        self.trn_rate_String = trn_rate_String

        self.rate_PPC_Params = PPC_Params_Dict
        self.rate_PUL_Params = PUL_Params_Dict
        self.rate_TRN_Params = TRN_Params_Dict

   
        assert(self.prop_inh >= 0 and self.prop_inh <= 1)
        assert(self.prop_som >= 0 and self.prop_som <= 1)

                
        # ------Number-------
        # PPC
        self.PPC_N = PPC_N 
        self.PPCexh_N = PPCexh_N
        self.PPCinh_N = PPCinh_N
        self.PPCsomM_N = PPCsomM_N
        # Thalamus
        self.PULexh_N = PULexh_N
        self.TRNinh_N = TRNinh_N
        

        # ---- Structure Details ----
        # PPC
        self.PPC_idx = PPC_idx
        self.PPCexh_idx = PPCexh_idx
        self.PPCinh_idx = PPCinh_idx
        self.PPCsomM_idx = PPCsomM_idx
        # Thalamus
        self.PULexh_idx = PULexh_idx 
        self.TRNinh_idx = TRNinh_idx

        # Create Synaptic Mask which convert columns 
        
        
        self.StructureMask = torch.tensor(StructureMask, dtype=self.dType, device=self.device, requires_grad=False)
        self.SynapticTypeMask = torch.tensor(SynapticTypeMask, dtype=self.dType, device=self.device, requires_grad=False)
        self.SynapseMask = torch.tensor(SynapseMask, dtype=self.dType, device=self.device, requires_grad=False)
  
        self.Win = torch.tensor(Win_init, dtype=self.dType, device=self.device, requires_grad=False)
        self.Wr_init = torch.tensor(Wr_init, dtype=self.dType, device=self.device, requires_grad=False)
        
        

    
        # Parameters that are learned 
        self.Wr = nn.Parameter(torch.tensor(Wr_init, dtype=self.dType, device=self.device), requires_grad=True)
        self.Wout = nn.Parameter(torch.tensor(Wout_init, dtype=self.dType, device=self.device), requires_grad=True)
        self.tauS = nn.Parameter(torch.tensor(tauS_init, dtype=self.dType, device=self.device), requires_grad=True)
        self.b_out = nn.Parameter(torch.tensor(np.random.rand(1)[0], dtype=self.dType, device=self.device), requires_grad=True)
        self.Wnoise = nn.Parameter(torch.tensor(Wnoise_init, dtype=self.dType, device=self.device), requires_grad=True)

        # Optogentic Mask to Suppress Neurons
        self.OptoMask = OptoMask
        self.OptoMaskIn = OptoMaskIn

        # Performance on Evaluation
        self.EvalTrainTotalPerform = None

        self.EvalTrainDiscrimPerform = None     
        self.EvalTrainDiscrimPerformRight = None
        self.EvalTrainDiscrimPerformLeft = None

        self.EvalTrainCatchPerform =  None
        self.EvalTrainCatchPerformRight = None
        self.EvalTrainCatchPerformLeft = None

        


        
        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #||||||||| CREATE MODEL SIGNITURE AND DATABASE |||||||||
        #|||||||||||||||||||||||||||||||||||||||||||||||||||||||

        if db_path == None:

            # Create the new folder if it doesn't already exist
            if not os.path.exists(self.dataBasePath):
                os.makedirs(self.dataBasePath)
                print(f"Directory '{self.modelSigniture}' created.")

            #Create DB for Storage
            self.createEpochDB()
            Validation = {"TotalPerform_mean":0,
            "Discrim_mean":0,                   
            "DiscrimRight_mean":0,
            "DiscrimLeft_mean":0,
            "Catch_mean":0,
            "CatchRight_mean":0,
            "CatchLeft_mean":0,
            "CatchAmbig_mean":0}
            self.save_rnn_to_db(Validation)

        else:
            self.db_path = db_path
            self.load_rnn_from_db(db_path)



    def load_data(self, filename: str) -> None:
        """Loads object from a file."""
        with open(filename, 'rb') as file:
            loaded_obj = pickle.load(file)
            self.__dict__.update(loaded_obj.__dict__)


    def freeze_parameter(self, param_name):
        # Check if the attribute exists and is an nn.Parameter
        if hasattr(self, param_name) and isinstance(getattr(self, param_name), nn.Parameter):
            getattr(self, param_name).requires_grad = False
            print(f"Parameter '{param_name}' is now frozen.")
        else:
            print(f"No parameter named '{param_name}' found.")

    def unfreeze_parameter(self, param_name):
        # Check if the attribute exists and is an nn.Parameter
        if hasattr(self, param_name) and isinstance(getattr(self, param_name), nn.Parameter):
            getattr(self, param_name).requires_grad = True
            print(f"Parameter '{param_name}' is now unfrozen.")
        else:
            print(f"No parameter named '{param_name}' found.")
    

    def resetActivation(self,BrainArea,rateName: str) -> None:
        if rateName not in ['leaky_relu','alpha_leaky_relu','relu','alpha_relu','sigmoid','generalized_sigmoid']:
            ValueError(f"{rateName} is not a valid activation. Choose: 'leaky_relu','alpha_leaky_relu','relu','alpha_relu','sigmoid','generalized_sigmoid'")
        if BrainArea == "PPC":
            self.ppc_rate_String = rateName
        elif BrainArea == "PUL":
            self.pul_rate_String = rateName
        elif BrainArea == "TRN":
            self.trn_rate_String = rateName
        elif BrainArea == "ALL":
            self.ppc_rate_String = rateName
            self.pul_rate_String = rateName
            self.trn_rate_String = rateName
        else:
            ValueError(f"{BrainArea} is not a Valid Brain Area. Choose: 'PPC', 'PUL', 'TRN', 'ALL' ")

        
    def randomize_tau_values(self, p, value_range=(4,20)):
        """
        Randomly sets elements of tau to new values within the specified range with probability p.

        Args:
            p (float): The probability of each element being replaced.
            value_range (tuple): The range of the new values (default is (4, 20)).

        Returns:
            torch.Tensor: The modified tau tensor.
        """
        # Generate a mask with probability p for each element
        mask = torch.bernoulli(torch.full(self.tauS.size(), p))

        # Generate random values within the specified range
        random_values = torch.Tensor(self.tauS.size()).uniform_(*value_range)

        # Replace the selected elements in tau with the random values
        tau = self.tauS * (1 - mask.to(self.device)) + random_values.to(self.device) * mask.to(self.device)
        
        return tau

    def forwardTrain(self,settings,u):

        batch_size = u.shape[0] 
        N = self.N
        Device = self.device
        blockSteps = u.shape[1]
     
        PPCn = self.PPC_N
        PULn = self.PPC_N + self.PULexh_N
        TRNn = self.PPC_N + self.PULexh_N + self.TRNinh_N



        xu = torch.randn((batch_size,N, 1), dtype=self.dType, requires_grad=False,device=Device)/500; 
        ru = torch.zeros((batch_size,N, 1), dtype=self.dType, requires_grad=False,device=Device); 

        ru_new = torch.zeros_like(ru)
        ru_new[0:PPCn]    = self.rateFunction(xu[0:PPCn]    , BrainArea = "PPC", RateParams = self.rate_PPC_Params) # PPC Rate Function
        ru_new[PPCn:PULn] = self.rateFunction(xu[PPCn:PULn] , BrainArea = "PUL", RateParams = self.rate_PUL_Params) # PUL Rate Function
        ru_new[PULn:TRNn] = self.rateFunction(xu[PULn:TRNn] , BrainArea = "TRN", RateParams = self.rate_TRN_Params) # TRN Rate Function
        ru = ru_new

        ou = self.Wout@ru + self.b_out

        # Set Up State Vectors and Output for analysis after Compuatation
        x = torch.zeros(batch_size, N, blockSteps,device="cpu",requires_grad=False)
        r = torch.zeros(batch_size, N, blockSteps,device="cpu",requires_grad=False)
        o = torch.zeros(batch_size, 1, blockSteps,device="cuda",requires_grad=False)
        
        x[:, :, 0] = xu.squeeze(-1).detach() # Storing Variable
        r[:, :, 0] = ru.squeeze(-1).detach() # Storing Variable
        o[:, :, 0] = ou.squeeze(-1)
        dt = 1
        for t in range(1,blockSteps):
            # Apply Neuronal Properties to to Recurrent Matrix        
            if self.apply_dale == True:
                WrDale = torch.relu(self.Wr)
            else:
                WrDale = self.Wr

            WrReal = torch.mul( WrDale, self.SynapseMask); 

            # Euler–Maruyama discretization
            mu_InternalNoise = settings["mu_InternalNoise"]
            sigma_InternalNoise = settings["sigma_InternalNoise"]
            InternalNoise = (torch.randn((batch_size,self.noise_channels,1),device=self.device)* sigma_InternalNoise + mu_InternalNoise) * torch.sqrt(torch.tensor(sigma_InternalNoise**2 / dt))
            mu_ExternalNoise = settings["mu_ExternalNoise"]
            sigma_ExternalNoise = settings["sigma_ExternalNoise"]
            ExternalNoise = (torch.randn((batch_size,self.N,1),device=self.device)* sigma_ExternalNoise +  mu_ExternalNoise)* torch.sqrt(torch.tensor(sigma_ExternalNoise**2 / dt))

            #A = torch.mul(1 - dt / self.tauS, xu)
            #B = torch.mul(dt / self.tauS,  torch.matmul(WrReal, ru))
            #C = torch.einsum("nc,bck->bnk", self.Wnoise, InternalNoise)
            #D = torch.mul(u[:, t - 1], self.Win).t().unsqueeze(-1)

            #print(A.shape)
            #print(B.shape)
            #print(C.shape)
            #print(D.shape)
            #print(ExternalNoise.shape)

            #    Previous Step                       Recurrent Step                                          Internal Noise                                                Input
            xu = torch.mul(1 - dt / self.tauS, xu) + torch.mul(dt / self.tauS,  torch.matmul(WrReal, ru))  + torch.einsum("nc,bck->bnk", self.Wnoise, InternalNoise)  + torch.mul(u[:, t - 1], self.Win).t().unsqueeze(-1) + ExternalNoise; 

            ru_new = torch.zeros_like(ru)
            ru_new[0:PPCn]    = self.rateFunction(xu[0:PPCn]    , BrainArea = "PPC", RateParams = self.rate_PPC_Params) # PPC Rate Function
            ru_new[PPCn:PULn] = self.rateFunction(xu[PPCn:PULn] , BrainArea = "PUL", RateParams = self.rate_PUL_Params) # PUL Rate Function
            ru_new[PULn:TRNn] = self.rateFunction(xu[PULn:TRNn] , BrainArea = "TRN", RateParams = self.rate_TRN_Params) # TRN Rate Function
            ru = ru_new


            ou = self.Wout@ru + self.b_out; # Readout Layer

            x[:, :, t] = xu.squeeze(-1).detach()
            r[:, :, t] = ru.squeeze(-1).detach()
            o[:, :, t] = ou.squeeze(-1)

        return o.squeeze(), x, r

    def train_block(self, u, z, settings,nB):

        num_epochs = settings["num_epochs"]
        lr = settings["lr"]
        betas = settings["betas"]
        weight_decay = settings["weight_decay"]


        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay,fused=True)

        if settings["LR_Scheduler"] == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs//settings["LrDips"],eta_min=lr*settings["eta_min"])
        elif settings["LR_Scheduler"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=settings["factor"], patience=settings["patience"], verbose=False,threshold=0.0001,min_lr=lr*settings["eta_min"])
        

        dataset = CustomDataset(u,z)
        batch_size = settings["batch_size"]  # Choose an appropriate batch size

        # Create a DataLoader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        optimizer.zero_grad()
        accum_steps = settings["accum_steps"]
        accumulation_steps = min(accum_steps,len(data_loader))  # Adjust as per requirement


        blockSteps = u.shape[1]


        # Initialize TensorBoard writer
        writer = SummaryWriter()    

        BreakFlag = 0
        CatchPerformHistory = list()
        for epoch in tqdm(range(num_epochs),desc="Epoch",leave=True,disable=True):
            print_loading_bar(epoch, num_epochs)
            if settings["LR_Scheduler"] == "CosineAnnealingLR":
                current_lr = scheduler.get_last_lr()[0]
            elif settings["LR_Scheduler"] == "ReduceLROnPlateau":
                current_lr = optimizer.param_groups[0]['lr']
            else:
                current_lr = optimizer.param_groups[0]['lr']

            lossEpoch =[]
            for batch_idx, sample in enumerate(data_loader):


                us,zs = sample[0].to(self.device), sample[1].to(self.device)
                # Get batch size
                batch_size = u.size(0)
    
                o, x, r = self.forwardTrain(settings,us)
                 
                loss = self.compute_loss_with_penalty( o, zs, settings)    

                # Normalize the gradients
                loss = loss/accumulation_steps

                # Backward pass
                loss.backward()

                # Detect exploding gradients  
                if settings["PrintGradientNorms"]:           
                    self.check_gradients_for_issues(exploding_threshold=settings["max_gradient_norm"])

                if settings["GradientClipping"]:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=settings["max_gradient_norm"], norm_type=2)
                    
                # Update parameters
                if (batch_idx + 1) % accumulation_steps != 0 and (batch_idx + 1 < len(data_loader)):
                    continue

                
                lossEpoch.append(loss.item())    

                if (batch_idx + 1 == len(data_loader)):
                    Loss = sum(lossEpoch)/len(data_loader)
                    # Log the loss to TensorBoard
                    writer.add_scalar('Loss/train', Loss, epoch)
                    writer.add_scalar('Learning Rate', current_lr, epoch)
                    # Log gradients for each parameter
                    for name, param in self.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            # Log gradient norms
                            writer.add_scalar(f'Gradients/{name}_norm', param.grad.norm(), epoch)
                            writer.add_histogram(f'Param_Dist/{name}_norm', param, epoch)
                                

            
                            



                
                optimizer.step()
                optimizer.zero_grad()
                

                if settings["Clamping"]:
                    # Optionally, clamp tauS
                    with torch.no_grad():

                        if (self.tauS > self.tausRange[1]).any():
                            self.tauS.clamp_(max = self.tausRange[1])

                        if (self.tauS < self.tausRange[0]).any():
                            self.tauS.clamp_(min = self.tausRange[0])

                        if (self.Wr < 0 ).any():
                            self.Wr.clamp_( min=0, max=None)
                    

                # --------- End Of Batch----------------------------


                        
            if settings["num_epochs"] > 3:
                if settings["LR_Scheduler"] == "CosineAnnealingLR":
                    scheduler.step()  
                if settings["LR_Scheduler"] == "ReduceLROnPlateau":
                    scheduler.step(loss)


            self.Tepoch = self.Tepoch + 1

            # validate

            if (settings["Validate"] == True and epoch % settings["ValFreq"] == 0):
                finalEval = False
                with torch.no_grad():
                    EvalOutput = self.EvalModel(settings,finalEval,[],False)

                    writer.add_scalar(f'Validation/Discrim', EvalOutput["eval_perf_discrim_mean"], epoch)
                    writer.add_scalar(f'Validation/Catch', EvalOutput["eval_perf_catch_mean"], epoch)
                    CatchPerformHistory.append(EvalOutput["eval_perf_catch_mean"])

                    CatchAverage = 1
                    if len(CatchPerformHistory) >= 3:
                        CatchAverage = sum(CatchPerformHistory)/len(CatchPerformHistory)
                        CatchPerformHistory.pop(0)

                    
                    #print("Catch Average: ", CatchAverage)

                    if  (CatchAverage < 0.2 and EvalOutput["eval_perf_catch_mean"] < 0.2) and ( (settings["Tdelay"] > 0 and settings["Delay_Type"] == "Fixed" ) or (settings["Delay_Type"] == "Random") ):
                        print("\n")
                        print("Unrecoverable Catch Performance! Break")
                        BreakFlag = 2
                        break
                        
                    if EvalOutput["eval_perf_discrim_mean"] == 1 and EvalOutput["eval_perf_catch_mean"] >= settings["catch_perf"]:
                        print("\n")
                        print("Performance Crieteia Reached! Break")
                        BreakFlag = 1
                        break

            #--------------Continual Learning Dohare----------------------
            
            self.reinitNeurons()       

            #--------------End Of Epoch-----------------------------------
        
        Validation = {"TotalPerform_mean":EvalOutput["eval_perf_mean"],
                      "Discrim_mean":EvalOutput["eval_perf_discrim_mean"],                   
                      "DiscrimRight_mean":EvalOutput["eval_perf_discrim_right_mean"],
                      "DiscrimLeft_mean":EvalOutput["eval_perf_discrim_left_mean"],
                      "Catch_mean":EvalOutput["eval_perf_catch_mean"],
                      "CatchRight_mean":EvalOutput["eval_perf_catch_right_mean"],
                      "CatchLeft_mean":EvalOutput["eval_perf_catch_left_mean"],
                      "CatchAmbig_mean":EvalOutput["eval_perf_catch_ambig_mean"]}  
                
         
        # Extract the state dictionary (parameters)
        self.save_rnn_to_db(Validation)

        #--------------End Of Session-----------------------------------
        # Close the writer when done
        writer.close()
        return o, Loss, BreakFlag
    
    def reinitNeurons(self,rateOutput,settings):

        continual_on = settings["Continual"]
        eligible_age =  settings["Eligible Age"]
        replacement_rate = settings["Replacement_Rate"]
        decay_rate = settings["DecayRate"]



        with torch.no_grad():

            
            Wr_contrib = (1-decay_rate) * torch.abs(rateOutput) * torch.abs( torch.sum(WrReal,dim = 1) )
            Tau_contrib = = (1-decay_rate) * torch.abs(rateOutput) * torch.abs( self.tauS)
            utility = decay_rate * utility + 

    
    
    def stimInputTrial_Train(self,settings, nB,stimHistCalc):
        
        # Pips - Total Number of High and Low Pips
        # T - Number of Timesteps per Trial
        # stim_on -  Timestep when stim turns on
        # stim_dur - Timesteps of stim on
        # nB - Block Number
        # histConst - History Constraint
        # stimHist - stimHistory in block
        Pips = settings["Pips"]
        T = settings["T"]
        stim_on = settings["stim_on"]
        stim_dur = settings["stim_dur"]
        hist = settings["BlockLength"] - 1
        blocks = settings["Blocks"]
        Sample = settings["SampleTrain"]
        
        
        if Sample == "Uniform":
            ProbArray = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        if Sample == "Pure":
            ProbArray = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.00]
        if Sample == "Standard":
            #               [55%, 60%, 65%, 70%, 75%, 80%, 85%, 90%, 95%, 100%]
            if nB < blocks*0.2:
                ProbArray = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.00]
            elif nB < blocks*0.4:
                ProbArray = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.10, 0.80]
            elif nB < blocks*0.6:
                ProbArray = [0.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.10, 0.10, 0.10, 0.60]
            elif nB < blocks*0.8:
                ProbArray = [0.0, 0.0, 0.0, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.40]
            else:
                ProbArray = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]

        SideChoice = np.random.rand(); #print("SideChoice: ", SideChoice)
        if SideChoice > 0.5:
            # Right Stim
            n = np.random.choice([11,12,13,14,15,16,17,18,19,20],1,replace=True,p=ProbArray).item()
            label = 1
        else:
            # Left Stim
            n = np.random.choice([9,8,7,6,5,4,3,2,1,0],1,replace=True,p=ProbArray).item()
            label = -1

        if len(stimHistCalc) == hist:
            n = 10
            label = 0
        
        stimMix = n/Pips;#print(stimMix) 
        if label != 0:
            stimHistCalc.append(stimMix)

        #print("stimHist: ",stimHist)
        RightStim = np.ones((1,n))
        LeftStim  = -1*np.ones((1,Pips-n))

        stim = np.concatenate((RightStim, LeftStim), axis=1)
        shuffled_stimIn = np.random.permutation(stim.flatten())
        Map = stim_dur/Pips
        repeat_stimIn = np.repeat(shuffled_stimIn ,Map)
        stimIn = repeat_stimIn.reshape((1,int(Pips*Map)))
        
        u = np.zeros((1, T))
        u[0, stim_on:stim_on+stim_dur] = stimIn

        return u, label, stimHistCalc
       
    def targetOutTrial_Train(self,settings,label,stimHistCalc):

        T = settings['T']
        stim_on = settings['stim_on']
        stim_dur = settings['stim_dur']
        stim_delay = settings['stim_delay']
        catch_decision_Train_on = settings['catch_decision_Trian_on']
        
        stim_resp = stim_on+stim_dur+stim_delay

        z = np.zeros((1, T))
        if label == 1: # Right Trial
            #print("Go Right")
            z[0, stim_resp:] = 1
        elif label == -1: # Left Trial
            #print("Go Left")
            z[0, stim_resp:] = -1
        else: # Catch Trial
            if catch_decision_Train_on == True:
                stimMean = np.mean(stimHistCalc)
                if stimMean >= 0.5:
                    #print("Catch Go Right")
                    z[0, stim_resp:] = 1
                else:
                    #print("Catch Go Left")
                    z[0, stim_resp:] = -1

        if (settings["TrainCatchTrialOnly"] == True) and label != 0:
            z = np.zeros((1, T))

        return z


    
    def dataBlock_Train(self,settings, nB):
        uTotal = np.array([[]])
        zTotal = np.array([[]])

        BlockLength= settings["BlockLength"]
        Hist = BlockLength - 1

        stimHistCalc = []
        blockLabel = []
        Start_Eval = []
        End_Eval = []
        StimDurStart = []
        StimDurStop = []
        DelayStart = []
        DelayStop = []


        if settings["Delay_Type"] == "Random":
            DelayOptions = np.array(settings["Delays"])
            DelayBlocks = BlockLength-1
            Hist = BlockLength - 1
            DelayProbs = settings["DelayProbs"]
            DelayBins = np.random.choice(DelayOptions,DelayBlocks,replace=True,p=DelayProbs)
        else:
            delayBins = settings["Tdelay"]

        SequenceLength = 0
        for nT in range(BlockLength):
            
            # ------- Stim Input -----
            u_np, label, stimHistCalc = self.stimInputTrial_Train( settings, nB,stimHistCalc)
            u_np = u_np + np.random.normal(loc=0,scale=0.001, size = u_np.shape )
            uTotal = np.concatenate((uTotal,u_np),axis=1)
            SequenceLength = SequenceLength + settings["stim_on"]
            StimDurStart.append(SequenceLength)
            SequenceLength = SequenceLength + settings["stim_dur"]
            StimDurStop.append(SequenceLength)

            SequenceLength = SequenceLength + settings["stim_delay"]
            Start_Eval.append(SequenceLength)
            SequenceLength = SequenceLength + ( settings["T"] - (settings["stim_on"]+settings["stim_dur"]+settings["stim_delay"]) )
            End_Eval.append(SequenceLength)



            if nT < Hist:

                if settings["Delay_Type"] == "Random":
                    delayBins = DelayBins[nT]

                delayPeriod = np.zeros((1,delayBins)) + np.random.normal(loc=0,scale=0.001, size = (1,delayBins) )   
                uTotal = np.concatenate((uTotal,delayPeriod),axis=1)

                DelayStart.append(SequenceLength)
                SequenceLength = SequenceLength + delayBins
                DelayStop.append(SequenceLength)

            blockLabel.append(label)

            # ------- Response -------
            z_np = self.targetOutTrial_Train(settings,label,stimHistCalc)
            z_np = z_np + np.random.normal(loc=0,scale=0.001, size = z_np.shape )
            zTotal = np.concatenate((zTotal,z_np),axis=1)
            if nT < Hist:
                delayPeriod = np.zeros((1,delayBins)) + np.random.normal(loc=0,scale=0.001, size = (1,delayBins) )
                zTotal = np.concatenate((zTotal,delayPeriod),axis=1)

        u = torch.tensor(uTotal,dtype=self.dType,requires_grad=False,device="cpu")
        z = torch.tensor(zTotal,dtype=self.dType,requires_grad=False,device="cpu")

        return u,z,blockLabel,stimHistCalc,Start_Eval,End_Eval,StimDurStart,StimDurStop,DelayStart,DelayStop
    
    def dataSession_Train(self,settings):

        u = list()
        z = list()
        blockLabel = list()
        stimHistCalc = list()
        Start_Eval = list()
        End_Eval = list()
        StimDurStart = list()
        StimDurStop = list()
        DelayStart = list()
        DelayStop = list()

        for nB in range(settings["Blocks"]):

            
            ub,zb,blockLabelnB,stimHistCalcnB,Start_EvalB,End_EvalB,StimDurStartB,StimDurStopB,DelayStartB,DelayStopB = self.dataBlock_Train(settings, nB)

            u.append(ub)
            z.append(zb)
            blockLabel.append(blockLabelnB)
            stimHistCalc.append(stimHistCalcnB)

            Start_Eval.append(Start_EvalB)
            End_Eval.append(End_EvalB)
            StimDurStart.append(StimDurStartB)
            StimDurStop.append(StimDurStopB)
            DelayStart.append(DelayStartB)
            DelayStop.append(DelayStopB)
          


        if settings["Delay_Type"] == "Fixed":
            u = torch.stack(u).squeeze(1)
            z = torch.stack(z).squeeze(1)

        if settings["Delay_Type"] == "Random":
            ArrayLengths = [array.size(dim=1) for array in u]
            target_length = max(ArrayLengths)
            padding =[ target_length - ArrayLength for ArrayLength in ArrayLengths]
            u_padded = [torch.nn.functional.pad(ui,(0,pad)) for ui,pad in zip(u,padding)]
            u = torch.stack(u_padded).squeeze(1)
            z_padded = [torch.nn.functional.pad(zi,(0,pad)) for zi,pad in zip(z,padding)]
            z = torch.stack(z_padded).squeeze(1)



        return u,z,blockLabel,stimHistCalc,Start_Eval,End_Eval,StimDurStart,StimDurStop,DelayStart,DelayStop
    
    def ComputeWeights(self,settings,z):

        stim_on = settings["stim_on"]
        stim_dur = settings["stim_dur"]
        stim_delay = settings["stim_delay"]
        T = settings["T"]
        Weight = settings["weight"]
        catch_decision_Train_on = settings['catch_decision_Trian_on']
        stim_resp = stim_on+stim_dur+stim_delay
        resp_time = T - stim_resp

        zd = torch.diff(torch.abs(z))
        # Define a threshold for what you consider a 'positive step change'
        threshold = 0.5

        # Find indices where the difference exceeds the threshold
        zdt = zd  > threshold
        zdt = zdt.int()

        zI = torch.where(zdt==1)
        Idx = np.linspace(0,len(zI[0])-1,len(zI[0]),dtype=np.int8)
        Idx5 = Idx[(settings["BlockLength"]-1)::settings["BlockLength"]]
        catchStart = zI[1][Idx5]
        catchEnd = zI[1][Idx5] + resp_time
        
        if catch_decision_Train_on == True:
    
            Weights = np.zeros((z.shape[0],z.shape[1]))

            for i in range(Weights.shape[0]):
                Weights[i,0:catchStart[i]] = 1
                Weights[i,catchStart[i]:catchEnd[i]] = Weight
                Weights[i,catchEnd[i]:] = 1


        else:

            Weights = np.zeros((z.shape[0],z.shape[1]))

            for i in range(Weights.shape[0]):
                Weights[i,:] = 1
        

        WeightsTorch = torch.from_numpy(Weights)

        return WeightsTorch
    
    #------------------------------------------------------------------------
    #----------------------- Eval Methods -----------------------------------
    #------------------------------------------------------------------------

    def forwardEval(self, settings, u, DelayStart_b,DelayStop_b,StimDurStart_b,StimDurStop_b):

        

        with torch.no_grad():
            batch_size = u.shape[0] 
            N = self.N
            Device = self.device
            blockSteps = u.shape[1]

            PPCn = self.PPC_N
            PULn = self.PPC_N + self.PULexh_N
            TRNn = self.PPC_N + self.PULexh_N + self.TRNinh_N

            DelayStart_m = np.array(DelayStart_b)
            DelayStop_m = np.array(DelayStop_b)

            StimStart_m = np.array(StimDurStart_b)
            StimStop_m = np.array(StimDurStop_b)

            xu = torch.randn((batch_size,N, 1), dtype=self.dType, requires_grad=False,device=Device)/500; #print("xu.Shape",xu.shape)
            ru = torch.zeros((batch_size,N, 1), dtype=self.dType, requires_grad=False,device=Device); 


            ru[0:PPCn]    = self.rateFunction(xu[0:PPCn]    , BrainArea = "PPC", RateParams = self.rate_PPC_Params) # PPC Rate Function
            ru[PPCn:PULn] = self.rateFunction(xu[PPCn:PULn] , BrainArea = "PUL", RateParams = self.rate_PUL_Params) # PUL Rate Function
            ru[PULn:TRNn] = self.rateFunction(xu[PULn:TRNn] , BrainArea = "TRN", RateParams = self.rate_TRN_Params) # TRN Rate Function

            ou = self.Wout@ru + self.b_out

            # Set Up State Vectors and Output for analysis after Compuatation
            x = torch.zeros(batch_size, N, blockSteps,device="cpu",requires_grad=False)
            r = torch.zeros(batch_size, N, blockSteps,device="cpu",requires_grad=False)
            o = torch.zeros(batch_size, 1, blockSteps,device="cuda",requires_grad=False)
            
            x[:, :, 0] = xu.squeeze(-1).detach() # Storing Variable
            r[:, :, 0] = ru.squeeze(-1).detach() # Storing Variable
            o[:, :, 0] = ou.squeeze(-1)
            dt = 1

            i = 0
            SuppressArray = []
            if settings["Manipulation"] in ["SuppressNeurons","AllPruneSynapse","PostPruneSynapse","PrePruneSynapse"]:
                AnySuppression = True
            else:
                AnySuppression = False 

            for t in range(1,blockSteps):

                # Apply Neuronal Properties to to Recurrent Matrix        
                if self.apply_dale == True:
                    WrDale = torch.relu(self.Wr)
                else:
                    WrDale = self.Wr

                WrReal = torch.mul( WrDale, self.SynapseMask); #print("WrReal.Shape",WrReal.shape,"WrReal.Device",WrReal.device,"WrReal.type",WrReal.type())
                WinReal = self.Win

                assert not (settings["DelaySuppressOn"] and settings["StimSuppressOn"]), "Stim and Delay Suppress cannot both be True"

                # Handling Suppression Index
                if   (settings["DelaySuppressOn"] == True)  and (settings["StimSuppressOn"] == False) and (settings["SuppressCatchOnly"] == False) and (t == DelayStop_m[0,i]) and (AnySuppression):
                    i += 1
                    i = min(len(DelayStart_m[0,:])-1,i)
                elif (settings["DelaySuppressOn"] == True)  and (settings["StimSuppressOn"] == False) and (settings["SuppressCatchOnly"] == True) and (AnySuppression):
                    i = len(DelayStart_m[0,:])-1
                elif (settings["DelaySuppressOn"] == False) and (settings["StimSuppressOn"] == True)  and (settings["SuppressCatchOnly"] == False) and (t == StimStop_m[0,i]) and (AnySuppression):
                    i += 1
                    i = min(len(StimStart_m[0,:])-1,i)
                elif (settings["DelaySuppressOn"] == False) and (settings["StimSuppressOn"] == True)  and (settings["SuppressCatchOnly"] == True) and (AnySuppression):
                    i = len(StimStart_m[0,:])-1
                else:
                    i = i

                if (settings["DelaySuppressOn"] == True) and (settings["StimSuppressOn"] == False) and (t >= DelayStart_m[0,i]) and ( t <= ( settings["SuppressLength"] + DelayStart_m[0,i] ) ) and (AnySuppression):
                    assert settings["Delay_Type"] == "Fixed", "Delay Suppress only implemented for Fixed delays"

                    WrReal = torch.mul(WrReal,self.OptoMask)
                    WinReal = torch.mul(self.Win,self.OptoMaskIn)
                    SuppressArray.append(1)

                elif (settings["DelaySuppressOn"] == False) and (settings["StimSuppressOn"] == True) and (t >= StimStart_m[0,i]) and ( t <= ( StimStop_m[0,i] ) ) and (AnySuppression):
                    assert settings["Delay_Type"] == "Fixed", "Stim Suppress only implemented for Fixed delays"

                    WrReal = torch.mul(WrReal,self.OptoMask)
                    WinReal = torch.mul(self.Win,self.OptoMaskIn)
                    SuppressArray.append(1)

                elif (settings["DelaySuppressOn"] == False) and (settings["StimSuppressOn"] == False)  and (AnySuppression):

                    WrReal = torch.mul(WrReal,self.OptoMask)
                    WinReal = torch.mul(self.Win,self.OptoMaskIn)
                    SuppressArray.append(1)
                    

                else:
                    SuppressArray.append(0)
                    
                # Euler–Maruyama discretization
                mu_InternalNoise = settings["mu_InternalNoise"]
                sigma_InternalNoise = settings["sigma_InternalNoise"]
                InternalNoise = (torch.randn((batch_size,self.N,1),device=self.device)* sigma_InternalNoise + mu_InternalNoise) * torch.sqrt(torch.tensor(sigma_InternalNoise**2 / dt))
                mu_ExternalNoise = settings["mu_ExternalNoise"]
                sigma_ExternalNoise = settings["sigma_ExternalNoise"]
                ExternalNoise = (torch.randn((batch_size,self.N,1),device=self.device)* sigma_ExternalNoise +  mu_ExternalNoise)* torch.sqrt(torch.tensor(sigma_ExternalNoise**2 / dt))

                #A = torch.mul(1 - dt / self.tauS, xu)
                #B = torch.mul(dt / self.tauS,  torch.matmul(WrReal, ru))

                #D = torch.mul(u[:, t - 1], self.Win).t().unsqueeze(-1)

                #print(A.shape)
                #print(B.shape)
                #print(D.shape)
                #print(InternalNoise.shape)
                #print(ExternalNoise.shape)

                #    Previous Step                       Recurrent Step                                          Internal Noise    Input                                               External Noise
                xu = torch.mul(1 - dt / self.tauS, xu) + torch.mul(dt / self.tauS,  torch.matmul(WrReal, ru))  + InternalNoise  +  torch.mul(u[:, t - 1], WinReal).t().unsqueeze(-1) + ExternalNoise; 

                ru[0:PPCn]    = self.rateFunction(xu[0:PPCn]    , BrainArea = "PPC", RateParams = self.rate_PPC_Params) # PPC Rate Function
                ru[PPCn:PULn] = self.rateFunction(xu[PPCn:PULn] , BrainArea = "PUL", RateParams = self.rate_PUL_Params) # PUL Rate Function
                ru[PULn:TRNn] = self.rateFunction(xu[PULn:TRNn] , BrainArea = "TRN", RateParams = self.rate_TRN_Params) # TRN Rate Function

                ou = self.Wout@ru + self.b_out; # Readout Layer

                x[:, :, t] = xu.squeeze(-1).detach()
                r[:, :, t] = ru.squeeze(-1).detach()
                o[:, :, t] = ou.squeeze(-1)


        return o.squeeze(1), x.squeeze(), r.squeeze(), SuppressArray

    
    def stimInputTrial_Eval(self,settings, nB,stimHistCalc,blockMultiplier):

        # Pips - Total Number of High and Low Pips
        # T - Number of Timesteps per Trial
        # stim_on -  Timestep when stim turns on
        # stim_dur - Timesteps of stim on
        # nB - Block Number
        # histConst - History Constraint
        # stimHist - stimHistory in block
        Pips = settings["Pips"]
        T = settings["T"]
        stim_on = settings["stim_on"]
        stim_dur = settings["stim_dur"]
        hist = settings["BlockLength"] - 1
        block = settings["GPU_Eval_Blocks"]
        Sample = settings["SampleTest"]

        RandCatchOn = settings["RandCatchTest"]
        
        if Sample == "Uniform":
            ProbArray = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        if Sample == "Pure":
            ProbArray = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.00]
        if Sample == "Standard":
            #               [55%, 60%, 65%, 70%, 75%, 80%, 85%, 90%, 95%, 100%]
            if nB < (block*0.2)*blockMultiplier:
                ProbArray = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.00]
            elif nB < (block*0.4)*blockMultiplier:
                ProbArray = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.90]
            elif nB < (block*0.6)*blockMultiplier:
                ProbArray = [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.80]
            elif nB < (block*0.8)*blockMultiplier:
                ProbArray = [0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.70]
            else:
                ProbArray = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
        if Sample == "Select":
            ProbArrayInput = settings["ProbArrayInput"]
            assert len(ProbArrayInput) == 10,"Select Prob Array Length is not lenght of 10"
            ProbArray = ProbArrayInput

    
        SideChoice = np.random.rand(); #print("SideChoice: ", SideChoice)
        if SideChoice > 0.5:
            # Right Stim
            n = np.random.choice([11,12,13,14,15,16,17,18,19,20],1,replace=True,p=ProbArray).item()
            label = 1
        else:
            # Left Stim
            n = np.random.choice([9,8,7,6,5,4,3,2,1,0],1,replace=True,p=ProbArray).item()
            label = -1

        CatchTrial = False
        if RandCatchOn == True:
            CatchTrial = np.random.choice([False,True],1,replace=True,p=[0.2,0.8]).item()

        if (len(stimHistCalc) == hist) or (CatchTrial == True) and (0.5 not in stimHistCalc):
 
            n = 10
            label = 0
    
            

        stimMix = n/Pips
        if label != 0:
            stimHistCalc.append(stimMix)

        RightStim = np.ones((1,n))
        LeftStim  = -1*np.ones((1,Pips-n))

        stim = np.concatenate((RightStim, LeftStim), axis=1)
        shuffled_stimIn = np.random.permutation(stim.flatten())
        Map = stim_dur/Pips
        repeat_stimIn = np.repeat(shuffled_stimIn ,Map)
        stimIn = repeat_stimIn.reshape((1,int(Pips*Map)))
        
        u = np.zeros((1, T))
        u[0, stim_on:stim_on+stim_dur] = stimIn

        if ( (label != 0)  ) and (settings["EvalCatchTrialOnly"] == True):
            u = np.zeros((1, T))

        return u, label, stimHistCalc

    def CreateFixedDelayBins(self,settings):

        
        #delayBins = settings['Tdelay']
        BlockLength = settings["BlockLength"]

        if settings["Delay_Type"] == "Random":
            DelayOptions = np.array(settings["Delays"])
            BlockLength = 6
            DelayBlocks = BlockLength-1
            
            DelayProbs = settings["DelayProbs"]
            DelayBins = np.concatenate(([0],np.random.choice(DelayOptions,DelayBlocks,replace=True,p=DelayProbs)),axis=0)
        else:
            DelayBlocks = BlockLength-1
            DelayBins = np.concatenate(([0],np.repeat(settings["Tdelay"],DelayBlocks)),axis=0)

        ResponseDuration = settings["T"] - settings["stim_on"] - settings["stim_dur"] - settings["stim_delay"]
        ResponseTime = settings["T"] - ResponseDuration


        TrialTime = np.repeat(settings["T"],BlockLength)
        RespTArray = np.repeat(ResponseTime,BlockLength)
        RespDArray = np.concatenate(([0],np.repeat(ResponseDuration,DelayBlocks)),axis=0)

        Start_Eval = list(accumulate(DelayBins[i] + RespTArray[i] + RespDArray[i] for i in range(BlockLength))) 
        Start_Eval =  [x - 1 for x in Start_Eval]
        End_Eval  = list(accumulate(TrialTime[i] + DelayBins[i] for i in range(BlockLength)))
        End_Eval = [x - 1 for x in End_Eval]

        StimDurStart = [x - (settings["stim_dur"] + settings["stim_delay"])  for x in Start_Eval]
        StimDurStop = [x - (settings["stim_delay"])  for x in Start_Eval]

        DelayStart = list(accumulate([settings["T"] + DelayBins[i] for i in range(0,len(DelayBins)-1)]))
        DelayStop = list(accumulate([settings["T"] + DelayBins[i] for i in range(1,len(DelayBins))]))

        DelayBins = np.delete(DelayBins, 0)

        delayDict = {"Start_Eval":Start_Eval, "End_Eval":End_Eval,\
                     "StimDurStart":StimDurStart, "StimDurStop":StimDurStop,\
                     "DelayStart":DelayStart, "DelayStop":DelayStop,\
                     "DelayBins":DelayBins}

        return delayDict
    

    def dataBlock_Eval(self,settings,nB,finalEval,delayDict):

        uTotal = np.array([[]])
        zTotal = np.array([[]])
        stimHistCalc = []
        
        BlockLength = settings["BlockLength"]
        Hist = BlockLength - 1
        labelTotal = []

        blockMultiplier = 1
        if finalEval == True:
            blockMultiplier = settings["BlockMultiplier"]
        
        if settings["Delay_Type"] == "Random":

            delayDict = self.CreateFixedDelayBins(settings)
            Start_Eval = delayDict["Start_Eval"]
            End_Eval = delayDict["End_Eval"]
            StimDurStart = delayDict["StimDurStart"]
            StimDurStop = delayDict["StimDurStop"]
            DelayStart = delayDict["DelayStart"]
            DelayStop = delayDict["DelayStop"]
            DelayBins = delayDict["DelayBins"]
        else:
            Start_Eval = delayDict["Start_Eval"]
            End_Eval = delayDict["End_Eval"]
            StimDurStart = delayDict["StimDurStart"]
            StimDurStop = delayDict["StimDurStop"]
            DelayStart = delayDict["DelayStart"]
            DelayStop = delayDict["DelayStop"]
            DelayBins = delayDict["DelayBins"]
        
        for nT in range(BlockLength):
            
            # ------- Stim Input -------------
            u_np,label,stimHistCalc = self.stimInputTrial_Eval( settings, nB, stimHistCalc, blockMultiplier)
            uTotal = np.concatenate((uTotal,u_np),axis=1)
      
            if nT < Hist:    

                delayBins = DelayBins[nT]
                delayPeriod = np.zeros((1,delayBins))

                uTotal = np.concatenate((uTotal,delayPeriod),axis=1)

            labelTotal.append(label)

            # ------- Response -------
            z_np = self.targetOutTrial_Train(settings,label,stimHistCalc)
            zTotal = np.concatenate((zTotal,z_np),axis=1)

            if nT < Hist:

                delayBins = DelayBins[nT]
                delayPeriod = np.zeros((1,delayBins))
                zTotal = np.concatenate((zTotal,delayPeriod),axis=1)
            
            
        
        u = torch.tensor(uTotal,dtype=self.dType,requires_grad=False,device="cpu")
        z = torch.tensor(zTotal,dtype=self.dType,requires_grad=False,device="cpu")
        return u, z, labelTotal, stimHistCalc,Start_Eval,End_Eval,StimDurStart,StimDurStop,DelayStart,DelayStop
    


    def EvalModel(self,settings,finalEval, opto_idx = [],PrintResult = True,train_data = torch.empty(0)):
        
        TrainDataEval = train_data.numel() > 0
        if TrainDataEval == False and PrintResult == True:
            OptoMaskIn,OptoMask = self.RnnManipulations(settings,opto_idx)
            self.OptoMaskIn = OptoMaskIn
            self.OptoMask = OptoMask


        with torch.no_grad():

            eval_perf = []
            eval_perf_catch = []
            eval_perf_discrim = []
            TotalLeftDiscrim = []
            TotalRightDiscrim = []
            TotalLeftCatch = []
            TotalRightCatch = []
            TotalCatchAmbig = []

            Thres = settings["Eval_Threshold"]

            uEvals = []
            zEvals = []
            oEvals = []
            xEvals = []
            rEvals = []
            labelEvals = []
            stimHistCalcEvals = []
            MaxOutputEvals = []
            MinOutputEvals = []
            MaxStimEvals = []
            MinStimEvals = []
            MaxDelayEvals = []
            MinDelayEvals = []
            StartEval = []
            StopEval = []
            StimDurStartEval = []
            StimDurStopEval = []
            DelayStartEval = []
            DelayStopEval = []
            SuppressArrayEvals = []

            Blocks   = 1 
            Blocks_Gpu = settings["GPU_Eval_Blocks"] 
            if finalEval == True:
                Blocks = settings["BlockMultiplier"]

            if settings["Delay_Type"] == "Fixed":
                delayDict = self.CreateFixedDelayBins(settings)
            else:
                delayDict = self.CreateFixedDelayBins(settings)

            if TrainDataEval == True:
                Blocks = TrainDataEval["u"].shape[0]

            for nb in range(Blocks):

                if TrainDataEval == False:
                    u_b,z_b,labelTotal_b,stimHistCalc_b,Start_Eval_b,End_Eval_b,StimDurStart_b,StimDurStop_b,DelayStart_b,DelayStop_b = self.dataSession_Eval(settings,Blocks_Gpu,delayDict,finalEval=True)

                if TrainDataEval == True:
                    u_b = train_data["u"][nb:nb+Blocks_Gpu]
                    z_b = train_data["z"][nb:nb+Blocks_Gpu]
                    labelTotal_b = train_data["labelTotal"][nb:nb+Blocks_Gpu]
                    stimHistCalc_b = train_data["stimHistCalc"][nb:nb+Blocks_Gpu] 
                    
                    Start_Eval_b = train_data["Start_Eval"][nb:nb+Blocks_Gpu]
                    End_Eval_b = train_data["End_Eval"][nb:nb+Blocks_Gpu ]
                    StimDurStart_b = train_data["StimDurStart"][nb:nb+Blocks_Gpu]
                    StimDurStop_b = train_data["StimDurStop"][nb:nb+Blocks_Gpu]
                    DelayStart_b = train_data["DelayStart"][nb:nb+Blocks_Gpu]
                    DelayStop_b =train_data["DelayStop"][nb:nb+Blocks_Gpu]


                u_i = u_b.to(self.device)
                oi,xi,ri,SuppressArray = self.forwardEval(settings,u_i,DelayStart_b,DelayStop_b,StimDurStart_b,StimDurStop_b)

                # Move the tensor to the CPU using .to("cpu")
                o = oi.to("cpu")
                x = xi.to("cpu")
                r = ri.to("cpu")
                

                # Remove the tensor from the GPU
                del oi,xi,ri,u_i
                gc.collect()  # Ensure garbage collector releases the GPU memory
                torch.cuda.empty_cache()  # Free up the cached memory
                labelTotal = torch.tensor(labelTotal_b,device="cpu")
                blocks,trials = labelTotal.shape


                MaxStimTensor = torch.zeros((Blocks_Gpu,trials),device="cpu")
                MinStimTensor = torch.zeros((Blocks_Gpu,trials),device="cpu")

                
                MaxDelayTensor = torch.zeros((Blocks_Gpu,trials),device="cpu")
                MinDelayTensor = torch.zeros((Blocks_Gpu,trials),device="cpu")
              
                MaxOutputTensor = torch.zeros((Blocks_Gpu,trials),device="cpu")
                MinOutputTensor = torch.zeros((Blocks_Gpu,trials),device="cpu")

                stimHistCalc = torch.tensor(stimHistCalc_b,device="cpu")
                StimAvg = torch.mean(stimHistCalc,dim=1)

                Start_Eval_b = torch.tensor(Start_Eval_b,device="cpu")
                End_Eval_b = torch.tensor(End_Eval_b,device="cpu")

                StimDurStart_b = torch.tensor(StimDurStart_b,device="cpu")
                StimDurStop_b = torch.tensor(StimDurStop_b,device="cpu")

                DelayStart_b = torch.tensor(DelayStart_b,device="cpu")
                DelayStop_b = torch.tensor(DelayStop_b,device="cpu")


                for nb_p in range(Blocks_Gpu):
                    

                    for tr in range(trials):
                        #---- For Data Collection--------------
                        MaxStimTensor[nb_p,tr] = torch.max(o[nb_p,StimDurStart_b[nb_p,tr]:StimDurStop_b[nb_p,tr]])
                        MinStimTensor[nb_p,tr] = torch.min(o[nb_p,StimDurStart_b[nb_p,tr]:StimDurStop_b[nb_p,tr]])
                        try:
                            MaxDelayTensor[nb_p,tr] = torch.max(o[nb_p,DelayStart_b[nb_p,tr]:DelayStop_b[nb_p,tr]])
                            MinDelayTensor[nb_p,tr] = torch.min(o[nb_p,DelayStart_b[nb_p,tr]:DelayStop_b[nb_p,tr]])
                        except:
                            MaxDelayTensor[nb_p,tr] = 0
                            MinDelayTensor[nb_p,tr] = 0

                        #--------------------------------------

                        MaxOutputTensor[nb_p,tr] = torch.max(o[nb_p,Start_Eval_b[nb_p,tr]:End_Eval_b[nb_p,tr]])
                        MinOutputTensor[nb_p,tr] = torch.min(o[nb_p,Start_Eval_b[nb_p,tr]:End_Eval_b[nb_p,tr]])

                        if labelTotal[nb_p,tr] ==  1:

                            if MaxOutputTensor[nb_p,tr] >= Thres and MinOutputTensor[nb_p,tr] >= -Thres:
                                eval_perf.append(1)
                                eval_perf_discrim.append(1)
                                TotalRightDiscrim.append(1)
                            else:
                                eval_perf.append(0)
                                eval_perf_discrim.append(0)
                                TotalRightDiscrim.append(0)


                        if labelTotal[nb_p,tr] == -1:

                            if MaxOutputTensor[nb_p,tr] <= Thres and MinOutputTensor[nb_p,tr] <= -Thres:
                                eval_perf.append(1)
                                eval_perf_discrim.append(1)
                                TotalLeftDiscrim.append(1)
                            else:
                                eval_perf.append(0)
                                eval_perf_discrim.append(0)
                                TotalLeftDiscrim.append(0)


                        if labelTotal[nb_p,tr] ==  0:

                            if StimAvg[nb_p] > 0.5:

                                if MaxOutputTensor[nb_p,tr] >= Thres and MinOutputTensor[nb_p,tr] >= -Thres:
                                    eval_perf.append(1)
                                    eval_perf_catch.append(1)
                                    TotalRightCatch.append(1)
                                else:
                                    eval_perf.append(0)
                                    eval_perf_catch.append(0)
                                    TotalRightCatch.append(0)

                            if StimAvg[nb_p]  < 0.5:
                                if MaxOutputTensor[nb_p,tr] <= Thres and MinOutputTensor[nb_p,tr] <= -Thres:
                                    eval_perf.append(1)
                                    eval_perf_catch.append(1)
                                    TotalLeftCatch.append(1)
                                else:
                                    eval_perf.append(0)
                                    eval_perf_catch.append(0)
                                    TotalLeftCatch.append(0)

                            if StimAvg[nb_p]  == 0.5:
                                if MaxOutputTensor[nb_p,tr] < Thres and MinOutputTensor[nb_p,tr] > -Thres:
                                    eval_perf.append(1)
                                    eval_perf_catch.append(1)
                                    TotalCatchAmbig.append(1)
                                else:
                                    eval_perf.append(0)
                                    eval_perf_catch.append(0)
                                    TotalCatchAmbig.append(0)
                        


                uEvals.append(u_b)
                zEvals.append(z_b)
                oEvals.append(o)
                xEvals.append(x)
                rEvals.append(r)
                SuppressArrayEvals.append(SuppressArray)

                labelEvals.append(labelTotal)
                stimHistCalcEvals.append(stimHistCalc)

                MaxOutputEvals.append(MaxOutputTensor)
                MinOutputEvals.append(MinOutputTensor)

                MaxStimEvals.append(MaxStimTensor)
                MinStimEvals.append(MinStimTensor)
                
                MaxDelayEvals.append(MaxDelayTensor)
                MinDelayEvals.append(MinDelayTensor)

                


                StartEval.append(Start_Eval_b)
                StopEval.append(End_Eval_b)
                StimDurStartEval.append(StimDurStart_b)
                StimDurStopEval.append(StimDurStop_b)
                DelayStartEval.append(DelayStart_b)
                DelayStopEval.append(DelayStop_b)
                
            eval_perf_mean = np.nanmean(eval_perf, 0)
            eval_perf_catch_mean = np.nanmean(eval_perf_catch, 0)
            eval_perf_discrim_mean = np.nanmean(eval_perf_discrim,0)
            eval_perf_discrim_right_mean = np.nanmean(TotalRightDiscrim,0)
            eval_perf_discrim_left_mean = np.nanmean(TotalLeftDiscrim,0)
            eval_perf_catch_right_mean =np.nanmean(TotalRightCatch,0)
            eval_perf_catch_left_mean =np.nanmean(TotalLeftCatch,0)
            with warnings.catch_warnings(action="ignore"):
                eval_perf_catch_ambig_mean =np.nanmean(TotalCatchAmbig,0)
            Discrim_Total_Right = len(TotalRightDiscrim)
            Discrim_Total_Left = len(TotalLeftDiscrim)
            Catch_Total_Right = len(TotalRightCatch)
            Catch_Total_Left = len(TotalLeftCatch)
            Catch_Total_Ambig = len(TotalCatchAmbig)


            now = datetime.datetime.now()
            formatted_now = now.strftime("%Y.%m.%d - %H:%M") 
            if PrintResult == True: 
                print("\n")
                print(f"{formatted_now}   Total Performance: {eval_perf_mean:.3f}   Discrimination Performance: {eval_perf_discrim_mean:.3f}   Catch Performance: {eval_perf_catch_mean:.3f}")
                print(f"Discrimination Left Performance: {eval_perf_discrim_left_mean:.3f}   Discrimination Right Performance: {eval_perf_discrim_right_mean:.3f}")
                print(f"Catch Left Performance: {eval_perf_catch_left_mean:.3f}   Catch Right Performance: {eval_perf_catch_right_mean:.3f}")

            EvalOutput = {"eval_perf_mean":eval_perf_mean,
                        "eval_perf_catch_mean":eval_perf_catch_mean,
                        "eval_perf_discrim_mean":eval_perf_discrim_mean,
                        "eval_perf":eval_perf,
                        "eval_perf_catch":eval_perf_catch,
                        "eval_perf_discrim":eval_perf_discrim,
                        "uEvals":uEvals,
                        "zEvals":zEvals,
                        "oEvals":oEvals,
                        "xEvals":xEvals,
                        "rEvals":rEvals,
                        "labelEvals":labelEvals,
                        "stimHistCalcEvals":stimHistCalcEvals,
                        "MaxOutputEvals":MaxOutputEvals,
                        "MinOutputEvals":MinOutputEvals,
                        "MaxStimEvals":MaxStimEvals,
                        "MinStimEvals":MinStimEvals,
                        "MaxDelayEvals":MaxDelayEvals,
                        "MinDelayEvals":MinDelayEvals,
                        "StartEval":StartEval,
                        "StopEval":StopEval,
                        "StimDurStartEval":StimDurStartEval,
                        "StimDurStopEval":StimDurStopEval,
                        "DelayStartEval":DelayStartEval,
                        "DelayStopEval":DelayStopEval,
                        "Wr":self.Wr.to("cpu"),
                        "Win":self.Win.to("cpu"),
                        "Wout":self.Wout.to("cpu"),
                        "bout":self.b_out.to("cpu"),
                        "SynapseMask":self.SynapseMask.to("cpu"),
                        "tauS":self.tauS.to("cpu"),
                        "opto_idx":opto_idx,
                        "OptoMaskIn":self.OptoMaskIn,
                        "OptoMask":self.OptoMask,
                        "eval_perf_discrim_right_mean":eval_perf_discrim_right_mean,
                        "eval_perf_discrim_left_mean":eval_perf_discrim_left_mean,
                        "eval_perf_catch_right_mean":eval_perf_catch_right_mean,
                        "eval_perf_catch_left_mean":eval_perf_catch_left_mean,
                        "eval_perf_catch_ambig_mean":eval_perf_catch_ambig_mean,
                        "Discrim_Total_Right":Discrim_Total_Right,
                        "Discrim_Total_Left":Discrim_Total_Left,
                        "Catch_Total_Right":Catch_Total_Right,
                        "Catch_Total_Left":Catch_Total_Left,
                        "Catch_Total_Ambig":Catch_Total_Ambig,
                        "SuppressArrayEvals":SuppressArrayEvals
                        }    
            
            
        return EvalOutput


    def dataSession_Eval(self,settings,Blocks,delayDict,finalEval):

        u = list()
        z = list()
        blockLabel = list()
        stimHistCalc = list()
        Start_Eval = list()
        End_Eval = list()
        StimDurStart = list()
        StimDurStop = list()
        DelayStart = list()
        DelayStop = list()


        for nB in range(Blocks):
            ub, zb, labelTotalB, stimHistCalcB,Start_EvalB,End_EvalB,StimDurStartB,StimDurStopB,DelayStartB,DelayStopB = self.dataBlock_Eval(settings, nB,finalEval, delayDict)
            u.append(ub)
            z.append(zb)
            blockLabel.append(labelTotalB)
            stimHistCalc.append(stimHistCalcB)
            Start_Eval.append(Start_EvalB)
            End_Eval.append(End_EvalB)
            StimDurStart.append(StimDurStartB)
            StimDurStop.append(StimDurStopB)
            DelayStart.append(DelayStartB)
            DelayStop.append(DelayStopB)

        if settings["Delay_Type"] == "Fixed":
            u = torch.stack(u).squeeze(1)
            z = torch.stack(z).squeeze(1)

        
        if settings["Delay_Type"] == "Random":
            ArrayLengths = [array.size(dim=1) for array in u]
            target_length = max(ArrayLengths)
            padding =[ target_length - ArrayLength for ArrayLength in ArrayLengths]
            u_padded = [torch.nn.functional.pad(ui,(0,pad)) for ui,pad in zip(u,padding)]
            u = torch.stack(u_padded).squeeze(1)
            z_padded = [torch.nn.functional.pad(zi,(0,pad)) for zi,pad in zip(z,padding)]
            z = torch.stack(z_padded).squeeze(1)

        
            
        return u,z,blockLabel,stimHistCalc,Start_Eval,End_Eval,StimDurStart,StimDurStop,DelayStart,DelayStop
    
         
    def train_model(self,settings):

        if settings["PrintSettings"]:
            print("\n")
            print("==========================Settings======================================")
            print("\n")
            try:
                self.check_required_keys(settings)
            except KeyError as e:
                print(e)
            
            count = 0
            line = ""

            for key, value in settings.items():
                line += f"{key}: {value}  "
                count += 1
                if count % 5 == 0:  # After every 5th pair
                    print(line.strip())
                    line = ""  # Reset the line

            if line:  # Print any remaining pairs (if the total number is not a multiple of 5)
                print(line.strip())
                    
            print("\n")
            print("=========================================================================")
            print("\n")
        FitSessions = settings["FitSessions"]
        eval_freq = settings["Eval_Freq"]

        print("-------------Initial Fit Model-------------------------------")
        u,z,blockLabel,stimHistCalc,Start_Eval,End_Eval,StimDurStart,StimDurStop,DelayStart,DelayStop = self.dataSession_Train(settings)
        trainData = {"u":u,"z":z,"blockLabel":blockLabel,"stimHistCalc":stimHistCalc,"Start_Eval":Start_Eval,
                "End_Eval":End_Eval,"StimDurStart":StimDurStart,"StimDurStop":StimDurStop,"DelayStart":DelayStart,"DelayStop":DelayStop}
        output,loss,BreakFlag = self.train_block(u, z, settings,0)
        self.NumberOfFits += 1
        now = datetime.datetime.now()
        print("\n")
        formatted_now = now.strftime("%Y.%m.%d - %H:%M")
        print(formatted_now,"   Sessions Fit ",0,"|| Loss: ", loss)

        
        finalEval = False
        if settings["TrainSetEval"] == True:
            print("================Evaluate Model on Training Set=======================")
            with torch.no_grad():
                EvalOutput = self.EvalModel(settings,finalEval,trainData)
        if BreakFlag == 0:
            for nFS in range(1,FitSessions):

                if nFS%eval_freq == 0:

                    print("================Evaluate Model=======================")
                    finalEval = False
                    with torch.no_grad():
                        EvalOutput = self.EvalModel(settings,finalEval)
                        self.updateEvalTable(EvalOutput)
                    
                    if EvalOutput["eval_perf_mean"] >= settings["perf"] and EvalOutput["eval_perf_catch_mean"] >= settings["catch_perf"]and nFS > 0:
                        print("Evaluation Criteria Achieved")
                        break

                print("-------------Fit Model--------------------------")
                u,z,blockLabel,stimHistCalc,Start_Eval,End_Eval,StimDurStart,StimDurStop,DelayStart,DelayStop = self.dataSession_Train(settings)
                trainData = {"u":u,"z":z,"blockLabel":blockLabel,"stimHistCalc":stimHistCalc,"Start_Eval":Start_Eval,
                            "End_Eval":End_Eval,"StimDurStart":StimDurStart,"StimDurStop":StimDurStop,"DelayStart":DelayStart,"DelayStop":DelayStop}
                
                output,loss,BreakFlag= self.train_block(u, z, settings,nFS)
                self.NumberOfFits += 1
                if BreakFlag > 0:
                    break

                # Get the current time
                now = datetime.datetime.now()
                print("\n")
                formatted_now = now.strftime("%Y.%m.%d - %H:%M")
                print(formatted_now,"  Sessions Fit ",nFS,"|| Loss: ", loss)

                
                finalEval = False
                if settings["TrainSetEval"] == True:
                    print("================Evaluate Model on Training Set=======================")
                    with torch.no_grad():
                        EvalOutput = self.EvalModel(settings,finalEval,trainData)

        print("================Final Evaluate Model=======================")
        finalEval = True
        with torch.no_grad():
            EvalOutput = self.EvalModel(settings,finalEval)
            self.updateEvalTable(EvalOutput)
        torch.cuda.empty_cache()
        return EvalOutput,BreakFlag
    
    def weighted_mse_loss(self,prediction, target, weight):
        """
        Weighted Mean Squared Error Loss

        Parameters:
        prediction (Tensor): Predicted output (batch_size x num_columns).
        target (Tensor): Ground truth (batch_size x num_columns).
        weight (Tensor): Weights for each column (1 x num_columns).

        Returns:
        Tensor: Weighted MSE loss.
        """
        
        if prediction.dim() < 2:
            prediction = prediction.unsqueeze(0)
        

        # Check if the dimensions match
        if prediction.dim() > 1 and target.dim() > 1 and weight.dim() > 1:
            if prediction.size(1) != target.size(1) or target.size(1) != weight.size(1):
                raise ValueError("Dimension mismatch between prediction, target, and weight")
        else:
            if prediction.size(0) != target.size(0) or target.size(0) != weight.size(0):
                raise ValueError("Dimension mismatch between prediction, target, and weight")
        # Compute MSE loss for each element and then apply weights
        diff = prediction - target
        weighted_sq_diff = (diff ** 2)*weight


        # Calculate the mean across all batches and columns
        loss = weighted_sq_diff.mean()
        return loss
    
    
    def RnnManipulations(self,settings,opto_idx):

        
        if settings["Manipulation"] == "SuppressNeurons":
            print("Suppress Neurons")

            WrSize = self.Wr.shape
            OptoMask = torch.ones(WrSize,device=self.device)
            OptoMask[opto_idx,:] = 0
            OptoMask[:,opto_idx] = 0

            WinSize = self.Win.shape
            OptoMaskIn = torch.ones(WinSize,device=self.device)
            OptoMaskIn[opto_idx,:] = 0

        elif settings["Manipulation"] == "AllPruneSynapse":
            print("Pruning Recurrent Synapses")

            WrSize = self.Wr.shape
            OptoMask = torch.ones(WrSize,device=self.device)
            self.OptoMask = OptoMask
            for i in opto_idx:
                #opto_idx = torch.tensor(opto_idx, dtype=torch.int64)  
                OptoMask[opto_idx,i] = 0

            WinSize = self.Win.shape
            OptoMaskIn = torch.ones(WinSize,device=self.device)

        elif settings["Manipulation"] == "PostPruneSynapse":
            print("Pruning Recurrent PostSynapses")

            WrSize = self.Wr.shape
            OptoMask = torch.ones(WrSize,device=self.device)
            self.OptoMask = OptoMask
            assert all(isinstance(i, list) for i in opto_idx ), f"Error: {opto_idx} Should Be A List Of Lists"
            assert len(opto_idx) == 2 or len(opto_idx) == 4, f"Error: {opto_idx} Should Be A List With Two or Four Elements"
         
            rowIdx = opto_idx[0]
            colIdx = opto_idx[1]
            for ci in range(len(colIdx)):
                for ri in range(len(rowIdx )):
                    OptoMask[rowIdx[ri],colIdx[ci]] = 0

            WinSize = self.Win.shape
            OptoMaskIn = torch.ones(WinSize,device=self.device)

        elif settings["Manipulation"] == "PrePruneSynapse":
            print("Pruning Recurrent PreSynapses")

            WrSize = self.Wr.shape
            OptoMask = torch.ones(WrSize,device=self.device)
            self.OptoMask = OptoMask
            assert all(isinstance(i, list) for i in opto_idx ), f"Error: {opto_idx} Should Be A List Of Lists"
            assert len(opto_idx) == 2 or len(opto_idx) == 4, f"Error: {opto_idx} Should Be A List With Two or Four Elements"


            rowIdx = opto_idx[0]
            colIdx = opto_idx[1]
            for ci in range(len(colIdx)):
                for ri in range(len(rowIdx )):
                    OptoMask[rowIdx[ri],colIdx[ci]] = 0

            WinSize = self.Win.shape
            OptoMaskIn = torch.ones(WinSize,device=self.device)

        else:
            print("No Manipulation Occurs")
            
            WrSize = self.Wr.shape
            OptoMask = torch.ones(WrSize,device=self.device)

            WinSize = self.Win.shape
            OptoMaskIn = torch.ones(WinSize,device=self.device)

        return OptoMaskIn, OptoMask
    
    def check_required_keys(self,settings):
        # Define the list of required keys
        required_keys = [
            "Pips", "T", "Tdelay", "Blocks", "BlockLength", "stim_on", "stim_dur", "stim_delay",
            "BioConstraints",
            "layerNorm",
            "LR_Scheduler", "factor", "patience",
            "Wr_Dropout", "tauS_Dropout", "Wout_Dropout", "bout_Dropout", "pDropout",
            "Delay_Type", "Delays", "DelayProbs",
            "FitSessions", "num_epochs",
            "SampleTrain", "accum_steps", "batch_size", "Loss", "Regularization", "lambda", "lr","LrDips","eta_min",
            "Continual","Eligible Age","Replacement_Rate","DecayRate",
            "betas","weight_decay",
            "EWC","lambda_ewc",
            "Manipulation",
            "max_gradient_norm","GradientClipping","PrintGradientNorms",
            "SuppressLength", "DelaySuppressOn", "SuppressCatchOnly", "TrainCatchTrialOnly", "EvalCatchTrialOnly", "catch_decision_Trian_on",
            "Eval_Threshold", "Eval_Freq", "GPU_Eval_Blocks", "BlockMultiplier",
            "weight", "SampleTest", "RandCatchTest",
            "perf", "catch_perf",
            "InitEvaluate", "filename", "EpochSaveFreq"
        ]
        
        # Check for missing keys
        missing_keys = [key for key in required_keys if key not in settings]

        # Check for unexpected keys
        unexpected_keys = [key for key in settings if key not in required_keys]
        
        if missing_keys:
            raise KeyError(f"Missing required keys: {missing_keys}")
        
        if unexpected_keys:
            raise KeyError(f"Unexpected keys found: {unexpected_keys}")
        
        print("All required keys are present, and no unexpected keys were found.")
        
        return True
    







    def compute_loss_with_penalty(self, o, z, settings):

        batch_size = z.size(0)

        weight = self.ComputeWeights(settings,z)
        weights = weight.to(self.device)

        # Compute MSE/MAE loss
        if settings["Loss"] == "MSE":
            loss = self.weighted_mse_loss(o, z, weights)
        elif settings["Loss"] == "MAE":
            loss = self.weighted_mae_loss(o, z, weights)
        elif settings["Loss"] == "LogCosh":
            loss = self.weighted_logcosh_loss(o, z, weights)
        else:
            raise Exception("No Loss Function Defined in Settings. Loss == MSE or MAE")

        # Apply L1 or L2 regularization
        if settings["Regularization"] == "L1":
            loss += settings["lambda"] * sum(torch.sum(torch.abs(p)) for p in self.parameters()) / batch_size 
        elif settings["Regularization"] == "L2":
            loss += settings["lambda"] * sum(torch.sum(p ** 2) for p in self.parameters()) / batch_size 

      
        return loss
    
    def createEpochDB(model) -> None: 
        with sqlite3.connect(model.db_path , timeout=10) as conn:
            cursor = conn.cursor()

            # Create EvalTable if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS EvalTable (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    NumberOfFits INTEGER,
                    Session INTEGER,
                    Epoch INTEGER,
                    Input BLOB,
                    Output BLOB,
                    StimHist BLOB,
                    Perf INTEGER, 
                    Perf_Discrim INTEGER,
                    Perf_Catch,
                    Perf_Discrim_Right,
                    Perf_Discrim_Left,
                    Perf_Catch_Right, 
                    Perf_Catch_Left, 
                    Perf_Catch_Ambig
              
                )
            ''')

    
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS currentRNNParameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    NumberOfFits INTEGER,
                    epoch INTEGER,
                    Session INTEGER,
                    
                    modelSigniture STRING,
                    dataBasePath STRING,
                    db_path STRING,
                    dType STRING,
                    device STRING,
                
                    N INTEGER,
                    apply_dale BOOLEAN,
                    prop_inh FLOAT,
                    prop_som FLOAT,
                    prob_rec FLOAT,
                    Prop_PUL FLOAT,
                    Prop_TRN FLOAT,
                    gain FLOAT,
                    tausRange BLOB,
                    
                    ppc_rate STRING, 
                    pul_rate STRING, 
                    trn_rate STRING,

                    PPC_idx BLOB,
                    PPCexh_idx BLOB,
                    PPCinh_idx BLOB,
                    PPCsomM_idx BLOB,
                           
                    PULexh_idx BLOB,
                    TRNinh_idx BLOB,

                    PPC_N INTEGER,       
                    PPCexh_N INTEGER,
                    PPCinh_N INTEGER,
                    PPCsomM_N INTEGER,

                    PULexh_N INTEGER,
                    TRNinh_N INTEGER,

                    StructureMask BLOB,      
                    SynapticTypeMask BLOB,
                    SynapseMask BLOB,
                           
                    Win BLOB,
                    WrInit BLOB,
                           
                    Wr BLOB,
                    TauS BLOB,
                    Wout BLOB,
                    b_out BLOB,
                    
                    OptoMask BLOB,
                    OptoMaskIn BLOB,
                           
                    TotalPerform_mean FLOAT,
                           
                    Discrim_Mean FLOAT,
                    Discrim_Mean_Left FLOAT,
                    Discrim_Mean_Right FLOAT,
                           
                    Catch_Mean FLOAT,
                    Catch_Mean_Left FLOAT,
                    Catch_Mean_RIGHT FLOAT,
                    Catch_Mean_AMBIG FLOAT
        
                )
            ''')

            conn.commit()



    def updateEvalTable(self, EvalOutput) -> None:
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            cursor = conn.cursor()

            # Check if all the 'perf_' columns are present and add them if necessary
            cursor.execute("PRAGMA table_info(EvalTable)")
            existing_columns = [info[1] for info in cursor.fetchall()]  # Get a list of column names
            
            # List of perf-related columns that should be in the table
            required_columns = [
                ("Perf","INTEGER"),
                ("Perf_Discrim", "INTEGER"),
                ("Perf_Catch", "INTEGER"),
                ("Perf_Discrim_Right", "INTEGER"),
                ("Perf_Discrim_Left", "INTEGER"),
                ("Perf_Catch_Right", "INTEGER"),
                ("Perf_Catch_Left", "INTEGER"),
                ("Perf_Catch_Ambig", "INTEGER")
            ]

            # Add any missing columns
            for col_name, col_type in required_columns:
                if col_name not in existing_columns:
                    cursor.execute(f"ALTER TABLE EvalTable ADD COLUMN {col_name} {col_type}")

            # Create EvalTable if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS EvalTable (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    NumberOfFits INTEGER,
                    Session INTEGER,
                    Epoch INTEGER,
                    Input BLOB,
                    Output BLOB,
                    StimHist BLOB,
                    Perf INTEGER, 
                    Perf_Discrim INTEGER,
                    Perf_Catch,
                    Perf_Discrim_Right,
                    Perf_Discrim_Left,
                    Perf_Catch_Right, 
                    Perf_Catch_Left, 
                    Perf_Catch_Ambig            
                )
            ''')

            # Convert tensors to binary data for SQLite storage
            uEvals_blob = pickle.dumps(EvalOutput["uEvals"])
            oEvals_blob = pickle.dumps(EvalOutput["oEvals"])
            stimHistCalcEvals_blob = pickle.dumps(EvalOutput["stimHistCalcEvals"])

            perf = EvalOutput["eval_perf_mean"]
            perf_discrim = EvalOutput["eval_perf_discrim_mean"]
            perf_catch = EvalOutput["eval_perf_catch_mean"]
            
            perf_discrim_right = EvalOutput["eval_perf_discrim_right_mean"]
            perf_discrim_left = EvalOutput["eval_perf_discrim_left_mean"]

            perf_catch_right = EvalOutput["eval_perf_catch_right_mean"]
            perf_catch_left = EvalOutput["eval_perf_catch_left_mean"]
            perf_catch_ambig = EvalOutput["eval_perf_catch_ambig_mean"]

            # Insert values into the EvalTable
            cursor.execute('''
                INSERT INTO EvalTable 
                    (NumberOfFits,Session, Epoch, Input, Output, StimHist, Perf, Perf_Discrim, Perf_Catch, Perf_Discrim_Right, Perf_Discrim_Left, Perf_Catch_Right, Perf_Catch_Left, Perf_Catch_Ambig)
                VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', 
            (self.NumberOfFits,self.Session, self.Tepoch, uEvals_blob, oEvals_blob, stimHistCalcEvals_blob,perf,perf_discrim,perf_catch,perf_discrim_right, perf_discrim_left,perf_catch_right,perf_catch_left,perf_catch_ambig))


            conn.commit()


    def save_rnn_to_db(self,Validation):
        # Connect to the SQLite database
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            cursor = conn.cursor()

            # Create the RNNParameters table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS currentRNNParameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                           
                    NumberOfFits INTEGER,
                    epoch INTEGER,
                    Session INTEGER,
                    
                    modelSigniture STRING,
                    dataBasePath STRING,
                    db_path STRING,
                    dType STRING,
                    device STRING,
                
                    N INTEGER,
                    apply_dale BOOLEAN,
                    prop_inh FLOAT,
                    prop_som FLOAT,
                    prob_rec FLOAT,
                    Prop_PUL FLOAT,
                    Prop_TRN FLOAT,
                    gain FLOAT,
                    tausRange BLOB,
                           
                    ppc_rate STRING, 
                    pul_rate STRING, 
                    trn_rate STRING,

                    PPC_idx BLOB,
                    PPCexh_idx BLOB,
                    PPCinh_idx BLOB,
                    PPCsomM_idx BLOB,
                           
                    PULexh_idx BLOB,
                    TRNinh_idx BLOB,

                    PPC_N INTEGER,       
                    PPCexh_N INTEGER,
                    PPCinh_N INTEGER,
                    PPCsomM_N INTEGER,

                    PULexh_N INTEGER,
                    TRNinh_N INTEGER,

                    StructureMask BLOB,      
                    SynapticTypeMask BLOB,
                    SynapseMask BLOB,
                           
                    Win BLOB,
                    WrInit BLOB,
                           
                    Wr BLOB,
                    TauS BLOB,
                    Wout BLOB,
                    b_out BLOB,
                    
                    OptoMask BLOB,
                    OptoMaskIn BLOB,
                           
                    TotalPerform_mean FLOAT,
                           
                    Discrim_Mean FLOAT,
                    Discrim_Mean_Left FLOAT,
                    Discrim_Mean_Right FLOAT,
                           
                    Catch_Mean FLOAT,
                    Catch_Mean_Left FLOAT,
                    Catch_Mean_RIGHT FLOAT, 
                    Catch_Mean_AMBIG FLOAT                                     
                )
            ''')

            # Delete any existing row to ensure only one row is in the table
            #cursor.execute('DELETE FROM currentRNNParameters')

            # Convert the attributes to binary (BLOB) format
            tausRange_blob = pickle.dumps(self.tausRange)
            

            SynapseMask_blob = pickle.dumps(self.SynapseMask.cpu().numpy())
            StructureMask_blob = pickle.dumps(self.StructureMask.cpu().numpy())
            SynapticTypeMask_blob = pickle.dumps(self.SynapticTypeMask.cpu().numpy())

            PPC_idx_blob = pickle.dumps(self.PPC_idx)
            PPCexh_idx_blob = pickle.dumps(self.PPCexh_idx)
            PPCinh_idx_blob = pickle.dumps(self.PPCinh_idx)
            PPCsomM_idx_blob = pickle.dumps(self.PPCsomM_idx)
            PULexh_idx_blob = pickle.dumps(self.PULexh_idx)
            TRNinh_idx_blob = pickle.dumps(self.TRNinh_idx)

            WrInit_blob = pickle.dumps(self.Wr_init.cpu().numpy())

            Win_blob = pickle.dumps(self.Win.detach().cpu().numpy())

            Wout_blob = pickle.dumps(self.Wout.detach().cpu().numpy())
            TauS_blob = pickle.dumps(self.tauS.detach().cpu().numpy())    
            Wr_blob = pickle.dumps(self.Wr.detach().cpu().numpy())
            b_out_blob = pickle.dumps(self.b_out.detach().cpu().numpy())

            OptoMask_blob = pickle.dumps(self.OptoMask.cpu().numpy())
            OptoMaskIn_blob = pickle.dumps(self.OptoMaskIn.cpu().numpy())

            

            
            # Insert the model parameters into the table
            cursor.execute('''
                INSERT INTO currentRNNParameters (
                    NumberOfFits, epoch, Session,    
                                          
                    modelSigniture, dataBasePath, db_path, dType, device,
                                  
                    N, apply_dale, prop_inh, prop_som, prob_rec, Prop_PUL, Prop_TRN, gain, tausRange, ppc_rate, pul_rate, trn_rate,
                
                    PPC_idx, PPCexh_idx, PPCinh_idx, PPCsomM_idx,
                                               
                    PULexh_idx, TRNinh_idx,
                    
                    PPC_N, PPCexh_N, PPCinh_N, PPCsomM_N,     
                                       
                    PULexh_N, TRNinh_N,
                    
                    StructureMask, SynapticTypeMask, SynapseMask,                         
                           
                    Win, WrInit,
                                            
                    Wr, TauS, Wout, b_out,
                                       
                    OptoMask, OptoMaskIn,
                                            
                    TotalPerform_mean,
                           
                    Discrim_Mean, Discrim_Mean_Left, Discrim_Mean_Right,
                                           
                    Catch_Mean, Catch_Mean_Left, Catch_Mean_Right, Catch_Mean_Ambig
                    
                    
                )VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                self.NumberOfFits, self.Tepoch, self.Session,
                self.modelSigniture, self.dataBasePath, self.db_path, str(self.dType), self.device,
                self.N, self.apply_dale, self.prop_inh, self.prop_som, self.prob_rec,self.Prop_PUL, self.Prop_TRN,self.gain, tausRange_blob, self.ppc_rate_String, self.pul_rate_String, self.trn_rate_String,
                PPC_idx_blob, PPCexh_idx_blob, PPCinh_idx_blob, PPCsomM_idx_blob,
                PULexh_idx_blob, TRNinh_idx_blob,
                self.PPC_N, self.PPCexh_N, self.PPCinh_N, self.PPCsomM_N,     
                self.PULexh_N, self.TRNinh_N,
                StructureMask_blob, SynapticTypeMask_blob, SynapseMask_blob,
                Win_blob, WrInit_blob,
                Wr_blob, TauS_blob, Wout_blob, b_out_blob,
                OptoMask_blob, OptoMaskIn_blob,
                Validation["TotalPerform_mean"],
                Validation["Discrim_mean"], Validation["DiscrimLeft_mean"], Validation["DiscrimRight_mean"],
                Validation["Catch_mean"], Validation["CatchLeft_mean"], Validation["CatchRight_mean"], Validation["CatchAmbig_mean"]
                
            ))

            conn.commit()
            print("\n")
            print(f"Model parameters saved to database.")
            """"
            Validation = {"TotalPerform_mean":EvalOutput["eval_perf_mean"],
                "Discrim_mean":EvalOutput["eval_perf_discrim_mean"],                   
                "DiscrimRight_mean":EvalOutput["eval_perf_discrim_right_mean"],
                "DiscrimLeft_mean":EvalOutput["eval_perf_discrim_left_mean"],
                "Catch_mean":EvalOutput["eval_perf_catch_mean"],
                "CatchRight_mean":EvalOutput["eval_perf_catch_right_mean"],
                "CatchLeft_mean":EvalOutput["eval_perf_catch_left_mean"],
                "CatchAmbig_mean":EvalOutput["eval_perf_catch_ambig_mean"]}  
            """

    def load_rnn_from_db(self, db_path,RowSelect=None):
            # Connect to the SQLite database
            # Add the custom class as a safe global
            if RowSelect != None:
                assert RowSelect > 0,"RowSelect is 1 Index"
            with sqlite3.connect(db_path, timeout=10) as conn:
                cursor = conn.cursor()

                # Determine which row to select
                if RowSelect is None:
                    # Default to the last row
                    cursor.execute('SELECT * FROM currentRNNParameters ORDER BY id DESC LIMIT 1')
                else:
                    # Fetch the specified row
                    cursor.execute('SELECT * FROM currentRNNParameters WHERE id = ?', (RowSelect,))
                
                # Fetch the row from the database
                row = cursor.fetchone()

                if row is None:
                    raise ValueError("No saved model found in the database with the specified selection.")
                
                # Query the table to fetch column information
                column_names = [description[0] for description in cursor.description]  # Extract column names
                number_of_columns = len(column_names)

                print(f"Number of columns: {number_of_columns}")
                print(f"Column names: {column_names}")

                try:
                    # Extract data from the fetched row
                    (
                        NumberOfFits,Tepoch, Session,
                        modelSigniture, dataBasePath, db_path, dType, device,
                        N, apply_dale, prop_inh, prop_som, prob_rec, Prop_PUL, Prop_TRN, gain, tausRange_blob, ppc_rate_String, pul_rate_String, trn_rate_String,
                        PPC_idx_blob,PPCexh_idx_blob,PPCinh_idx_blob,PPCsomM_idx_blob,
                        PULexh_idx_blob,TRNinh_idx_blob,
                        PPC_N, PPCexh_N, PPCinh_N, PPCsomM_N,     
                        PULexh_N, TRNinh_N,
                        StructureMask_blob, SynapticTypeMask_blob, SynapseMask_blob,
                        Win_blob, WrInit_blob,
                        Wr_blob, TauS_blob, Wout_blob, b_out_blob,
                        OptoMask_blob, OptoMaskIn_blob,
                        TotalPerform_mean,
                        Discrim_mean, DiscrimLeft_mean, DiscrimRight_mean,
                        Catch_mean, CatchLeft_mean, CatchRight_mean, CatchAmbig_mean
                    ) = row[1:]  # Skip the `id` field
                except ValueError:
                    # Extract data from the fetched row
                    print("Incorrect DATABASE")

                
                self.NumberOfFits = NumberOfFits
                self.Tepoch = Tepoch
                self.Session = Session

                self.dataBasePath = dataBasePath
                self.modelSigniture = modelSigniture
                self.db_path = db_path          
                self.dType =torch.float32 if dType == "torch.float32" else torch.float64

                # Initialize device as None
                self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
                print("Device Selected: ", self.device)


                self.N = N
                self.apply_dale = apply_dale
                self.prop_inh = prop_inh
                self.prop_som = prop_som
                self.prob_rec = prob_rec
                self.Prop_PUL = Prop_PUL
                self.Prop_TRN = Prop_TRN
                self.gain = gain
                self.tausRange = pickle.loads(tausRange_blob)

                self.ppc_rate_String = ppc_rate_String 
                self.pul_rate_String = pul_rate_String
                self.trn_rate_String = trn_rate_String

                assert(self.prop_inh >= 0 and self.prop_inh <= 1)
                assert(self.prop_som >= 0 and self.prop_som <= 1)

                        
                # ------Number-------
                # PPC
                self.PPC_N = PPC_N 
                self.PPCexh_N = PPCexh_N
                self.PPCinh_N = PPCinh_N
                self.PPCsomM_N = PPCsomM_N
                # Thalamus
                self.PULexh_N = PULexh_N
                self.TRNinh_N = TRNinh_N
                

                # ---- Structure Details ----
                # PPC
                self.PPC_idx = pickle.loads(PPC_idx_blob)
                self.PPCexh_idx = pickle.loads(PPCexh_idx_blob)
                self.PPCinh_idx = pickle.loads(PPCinh_idx_blob)
                self.PPCsomM_idx = pickle.loads(PPCsomM_idx_blob)
                # Thalamus
                self.PULexh_idx = pickle.loads(PULexh_idx_blob)
                self.TRNinh_idx = pickle.loads(TRNinh_idx_blob)

                # Create Synaptic Mask which convert columns 
                
                
                self.StructureMask = torch.tensor(pickle.loads(StructureMask_blob), dtype=self.dType, device=self.device, requires_grad=False)
                self.SynapticTypeMask = torch.tensor(pickle.loads(SynapticTypeMask_blob), dtype=self.dType, device=self.device, requires_grad=False)
                self.SynapseMask = torch.tensor(pickle.loads(SynapseMask_blob), dtype=self.dType, device=self.device, requires_grad=False)
        
                self.Win = torch.tensor(pickle.loads(Win_blob), dtype=self.dType, device=self.device, requires_grad=False)
                self.Wr_init = torch.tensor(pickle.loads(WrInit_blob), dtype=self.dType, device=self.device, requires_grad=False)
                
                  
                # Parameters that are learned 
                self.Wr = nn.Parameter(torch.tensor(pickle.loads(Wr_blob), dtype=self.dType, device=self.device), requires_grad=True)
                self.Wout = nn.Parameter(torch.tensor(pickle.loads(Wout_blob), dtype=self.dType, device=self.device), requires_grad=True)
                self.tauS = nn.Parameter(torch.tensor(pickle.loads(TauS_blob), dtype=self.dType, device=self.device), requires_grad=True)
                self.b_out = nn.Parameter(torch.tensor(pickle.loads(b_out_blob), dtype=self.dType, device=self.device), requires_grad=True)

                # Optogentic Mask to Suppress Neurons
                self.OptoMask = torch.tensor(pickle.loads(OptoMask_blob), dtype=self.dType, device=self.device, requires_grad=False)
                self.OptoMaskIn = torch.tensor(pickle.loads(OptoMaskIn_blob), dtype=self.dType, device=self.device, requires_grad=False)

                # Performance on Evaluation
                self.EvalTrainTotalPerform = TotalPerform_mean

                self.EvalTrainDiscrimPerform =  Discrim_mean    
                self.EvalTrainDiscrimPerformRight = DiscrimLeft_mean 
                self.EvalTrainDiscrimPerformLeft = DiscrimRight_mean

                self.EvalTrainCatchPerform = Catch_mean
                self.EvalTrainCatchPerformRight = CatchLeft_mean
                self.EvalTrainCatchPerformLeft = CatchRight_mean
                self.EvalTrainCatchPerformAmbig = CatchAmbig_mean

                
                print("\n")
                print(f"Model parameters loaded from database for epoch {self.Tepoch}.")
                print(f"Model parameters loaded from database for Idx {RowSelect}.")
                if 'Catch_mean' in locals():
                    print(f"Model parameters loaded from database for Catch Perf {Catch_mean}.")
                print("\n")
    
    def reload_rnn_from_db(self, db_path,RowSelect=None):
        # Connect to the SQLite database
        # Add the custom class as a safe global
        if RowSelect != None:
            assert RowSelect > 0,"RowSelect is 1 Index"
        with sqlite3.connect(db_path, timeout=10) as conn:
            cursor = conn.cursor()

            # Execute the DELETE query
            cursor.execute("DELETE FROM currentRNNParameters WHERE id > ?", (RowSelect,))

            # Commit the changes
            conn.commit()

            # Determine which row to select
            if RowSelect is None:
                # Default to the last row
                cursor.execute('SELECT * FROM currentRNNParameters ORDER BY id DESC LIMIT 1')
            else:
                # Fetch the specified row
                cursor.execute('SELECT * FROM currentRNNParameters WHERE id = ?', (RowSelect,))
            
            # Fetch the row from the database
            row = cursor.fetchone()

            if row is None:
                raise ValueError("No saved model found in the database with the specified selection.")
            
            # Query the table to fetch column information
            column_names = [description[0] for description in cursor.description]  # Extract column names
            number_of_columns = len(column_names)

            print(f"Number of columns: {number_of_columns}")
            print(f"Column names: {column_names}")

            try:
                # Extract data from the fetched row
                (
                    NumberOfFits,Tepoch, Session,
                    modelSigniture, dataBasePath, db_path, dType, device,
                    N, apply_dale, prop_inh, prop_som, prob_rec, Prop_PUL, Prop_TRN, gain, tausRange_blob,  ppc_rate_String, pul_rate_String, trn_rate_String,
                    PPC_idx_blob,PPCexh_idx_blob,PPCinh_idx_blob,PPCsomM_idx_blob,
                    PULexh_idx_blob,TRNinh_idx_blob,
                    PPC_N, PPCexh_N, PPCinh_N, PPCsomM_N,     
                    PULexh_N, TRNinh_N,
                    StructureMask_blob, SynapticTypeMask_blob, SynapseMask_blob,
                    Win_blob, WrInit_blob,
                    Wr_blob, TauS_blob, Wout_blob, b_out_blob,
                    OptoMask_blob, OptoMaskIn_blob,
                    TotalPerform_mean,
                    Discrim_mean, DiscrimLeft_mean, DiscrimRight_mean,
                    Catch_mean, CatchLeft_mean, CatchRight_mean, CatchAmbig_mean
                ) = row[1:]  # Skip the `id` field
            except ValueError:
                # Extract data from the fetched row
                print("Incorrect DATABASE")

            
            self.NumberOfFits = NumberOfFits
            self.Tepoch = Tepoch
            self.Session = Session

            self.dataBasePath = dataBasePath
            self.modelSigniture = modelSigniture
            self.db_path = db_path          
            self.dType =torch.float32 if dType == "torch.float32" else torch.float64

            # Initialize device as None
            self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
            print("Device Selected: ", self.device)


            self.N = N
            self.apply_dale = apply_dale
            self.prop_inh = prop_inh
            self.prop_som = prop_som
            self.prob_rec = prob_rec
            self.Prop_PUL = Prop_PUL
            self.Prop_TRN = Prop_TRN
            self.gain = gain
            self.tausRange = pickle.loads(tausRange_blob)

            self.ppc_rate_String = ppc_rate_String 
            self.pul_rate_String = pul_rate_String
            self.trn_rate_String = trn_rate_String
    
            assert(self.prop_inh >= 0 and self.prop_inh <= 1)
            assert(self.prop_som >= 0 and self.prop_som <= 1)

                    
            # ------Number-------
            # PPC
            self.PPC_N = PPC_N 
            self.PPCexh_N = PPCexh_N
            self.PPCinh_N = PPCinh_N
            self.PPCsomM_N = PPCsomM_N
            # Thalamus
            self.PULexh_N = PULexh_N
            self.TRNinh_N = TRNinh_N
            

            # ---- Structure Details ----
            # PPC
            self.PPC_idx = pickle.loads(PPC_idx_blob)
            self.PPCexh_idx = pickle.loads(PPCexh_idx_blob)
            self.PPCinh_idx = pickle.loads(PPCinh_idx_blob)
            self.PPCsomM_idx = pickle.loads(PPCsomM_idx_blob)
            # Thalamus
            self.PULexh_idx = pickle.loads(PULexh_idx_blob)
            self.TRNinh_idx = pickle.loads(TRNinh_idx_blob)

            # Create Synaptic Mask which convert columns 
            
            
            self.StructureMask = torch.tensor(pickle.loads(StructureMask_blob), dtype=self.dType, device=self.device, requires_grad=False)
            self.SynapticTypeMask = torch.tensor(pickle.loads(SynapticTypeMask_blob), dtype=self.dType, device=self.device, requires_grad=False)
            self.SynapseMask = torch.tensor(pickle.loads(SynapseMask_blob), dtype=self.dType, device=self.device, requires_grad=False)
    
            self.Win = torch.tensor(pickle.loads(Win_blob), dtype=self.dType, device=self.device, requires_grad=False)
            self.Wr_init = torch.tensor(pickle.loads(WrInit_blob), dtype=self.dType, device=self.device, requires_grad=False)
            
                
            # Parameters that are learned 
            self.Wr = nn.Parameter(torch.tensor(pickle.loads(Wr_blob), dtype=self.dType, device=self.device), requires_grad=True)
            self.Wout = nn.Parameter(torch.tensor(pickle.loads(Wout_blob), dtype=self.dType, device=self.device), requires_grad=True)
            self.tauS = nn.Parameter(torch.tensor(pickle.loads(TauS_blob), dtype=self.dType, device=self.device), requires_grad=True)
            self.b_out = nn.Parameter(torch.tensor(pickle.loads(b_out_blob), dtype=self.dType, device=self.device), requires_grad=True)

            # Optogentic Mask to Suppress Neurons
            self.OptoMask = torch.tensor(pickle.loads(OptoMask_blob), dtype=self.dType, device=self.device, requires_grad=False)
            self.OptoMaskIn = torch.tensor(pickle.loads(OptoMaskIn_blob), dtype=self.dType, device=self.device, requires_grad=False)

            # Performance on Evaluation
            self.EvalTrainTotalPerform = TotalPerform_mean

            self.EvalTrainDiscrimPerform =  Discrim_mean    
            self.EvalTrainDiscrimPerformRight = DiscrimLeft_mean 
            self.EvalTrainDiscrimPerformLeft = DiscrimRight_mean

            self.EvalTrainCatchPerform = Catch_mean
            self.EvalTrainCatchPerformRight = CatchLeft_mean
            self.EvalTrainCatchPerformLeft = CatchRight_mean
            self.EvalTrainCatchPerformAmbig = CatchAmbig_mean

            
            
            print("\n")
            print(f"Model parameters loaded from database for epoch {self.Tepoch}.")
            print(f"Model parameters loaded from database for Idx {RowSelect}.")
            if 'Catch_mean' in locals():
                print(f"Model parameters loaded from database for Catch Perf {Catch_mean}.")
            print("\n")
    

    def get_last_index(self,database_path, index_column="id"):
        """
        Get the last index (maximum value) of a specified column in an SQLite table.

        Parameters:
            database_path (str): Path to the SQLite database file.
            table_name (str): Name of the table to query.
            index_column (str): Name of the column to check for the maximum value. Defaults to "id".

        Returns:
            int or None: The last index (maximum value) if found, otherwise None if the table is empty.
        """
        try:
            # Connect to the database
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()

            # Prepare the query
            query = f"SELECT MAX({index_column}) FROM currentRNNParameters"

            # Execute the query
            cursor.execute(query)

            # Fetch the result
            last_index = cursor.fetchone()[0]

            return last_index

        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return None

        finally:
            # Ensure the connection is closed
            if conn:
                conn.close()


    
    # Function to compute gradient norms
    def compute_grad_norms(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)  # L2 norm of the gradient
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def check_gradients_for_issues(self, exploding_threshold=10.0, vanishing_threshold=1e-5):
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                if grad_norm > exploding_threshold:
                    print("\n")
                    print(f"Exploding gradients detected in {name}, norm: {grad_norm:.5f},\n")
                elif grad_norm < vanishing_threshold:
                    print("\n")
                    print(f"Vanishing gradients detected in {name}, norm: {grad_norm:.5f},\n")

    
    def rateFunction(self,x,BrainArea,RateParams):

        if BrainArea == "PPC":

            name = self.ppc_rate_String
        elif BrainArea == "PUL":
            name = self.pul_rate_String
        elif BrainArea == "TRN":
            name = self.trn_rate_String
        else:
            raise ValueError(f"Unknown Brain Area function: {BrainArea}") 

        if name == 'leaky_relu':
            
            missing_keys = [key for key in ["negative_slope", "max"] if key not in RateParams]
            if missing_keys:
                KeyError(f"Rate Param Missing: {missing_keys}")
            else:             
                NS = RateParams["negative_slope"]
                MAX = RateParams["max"]

            return torch.clamp(torch.nn.functional.leaky_relu(x,negative_slope=NS), max=MAX)
        
        elif name == 'alpha_leaky_relu':

            missing_keys = [key for key in ["negative_slope", "max", "alpha"] if key not in RateParams]
            if missing_keys:
                KeyError(f"Rate Param Missing: {missing_keys}")
            else:             
                NS = RateParams["negative_slope"]
                MAX = RateParams["max"]
                alpha = RateParams["alpha"]
                centre =  RateParams["centre"]
                      
            return torch.clamp(alpha*torch.nn.functional.leaky_relu(x-centre,negative_slope=NS), max=MAX)
        
        elif name == 'relu':

            missing_keys = [key for key in ["max"] if key not in RateParams]
            if missing_keys:
                KeyError(f"Rate Param Missing: {missing_keys}")
            else:             
                MAX = RateParams["max"]

            return torch.clamp(torch.nn.functional.relu(x),min=0 ,max=MAX) 
        
        elif name == 'alpha_relu':

            missing_keys = [key for key in ["max", "alpha"] if key not in RateParams]
            if missing_keys:
                KeyError(f"Rate Param Missing: {missing_keys}")
            else:             
                MAX = RateParams["max"]
                alpha = RateParams["alpha"]
                centre =  RateParams["centre"]

            return torch.clamp(alpha*torch.nn.functional.relu(x-centre),min=0 ,max=MAX) 
        
        elif name == 'sigmoid':


            return torch.nn.functional.sigmoid(x)
        
        elif name == "generalized_sigmoid":

            missing_keys = [key for key in ["fmax", "beta", "c"] if key not in RateParams]
            if missing_keys:
                KeyError(f"Rate Param Missing: {missing_keys}")
            else:             
                fmax = RateParams["fmax"]
                beta = RateParams["beta"] 
                c = RateParams["c"]

            return fmax*torch.nn.functional.sigmoid(beta * (x - c))
        
        elif name == 'softplus':

            return torch.clamp(torch.nn.functional.softplus(x), min=0, max=20) # Clamping is required to stop runaway accumulation of Snapytic currents
        
        elif name == 'surrogate_sigmoid':

            return surrogate_sigmoid(x)
        
        elif name == 'surrogate_generalized_sigmoid':

            missing_keys = [key for key in ["fmax", "beta", "c"] if key not in RateParams]
            if missing_keys:
                raise KeyError(f"Rate Param Missing: {missing_keys}")
            else:             
                fmax = torch.tensor(RateParams["fmax"],device=x.device,requires_grad=False)
                beta = torch.tensor(RateParams["beta"],device=x.device,requires_grad=False)
                c = torch.tensor(RateParams["c"],device=x.device,requires_grad=False)

            return surrogate_generalized_sigmoid(x,fmax,beta,c)
        
        else:
            raise ValueError(f"Unknown activation function: {name}")   



# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# |||||||||||||||||||||||||||| OUTSIDE CLASS ||||||||||||||||||||||||||||||||||||||||||||||| 
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||       
def GpuMemory():
    if torch.cuda.is_available():
        # Get the total GPU memory in bytes and convert it to gigabytes
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Get the current memory usage in bytes and convert it to gigabytes
        current_memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        # Calculate the available memory
        available_memory = total_memory - current_memory_allocated

        print(f"Total GPU Memory: {total_memory:.2f} GB")
        print(f"Currently Allocated Memory: {current_memory_allocated:.2f} GB")
        print(f"Available GPU Memory: {available_memory:.2f} GB")
    else:
        print("No CUDA-capable GPU is available.")   


    




#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#------------------------------A Random Collection Of Functions--------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


def matchList(list_list,pattern):

    idx_matches = []
    ary_matches = []
    for idx, sublist in enumerate(list_list):
        if sublist == pattern:
            ary_matches.append(sublist)
            idx_matches.append(idx)
    return ary_matches, idx_matches



def print_loading_bar(iteration, total, length=40):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\rSession Progress: |{bar}| {percent}% Complete', end='\r')
    # Print New Line on Complete
    if iteration == total: 
        print()



