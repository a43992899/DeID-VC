from model_vc import Generator
import torch
from torch import nn
import torch.nn.functional as F
import time
import datetime
from tqdm import tqdm
import os
from hparams import DATA_PATH, PROJECT_ROOT, AUTOVC_ROOT
from model_bl import D_VECTOR, Proj
from collections import OrderedDict


class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.resume = config.resume
        self.resume_iter = 0
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step

        # Build the model and tensorboard.
        self.build_model()

            
    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        
        self.G.to(self.device)

        if self.resume is not None:
            g_checkpoint = torch.load(self.resume, map_location=self.device)
            self.G.load_state_dict(g_checkpoint['model'])
            self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
            self.resume_iter = g_checkpoint['iter']

        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        best_loss = 1e+10
        for i in range(self.resume_iter, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)
            
            
            x_real = x_real.to(self.device) 
            emb_org = emb_org.to(self.device) 
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
            g_loss_id = F.mse_loss(x_real, x_identic.squeeze(1))   
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt.squeeze(1))   
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)

            if (i+1) % 5000 == 0:
                l = g_loss.item()
                if l < best_loss: 
                    best_loss = l
                    PATH = os.path.join(DATA_PATH, 'model', f'autovc_best.ckpt')
                    torch.save({
                    'iter': i,
                    'model': self.G.state_dict(),
                    'optimizer': self.g_optimizer.state_dict(),
                    'loss': loss,
                    }, PATH)

                PATH = os.path.join(DATA_PATH, 'model', f'autovc_epoch_{int((i+1)/5000)}_loss_{l:4.4f}.ckpt')
                torch.save({
                    'iter': i,
                    'model': self.G.state_dict(),
                    'optimizer': self.g_optimizer.state_dict(),
                    'loss': loss,
                    }, PATH)
                

class Solver_enhanced(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd_reconst = config.lambda_cd_reconst
        self.lambda_cd_convert = config.lambda_cd_convert
        self.lambda_emb_convert = config.lambda_emb_convert
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.num_iters = config.num_iters
        self.resume = config.resume
        self.resume_iter = 0
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse = nn.MSELoss(reduction='sum')
        self.l1 = nn.L1Loss(reduction='sum')
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step

        # Build the model and tensorboard.
        self.build_model()

            
    def build_model(self):
        
        # Build G.
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.lr)
        self.G.to(self.device)

        if self.resume is not None:
            g_checkpoint = torch.load(self.resume, map_location=self.device)
            self.G.load_state_dict(g_checkpoint['model'])
            self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
            self.resume_iter = g_checkpoint['iter']
        
        # Build Es.
        if self.lambda_emb_convert > 0:
            self.Es = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).to(self.device)
            es_checkpoint = torch.load(DATA_PATH+'3000000-BL.ckpt', map_location=self.device)
            new_state_dict = OrderedDict()
            for key, val in es_checkpoint['model_b'].items():
                new_key = key[7:]
                new_state_dict[new_key] = val
            self.Es.load_state_dict(new_state_dict)
            for p in self.Es.parameters():
                p.requires_grad = False

        # Build linear projection.
        # self.Proj = Proj(in_dim=256, out_dim=256).to(self.device)

        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd_reconst','G/loss_cd_convert',
                'G/loss_emb_convert']
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        best_loss = 1e+10
        for i in range(self.resume_iter, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)
            
            
            x_real = x_real.to(self.device)
            emb_org = emb_org.to(self.device)
            emb_trg = emb_org.clone().to(self.device)
            emb_trg = emb_trg[torch.randperm(emb_trg.shape[0])]
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()
            # self.Proj = self.Proj.train()
            
            """ Step 1: X1->1. """
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
            g_loss_id = F.l1_loss(x_real, x_identic.squeeze(1))   
            g_loss_id_psnt = F.l1_loss(x_real, x_identic_psnt.squeeze(1))   
            
            # Content code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd_reconst = F.l1_loss(code_real, code_reconst)

            """ Step 2: X1->2. """
            # Compute X1->2
            x_convert, x_convert_psnt, code_real = self.G(x_real, emb_org, emb_trg)
            
            # Content code semantic loss.
            code_convert = self.G(x_convert_psnt, emb_trg, None)
            g_loss_cd_convert = F.l1_loss(code_real, code_convert)

            # Speaker embedding semantic loss.
            if self.lambda_emb_convert > 0:
                emb_convert = self.Es(x_convert_psnt.squeeze(1))
                # emb_convert = self.Proj(emb_convert)
                cos_sim = self.cos(emb_trg, emb_convert)
                cos_sim[cos_sim > 0.3] = 1.0
                cos_sim[cos_sim < 0.0] = 0.0
                g_loss_emb_convert = (1 - cos_sim).mean() # + F.l1_loss(emb_trg, emb_convert)

            """ Step 3: Compute loss. """
            # Backward and optimize.
            if self.lambda_emb_convert > 0:
                g_loss = g_loss_id + g_loss_id_psnt + \
                        self.lambda_cd_reconst * g_loss_cd_reconst + \
                        self.lambda_cd_convert * g_loss_cd_convert + \
                        self.lambda_emb_convert * g_loss_emb_convert
            else:
                g_loss = g_loss_id + g_loss_id_psnt + \
                        self.lambda_cd_reconst * g_loss_cd_reconst + \
                        self.lambda_cd_convert * g_loss_cd_convert
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd_reconst'] = g_loss_cd_reconst.item()
            loss['G/loss_cd_convert'] = g_loss_cd_convert.item()
            loss['G/loss_emb_convert'] = g_loss_emb_convert.item() if self.lambda_emb_convert > 0 else 0.0

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)

            if (i+1) % 100 == 0:
                l = g_loss.item()
                if l < best_loss: 
                    best_loss = l
                    PATH = os.path.join(DATA_PATH, 'model', f'autovc_best.ckpt')
                    torch.save({
                    'iter': i,
                    'model': self.G.state_dict(),
                    'optimizer': self.g_optimizer.state_dict(),
                    # 'proj': self.Proj.state_dict(),
                    'loss': loss,
                    }, PATH)

                PATH = os.path.join(DATA_PATH, 'model', 
                                    f'autovc_epoch_{int((i+1)/5000)}_loss_{l:4.4f}.ckpt')
                torch.save({
                    'iter': i,
                    'model': self.G.state_dict(),
                    'optimizer': self.g_optimizer.state_dict(),
                    # 'proj': self.Proj.state_dict(),
                    'loss': loss,
                    }, PATH)
                
    

    