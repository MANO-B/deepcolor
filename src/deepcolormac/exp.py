# evndyn
import torch
from .commons import make_batch_to_device
from .modules import VaeSm, scVAE
from .funcs import calc_kld, calc_poisson_loss, calc_nb_loss
from .dataset import VaeSmDataSet, VaeSmDataManager, VaeSmDataManagerDPP, ConcatDataset
from torch.utils.data import DataLoader
import numpy as np
import copy
import torch.nn as nn
import matplotlib.pyplot as plt

class VaeSmExperiment:
    def __init__(self, model_params, lr, x, s, test_ratio, x_batch_size, s_batch_size, num_workers, b=None, validation_ratio=0.1, device='auto', use_poisson=False):
        if device == 'auto':
            self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.sedm = VaeSmDataManager(s, test_ratio, x_batch_size, num_workers, validation_ratio=validation_ratio, b=b)
        self.xedm = VaeSmDataManager(x, test_ratio, s_batch_size, num_workers, validation_ratio=validation_ratio, b=b)
        self.model_params = model_params
        self.vaesm = VaeSm(self.sedm.x.size()[0], **self.model_params)
        self.vaesm.to(self.device)
        self.vaesm_optimizer = torch.optim.Adam(self.vaesm.parameters(), lr=lr)
        self.train_loss_list = []
        self.test_loss_list = []
        self.best_loss = None
        s = self.sedm.x
        snorm_mat = self.sedm.xnorm_mat
        self.s = s.to(self.device)
        self.snorm_mat = snorm_mat.to(self.device)
        self.mode = 'sc'
        self.use_poisson = use_poisson

    def elbo_loss(self, batch, s, snorm_mat):
        x, xnorm_mat, *others = batch
        xz, qxz, xld, p, sld, theta_x, theta_s = self.vaesm(batch)
        print(f"theta_x raw output from vaesm: {theta_x}")
        print(f"Is theta_x NaN? {torch.isnan(theta_x).any()}")
        print(f"theta_x min: {torch.min(theta_x)}, max: {torch.max(theta_x)}")
        elbo_loss = 0
        if self.mode != 'sp':        
            # kld of pz and qz
            elbo_loss += calc_kld(qxz).sum()
            # reconst loss of x
            elbo_loss += calc_nb_loss(xld, xnorm_mat, theta_x, x).sum()
        if self.mode != 'sc':            
            # reconst loss of s
            if self.use_poisson:
                elbo_loss += calc_poisson_loss(sld, snorm_mat, s).sum()
            else:
                elbo_loss += calc_nb_loss(sld, snorm_mat, theta_s, s).sum()
        return(elbo_loss)
        
    def train_epoch(self):
        total_loss = 0
        entry_num = 0
        for batch in self.xedm.train_loader:
            batch = make_batch_to_device(batch, self.device)
            self.vaesm_optimizer.zero_grad()
            loss = self.elbo_loss(
                batch, self.s, self.snorm_mat)
            loss.backward()
            self.vaesm_optimizer.step()
            total_loss += loss.item()
            entry_num += batch[0].size(0)
        return total_loss / entry_num
        
    def evaluate(self, mode = 'test'):
        with torch.no_grad():
            self.vaesm.eval()
            if mode == 'test':            
                batch = make_batch_to_device(self.xedm.get_test_item(), self.device)
            else:
                batch = make_batch_to_device(self.xedm.get_validation_item(), self.device)
            x, xnorm_mat, *others = batch
            loss = self.elbo_loss(
                batch, self.s, self.snorm_mat)
            entry_num = x.shape[0]
            loss_val = loss / entry_num
        return(loss_val)

    def train_total(self, epoch_num, early_stop_patience=10, min_delta=0.01):
        self.vaesm.train()
        self.train_loss_list = []
        self.test_loss_list = []
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(epoch_num):
            train_loss = self.train_epoch()
            self.train_loss_list.append(train_loss)
            val_loss = self.evaluate(mode='validation')
            self.test_loss_list.append(val_loss)
    
            if epoch % 10 == 0:
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.vaesm.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                break
    
        if best_model_state is not None:
            self.vaesm.load_state_dict(best_model_state)
    def initialize_optimizer(self, lr):
        self.vaesm_optimizer = torch.optim.Adam(self.vaesm.parameters(), lr=lr)

    def initialize_loader(self, x_batch_size, s_batch_size):
        self.xedm.initialize_loader(x_batch_size)
        self.sedm.initialize_loader(s_batch_size)

    def mode_change(self, mode):
        self.mode = mode
        if mode == 'sc':
            self.vaesm.sc_mode()        
        if mode == 'sp':
            self.vaesm.sp_mode()        
        if mode == 'dual':
            self.vaesm.dual_mode()
            
    def plot_loss_curve(self):
        if not self.train_loss_list or not self.test_loss_list:
            print("Loss lists are empty. Train the model first.")
            return
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_loss_list, label="Train Loss")
        plt.plot(self.test_loss_list, label="Validation Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
class VaeSmExperimentMB(VaeSmExperiment):
    def __init__(self, model_params, lr, x, s, test_ratio, x_batch_size, s_batch_size, num_workers, validation_ratio=0.1, device='auto'):
        if device == 'auto':
            self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.sedm = VaeSmDataManagerMB(s, torch.zeros(s.shape[0]), test_ratio, s_batch_size, num_workers, validation_ratio=validation_ratio)
        self.xedm = VaeSmDataManagerMB(x, torch.zeros(x.shape[0]), test_ratio, x_batch_size, num_workers, validation_ratio=validation_ratio)
        self.model_params = model_params
        self.vaesm = VaeSm(self.sedm.x.size()[0], **self.model_params)
        self.vaesm.to(self.device)
        self.vaesm_optimizer = torch.optim.Adam(self.vaesm.parameters(), lr=lr)
        self.train_loss_list = []
        self.test_loss_list = []
        self.best_loss = None
        self.early_stop_count = 0
        self.early_stop_limit = 10
        self.mode = 'sc'
        self.use_poisson = False

    def elbo_loss(self, x, xnorm_mat, s, snorm_mat):
        xz, qxz, xld, p, sld, theta_x, theta_s = self.vaesm((x, xnorm_mat))
        elbo_loss = 0
        if self.mode != 'sp':
            elbo_loss += calc_kld(qxz).sum()
            elbo_loss += calc_nb_loss(xld, xnorm_mat, theta_x, x).sum()
        if self.mode != 'sc':
            if self.use_poisson:
                elbo_loss += calc_poisson_loss(sld, snorm_mat, s).sum()
            else:
                elbo_loss += calc_nb_loss(sld, snorm_mat, theta_s, s).sum()
        return elbo_loss

    def train_epoch(self):
        total_loss = 0
        entry_num = 0
        self.vaesm.train()
        for (x, xnorm_mat, x_batch_idx), (s, snorm_mat, s_batch_idx) in zip(self.xedm.train_loader, self.sedm.train_loader):
            x = x.to(self.device)
            xnorm_mat = xnorm_mat.to(self.device)
            s = s.to(self.device)
            snorm_mat = snorm_mat.to(self.device)
            self.vaesm_optimizer.zero_grad()
            loss = self.elbo_loss(x, xnorm_mat, s, snorm_mat)
            loss.backward()
            self.vaesm_optimizer.step()
            total_loss += loss.item()
            entry_num += x.shape[0]
        avg_loss = total_loss / entry_num
        self.train_loss_list.append(avg_loss)
        return avg_loss

    def evaluate(self, mode='test'):
        self.vaesm.eval()
        with torch.no_grad():
            if mode == 'test':
                x = self.xedm.test_x.to(self.device)
                xnorm_mat = self.xedm.test_xnorm_mat.to(self.device)
                s = self.sedm.test_x.to(self.device)
                snorm_mat = self.sedm.test_xnorm_mat.to(self.device)
            else:
                x = self.xedm.validation_x.to(self.device)
                xnorm_mat = self.xedm.validation_xnorm_mat.to(self.device)
                s = self.sedm.validation_x.to(self.device)
                snorm_mat = self.sedm.validation_xnorm_mat.to(self.device)
            loss = self.elbo_loss(x, xnorm_mat, s, snorm_mat)
            loss_val = loss.item() / x.shape[0]
            self.test_loss_list.append(loss_val)
            return loss_val

    def train_total(self, epoch_num, min_delta=0.01):
        for epoch in range(epoch_num):
            state_dict = copy.deepcopy(self.vaesm.state_dict())
            loss = self.train_epoch()
            if np.isnan(loss):
                self.vaesm.load_state_dict(state_dict)
                break
            val_loss = self.evaluate(mode='validation')
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}')

            # Early stopping
            if self.best_loss is None or val_loss < self.best_loss - min_delta:
                self.best_loss = val_loss
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1
                if self.early_stop_count >= self.early_stop_limit:
                    print("Early stopping triggered.")
                    break

    def plot_training_curve(self):
        plt.figure(figsize=(6,4))
        plt.plot(self.train_loss_list, label="Train Loss")
        plt.plot(self.test_loss_list, label="Validation Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
class VaeSmExperimentDPP(VaeSmExperiment):
    def __init__(self, gpu, gpu_num, model_params, lr, x, s, test_ratio, x_batch_size, s_batch_size, num_workers, validation_ratio=0.1, device=None):
        # Wrap the model
        self.device = torch.device(f'cuda:{gpu}')
        self.gpu = gpu
        torch.cuda.set_device(gpu)
        self.model_params = model_params
        self.vaesm = VaeSm(s.size()[0],  **self.model_params)
        self.vaesm.to(self.device)
        self.vaesm = nn.parallel.DistributedDataParallel(self.vaesm, device_ids=[gpu], find_unused_parameters=True)
        self.sedm = VaeSmDataManagerDPP(gpu, gpu_num, s, test_ratio, x_batch_size, num_workers, validation_ratio=validation_ratio)
        self.xedm = VaeSmDataManagerDPP(gpu, gpu_num, x, test_ratio, s_batch_size, num_workers, validation_ratio=validation_ratio)
        self.vaesm_optimizer = torch.optim.Adam(self.vaesm.parameters(), lr=lr)
        self.train_loss_list = []
        self.test_loss_list = []
        self.best_loss = None
        s = self.sedm.x
        snorm_mat = self.sedm.xnorm_mat
        self.s = s.to(self.device)
        self.snorm_mat = snorm_mat.to(self.device)
        self.mode = 'sc'

    def initialize_optimizer(self, lr):
        self.vaesm_optimizer = torch.optim.Adam(self.vaesm.parameters(), lr=lr)

    def initialize_loader(self, x_batch_size, s_batch_size):
        self.xedm.initialize_loader(x_batch_size)
        self.sedm.initialize_loader(s_batch_size)

    def mode_change(self, mode):
        state_dict = copy.deepcopy(self.vaesm.module.state_dict())
        self.vaesm = VaeSm(self.sedm.x.size()[0], **self.model_params)
        self.vaesm.load_state_dict(state_dict)
        self.vaesm.to(self.device)
        self.mode = mode
        if mode == 'sc':
            self.vaesm.sc_mode()        
        if mode == 'sp':
            self.vaesm.sp_mode()        
        if mode == 'dual':
            self.vaesm.dual_mode()
        self.vaesm = nn.parallel.DistributedDataParallel(self.vaesm, device_ids=[self.gpu], find_unused_parameters=True)
        
    def train_epoch(self, epoch=None):
        self.vaesm.train()
        total_loss = 0
        entry_num = 0
    
        # DDP のための set_epoch（GPU 1つでも問題なし）
        if hasattr(self.xedm, 'train_sampler') and self.xedm.train_sampler is not None:
            self.xedm.train_sampler.set_epoch(epoch if epoch is not None else 0)
        if hasattr(self.sedm, 'train_sampler') and self.sedm.train_sampler is not None:
            self.sedm.train_sampler.set_epoch(epoch if epoch is not None else 0)
    
        for (x, xnorm_mat, x_batch_idx), (s, snorm_mat, s_batch_idx) in zip(self.xedm.train_loader, self.sedm.train_loader):
            x = x.to(self.device)
            xnorm_mat = xnorm_mat.to(self.device)
            x_batch_idx = x_batch_idx.to(self.device)
            s = s.to(self.device)
            snorm_mat = snorm_mat.to(self.device)
            s_batch_idx = s_batch_idx.to(self.device)
    
            self.vaesm_optimizer.zero_grad()
    
            # モデルが DDP か単体かにかかわらず呼び方は同じ
            loss = self.elbo_loss(x, xnorm_mat, s, snorm_mat)
    
            if torch.isnan(loss):
                print("NaN detected in loss — skipping batch.")
                continue
    
            loss.backward()
            self.vaesm_optimizer.step()
    
            total_loss += loss.item()
            entry_num += x.size(0)
    
        avg_loss = total_loss / max(entry_num, 1)
        self.train_loss_list.append(avg_loss)
        return avg_loss
        
    def train_total(self, epoch_num, patience=20, min_delta=0.01):
        best_val_loss = float('inf')
        best_state_dict = copy.deepcopy(self.vaesm.module.state_dict() if isinstance(self.vaesm, nn.parallel.DistributedDataParallel) else self.vaesm.state_dict())
        patience_counter = 0
    
        for epoch in range(epoch_num):
            loss = self.train_epoch(epoch)
            if np.isnan(loss):
                print("NaN in loss, reverting to previous state.")
                self.vaesm.load_state_dict(best_state_dict)
                break
    
            val_loss = self.evaluate(mode='validation')
            self.test_loss_list.append(val_loss.item())
    
            print(f"[Epoch {epoch}] Train loss: {loss:.4f} | Validation loss: {val_loss:.4f}")
    
            # Early stopping logic
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(self.vaesm.module.state_dict() if isinstance(self.vaesm, nn.parallel.DistributedDataParallel) else self.vaesm.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  EarlyStopping counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    self.vaesm.load_state_dict(best_state_dict)
                    break
