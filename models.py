import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from torch.nn.functional import mse_loss
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def create_net(in_dim, out_dim, depth, width, activation=torch.nn.Tanh()):
        list_modules = []
        for i in range(depth):
            if i == 0:
                layer = torch.nn.Linear(in_dim, width)
                xavier_std = np.sqrt(2 / (in_dim+width))
                list_modules.append(layer)
                list_modules.append(activation)
            elif i==depth-1:
                layer = torch.nn.Linear(width, out_dim)
                xavier_std = np.sqrt(2 / (width+out_dim))
                list_modules.append(layer)
            else:
                layer = torch.nn.Linear(width, width)
                xavier_std = np.sqrt(2 / (width + width))
                list_modules.append(layer)
                list_modules.append(activation)
            torch.nn.init.normal_(layer.weight, mean=0.0, std=xavier_std)
        model = torch.nn.Sequential(*list_modules)
        return model

def grad(output, var, create_graph=True, retain_graph=True):
    return torch.autograd.grad(output, var, torch.ones_like(output), create_graph=create_graph, retain_graph=retain_graph)[0]
    
def compute_gradients(output, x, y, create_graph=True, retain_graph=True):
    return grad(output, x, create_graph=create_graph, retain_graph=retain_graph), \
            grad(output, y, create_graph=create_graph, retain_graph=retain_graph)

def get_grid(x,y):
    xg, yg = torch.meshgrid((x, y), indexing='xy')
    xg = xg.flatten().to(device)
    yg = yg.flatten().to(device)
    return xg, yg
    
class PINN(nn.Module):
    def __init__(self, in_dim, out_dim, data, depth, width, my_lr=0.001, rff_dim=6, sigma=3):
        super(PINN, self).__init__()
        self.model = create_net(in_dim if rff_dim is None else 2*rff_dim, out_dim, depth, width, activation=torch.nn.SiLU())
        
        # Misc functions
        self.sum_square = lambda x: torch.sum(torch.square(x))
        self.logcosh_loss = lambda x,y: torch.mean(torch.log(torch.cosh(x - y)))

        # Training data
        self.data = data
        self.pos_train = data[0]
        self.sol_train = data[1]
        self.x_train = self.pos_train[:,0]
        self.y_train = self.pos_train[:,1]
        self.rff_dim = rff_dim
        if rff_dim:
            self.B = sigma*torch.randn((rff_dim, in_dim)).to(device)
            
        # Self-annealing weights
        self.lambda_data = 1. 
        self.lambda_bc = 1.
        self.lambda_pde = 1.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=my_lr)

        # SAMPLE SIZES 
        self.BC_SAMPLE_SIZE = 50
        self.PDE_R_SAMPLE_SIZE = 20000
        self.der_grid_res = 600

        # Variables for imposing boundary conditions
        self.top_boundary = torch.ones(self.BC_SAMPLE_SIZE).to(device)
        self.bot_boundary = torch.zeros(self.BC_SAMPLE_SIZE).to(device)
        self.left_boundary = torch.zeros(self.BC_SAMPLE_SIZE).to(device)
        self.right_boundary = torch.ones(self.BC_SAMPLE_SIZE).to(device)

        # Variables for gradient estimation
        self.x_pde, self.y_pde = torch.linspace(0,1,self.der_grid_res).to(device), torch.linspace(0,1,self.der_grid_res).to(device)
        self.xg_pde, self.yg_pde = get_grid(self.x_pde,self.y_pde)
        self.dr = self.x_pde[1] - self.x_pde[0] # spatial increments

        # Initialize losses
        self.L_bc = torch.tensor([0.]).float()

    def forward(self, x, y):
        if not self.rff_dim is None:
            pos = torch.stack((x,y))
            vp = 2 * torch.pi * self.B@pos
            pos = torch.concat((torch.sin(vp), torch.cos(vp)), dim=0).T
        else:
            pos = torch.stack((x,y)).T
        uvp = self.model(pos)
        return uvp

    def compute_derivatives(self):
        # Esimate Gradients with finite difference
        sol_pde = self.forward(self.xg_pde, self.yg_pde)
        sol_pde = sol_pde.reshape(self.der_grid_res, self.der_grid_res, 3)
        x_der = (sol_pde - torch.roll(sol_pde, shifts=1, dims=1)) / self.dr
        y_der = (sol_pde - torch.roll(sol_pde, shifts=1, dims=0)) / self.dr
        x_der_2 = (torch.roll(sol_pde, shifts=1, dims=1) + torch.roll(sol_pde, shifts=-1, dims=1) - 2*sol_pde) / self.dr**2
        y_der_2 = (torch.roll(sol_pde, shifts=1, dims=0) + torch.roll(sol_pde, shifts=-1, dims=0) - 2*sol_pde) / self.dr**2

        # Remove the border to avoid large outlier artifacts from the boundaries
        x_der = x_der[1:-1, 1:-1, :]
        x_der_2 = x_der_2[1:-1, 1:-1, :]
        y_der = y_der[1:-1, 1:-1, :]
        y_der_2 = y_der_2[1:-1, 1:-1, :]
        sol_pde = sol_pde[1:-1, 1:-1]
        self.u_hat_pde = sol_pde[:,:,0].flatten()
        self.v_hat_pde = sol_pde[:,:,1].flatten()
        self.p_hat_pde = sol_pde[:,:,2].flatten()

        # Store the results of computation
        self.u_x, self.u_y = x_der[:,:,0].flatten(), y_der[:,:,0].flatten()
        self.v_x, self.v_y = x_der[:,:,1].flatten(), y_der[:,:,1].flatten()
        self.p_x, self.p_y = x_der[:,:,2].flatten(), y_der[:,:,2].flatten()
        self.u_xx, self.u_yy = x_der_2[:,:,0].flatten(), y_der_2[:,:,0].flatten()
        self.v_xx, self.v_yy = x_der_2[:,:,1].flatten(), y_der_2[:,:,1].flatten()  

    def compute_boundaries(self, sample_size):
        self.xx = torch.rand(self.BC_SAMPLE_SIZE).to(device)
        self.yy = torch.rand(self.BC_SAMPLE_SIZE).to(device)

        self.top_sol = self.forward(self.xx, self.top_boundary)
        self.bot_sol = self.forward(self.xx, self.bot_boundary)
        
        self.left_sol = self.forward(self.left_boundary, self.yy)
        self.right_sol = self.forward(self.right_boundary, self.yy)

    def compute_pde_residues(self, random=True):
        pde_r_x = self.u_hat_pde*self.u_x + self.v_hat_pde*self.u_y + self.p_x - (self.u_xx + self.u_yy)/100
        pde_r_y = self.u_hat_pde*self.v_x + self.v_hat_pde*self.v_y + self.p_y - (self.v_xx + self.v_yy)/100
        pde_r_incompress = self.u_x + self.v_y
        residue_rand_pts = torch.randint(0, len(pde_r_x), (self.PDE_R_SAMPLE_SIZE,))
        if random:
            # self.L_pde = self.logcosh_loss(torch.abs(pde_r_x[residue_rand_pts]) + torch.abs(pde_r_y[residue_rand_pts])\
            #               + torch.abs(pde_r_incompress[residue_rand_pts]), 0) 
            self.L_pde = torch.mean(pde_r_x[residue_rand_pts]**2) + torch.mean(pde_r_y[residue_rand_pts]**2)\
                          + torch.mean(pde_r_incompress[residue_rand_pts]**2) 
        else:
            self.L_pde = self.logcosh_loss(torch.abs(pde_r_x) + torch.abs(pde_r_y) + torch.abs(pde_r_incompress), 0)
            
    def compute_bc_loss(self):
        self.L_bc = self.logcosh_loss((torch.abs(self.top_sol[:,0] - 1) \
                + torch.abs(self.left_sol[:,0]) + torch.abs(self.right_sol[:,0]) + torch.abs(self.bot_sol[:,0]) \
                + torch.abs(self.top_sol[:,1]) + torch.abs(self.left_sol[:,1]) 
                + torch.abs(self.right_sol[:,1]) + torch.abs(self.bot_sol[:,1])), 0)
        
    def train(self, num_epochs, losses=None, adapt_weights=False, F=250):
        ALPHA=0.9
        
        for epoch in (pbar := tqdm(range(num_epochs))):
            self.optimizer.zero_grad()
            
            # =========================== PDE Equation Loss =============================
            self.compute_derivatives()
            self.compute_pde_residues(random=True)
                
            # ======================= Boundary Condition Loss =======================
            if not self.recon:
                self.compute_boundaries(self.BC_SAMPLE_SIZE)
                self.compute_bc_loss()
            
            # ========================== Data loss =============================
            with torch.no_grad():
                sol_hat = self.forward(self.x_train, self.y_train)
                L_u = mse_loss(sol_hat[:,0], self.sol_train[:,0])
                L_v = mse_loss(sol_hat[:,1], self.sol_train[:,1])
                L_p = mse_loss(sol_hat[:,2], self.sol_train[:,2])

            sol_recon_hat = self.forward(self.x_recon, self.y_recon)
            self.L_recon = mse_loss(sol_recon_hat[:,0], self.sol_recon[:,0])\
                         + mse_loss(sol_recon_hat[:,1], self.sol_recon[:,1])\
                         + mse_loss(sol_recon_hat[:,2], self.sol_recon[:,2])

            # ============================ Aggregate Losses ===============================
            # if epoch%F==0 and adapt_weights:
            #     self.L_pde.retain_grad()
            #     self.L_bc.retain_grad()
                
            #     # Balance loss function from time to time
            #     w_pde, w_bc, w_data = 0., 0., 0.
            #     L_pde_grad = torch.autograd.grad(self.L_pde, self.model.parameters(), retain_graph=True)
            #     for g in L_pde_grad:
            #         w_pde += torch.linalg.vector_norm(g).item()
                    
            #     L_bc_grad = torch.autograd.grad(self.L_bc, self.model.parameters(), retain_graph=True)
            #     for g in L_bc_grad:
            #         w_bc += torch.linalg.vector_norm(g).item()
      
            #     print("WEIGHTS:", w_pde, w_bc, w_data)
                
            #     lambda_pde_hat = (w_bc + w_pde) / w_pde
            #     lambda_bc_hat = (w_bc + w_pde) / w_bc
            #     self.lambda_pde = ALPHA*self.lambda_pde + (1-ALPHA)*lambda_pde_hat
            #     self.lambda_bc = ALPHA*self.lambda_bc + (1-ALPHA)*lambda_bc_hat
            #     print("LAMBDAs:", self.lambda_pde, self.lambda_bc, self.lambda_data)
            
            if self.recon:
                # loss = self.L_recon + self.lambda_pde*self.L_pde
                loss = self.L_recon + self.lambda_pde*self.L_pde
            else:
                loss = self.lambda_pde*self.L_pde + self.lambda_bc*self.L_bc
            loss.backward()
            self.optimizer.step()

            # Record and update losses
            if not losses is None and self.recon:
                losses[epoch, :] = [self.L_pde.item(), self.L_recon.item()]
            else:
                losses[epoch, :] = [self.L_pde.item(), self.L_bc.item(), L_u.item(), L_v.item(), L_p.item()]
                
            if epoch%100==0:
                if self.recon: 
                    print(f"EPOCH {epoch}: LOSS={np.round(loss.item(),9)}, PDE RES={np.round(self.L_pde.item(),9)} | RECON LOSS={np.round(self.L_recon.item(),9)}  | LOSS u,v,p=({np.round(L_u.item(),4)}, {np.round(L_v.item(),4)}, {np.round(L_p.item(),4)})")
                    
                else:
                    print(f"EPOCH {epoch}: LOSS={np.round(self.L_pde.item() + self.L_bc.item(), 4)} | LOSS u,v,p=({np.round(L_u.item(),4)}, {np.round(L_v.item(),4)}, {np.round(L_p.item(),4)})")

class ReconPINN(nn.Module):
    def __init__(self, in_dim, out_dim, data, depth, width, optimizer='adam', loss_fn='mse', recon_num_obs=10, my_lr=0.001):
        super(ReconPINN, self).__init__()
        self.model = create_net(in_dim, out_dim, depth, width, activation=torch.nn.SiLU())
        
        # Training data
        self.data = data
        self.pos_train = data[0]
        self.sol_train = data[1]
        self.loss_fn = loss_fn

        self.logcosh = lambda x: torch.mean(torch.log(torch.cosh(x)))

        # Recon data
        self.recon_num_obs = recon_num_obs
        obs_idx = (self.pos_train[:,0]>=0.2) & (self.pos_train[:,0]<=0.8) & (self.pos_train[:,1]>=0.2) & (self.pos_train[:,1]<=0.8)
        self.pos_recon = self.pos_train[obs_idx, :]
        self.sol_recon = self.sol_train[obs_idx, :]
        rand_idx = torch.randint(0, len(self.pos_recon), (recon_num_obs,))
        self.pos_recon = self.pos_recon[rand_idx].detach().clone()
        self.sol_recon = self.sol_recon[rand_idx].detach().clone()
        
        self.x_train = self.pos_train[:,0]
        self.y_train = self.pos_train[:,1]
        self.x_recon = self.pos_recon[:,0]
        self.y_recon = self.pos_recon[:,1]

        if optimizer=="lbfgs":
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=my_lr)
        elif optimizer=="adam": 
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=my_lr)

        # SAMPLE SIZES
        self.PDE_R_SAMPLE_SIZE = 20000
        self.der_grid_res = 600
        
        # Variables for gradient estimation
        self.x_pde, self.y_pde = torch.linspace(0,1,self.der_grid_res).to(device), torch.linspace(0,1,self.der_grid_res).to(device)
        self.xg_pde, self.yg_pde = get_grid(self.x_pde,self.y_pde)
        self.dr = self.x_pde[1] - self.x_pde[0] # spatial increments

    def forward(self, x, y):
        pos = torch.stack((x,y)).T
        uvp = self.model(pos)
        return uvp

    def compute_derivatives(self):
        # Esimate Gradients with finite difference
        sol_pde = self.forward(self.xg_pde, self.yg_pde)
        sol_pde = sol_pde.reshape(self.der_grid_res, self.der_grid_res, 3)
        x_der = (sol_pde - torch.roll(sol_pde, shifts=1, dims=1)) / self.dr
        y_der = (sol_pde - torch.roll(sol_pde, shifts=1, dims=0)) / self.dr
        x_der_2 = (torch.roll(sol_pde, shifts=1, dims=1) + torch.roll(sol_pde, shifts=-1, dims=1) - 2*sol_pde) / self.dr**2
        y_der_2 = (torch.roll(sol_pde, shifts=1, dims=0) + torch.roll(sol_pde, shifts=-1, dims=0) - 2*sol_pde) / self.dr**2

        # Remove the border to avoid large outlier artifacts from the boundaries
        x_der = x_der[1:-1, 1:-1, :]
        x_der_2 = x_der_2[1:-1, 1:-1, :]
        y_der = y_der[1:-1, 1:-1, :]
        y_der_2 = y_der_2[1:-1, 1:-1, :]
        sol_pde = sol_pde[1:-1, 1:-1]
        self.u_hat_pde = sol_pde[:,:,0].flatten()
        self.v_hat_pde = sol_pde[:,:,1].flatten()
        self.p_hat_pde = sol_pde[:,:,2].flatten()

        # Store the results of computation
        self.u_x, self.u_y = x_der[:,:,0].flatten(), y_der[:,:,0].flatten()
        self.v_x, self.v_y = x_der[:,:,1].flatten(), y_der[:,:,1].flatten()
        self.p_x, self.p_y = x_der[:,:,2].flatten(), y_der[:,:,2].flatten()
        self.u_xx, self.u_yy = x_der_2[:,:,0].flatten(), y_der_2[:,:,0].flatten()
        self.v_xx, self.v_yy = x_der_2[:,:,1].flatten(), y_der_2[:,:,1].flatten()  

    def compute_pde_residues(self, random=True):
        pde_r_x = self.u_hat_pde*self.u_x + self.v_hat_pde*self.u_y + self.p_x - (self.u_xx + self.u_yy)/100
        pde_r_y = self.u_hat_pde*self.v_x + self.v_hat_pde*self.v_y + self.p_y - (self.v_xx + self.v_yy)/100
        pde_r_incompress = self.u_x + self.v_y
        if random:
            residue_rand_pts = torch.randint(0, len(pde_r_x), (self.PDE_R_SAMPLE_SIZE,))
            if self.loss_fn == 'mse':
                self.L_pde = torch.mean(pde_r_x[residue_rand_pts]**2) + torch.mean(pde_r_y[residue_rand_pts]**2)\
                              + torch.mean(pde_r_incompress[residue_rand_pts]**2) 
            elif self.loss_fn == 'logcosh':
                self.L_pde = self.logcosh(pde_r_x[residue_rand_pts]) + self.logcosh(pde_r_y[residue_rand_pts])\
                              + self.logcosh(pde_r_incompress[residue_rand_pts])
        else:
            if self.loss_fn == 'mse':
                self.L_pde = torch.mean(pde_r_x**2) + torch.mean(pde_r_y**2) + torch.mean(pde_r_incompress**2)
            elif self.loss_fn == 'logcosh':
                self.L_pde = self.logcosh(pde_r_x) + self.logcosh(pde_r_y) + self.logcosh(pde_r_incompress)

    def print_losses(self, epoch):
        with torch.no_grad():
            self.sol_hat = self.forward(self.x_train, self.y_train)
            self.L_u = mse_loss(self.sol_hat[:,0], self.sol_train[:,0])
            self.L_v = mse_loss(self.sol_hat[:,1], self.sol_train[:,1])
            self.L_p = mse_loss(self.sol_hat[:,2], self.sol_train[:,2])
        print(f"EPOCH {epoch}, LOSS={np.round(self.loss.item(),9)}, PDE RES={np.round(self.L_pde.item(),9)} | LOSS u,v,p=({np.round(self.L_u.item(),4)}, {np.round(self.L_v.item(),4)}, {np.round(self.L_p.item(),4)})")    
    
    def train(self, max_epoch, tol=5e-5, losses=None):
        
        ALPHA=0.9

        self.loss = torch.inf
        def closure():
            self.optimizer.zero_grad()
            
            # =========================== PDE Equation Loss =============================
            self.compute_derivatives()
            self.compute_pde_residues(random=False)
            
            # ========================== Recon Data loss =============================
            self.sol_recon_hat = self.forward(self.x_recon, self.y_recon)
            if self.loss_fn == 'mse':
                self.L_recon = torch.mean((self.sol_recon_hat[:,0]-self.sol_recon[:,0])**2\
                         + (self.sol_recon_hat[:,1] - self.sol_recon[:,1])**2)
            elif self.loss_fn == 'logcosh':
                self.L_recon = self.logcosh(self.sol_recon_hat[:,0] - self.sol_recon[:,0])\
                           + self.logcosh(self.sol_recon_hat[:,1] - self.sol_recon[:,1])

            self.loss = self.L_recon + self.L_pde
            self.loss.backward()
            return self.loss
        epoch = 0
        with tqdm(range(max_epoch), 
                  desc=f"Recon:{self.recon_num_obs} samples, loss={self.loss_fn}.") as pbar:
            while self.loss > tol and epoch<max_epoch:
                self.optimizer.step(closure=closure)
            
                if epoch%1 == 0:
                    self.print_losses(epoch)
                epoch += 1
                pbar.update(1)
        if epoch!=max_epoch:
            print("Model Converged to Tolerance.")
        self.print_losses(epoch)
        
def train_field_recon_PINN(recon_num_obs, pos_train, sol_train, 
                           optimizer='adam', tol=5e-5, epochs=200, lr=0.1, loss_fn='mse', save=True):
    # torch.random.manual_seed(4365)
    recon_model = ReconPINN(in_dim=2, out_dim=3, data=(pos_train, sol_train), 
                            recon_num_obs=recon_num_obs, optimizer=optimizer,
                            my_lr=lr, loss_fn=loss_fn, depth=9, width=20)
    recon_model.to(device);
    torch.cuda.empty_cache()
    recon_model.train(epochs, tol=tol)
    if save:
        torch.save(recon_model.state_dict(), f"recon-models/recon-fields-{recon_num_obs}-pts-{loss_fn}-tol.model")
        
def eval_model_field(model, x, y, use_grid=True):
    if use_grid:
        xg, yg = get_grid(x,y)
        uvp_hat = model(xg, yg)
        uvp_hat = uvp_hat.reshape(len(y), len(x), 3)
    else:
        uvp_hat = model(x,y)
        uvp_hat.reshape(int(np.sqrt(len(y))), int(np.sqrt(len(x))), 3)
    u_hat = uvp_hat[:,:,0].detach().cpu().numpy()
    v_hat = uvp_hat[:,:,1].detach().cpu().numpy()
    p_hat = uvp_hat[:,:,2].detach().cpu().numpy()
    if use_grid:
        return xg.reshape(len(y), len(x)), yg.reshape(len(y), len(x)), u_hat, v_hat, p_hat
    else:
        return x,y, u_hat, v_hat, p_hat

def plot_fields(u, v, p):
    fig, axs = plt.subplots(1,3, figsize=(12, 5))
    cmap = plt.get_cmap('PuOr', 25)
    cmap2= plt.get_cmap('PuOr', 10)
    v_im = axs[1].imshow(v, origin='lower', cmap=cmap)
    u_im = axs[0].imshow(u, origin='lower', cmap=cmap)
    p_im = axs[2].imshow(p, origin='lower', cmap=cmap2)
                         
    plt.colorbar(u_im, ax=axs[0], fraction=0.045, pad=0.04)
    plt.colorbar(v_im, ax=axs[1],fraction=0.045, pad=0.04)
    plt.colorbar(p_im, ax=axs[2], fraction=0.045, pad=0.04)
    TITLES = ["$u$", "$v$", "$p$"]
    for i in range(3):
        axs[i].set_title(TITLES[i])
    plt.tight_layout()
    return fig

def recon_field(recon_model):
    with torch.no_grad():
        sol_hat = recon_model(recon_model.x_train, recon_model.y_train)
        u_err = mse_loss(sol_hat[:,0], recon_model.sol_train[:,0]) / torch.mean(torch.square(recon_model.sol_train[:,0]))
        v_err = mse_loss(sol_hat[:,1], recon_model.sol_train[:,1]).item() / torch.mean(torch.square(recon_model.sol_train[:,1]))
        p_err = mse_loss(sol_hat[:,2] - sol_hat[0,2], recon_model.sol_train[:,2]).item() / torch.mean(torch.square(recon_model.sol_train[:,2]))
        print("REL L2 ERRS:", np.round(u_err.item(),3), np.round(v_err.item(),3), np.round(p_err.item(),3))

        # Plot reconstructed field
        NUM_PTS = 3000
        x_test, y_test = torch.linspace(0,1,NUM_PTS).to(device), torch.linspace(0,1,NUM_PTS).to(device)
        x, y, u_hat, v_hat, p_hat = eval_model_field(recon_model, x_test, y_test)
        # plot_fields(u_hat, v_hat, p_hat)
        return x,y,u_hat, v_hat, p_hat