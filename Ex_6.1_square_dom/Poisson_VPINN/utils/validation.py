import numpy as np
import matplotlib.pyplot as plt
import torch
from . import groundtruth as gt
from . import tools
from scipy.interpolate import griddata
import os

resolution = 200
val_x1 = np.arange(0, 1, (1)/resolution).reshape(-1, 1)
val_x2 = np.arange(0, 1, (1)/resolution).reshape(-1, 1)

dtype = torch.float64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# t_vx1 = torch.from_numpy(val_x1).type(dtype).to(device)
# t_vx2 = torch.from_numpy(val_x2).type(dtype).to(device)

val_ms_x1, val_ms_x2 = np.meshgrid(val_x1, val_x2)

plot_val_x1 = np.ravel(val_ms_x1).reshape(-1, 1)
plot_val_x2 = np.ravel(val_ms_x2).reshape(-1, 1)

t_val_vx1, t_val_vx2 = tools.from_numpy_to_tensor(
    [plot_val_x1, plot_val_x2], [True, True], dtype=dtype)


# make the following concatination an n by 2 tensor

#t_plot_grids = torch.concatenate([t_val_vx1,t_val_vx2], dim=1)

t_plot_grids = torch.concatenate(
    [t_val_vx1.unsqueeze(-1), t_val_vx2.unsqueeze(-1)], dim=-1)

gt_gen = gt.gt_gen()
y_gt = gt_gen.generate(gt_gen.y, np.concatenate(
    [plot_val_x1, plot_val_x2], axis=1)).reshape(-1, 1)
deri_y_gt_x1 = gt_gen.generate(
    gt_gen.y_x1, np.concatenate([plot_val_x1, plot_val_x2], axis=1)).reshape(-1, 1)
deri_y_gt_x2 = gt_gen.generate(
    gt_gen.y_x2, np.concatenate([plot_val_x1, plot_val_x2], axis=1)).reshape(-1, 1)


# compute H1 norm of y_gt: compute
y_gt_H1_norm = np.sqrt(np.mean(np.square(y_gt)) + np.mean(
    np.square(deri_y_gt_x1)) + np.mean(np.square(deri_y_gt_x2)))

# compute L2 norm of y_gt
y_gt_L2_norm = np.sqrt(np.mean(np.square(y_gt)))


mse_loss = torch.nn.MSELoss()



class recorder():
    def __init__(self):
        self.epoch = 0
        self.losslist = []
        self.rel_vallist = []
        self.abs_vallist = []

    def update_epoch(self):
        self.epoch += 1

    def update_hist(self, newloss):
        self.losslist.append(newloss)

    # def plot_data(self,data,colx,coly,name=None):
    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')
    #     ax.scatter3D(colx, coly,data , c=data, cmap='Greens')
    #     if name is None:
    #         plt.savefig('figure/plot{}.png'.format(self.epoch))
    #     else:
    #         plt.savefig(name)
    #     plt.close()

    def plot_data(self, data, colx, coly, name=None):
        # Convert inputs to 1D arrays if needed
        colx = np.ravel(colx)
        coly = np.ravel(coly)
        data = np.ravel(data)

        # Create a uniform grid
        grid_x, grid_y = np.mgrid[
            np.min(colx):np.max(colx):100j,
            np.min(coly):np.max(coly):100j
        ]

        # Interpolate scattered data to grid
        grid_z = griddata((colx, coly), data, (grid_x, grid_y), method='cubic')

        # Plot using pcolor format
        fig = plt.figure(figsize=(6, 5))
        plt.pcolor(grid_x, grid_y, grid_z, cmap='jet', shading='auto')
        h = plt.colorbar()
        h.ax.tick_params(labelsize=20)
        plt.xticks([])
        plt.yticks([])

        # Save the figure
        if name is None:
            plt.savefig('figure/plot{}.png'.format(self.epoch),
                        bbox_inches='tight')
        else:
            plt.savefig(name, bbox_inches='tight')
        plt.close()

    # def plot(self, net, name=None):
    #     net_out = net(t_plot_grids).detach().squeeze(-1).squeeze(-1).numpy()
    #     self.plot_data(net_out, plot_val_x1, plot_val_x2, name)

    def plot(self, net, name=None):
        with torch.no_grad():
            net_out = net(t_plot_grids).cpu().numpy().reshape(
                resolution, resolution)
        self.plot_data(net_out, plot_val_x1, plot_val_x2, name)

    def validate(self, net):
        # make sure inputs track gradients
        # t_plot_grids.requires_grad_(True)

        net_out_pred = net(t_plot_grids)
        
        # take derivative of net_out_pred with respect to t_val_vx1  
        
        
        net_out_pred_derivative_x1 = torch.autograd.grad(net_out_pred,t_val_vx1,grad_outputs=torch.ones_like(net_out_pred),create_graph=True)[0]
        net_out_pred_derivative_x2 = torch.autograd.grad(net_out_pred,t_val_vx2,grad_outputs=torch.ones_like(net_out_pred),create_graph=True)[0]
        
        
        net_out_pred = net_out_pred.detach().numpy().reshape(-1,1)
        net_out_pred_derivative_x1 = net_out_pred_derivative_x1.detach().numpy().reshape(-1,1)
        net_out_pred_derivative_x2 = net_out_pred_derivative_x2.detach().numpy().reshape(-1,1)
        
        #now you compute the H1 error between net_out_pred and t_ygt and the derivatives using the following type  formula      
         
         #H1_err_sol = np.sqrt(np.mean(np.square(y_pred-y_gt)) + np.mean(np.square(dy_dx-deri_y_gt_dx)) + np.mean(np.square(dy_dy-deri_y_gt_dy)))
    
        H1_error = np.sqrt(np.mean(np.square(net_out_pred - y_gt)) + np.mean(np.square(net_out_pred_derivative_x1 - deri_y_gt_x1)) + np.mean(np.square(net_out_pred_derivative_x2 - deri_y_gt_x2)))
        relative_H1_err = H1_error / y_gt_H1_norm
        
        # compute L2 error
        L2_error = np.sqrt(np.mean(np.square(net_out_pred - y_gt)))
        relative_L2_err = L2_error / y_gt_L2_norm
        
        self.plot(net)
        return relative_L2_err, L2_error, relative_H1_err, H1_error

# # new validation function

#     def validate(self, net):
#         net.eval()

#     # L2 part: no grad needed
#         with torch.no_grad():
#             pred_no_grad = net(t_plot_grids).squeeze(-1)
#             relative_L2_err = torch.sqrt(
#                 torch.mean((pred_no_grad - t_ygt)**2)) / y_L2
#             abs_err = torch.sqrt(torch.mean((pred_no_grad - t_ygt)**2))

#     # H1 part: need grads w.r.t. inputs — do a single grad-enabled pass
#         t_plot_grids_req = t_plot_grids.detach().clone().requires_grad_(True)
#         pred = net(t_plot_grids_req).squeeze(-1)

#         grads = torch.autograd.grad(
#             outputs=pred,
#             inputs=t_plot_grids_req,
#             grad_outputs=torch.ones_like(pred),
#             create_graph=False,
#             retain_graph=False,
#             allow_unused=False
#         )[0]                                # shape (N, 2)

#         gx1, gx2 = grads[..., 0], grads[..., 1]

#         H1_error = torch.sqrt(
#             torch.mean((pred - t_ygt)**2) +
#             torch.mean((gx1 - t_deri_y_gt_x1)**2) +
#             torch.mean((gx2 - t_deri_y_gt_x2)**2)
#         )
#         relative_H1_err = H1_error / y_H1_norm

#     # plot rarely outside if you like
#     # self.plot(net)

#         return (relative_L2_err.item(),
#             abs_err.item(),
#             relative_H1_err.item(),
#             H1_error.item())

    def print_info(self, net, val_int=100):
        if self.epoch % val_int == 0:
            val_L2_rel, val_L2_abs, relative_H1_err, H1_err = self.validate(
                net)
            self.rel_vallist.append((val_L2_rel, relative_H1_err))
            self.abs_vallist.append((val_L2_abs, H1_err))

        # self.plot_data(torch.abs((net(t_plot_grids)).detach().squeeze(-1).squeeze(-1)-t_ygt).numpy(),plot_val_x1,plot_val_x2,'error{}.png'.format(self.epoch))
            print("Epoch:", self.epoch, "Loss:", self.losslist[-1], "validation_rel:", val_L2_rel,
                  "validation_abs:", val_L2_abs, " H1_rel:", relative_H1_err, " H1_abs:", H1_err)


#             ##### NWE CoDE
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from . import groundtruth as gt
# from . import tools
# from scipy.interpolate import griddata
# import os

# resolution = 200
# val_x1 = np.arange(0, 1, 1/resolution).reshape(-1,1)
# val_x2 = np.arange(0, 1, 1/resolution).reshape(-1,1)

# dtype = torch.float64
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# t_vx1 = torch.from_numpy(val_x1).type(dtype).to(device)
# t_vx2 = torch.from_numpy(val_x2).type(dtype).to(device)

# val_ms_x1, val_ms_x2 = np.meshgrid(val_x1, val_x2)

# plot_val_x1 = np.ravel(val_ms_x1).reshape(-1,1)
# plot_val_x2 = np.ravel(val_ms_x2).reshape(-1,1)

# t_val_vx1, t_val_vx2 = tools.from_numpy_to_tensor([plot_val_x1, plot_val_x2], [False, False], dtype=dtype)

# t_plot_grids = torch.concatenate([t_val_vx1.unsqueeze(-1), t_val_vx2.unsqueeze(-1)], dim=-1)

# gt_gen = gt.gt_gen()
# y_gt = gt_gen.generate(gt_gen.y, np.concatenate([plot_val_x1, plot_val_x2], axis=1))
# deri_y_gt_x1 = gt_gen.generate(gt_gen.y_x1, np.concatenate([plot_val_x1, plot_val_x2], axis=1))
# deri_y_gt_x2 = gt_gen.generate(gt_gen.y_x2, np.concatenate([plot_val_x1, plot_val_x2], axis=1))

# t_ygt = torch.from_numpy(y_gt)
# t_deri_y_gt_x1 = torch.from_numpy(deri_y_gt_x1)
# t_deri_y_gt_x2 = torch.from_numpy(deri_y_gt_x2)

# mse_loss = torch.nn.MSELoss()
# y_L2 = torch.sqrt(torch.mean(torch.square(t_ygt)))
# # Compute H1 norm of ground truth
# y_H1 = torch.sqrt(torch.mean(torch.square(t_ygt)) +
#                  torch.mean(torch.square(t_deri_y_gt_x1)) +
#                  torch.mean(torch.square(t_deri_y_gt_x2)))

# class recorder():
#     def __init__(self):
#         self.epoch = 0
#         self.losslist = []
#         self.rel_vallist = []
#         self.abs_vallist = []
#         self.rel_H1_vallist = []  # New list for H1 relative errors
#         self.abs_H1_vallist = []  # New list for H1 absolute errors

#     def update_epoch(self):
#         self.epoch += 1

#     def update_hist(self, newloss):
#         self.losslist.append(newloss)

#     def plot_data(self, data, colx, coly, name=None):
#         colx = np.ravel(colx)
#         coly = np.ravel(coly)
#         data = np.ravel(data)

#         grid_x, grid_y = np.mgrid[
#             np.min(colx):np.max(colx):100j,
#             np.min(coly):np.max(coly):100j
#         ]

#         grid_z = griddata((colx, coly), data, (grid_x, grid_y), method='cubic')

#         fig = plt.figure(figsize=(6, 5))
#         plt.pcolor(grid_x, grid_y, grid_z, cmap='jet', shading='auto')
#         h = plt.colorbar()
#         h.ax.tick_params(labelsize=20)
#         plt.xticks([])
#         plt.yticks([])

#         if name is None:
#             plt.savefig('figure/plot{}.png'.format(self.epoch), bbox_inches='tight')
#         else:
#             plt.savefig(name, bbox_inches='tight')
#         plt.close()

#     def plot(self, net, name=None):
#         net_out = net(t_plot_grids).detach().squeeze(-1).squeeze(-1).numpy()
#         self.plot_data(net_out, plot_val_x1, plot_val_x2, name)

#     def validate(self, net):
#         # Enable gradient computation for input coordinates
#         t_plot_grids.requires_grad_(True)

#         # Forward pass
#         net_out = net(t_plot_grids).squeeze()

#         # Compute gradients
#         gradients = torch.autograd.grad(
#             outputs=net_out,
#             inputs=t_plot_grids,
#             grad_outputs=torch.ones_like(net_out),
#             create_graph=False,
#             retain_graph=True
#         )[0]

#         net_deri_x1 = gradients[:, 0]
#         net_deri_x2 = gradients[:, 1]

#         # Disable gradient tracking for error computation
#         with torch.no_grad():
#             # L2 errors
#             abs_L2_err = torch.sqrt(torch.mean(torch.square(net_out - t_ygt)))
#             rel_L2_err = abs_L2_err / y_L2

#             # H1 errors
#             abs_H1_err = torch.sqrt(
#                 torch.mean(torch.square(net_out - t_ygt)) +
#                 torch.mean(torch.square(net_deri_x1 - t_deri_y_gt_x1)) +
#                 torch.mean(torch.square(net_deri_x2 - t_deri_y_gt_x2))
#             )
#             rel_H1_err = abs_H1_err / y_H1

#         self.plot(net)
#         return (rel_L2_err.numpy(), abs_L2_err.numpy(),
#                 rel_H1_err.numpy(), abs_H1_err.numpy())

#     def print_info(self, net, val_int=100):
#         if self.epoch % val_int == 0:
#             val_L2_rel, val_L2_abs, val_H1_rel, val_H1_abs = self.validate(net)
#             self.rel_vallist.append(val_L2_rel)
#             self.abs_vallist.append(val_L2_abs)
#             self.rel_H1_vallist.append(val_H1_rel)  # Store H1 relative error
#             self.abs_H1_vallist.append(val_H1_abs)  # Store H1 absolute error

#             print(f"Epoch: {self.epoch} Loss: {self.losslist[-1]:.6f} "
#                   f"L2_rel: {val_L2_rel:.6f} L2_abs: {val_L2_abs:.6f} "
#                   f"H1_rel: {val_H1_rel:.6f} H1_abs: {val_H1_abs:.6f}")
