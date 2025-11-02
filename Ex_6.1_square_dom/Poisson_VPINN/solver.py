import torch
import numpy as np
import torch.optim as opt
import pickle
import pandas as pd
from utils import model,groundtruth,tools,validation,pde
import pickle as pkl
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from time import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


torch.set_default_dtype(torch.float64)

max_iter = 10000


if not os.path.exists('figure/'):
    os.makedirs("figure/")


def loader(path,format='float64'):
    data = np.array(pd.read_fwf(path,header=None)).astype(format)
    return data

data_el = loader('data/mesh_elems.dat',int)-1
data_cr = loader('data/mesh_coord.dat')

nrElem = len(data_el)
#aug_cd = np.concatenate([np.ones([len(data_cr),1]),data_cr],axis=1)

mesh = tools.mesh(data_cr,data_el)
data_gt = groundtruth.gt_gen()
quad_dict = mesh.get_full_quad()


qp = torch.tensor(quad_dict['quadrature_points'],requires_grad=True)
w = torch.tensor(quad_dict['weight']).unsqueeze(-1)
cm = torch.tensor(quad_dict['coef_mat'])
print(qp.shape,w.shape,cm.shape)

rec = validation.recorder()




#Train boundary extension
bdry_col = torch.tensor(mesh.crd[~np.isin(np.arange(mesh.nrNode),mesh.domain_mark),:])
x1,x2 = mesh.crd[mesh.domain_mark,:].T
#t_x1,t_x2 = tools.from_numpy_to_tensor([x1,x2],[True,True])
#t_x1 = t_x1.unsqueeze(-1)
#t_x2 = t_x2.unsqueeze(-1)
print("boundary collocation:",bdry_col.shape)
bdrydat = torch.tensor(data_gt.generate(data_gt.bdry,bdry_col.detach().numpy())).unsqueeze(-1)

#rec.plot_data(bdrydat,bdry_col[:,0],bdry_col[:,1],'gtbdrt.png')

ub = model.NNbdry()
optimizer_bdry = torch.optim.Adam(ub.parameters(),lr=1e-3)
def bdry_closure():
    optimizer_bdry.zero_grad()
    bdry_pred = ub(bdry_col)
    #print(bdry_pred.shape,bdrydat.shape)
    #lap_pred = pde.laplacian(t_x1,t_x2,ub)
    loss_bdry = torch.mean(torch.square(bdry_pred-bdrydat))
    #loss_lap = torch.mean(torch.square(lap_pred))
    #loss = loss_lap + 100*loss_bdry
    loss_bdry.backward()
    return loss_bdry.detach().numpy()


print("Start training boundary extension...")
for epoch in range(4000):
    loss_bdry = bdry_closure()
    optimizer_bdry.step()
    if epoch%1000 == 0:
        print("bdry training at epoch:",epoch,"L2 error:",np.sqrt(loss_bdry))

ub.eval()
rec.plot(ub)
print('Boundary extension training finished.')


#Train PDE
u_net = model.NNsol()
def u(qp):
    return u_net(qp,ub)

#Caution: including ub as attribute of u, then parameters of ub are appended to parameters of u. ub will change when optimizing u.
#Hence ub as a fixed boundary extension should be passed to u as input variable of forward function, instead of as attribute.

optimizer = torch.optim.Adam(u_net.parameters(),lr=1e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5000,10000,15000],gamma=0.1)

f = torch.tensor(data_gt.generate(data_gt.f,quad_dict['quadrature_points'].reshape(-1,2)).reshape(mesh.nrElem,6,1))
print(f.shape)
#f is correct after reshape. Checked.

rec.plot(u,'figure/bdry.png')
with torch.no_grad():
    rhs = pde.pde_rhs(f,qp,w,cm)
    rhs_assemble = mesh.assemble(rhs)

def hook(optimizer,nploss):
    rec.update_epoch()
    rec.update_hist(nploss)
    rec.print_info(u)
    if rec.epoch % 100 ==0:
        rec.plot(ub,"figure/bdrynet.png")

def closure():
    optimizer.zero_grad()

    lhs = pde.pde_lhs(u,qp,w,cm)
    lhs_assemble= mesh.assemble(lhs)
    loss = torch.sum(torch.square((lhs_assemble-rhs_assemble)[mesh.domain_mark]))
    loss.backward()

    return loss.detach().numpy()

print("Start training PDE...")



start_time = time()

losslist = list()
#optimizer.step(closure)
for _ in range(max_iter):
    loss = closure()
    losslist.append(loss)
    optimizer.step()
    hook(optimizer,loss)
    
    
end_time = time()
elapsed_time = end_time - start_time
print(f"Elapsed time for PDE training: {elapsed_time:.2f} seconds")    

print("PDE training finished.")

with open("loss.pkl",'wb') as pfile:
    pkl.dump(losslist,pfile)












# # Compute gradient of neural network solution
# pt_x = Variable(torch.from_numpy(x_pts).float(), requires_grad=True)
# pt_t = Variable(torch.from_numpy(t_pts).float(), requires_grad=True)

# # Forward pass
# pt_u_nn = y(pt_x, pt_t)

# # Compute gradients
# grad_u_nn_x = torch.autograd.grad(pt_u_nn, pt_x, grad_outputs=torch.ones_like(pt_u_nn), 
#                                  create_graph=True)[0]
# grad_u_nn_t = torch.autograd.grad(pt_u_nn, pt_t, grad_outputs=torch.ones_like(pt_u_nn), 
#                                  create_graph=True)[0]

# # Convert to numpy
# grad_u_nn_x = grad_u_nn_x.data.cpu().numpy()
# grad_u_nn_t = grad_u_nn_t.data.cpu().numpy()



# deri_ugt_x1 = deri_ugt_x1.flatten()
# deri_ugt_x2 = deri_ugt_x2.flatten()


# # Compute L2 norm of gradient difference
# grad_error_squared = np.mean((deri_ugt_x1 - grad_u_nn_x)**2 + (deri_ugt_x2 - grad_u_nn_t)**2)

# # Compute H1 error
# h1_error = np.sqrt(l2_error**2 + grad_error_squared)

# # H1 relative error (need to compute H1 norm of ground truth)
# h1_norm_gt = np.sqrt(np.mean(ms_ugt_flat**2) + np.mean(deri_ugt_x1**2 + deri_ugt_x2**2))
# h1_relative_error = h1_error / h1_norm_gt

# print(f"H1 Error: {h1_error}")
# print(f"H1 Relative Error: {h1_relative_error}")



# #### Faster version of plotting code
# batch_size = 1000  # Tune this for your GPU/CPU memory
# n_points = x_pts.shape[0]
# grad_u_nn_x = []
# grad_u_nn_t = []

# for i in range(0, n_points, batch_size):
#     bx = x_pts[i:i+batch_size]
#     bt = t_pts[i:i+batch_size]
#     pt_x = Variable(torch.from_numpy(bx).float(), requires_grad=True)
#     pt_t = Variable(torch.from_numpy(bt).float(), requires_grad=True)
#     pt_u_nn = y(pt_x, pt_t)
#     grads = torch.autograd.grad(
#         pt_u_nn, [pt_x, pt_t],
#         grad_outputs=torch.ones_like(pt_u_nn),
#         create_graph=False, retain_graph=False
#     )
#     grad_u_nn_x.append(grads[0].detach().cpu().numpy())
#     grad_u_nn_t.append(grads[1].detach().cpu().numpy())

# grad_u_nn_x = np.concatenate(grad_u_nn_x).flatten()
# grad_u_nn_t = np.concatenate(grad_u_nn_t).flatten()





# deri_ugt_x1 = deri_ugt_x1.flatten()
# deri_ugt_x2 = deri_ugt_x2.flatten()


# # Compute L2 norm of gradient difference
# grad_error_squared = np.mean((deri_ugt_x1 - grad_u_nn_x)**2 + (deri_ugt_x2 - grad_u_nn_t)**2)

# # Compute H1 error
# h1_error = np.sqrt(l2_error**2 + grad_error_squared)

# # H1 relative error (need to compute H1 norm of ground truth)
# h1_norm_gt = np.sqrt(np.mean(ms_ugt_flat**2) + np.mean(deri_ugt_x1**2 + deri_ugt_x2**2))
# h1_relative_error = h1_error / h1_norm_gt

# print(f"H1 Error: {h1_error}")
# print(f"H1 Relative Error: {h1_relative_error}")


