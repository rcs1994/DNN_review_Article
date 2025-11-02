from time import time
from tracemalloc import start
import numpy as np
import torch
import torch.optim as opt
import matplotlib.pyplot as plt
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as Dataloader
from torch.autograd import Variable
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import model,pde,data,tools,g_tr,validation
from time import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


torch.set_default_dtype(torch.float32)

y = model.NN()
y.apply(model.init_weights)

dataname = '8000pts'
name = 'results/'

bw = 700

if not os.path.exists(name):
    os.makedirs(name)

if not os.path.exists(name+"y_plot/"):
    os.makedirs(name+"y_plot/")


params = list(y.parameters())



with open("dataset/"+dataname,'rb') as pfile:
    int_col = pkl.load(pfile)
    bdry_col = pkl.load(pfile)

print(int_col.shape,bdry_col.shape)

intx1,intx2 = np.split(int_col,2,axis=1)
bdx1,bdx2 = np.split(bdry_col,2,axis=1)

tintx1,tintx2,tbdx1,tbdx2 = tools.from_numpy_to_tensor([intx1,intx2,bdx1,bdx2],[True,True,False,False,],dtype=torch.float32)


with open("dataset/gt_on_{}".format(dataname),'rb') as pfile:
    y_gt = pkl.load(pfile)
    f_np = pkl.load(pfile)
    bdry_np = pkl.load(pfile)

f,bdrydat,ygt = tools.from_numpy_to_tensor([f_np,bdry_np,y_gt],[False,False,False],dtype=torch.float32)


optimizer = opt.Adam(params,lr=1e-4)

mse_loss = torch.nn.MSELoss()

scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer,patience=500)
loader = torch.utils.data.DataLoader([intx1,intx2],batch_size = 500,shuffle = True)


def closure():
    tot_loss = 0
    for i,subquad in enumerate(loader):
       optimizer.zero_grad()
       ttintx1 = Variable(subquad[0].float(),requires_grad = True)
       ttintx2 = Variable(subquad[1].float(),requires_grad = True)

       loss = pde.pdeloss(y,ttintx1,ttintx2,f,tbdx1,tbdx2,bdrydat,bw)
       loss.backward()
       optimizer.step()
       tot_loss = tot_loss + loss

    nploss = tot_loss.detach().numpy()
    scheduler.step(nploss)
    return nploss   



  

losslist = list()

start_time = time()

for epoch in range(10000):
    loss = closure()
    losslist.append(loss)
    if epoch %100==0:
        print("epoch: {}, loss:{}".format(epoch,loss))
        validation.plot_2D(y,name+"y_plot/"+'epoch{}'.format(epoch))

with open("results/losshist.pkl",'wb') as pfile:
    pkl.dump(losslist,pfile)


end_time = time() 

training_time  = end_time - start_time

# Print to console
print("\nTotal Training Time: {:.2f} seconds".format(training_time))


    
    
    




print("\nFinal Trained Weights and Biases:\n")

total_nonzero = 0
global_max_value = 0.0
global_max_name = ""
global_max_entry = None

for name, param in y.named_parameters():
    if param.requires_grad:
        arr = param.data.numpy()

        ## Print matrix shape and entries
        # print(f"{name}: {arr.shape}")
        # print(arr, "\n")

        ## Count nonzero entries
        nonzero_count = np.count_nonzero(arr)
        total_nonzero += nonzero_count
        #print(f"--> Non-zero entries in {name}: {nonzero_count}")

        # Find max modulus entry in this layer
        max_val = np.max(np.abs(arr))
        #print(f"--> Max |entry| in {name}: {max_val}\n")

        # Update global max
        if max_val > global_max_value:
            global_max_value = max_val
            global_max_name = name
            # store the actual entry (not only magnitude)
            idx = np.unravel_index(np.argmax(np.abs(arr)), arr.shape)
            global_max_entry = arr[idx]

# Final summary
# print("======================================")
# print(f"Total non-zero entries in the network: {total_nonzero}")
# print("======================================")
# print(f"Maximum modulus entry in the entire network is: {global_max_value}")
# print(f"This occurs in: {global_max_name}")
# print(f"The actual entry value (with sign) is: {global_max_entry}")
# print("======================================")




# ---- Save Everything to Text File ----
with open("training_time.txt", "w") as f:
    f.write("Total Training Time: {:.2f} seconds\n".format(training_time))
    f.write("======================================\n")
    f.write("Total non-zero entries in the network: {}\n".format(total_nonzero))
    f.write("======================================\n")
    f.write("Maximum modulus entry in the entire network is: {}\n".format(global_max_value))
    f.write("This occurs in: {}\n".format(global_max_name))
    f.write("The actual entry value (with sign) is: {}\n".format(global_max_entry))
    f.write("======================================\n")






x_pts = np.linspace(0,1,200)
y_pts = np.linspace(0,1,200)

ms_x, ms_y = np.meshgrid(x_pts,y_pts)

x_pts = np.ravel(ms_x).reshape(-1,1)
t_pts = np.ravel(ms_y).reshape(-1,1)

collocations = np.concatenate([x_pts,t_pts], axis=1)

u_gt1,f,deri_ugt_x1,deri_ugt_x2 = g_tr.data_gen_interior(collocations)


ms_ugt = u_gt1.reshape(ms_x.shape)

pt_x = Variable(torch.from_numpy(x_pts).float(),requires_grad=True)
pt_t = Variable(torch.from_numpy(t_pts).float(),requires_grad=True)

pt_y = y(pt_x,pt_t)
y_computed = pt_y.data.cpu().numpy()
ms_ysol = y_computed.reshape(ms_x.shape)



##L^2 error computation
# Flatten the arrays for easier computation
ms_ugt_flat = ms_ugt.flatten()
ms_ysol_flat = ms_ysol.flatten()

# Compute the L2 error
l2_error = np.sqrt(np.mean((ms_ugt_flat - ms_ysol_flat)**2))

l2_realtiveError= l2_error / (np.sqrt(np.mean((ms_ugt_flat)**2)))




print(f"L2 Error: {l2_error}")

print(f"L2relativeError : {l2_realtiveError}")



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



#### Faster version of plotting code
batch_size = 1000  # Tune this for your GPU/CPU memory
n_points = x_pts.shape[0]
grad_u_nn_x = []
grad_u_nn_t = []

for i in range(0, n_points, batch_size):
    bx = x_pts[i:i+batch_size]
    bt = t_pts[i:i+batch_size]
    pt_x = Variable(torch.from_numpy(bx).float(), requires_grad=True)
    pt_t = Variable(torch.from_numpy(bt).float(), requires_grad=True)
    pt_u_nn = y(pt_x, pt_t)
    grads = torch.autograd.grad(
        pt_u_nn, [pt_x, pt_t],
        grad_outputs=torch.ones_like(pt_u_nn),
        create_graph=False, retain_graph=False
    )
    grad_u_nn_x.append(grads[0].detach().cpu().numpy())
    grad_u_nn_t.append(grads[1].detach().cpu().numpy())

grad_u_nn_x = np.concatenate(grad_u_nn_x).flatten()
grad_u_nn_t = np.concatenate(grad_u_nn_t).flatten()





deri_ugt_x1 = deri_ugt_x1.flatten()
deri_ugt_x2 = deri_ugt_x2.flatten()


# Compute L2 norm of gradient difference
grad_error_squared = np.mean((deri_ugt_x1 - grad_u_nn_x)**2 + (deri_ugt_x2 - grad_u_nn_t)**2)

# Compute H1 error
h1_error = np.sqrt(l2_error**2 + grad_error_squared)

# H1 relative error (need to compute H1 norm of ground truth)
h1_norm_gt = np.sqrt(np.mean(ms_ugt_flat**2) + np.mean(deri_ugt_x1**2 + deri_ugt_x2**2))
h1_relative_error = h1_error / h1_norm_gt

print(f"H1 Error: {h1_error}")
print(f"H1 Relative Error: {h1_relative_error}")










   

fig_1 = plt.figure(1, figsize=(6, 5))
plt.pcolor(ms_x,ms_y,ms_ysol, cmap='jet')
h=plt.colorbar()
h.ax.tick_params(labelsize=20)
plt.xticks([])
plt.yticks([])
plt.savefig('NNsolution',bbox_inches='tight')


fig_2 = plt.figure(2, figsize=(6, 5))
plt.pcolor(ms_x,ms_y,ms_ugt, cmap='jet')
h=plt.colorbar()
h.ax.tick_params(labelsize=20)
plt.xticks([])
plt.yticks([])
plt.savefig('GTsolution',bbox_inches='tight')

fig_3 = plt.figure(3, figsize=(6, 5))
plt.pcolor(ms_x,ms_y,abs(ms_ugt-ms_ysol), cmap='jet')
h=plt.colorbar()
h.ax.tick_params(labelsize=20)
plt.xticks([])
plt.yticks([])
plt.savefig('Error',bbox_inches='tight')









