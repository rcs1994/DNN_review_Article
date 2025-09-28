from sched import scheduler
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
from utils import model,pde,tools,validation
import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


torch.set_default_dtype(torch.float64)

y = model.NN()
y.apply(model.init_weights)

exp_name = 'PINN_t2'

settings = {
        'dataname':'10000pts',
        'interior_data':10000,
        'boundary_data':2000,
        
        'max_iter': 100,
        'boundary_weight': 100,
        'lr': 1e-4,
        'NN_arc': [4,80,80,80,1],
        'optimizer': 'Adam',
        'scheduler': 'MultistepLr',
        'scheduler_params': {'milestones': [7000, 9000], 'gamma': 0.1},
        'datatype': 'float64',
}

dataname =settings['dataname']
name = 'results/' + exp_name +"/"
bw = settings['boundary_weight']



if not os.path.exists(name):
    os.makedirs(name)

if not os.path.exists(name+"y_plot/"):
    os.makedirs(name+"y_plot/")


params = list(y.parameters())



with open("dataset/"+dataname,'rb') as pfile:
    int_col = pkl.load(pfile)
    bdry_col = pkl.load(pfile)

print(int_col.shape,bdry_col.shape)

intx1,intx2,intx3,intx4,intx5,intx6 = np.split(int_col,6,axis=1)
bdx1,bdx2,bdx3,bdx4,bdx5,bdx6 = np.split(bdry_col,6,axis=1)

tintx1,tintx2,tintx3,tintx4,tintx5,tintx6,tbdx1,tbdx2,tbdx3,tbdx4,tbdx5,tbdx6 = tools.from_numpy_to_tensor([intx1,intx2,intx3,intx4,intx5,intx6,bdx1,bdx2,bdx3,bdx4,bdx5,bdx6],[True,True,True,True,True,True,False,False,False,False,False,False],dtype=torch.float64)


with open("dataset/gt_on_{}".format(dataname),'rb') as pfile:
    y_gt = pkl.load(pfile)
    f_np = pkl.load(pfile)
    bdry_np = pkl.load(pfile)


'''plt.scatter(bdx1,bdx2,c=bdry_np)
plt.savefig(name+"bdrydata")'''
f,bdrydat,ygt = tools.from_numpy_to_tensor([f_np,bdry_np,y_gt],[False,False,False],dtype=torch.float64)
bdrydat = bdrydat.view(-1,1)
f = f.view(-1,1)

optimizer = opt.Adam(params,lr=settings['lr'])

mse_loss = torch.nn.MSELoss()

#scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer,patience=500)
scheduler = opt.lr_scheduler.MultiStepLR(optimizer,**settings['scheduler_params'])

def closure():
    optimizer.zero_grad()
    loss,pres,bres = pde.pdeloss(y,tintx1,tintx2,tintx3,tintx4,tintx5,tintx6,f,tbdx1,tbdx2,tbdx3,tbdx4,tbdx5,tbdx6,bdrydat,bw)
    loss.backward()
    optimizer.step()
    nploss = loss.detach().numpy()
    scheduler.step()
    return nploss


start_time = time.time()

losslist = list()
for epoch in range(settings['max_iter']):
    loss = closure()
    losslist.append(loss)
    if epoch %1==0:
        print("epoch: {}, loss:{}".format(epoch,loss))
        #validation.plot_2D(y,name+"y_plot/"+'epoch{}'.format(epoch))

end_time = time.time()  

Training_time = end_time-start_time

torch.save(y,name+'y.pt')
# settings['rel_error_L2'] = l2relativ_err

# settings = {
#     exp_name:settings
# }

# validation.plot_2D(y,name+"y_plot/"+'final')
# rec = tools.recorder('exp_table.json')
# #rec.write(settings)
# rec.save()

l2relativ_err,l2_err,_ = validation.validate(y)

with open(name + "losshist_pinn.dat",'w') as f:
    f.write(f"L2_Error: {l2_err}\n")
    f.write(f"L2relative Error: {l2relativ_err}\n")
    f.write(f"Training Time: {Training_time}\n")
    



with open(name + "losshist.pkl",'wb') as pfile:
    pkl.dump(losslist,pfile)



'''loader = torch.utils.data.DataLoader([intx1,intx2],batch_size = 500,shuffle = True)


def closure():
    tot_loss = 0
    for i,subquad in enumerate(loader):
       optimizer.zero_grad()
       ttintx1 = Variable(subquad[0].float(),requires_grad = True)
       ttintx2 = Variable(subquad[1].float(),requires_grad = True)

       loss,pres,bres = pde.pdeloss(y,ttintx1,ttintx2,f,tbdx1,tbdx2,bdrydat,bw)
       loss.backward()
       optimizer.step()
       tot_loss = tot_loss + loss

    nploss = tot_loss.detach().numpy()
    scheduler.step(nploss)
    return nploss   '''



  











'''

x_pts = np.linspace(0,1,200)
y_pts = np.linspace(0,1,200)

ms_x, ms_y = np.meshgrid(x_pts,y_pts)

x_pts = np.ravel(ms_x).reshape(-1,1)
t_pts = np.ravel(ms_y).reshape(-1,1)

collocations = np.concatenate([x_pts,t_pts], axis=1)

u_gt1,f = g_tr.data_gen_interior(collocations)
#u_gt1 = [np.sin(np.pi*x_col)*np.sin(np.pi*y_col) for x_col,y_col in zip(x_pts,t_pts)]
#u_gt1 = [np.exp(x_col+y_col) for x_col,y_col in zip(x_pts,t_pts)]

#u_gt = np.array(u_gt1)

ms_ugt = u_gt1.reshape(ms_x.shape)

pt_x = Variable(torch.from_numpy(x_pts).float(),requires_grad=True)
pt_t = Variable(torch.from_numpy(t_pts).float(),requires_grad=True)

pt_y = y(pt_x,pt_t)
y = pt_y.data.cpu().numpy()
ms_ysol = y.reshape(ms_x.shape)




   

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
plt.savefig('Error',bbox_inches='tight')'''









