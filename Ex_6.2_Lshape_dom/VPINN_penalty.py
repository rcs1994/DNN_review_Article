import torch
import numpy as np
import torch.optim as opt
import pickle
import pandas as pd
from utils import groundtruth
from utils import pde_VPINN as pde
from utils import model,tools,validation
import pickle as pkl
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

exp_name = 'VPINN_t1'
settings = {
    'mashpath':'L_shape_domain_2.msh',
    'nNodes':2113,
    'nElems':4096,
    'max_iter_extension': 2000,
    'max_iter': 10000,
    'lr': 1e-3,
    'NN_arc': [2,30,30,1],
    'optimizer': 'Adam',
    'scheduler': 'MultistepLr',
    'scheduler_params': {'milestones': [5000, 7500, 9000], 'gamma': 0.1},
    'datatype': 'float64',
    'comment': 'boundary extension is used'
}

name = 'results/' + exp_name +"/"


torch.set_default_dtype(torch.float64)
max_iter = settings['max_iter']
bw = 100000

if not os.path.exists(name):
    os.makedirs(name)


'''def loader(path,format='float64'):
    data = np.array(pd.read_fwf(path,header=None)).astype(format)
    return data'''

data_cr,data_el,areas = tools.load_mesh('dataset/VPINN/'+settings['mashpath'])
#data_el = loader('dataset/VPINN/mesh1/mesh_elems.dat',int)-1
#data_cr = loader('dataset/VPINN/mesh1/mesh_coord.dat')

nrElem = len(data_el)
print("nrElem:",nrElem,'nrNode:',len(data_cr))
#aug_cd = np.concatenate([np.ones([len(data_cr),1]),data_cr],axis=1)

mesh = tools.mesh(data_cr,data_el)
data_gt = groundtruth.gt_gen()
quad_dict = mesh.get_full_quad()


qp = torch.tensor(quad_dict['quadrature_points'],requires_grad=True)
w = torch.tensor(quad_dict['weight']).unsqueeze(-1)
cm = torch.tensor(quad_dict['coef_mat'])
print(qp.shape,w.shape,cm.shape)


#Train boundary extension
#bdry_col = torch.tensor(mesh.crd[~np.isin(np.arange(mesh.nrNode),mesh.domain_mark),:])
bdry_col = torch.tensor(mesh.crd[mesh.bdry_mark,:])
#x1,x2 = mesh.crd[mesh.domain_mark,:].T
print(bdry_col.shape,'!!!')

bx1,bx2 = bdry_col.T
bx1 = bx1.unsqueeze(-1)
bx2 = bx2.unsqueeze(-1)


print("boundary collocation:",bdry_col.shape)
bdrydat = torch.tensor(data_gt.generate_by_cart(data_gt.bdry,bdry_col.detach().numpy())).unsqueeze(-1)
print(bdrydat.shape,"!!!")

#Train PDE
u = model.NN_v_penalty()

#Caution: including ub as attribute of u, then parameters of ub are appended to parameters of u. ub will change when optimizing u.
#Hence ub as a fixed boundary extension should be passed to u as input variable of forward function, instead of as attribute.

optimizer = torch.optim.Adam(u.parameters(),lr=settings['lr'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,**settings['scheduler_params'])

f = torch.tensor(data_gt.generate_by_cart(data_gt.f,quad_dict['quadrature_points'].reshape(-1,2)).reshape(mesh.nrElem,6,1))
#print(f.shape)
#f is correct after reshape. Checked.

#validation.plot_2D(u,'figure/bdry.png')
with torch.no_grad():
    rhs = pde.pde_rhs(f,qp,w,cm)
    rhs_assemble = mesh.assemble(rhs)

'''def hook(optimizer,nploss):
    rec.update_epoch()
    rec.update_hist(nploss)
    rec.print_info(u)
    if rec.epoch % 100 ==0:
        rec.plot(ub,"figure/bdrynet.png")'''

def closure():
    optimizer.zero_grad()

    lhs = pde.pde_lhs(u,qp,w,cm)
    lhs_assemble= mesh.assemble(lhs)
    loss = torch.sum(torch.square((lhs_assemble-rhs_assemble)[mesh.domain_mark])) + bw * torch.sum(torch.square(u(bdry_col)-bdrydat))
    loss.backward()

    return loss.detach().numpy()

print("Start training PDE...")
#optimizer.step(closure)
losslist = []
for epoch in range(max_iter):
    loss = closure()
    optimizer.step()
    scheduler.step()

    if epoch%100 == 0:
        validation.plot_2D_vpinn(u,name+"VPINN.png")

        vy = validation.validate_vpinn(u)
        losslist.append(loss)
        print("training at epoch:",epoch,"L2 error:",vy)
    #hook(optimizer,loss)

with open(name + "losshist.pkl",'wb') as pfile:
    pkl.dump(losslist,pfile)

torch.save(u,name+"u.pt")

vy = validation.validate_vpinn(u)

settings['rel_error_L2'] = vy 

settings = {
    exp_name:settings
}

rec = tools.recorder('exp_table.json')
rec.write(settings)
rec.save()

print("PDE training finished.")