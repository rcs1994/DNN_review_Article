import torch
from torch.autograd import Variable
import numpy as np

#This version allows training multiple neural networks withing PDE.
#gpu version still debugging

def from_numpy_to_tensor_with_grad(numpys,dtype = torch.float64):
    #numpys: a list of numpy arrays.
    outputs = list()
    for ind in range(len(numpys)):
        outputs.append(
            Variable(torch.from_numpy(numpys[ind]).float(),requires_grad=True).type(dtype)
        )

    return outputs

def from_numpy_to_tensor(numpys,require_grads,dtype=torch.float64):
    #numpys: a list of numpy arrays.
    #requires_grads: a list of boolean to indicate whether give gradients
    outputs = list()
    for ind in range(len(numpys)):
        outputs.append(
            Variable(torch.from_numpy(numpys[ind]),requires_grad=require_grads[ind]).type(dtype)
        )

    return outputs
    

    
class mesh(object):
    def __init__(self,coord,elem):
        self.crd = coord
        self.em = elem
        self.em_flattened = torch.tensor(self.em).reshape(-1)
        
        self.nrElem = len(elem)
        self.nrNode = len(coord)

        self.quad = np.array([[0.44594849091597, 0.4459849091597 , 0.22338158967801],
        [0.44594849091597, 0.10810301816807, 0.22338158967801],
        [0.10810301816807 ,0.44594849091597, 0.22338158967801],
        [0.09157621350977 ,0.09157621350977, 0.10995174365532],
        [0.09157621350977 ,0.81684757298046, 0.10995174365532],
        [0.81684757298046 ,0.09157621350977, 0.10995174365532]])

        self.domain_mark = self.identify_bdry()
        #shape: (6,3). 6 quadrature points, 2 coordinates + 1 weight.

    def identify_bdry(self):
        domain_mark = list()
        for ind in range(len(self.crd)):
            if not (np.abs(self.crd[ind,0]-0.0)<1e-6 or np.abs(self.crd[ind,1]-0.0)<1e-6 or np.abs(self.crd[ind,0]-1.0)<1e-6 or np.abs(self.crd[ind,1]-1.0)<1e-6):
                domain_mark.append(ind)
        return domain_mark

    def get_coef_mat(self,cd):
        #coord = [[x1,y1],[x2,y2],[x3,y3]], shape:(3,2)
        aug_cd = np.concatenate([np.ones((3,1)),cd],axis=1)
        L1 = np.linalg.solve(aug_cd,[1,0,0])
        L2 = np.linalg.solve(aug_cd,[0,1,0])
        L3 = np.linalg.solve(aug_cd,[0,0,1])
        
        return np.array([L1,L2,L3])
    
    
    def area(self,cx,cy):
        area = 0.5*np.abs(cx[0]*(cy[1]-cy[2])+cx[1]*(cy[2]-cy[0])+cx[2]*(cy[0]-cy[1])) #area is correct
        return area

    def quadrature(self,cx,cy,f,L,area=None):
        if area is None:
            area = 0.5*np.abs(cx[0]*(cy[1]-cy[2])+cx[1]*(cy[2]-cy[0])+cx[2]*(cy[0]-cy[1]))
        xq = (cx[0]*(1-self.quad[:,0]-self.quad[:,1]) + cx[1]*self.quad[:,0] + cx[2]*self.quad[:,1])
        yq = (cy[0]*(1-self.quad[:,0]-self.quad[:,1]) + cy[1]*self.quad[:,0] + cy[2]*self.quad[:,1])
        return area*np.sum(self.quad[:,2]*(f(xq,yq)*(L[0]+L[1]*xq+L[2]*yq)))
    
    def quadrature_points(self,cx,cy):
        xq = (cx[0]*(1-self.quad[:,0]-self.quad[:,1]) + cx[1]*self.quad[:,0] + cx[2]*self.quad[:,1])
        yq = (cy[0]*(1-self.quad[:,0]-self.quad[:,1]) + cy[1]*self.quad[:,0] + cy[2]*self.quad[:,1])
        return xq,yq
    
    def assemble(self,element_wise_staff):
    #shape of input: (nrElem,3,1)
    #assembling takes much time!!!
        node_wise_staff = torch.zeros([self.nrNode])
        node_wise_staff.index_add_(0,self.em_flattened,element_wise_staff.reshape(-1))

        return node_wise_staff.unsqueeze(-1)#[domain_mark]

    def get_full_quad(self):
        '''
        Returns dictionary:
            {
                'quadrature_points':[[x_i,y_i]]. numpy array with shape (nrElem,6,2) 
                'weight': [w_i]. numpy array with shape (nrElem,6). Area will be multiplied here.
                'coef_mat': [[a_i,b_i,c_i]]. numpy array with shape (nrElem,3,3). Each element has a 3x3 coef mat.
    
            }
        
        '''
        qp = []
        w = [] 
        cm = []
        for ind in np.arange(0,self.nrElem):
            curElem = self.em[ind,:]
            curCoord = self.crd[curElem,:] #shape: (3,2)

            coef_mat = self.get_coef_mat(curCoord) #[L1,L2,L3]

            area = self.area(curCoord[:,0],curCoord[:,1])
            x = (curCoord[0,0]*(1-self.quad[:,0]-self.quad[:,1]) + curCoord[1,0]*self.quad[:,0] + curCoord[2,0]*self.quad[:,1])
            y = (curCoord[0,1]*(1-self.quad[:,0]-self.quad[:,1]) + curCoord[1,1]*self.quad[:,0] + curCoord[2,1]*self.quad[:,1])
            
            emw = self.quad[:,2] * area

            qp.append(np.array([x,y]).T) #check shape
            w.append(emw) 
            cm.append(coef_mat) #Coef mat is correct (checked)

        output = {
            "quadrature_points":np.array(qp),
            "weight":np.array(w),
            "coef_mat":np.array(cm)
        }
        return output
