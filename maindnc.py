import torch

import tensorflow as tf
import numpy as np
from torch.nn import functional as F
import torch.distributions as tdist





#########################################################
## main dnc xsm code                                   ##
#########################################################

def maindnc(self, size, batch_index,z0):

    n = tdist.Normal(0,1.6)
    z1 =n.sample((size, self.z_dim)).to(self._device()) 


    #z1 = torch.randn(size, self.z_dim).to(self._device())
    #torch.save(z1, 'file.pt')
    #if batch_index==1:
    #read operation
        #z0=torch.load('file.pt')    

    z2=torch.cat((z0,z1,z1), 0)



    # operation on the dnc memory
    #z2=torch.unique(z2, dim=0)   

    if batch_index==2000:
        # write operation

        z2=memope(self, size,z2)


        torch.save(z2, 'dnc.pt')
    #z=torch.load('file.pt')


    #print('xsm z1 size',z1.size())
    #print('xsm z size',z2.size())
    #print('xsm z0 ',z0)
    #print('xsm z1 ',z1)
    #print('xsm z ',z2)


    return z2


def memope(self, size,z2):

    n = torch.distributions.Normal(0,1.7)

    z1 =n.sample((60, self.z_dim)).to(z2)


    #calculate the tensor similarity then increase dense 
    x = tf.constant(np.random.uniform(-1, 1, 10)) 
    y = tf.constant(np.random.uniform(-1, 1, 10))
    tensor_similarity=1-tf.losses.cosine_distance(tf.nn.l2_normalize(x, 0), tf.nn.l2_normalize(y, 0), dim=0)



    z2=torch.cat((z2,z1), 0)

 
    return z2


