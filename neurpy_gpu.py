import cudamat as cm
#import gnumpy as gpu

import time
import numpy as np
import neurpy as neur

class NeurpyGpu:
    def __init__(self):
        self.m = 10000
        self.sizes = [10,5,3]
        self.thetas = map(lambda x: cm.CUDAMatrix(x), neur.create_initial_thetas(self.sizes, 0.12))

        self.activ_layers = map(lambda x: cm.CUDAMatrix(np.zeros((self.m, x + 1))), self.sizes[0:-1])
        self.activ_layers.append(cm.CUDAMatrix(np.zeros((self.m, self.sizes[-1]))))
        for i in range(len(self.sizes)):
            self.activ_layers[i].set_col_slice(0,1,1)

        self.activ_layers_temp = map(lambda x: cm.empty((self.m, x + 1)), self.sizes[0:-1])
        self.activ_layers_temp.append(cm.empty((self.m, self.sizes[-1])))

        self.layer_expand_mask = map(lambda x: cm.CUDAMatrix(np.hstack([np.zeros((x, 1)), np.eye(x)])), self.sizes[0:-1])

        clear = np.zeros((11,11))
        clear[0,0] = 1
        self.clear_vec = cm.CUDAMatrix(clear[0:11,0:11])
        #print self.clear_vec.shape
        self.clear_vec2 = cm.CUDAMatrix(clear[0:6,0:6])
        
        self.z = [0] * (len(self.thetas) + 1)
        self.z[1] = cm.empty((self.m, 5))
        self.z[2] = cm.empty((self.m, 3))
        
    def forward_prop(self, x, thetas):
        num_thetas = len(thetas)
        # add ones to end
        cm.dot(x, self.layer_expand_mask[0], self.activ_layers_temp[0])
        cm.dot(self.activ_layers[0], self.clear_vec, self.activ_layers[0])
        self.activ_layers[0].add(self.activ_layers_temp[0])

        cm.dot(self.activ_layers[0], thetas[0].T, self.z[1])
        for i in range(1, num_thetas):
            cm.dot(self.z[i].apply_sigmoid(), self.layer_expand_mask[i], self.activ_layers_temp[i])
            cm.dot(self.activ_layers[i], self.clear_vec2, self.activ_layers[i])
            
            self.activ_layers[i].add(self.activ_layers_temp[i])
            cm.dot(self.activ_layers[i], thetas[i].T, self.z[i + 1])
        self.z[num_thetas].apply_sigmoid(self.activ_layers[num_thetas])
        #print self.activ_layers[num_thetas].asarray()
        return self.activ_layers[num_thetas], self.activ_layers

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(z * -1))

def forward_prop_compare(x, thetas):
    rows, columns = x.shape
    num_thetas = len(thetas)
    a = [0] * (num_thetas + 1)
    z = [0] * (num_thetas + 1)
    a[0] = np.hstack([np.ones((rows, 1)), x])
    
    z[1] = np.dot(a[0], thetas[0].transpose())

    for i in range(1, num_thetas):
        a[i] = np.hstack([np.ones((len(z[i]),1)), sigmoid(z[i])])
        z[i + 1] = np.dot(a[i], thetas[i].transpose())
    a[num_thetas] = sigmoid(z[num_thetas])
    out = a[num_thetas]
    #print out
    return out, a    


#cm.cuda_set_device(0)
cm.init()
cm.CUDAMatrix.init_random(seed = 42)

nrp = NeurpyGpu()

x = cm.empty((nrp.m, 10))


print x.fill_with_rand()


print nrp.forward_prop(x, nrp.thetas)
print nrp.forward_prop(x, nrp.thetas)


def repeater(iter, f):
    for i in range(iter):
        f()


times_to_run = 1000
import cProfile
cProfile.run('repeater(times_to_run, lambda: nrp.forward_prop(x, nrp.thetas))', None, 'time')

x = x.asarray()
thetas = map(lambda x: x.asarray(), nrp.thetas)

forward_prop_compare(x, thetas)
forward_prop_compare(x, thetas)

print "numpy time"
cProfile.run('repeater(times_to_run, lambda: forward_prop_compare(x, thetas))', None, 'time')


#print nrp.thetas
#cm.shutdown()
