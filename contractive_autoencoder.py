import numpy as np
import neurpy as neur

class Autoencoder:
    def __init__(self, size, hidden_layer_size, denoise = False, denoise_percent = 0.3, contractive = False):
        thetas = neur.create_initial_thetas([size, hidden_layer_size, size], 0.1)
        self.encode_weights = thetas[0]
        self.decode_weights = thetas[1]
        self.denoise = denoise
        self.corrupt = denoise_percent
        self.contractive = contractive

    def encode(self, x):
        # add ones
        m = len(x)
        x = np.hstack([np.ones((m,1)), x])
        return neur.sigmoid(np.dot(x, self.encode_weights.transpose()))

    def forward_prop(self, x):
        return neur.forward_prop_sigmoid(x, [self.encode_weights, self.decode_weights])

    def backprop(self, a, y):
        gradients = neur.backprop(a, y, [self.encode_weights, self.decode_weights], 0)
        if self.contractive:
            reg = 0.01 * self.contractive_reg_gradient(a[1][:,1:])
            gradients[0][:,1:] += reg[:,1:]
            gradients[1][:,1:] += reg[:,1:].transpose()
        return gradients
    
    def validate(self, result, orig):
        #return neur.logistic_squared_distance(result, orig)
        return np.sum((result - orig) ** 2) / (2 *  len(result))

    def contractive_reg(self, hidden_layer):
        h = hidden_layer
        return np.dot((h * (1 - h)) ** 2, self.encode_weights ** 2).sum()
        
    def contractive_reg_gradient(self, hidden_layer):
        accum = np.zeros_like(self.encode_weights)
        y = hidden_layer
        yy = (y * (1 - y)) ** 2
        width = self.encode_weights.shape[1]
        height = self.encode_weights.shape[0]
        for i in range(len(yy)):
            mask = yy[i,:].repeat(width).reshape(height, width)
            accum += mask * 2 * self.encode_weights
        return accum

    def optimize(self, train, iters = 100, learning_rate = 0.2, val_set = []):
        costs = []
        vcosts = []
        m = len(train)
        momentum_speed_e = np.zeros_like(self.encode_weights)
        momentum_speed_d = np.zeros_like(self.decode_weights)
        mini_batch_size = 100
        start_of_next_mini_batch = 0
        for i in range(iters):
            mini_batch = train[start_of_next_mini_batch:(start_of_next_mini_batch + mini_batch_size), :]
            mini_batch_y = train[start_of_next_mini_batch:(start_of_next_mini_batch + mini_batch_size), :]
            start_of_next_mini_batch = (start_of_next_mini_batch + mini_batch_size) % m
            
            if self.denoise:
                mini_batch   = mini_batch * np.random.binomial(1, 1.0-self.corrupt, size=mini_batch.shape)
                #mini_batch_y = mini_batch_y * np.random.binomial(1, 1.0-self.corrupt, size=mini_batch.shape)

            self.decode_weights[:, 1:] = self.encode_weights.transpose()[1:,:]            
            h_x, a = self.forward_prop(mini_batch)
            cost = self.validate(h_x, mini_batch_y) #+ 0.1 * self.contractive_reg(a[1][:,1:]) / len(h_x)
            costs.append(cost)            

            if len(val_set) > 0:
                vh_x, va = self.forward_prop(val_set)
                vcost = self.validate(vh_x, val_set) #+ 0.1 * self.contractive_reg(a[1][:,1:]) / len(h_x)
                vcosts.append(vcost)                


            gradients = self.backprop(a, mini_batch_y)


            
            momentum_speed_e = momentum_speed_e * 0.9 - gradients[0];
            self.encode_weights = self.encode_weights + (momentum_speed_e * learning_rate)

            momentum_speed_d = momentum_speed_d * 0.9 - gradients[1];
            self.decode_weights = self.decode_weights + (momentum_speed_d * learning_rate)

        return costs, vcosts
