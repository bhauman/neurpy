import numpy as np
import neurpy as neur

class Autoencoder:
    def __init__(self, size, hidden_layer_size, denoise = False, denoise_percent = 0.3):
        thetas = neur.create_initial_thetas([size, hidden_layer_size, size], 0.1)
        self.encode_weights = thetas[0]
        self.decode_weights = thetas[1]
        self.denoise = denoise
        self.corrupt = denoise_percent

    def forward_prop(self, x):
        return neur.forward_prop_sigmoid(x, [self.encode_weights, self.decode_weights])

    def backprop(self, a, y):
        return neur.backprop(a, y, [self.encode_weights, self.decode_weights], 0)
    
    def validate(self, result, orig):
        return np.sum((result - orig) ** 2) / (2 *  len(result))

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
            start_of_next_mini_batch = (start_of_next_mini_batch + mini_batch_size) % m
            
            if self.denoise:
                mini_batch = mini_batch * np.random.binomial(1, 1.0-self.corrupt, size=mini_batch.shape)

            h_x, a = self.forward_prop(mini_batch)
            cost = self.validate(h_x, mini_batch)
            costs.append(cost)            

            if val_set.any():
                vh_x, va = self.forward_prop(val_set)
                vcost = self.validate(vh_x, val_set)
                vcosts.append(vcost)                
            
            gradients = self.backprop(a, mini_batch)
            
            momentum_speed_e = momentum_speed_e * 0.9 - gradients[0];
            self.encode_weights = self.encode_weights + (momentum_speed_e * learning_rate)

            momentum_speed_d = momentum_speed_d * 0.9 - gradients[1];
            self.decode_weights = self.decode_weights + (momentum_speed_d * learning_rate)

        return costs, vcosts
    
    
