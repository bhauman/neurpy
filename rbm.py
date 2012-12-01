import numpy as np
import neurpy as neur

class RBM:
    def __init__(self, visible_size, hidden_size):
        self.weights = ((np.random.rand(hidden_size, visible_size) * 2) - 1) * 0.1

    def prop_up(self, vis_layer):
        return neur.sigmoid(np.dot(vis_layer, self.weights.transpose()))

    def prop_down(self, hid_layer):
        return neur.sigmoid(np.dot(hid_layer, self.weights))

    def goodness(self, vis_layer, hid_layer):
        return np.mean(np.sum(np.dot(vis_layer, self.weights.transpose()) * hid_layer, axis=1))

    def goodness_gradient(self, vis_layer, hid_layer):
        return np.dot(hid_layer.astype(float).transpose(), vis_layer) / float(vis_layer.shape[0])
        
    def random_binary_sample(self, layer):
        rand_layer = np.random.rand(layer.shape[0], layer.shape[1])
        return (layer > rand_layer).astype(int)
        
    def cd1(self, vis_layer):
        hid_prob_t0   = self.prop_up(self.random_binary_sample(vis_layer))
        hid_sample_t0 = self.random_binary_sample(hid_prob_t0)
        gradient_t0   = self.goodness_gradient(vis_layer, hid_sample_t0)
        
        vis_prob_t1   = self.prop_down(hid_sample_t0)
        vis_sample_t1 = self.random_binary_sample(vis_prob_t1)
        hid_prob_t1   = self.prop_up(vis_sample_t1)
        gradient_t1   = self.goodness_gradient(vis_sample_t1, hid_prob_t1)

        return (gradient_t0 - gradient_t1), hid_prob_t1

    def get_impression(self, input_layer):
        hidden_layer = self.prop_up([input_layer])
        return (self.prop_down(hidden_layer) > 0.5).astype(int)

    def optimize(self, train, iters = 100, learning_rate = 0.2):
        costs = []
        momentum_speed = np.zeros_like(self.weights)
        for i in range(iters):
            gradient, hid_prob = self.cd1(train)
            #print gradient
            hid_sample = self.random_binary_sample(hid_prob)
            cost = self.goodness(train, hid_sample)
            print "energy", cost
            print "validate squared_error",  self.validate(train)
            costs.append(cost)
            momentum_speed = momentum_speed * 0.9 + gradient;
            self.weights = self.weights + (momentum_speed * learning_rate)
        return costs
    
    def validate(self, val_set):
        hid_layer = self.prop_up(val_set)
        hid_sample =  self.random_binary_sample(hid_layer)
        vis_layer = self.prop_down(hid_layer)
        #vis_sample =  self.random_binary_sample(vis_layer)
        
        return np.sum((vis_layer - val_set) ** 2) / (2 *  len(vis_layer))
    
