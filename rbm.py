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
    
    def free_energy(self, vis_layer):
        # calc hidden before sigmoid 
        x = np.dot(vis_layer, self.weights.transpose())
        return np.mean(np.sum(np.log(np.exp(x) + 1), axis=1))
    
    def random_binary_sample(self, layer):
        rand_layer = np.random.rand(layer.shape[0], layer.shape[1])
        return (layer > rand_layer).astype(int)
        
    def cd1(self, vis_layer):
        # this is here so that if we wanto to we can take probabilities as input
        # should probably bastract it to another function
        vis_prob      = self.random_binary_sample(vis_layer)
        hid_prob_t0   = self.prop_up(vis_prob)
        hid_sample_t0 = self.random_binary_sample(hid_prob_t0)
        gradient_t0   = self.goodness_gradient(vis_layer, hid_prob_t0)
        
        # hinton states that you don't need to sample the visible probabilities but just use them
        vis_prob_t1   = self.prop_down(hid_sample_t0)
        # vis_sample_t1 = self.random_binary_sample(vis_prob_t1)
        hid_prob_t1   = self.prop_up(vis_prob_t1)
        # hinton also states that you don't need to sample the last hidden state
        # but you should samle all hidden states but the last
        gradient_t1   = self.goodness_gradient(vis_prob_t1, hid_prob_t1)

        return (gradient_t0 - gradient_t1), hid_prob_t1

    def get_impression(self, input_layer):
        hidden_layer = self.prop_up([input_layer])
        return (self.prop_down(hidden_layer) > 0.5).astype(int)

    def optimize(self, train, iters = 100, learning_rate = 0.2, val_set = []):
        costs = []
        vcosts = []
        m = len(train)
        momentum_speed = np.zeros_like(self.weights)
        mini_batch_size = 100
        start_of_next_mini_batch = 0
        for i in range(iters):
            mini_batch = train[start_of_next_mini_batch:(start_of_next_mini_batch + mini_batch_size), :]
            start_of_next_mini_batch = (start_of_next_mini_batch + mini_batch_size) % m
            gradient, hid_prob = self.cd1(mini_batch)
            print mini_batch.shape
            cost = self.goodness(mini_batch, hid_prob)
            costs.append(cost)
            print "energy", cost
            if len(val_set) > 100:
                print "train free energy", self.free_energy(train[0:100,:])
                print "val   free energy", self.free_energy(val_set)
            momentum_speed = momentum_speed * 0.9 + gradient;
            self.weights = self.weights + (momentum_speed * learning_rate)
            print "validate squared_error",  self.validate(train)

        return costs
    
    def validate(self, val_set):
        hid_layer = self.prop_up(val_set)
        hid_sample =  self.random_binary_sample(hid_layer)
        vis_layer = self.prop_down(hid_layer)
        #vis_sample =  self.random_binary_sample(vis_layer)
        
        return np.sum((vis_layer - val_set) ** 2) / (2 *  len(vis_layer))
    
