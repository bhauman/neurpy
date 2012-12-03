import numpy as np
import neurpy as neur

class RBM:
    def __init__(self, visible_size, hidden_size):
        self.visible_size = visible_size
        self.hidden_size  = hidden_size
        self.weights      = ((np.random.rand(hidden_size, visible_size) * 2) - 1) * 0.1
        self.hidden_bias  = ((np.random.rand(hidden_size)  * 2) - 1) * 0.1
        self.visible_bias = ((np.random.rand(visible_size) * 2) - 1) * 0.1

    def expand_hidden_bias(self, m):
        return np.dot(np.ones((m,1)), self.hidden_bias.reshape(1,self.hidden_size))

    def expand_visible_bias(self, m):
        return np.dot(np.ones((m,1)), self.visible_bias.reshape(1,self.visible_size))

    def prop_up(self, vis_layer):
        hid_bias = self.expand_hidden_bias(vis_layer.shape[0])
        res = np.dot(vis_layer, self.weights.transpose()) + hid_bias
        return neur.sigmoid(res)

    def prop_down(self, hid_layer):
        vis_bias = self.expand_visible_bias(hid_layer.shape[0])
        return neur.sigmoid(np.dot(hid_layer, self.weights) + vis_bias)

    def goodness(self, vis_layer, hid_layer):
        vis_layer_wb = vis_layer * self.visible_bias
        hid_layer_wb = hid_layer * self.hidden_bias
        sum_before_mean = np.sum(np.dot(vis_layer, self.weights.transpose()) * hid_layer, axis=1) + np.sum(vis_layer_wb, axis=1) + np.sum(hid_layer_wb, axis=1)
        return np.mean(sum_before_mean)

    def goodness_gradient(self, vis_layer, hid_layer):
        return np.dot(hid_layer.astype(float).transpose(), vis_layer) / float(vis_layer.shape[0])
    
    def goodness_gradient_layer(self, layer):
        return np.sum(layer, axis=0) / float(layer.shape[0])
    
    def random_binary_sample(self, layer):
        rand_layer = np.random.rand(layer.shape[0], layer.shape[1])
        return (layer > rand_layer).astype(int)
        
    def cd1(self, vis_layer):
        vis_sample_t0 = self.random_binary_sample(vis_layer)
        hid_prob_t0   = self.prop_up(vis_sample_t0)
        hid_sample_t0 = self.random_binary_sample(hid_prob_t0)

        visible_bias_grad_t0 = self.goodness_gradient_layer(vis_layer)
        hidden_bias_grad_t0  = self.goodness_gradient_layer(hid_prob_t0)
        gradient_t0          = self.goodness_gradient(vis_layer, hid_prob_t0)
        
        vis_prob_t1   = self.prop_down(hid_sample_t0)
        # vis_sample_t1 = self.random_binary_sample(vis_prob_t1)
        hid_prob_t1   = self.prop_up(vis_prob_t1)

        visible_bias_grad_t1 = self.goodness_gradient_layer(vis_prob_t1)
        hidden_bias_grad_t1  = self.goodness_gradient_layer(hid_prob_t1)
        gradient_t1          = self.goodness_gradient(vis_prob_t1, hid_prob_t1)

        return (gradient_t0 - gradient_t1), (visible_bias_grad_t0 - visible_bias_grad_t1), (hidden_bias_grad_t0 - hidden_bias_grad_t1), hid_prob_t1

    def get_impression(self, input_layer):
        hidden_layer = self.prop_up([input_layer])
        return (self.prop_down(hidden_layer) > 0.5).astype(int)

    def optimize(self, train, iters = 100, learning_rate = 0.2):
        costs = []
        m = len(train)
        momentum_speed = np.zeros_like(self.weights)
        v_momentum_speed = np.zeros(self.visible_size)
        h_momentum_speed = np.zeros(self.hidden_size)

        mini_batch_size = 100
        start_of_next_mini_batch = 0
        for i in range(iters):
            mini_batch = train[start_of_next_mini_batch:(start_of_next_mini_batch + mini_batch_size), :]
            start_of_next_mini_batch = (start_of_next_mini_batch + mini_batch_size) % m
            gradient, v_gradient, h_gradient, hid_prob = self.cd1(mini_batch)
            #print gradient
            hid_sample = self.random_binary_sample(hid_prob)
            cost = self.goodness(mini_batch, hid_sample)
            print "energy", cost
            print "validate squared_error",  self.validate(train)
            costs.append(cost)
            # weights update
            momentum_speed = momentum_speed * 0.9 + gradient;
            self.weights = self.weights + (momentum_speed * learning_rate)

            # visible bias update
            v_momentum_speed = v_momentum_speed * 0.9 + v_gradient;
            self.visible_bias = self.visible_bias + (v_momentum_speed * learning_rate)

            # hidden bias update
            h_momentum_speed = h_momentum_speed * 0.9 + h_gradient;
            self.hidden_bias = self.hidden_bias + (h_momentum_speed * learning_rate)
        return costs
    
    def validate(self, val_set):
        hid_layer = self.prop_up(val_set)
        hid_sample =  self.random_binary_sample(hid_layer)
        vis_layer = self.prop_down(hid_layer)
        #vis_sample =  self.random_binary_sample(vis_layer)
        
        return np.sum((vis_layer - val_set) ** 2) / (2 *  len(vis_layer))
    
