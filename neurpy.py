import numpy as np
#import gnumpy as gpu
import copy
import os.path
import pickle

def cross_validation_sets(X,y, pickle_name = False, cache_overwrite = False):
    if pickle_name:
        filename = './random_set_cache/data_' + pickle_name +'.pkl'
    if pickle_name and not cache_overwrite and os.path.exists(filename):
        print "Loading cached data"
        res_tuple = pickle.load(open(filename, 'r'))
    else:
        m = X.shape[0]
        indexes = np.random.permutation(np.arange(m))
        Xshuf = X[indexes, :]
        yshuf = y[indexes, :]
        first_split = int(m * 0.8)
        second_split = int(m * 0.9)
        X = Xshuf[0:first_split, :]
        y = yshuf[0:first_split, :]
        X_val = Xshuf[first_split:second_split, :]
        y_val = yshuf[first_split:second_split, :]
        X_test = Xshuf[second_split:m, :]
        y_test = yshuf[second_split:m, :]
        res_tuple = X, y, X_val, y_val, X_test, y_test
        if pickle_name:
            pickle.dump(res_tuple, open(filename,'w'))
    return res_tuple

def rand_init_theta(input_size, output_size, epsilon = 0.12):
    return np.random.rand(output_size, input_size + 1) * 2 * epsilon - epsilon

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(z * -1))

def softmax(inputs):
    res = np.max(inputs, axis=1)
    res = res.repeat(len(inputs[0, :]))
    res.shape = len(inputs), len(inputs[0, :])
    # this is so that the log function doesn't get out of control
    out =  np.exp(inputs - res)
    rout = np.sum(out, axis=1).repeat(len(inputs[0,:]))
    rout.shape = len(inputs), len(inputs[0, :])
    return out / rout
    
def forward_prop(x, thetas):
    rows, columns = x.shape
    num_thetas = len(thetas)
    a = [0] * (num_thetas + 1)
    z = [0] * (num_thetas + 1)
    a[0] = np.hstack([np.ones((rows, 1)), x])
    
    z[1] = np.dot(a[0], thetas[0].transpose())
    for i in range(1, num_thetas):
        a[i] = np.hstack([np.ones((len(z[i]),1)), sigmoid(z[i])])
        z[i + 1] = np.dot(a[i], thetas[i].transpose())
    # don't sigmoid and softmax !!!
    #a[num_thetas] = sigmoid(z[num_thetas])
    a[num_thetas] = softmax(z[num_thetas])
    out = a[num_thetas]
    return out, a

#def sigmoid_gpu(z):
#    return 1.0 / (1.0 + gpu.exp(z * -1))

def logistic_squared_distance(h_x, y):
    m = h_x.shape[0]
    #print h_x.shape
    #print y.shape
    
    return -1 * (y * np.log(h_x) + (1 - y) * np.log(1 - h_x)).sum() / m

# h_x is intended to be the output of a softmax
def cross_entropy_loss(h_x, y):
    return -np.mean(np.sum(y * np.log(h_x), axis=1))

def cost_function_weight_decay(thetas, lamb):
    theta_squared_sum = 0
    for i in range(len(thetas)):
      theta_squared_sum += (thetas[i][:, 1:thetas[i].shape[1]] ** 2).sum() 
    return (lamb/2.0) * theta_squared_sum;

def cross_entropy_loss_with_wd(h_x, y, thetas, lamb):
    return cross_entropy_loss(h_x, y) + cost_function_weight_decay(thetas, lamb)

def logistic_squared_distance_with_wd(h_x, y, thetas, lamb):
    m = h_x.shape[0]
    return logistic_squared_distance(h_x, y) + cost_function_weight_decay(thetas, lamb)

def logistic_squared_cost_function(X, y, thetas, lamb):
    h_x, a = forward_prop(X, thetas)
    return logistic_squared_distance_with_wd(h_x, y, thetas, lamb)

def backprop(activations, y, thetas, lamb):
  a = activations
  m = a[0].shape[0]

  #number of layers
  L = len(a);
  delta = [0] * L
  delta[L - 1] = a[L - 1] - y;
  # count back from last layer
  
  for layer in range(L - 2, 0, -1):
    derivative_of_previous_z = a[layer] * (1 - a[layer])
    delta[layer] = (np.dot(delta[layer + 1], thetas[layer])) * derivative_of_previous_z
    # these layers are all have to eliminate the first element of the deltas (accomodate bias)
    delta[layer] = delta[layer][:, 1:]

  # number of theta layers
  tL = L - 1;
  theta_derivatives = [0] * tL 
  for t_layer in range(tL):
    theta_derivatives[t_layer] = np.dot(a[t_layer].transpose(), delta[t_layer + 1]).transpose() / m

  # last layer different
  theta_derivatives[tL - 1] = np.dot(a[tL - 1].transpose(), delta[tL]).transpose() / m

  gradients = [0] * tL
  
  for t_layer in range(tL):
    theta_reg = thetas[t_layer].copy()
    theta_reg[:,0] = 0 # zero out first column
    gradients[t_layer] = theta_derivatives[t_layer] + lamb * theta_reg

  return gradients

def create_dropout_indices(thetas, percentage = 0.9):
    expanded_indices = []
    hid_layer_size = len(thetas[0])
    how_many = int(hid_layer_size * percentage)
    hid_layer_indices = np.random.permutation(range(hid_layer_size))[0:how_many]
    hid_layer_indices.sort()
    expanded_indices.append([hid_layer_indices, Ellipsis])
    row = False
    for i in range(1,len(thetas)):
        if row:
            hid_layer_size = len(thetas[i])
            how_many = int(hid_layer_size * percentage)
            # we will call this crazy dropout select rows randomly and duplicated
            # crazy thing is that it still works!!
            # hid_layer_indices = map(int,np.random.rand(np.ceil(hid_layer_size / 2)) * hid_layer_size)
            hid_layer_indices = np.random.permutation(range(hid_layer_size))[0:how_many]
            hid_layer_indices.sort()
            expanded_indices.append([hid_layer_indices, Ellipsis])
        else:
            column_indices = [0] + map(lambda x: x + 1, hid_layer_indices) # have to account for bias column
            expanded_indices.append([Ellipsis, column_indices])
        row = not row
    return expanded_indices

def create_dropout_indices_new(thetas, percentage = 0.9):
    expanded_indices = []
    hid_layer_size = len(thetas[0])
    how_many = int(hid_layer_size * percentage)
    hid_layer_indices = np.random.permutation(range(hid_layer_size))[0:how_many]
    hid_layer_indices.sort()
    expanded_indices.append([hid_layer_indices, Ellipsis])
    row = False
    for i in range(1,len(thetas)):
        column_indices = [0] + map(lambda x: x + 1, hid_layer_indices) # have to account for bias column
        hid_layer_indices = Ellipsis
        if i < len(thetas) - 1:
            hid_layer_size = len(thetas[i + 1])
            how_many = int(hid_layer_size * percentage)
            # we will call this crazy dropout select rows randomly and duplicated
            # crazy thing is that it still works!!
            # hid_layer_indices = map(int,np.random.rand(np.ceil(hid_layer_size / 2)) * hid_layer_size)
            hid_layer_indices = np.random.permutation(range(hid_layer_size))[0:how_many]
            hid_layer_indices.sort()
        expanded_indices.append([hid_layer_indices, column_indices])

    return expanded_indices

def dropout_indices_each(indices, f):
    return map(lambda i, drop_index: f(i, drop_index[0], drop_index[1]), range(len(indices)), indices)

def dropout_thetas(thetas, selected_indices = []):
    selected_indices = selected_indices if len(selected_indices) > 0 else create_dropout_indices(thetas)
    return dropout_indices_each(selected_indices, lambda i,r,c: thetas[i][r,c]), selected_indices

def recover_dropped_out_thetas(thetas, dropped_out_thetas, selected_indices):
    def task(i,r,c):
        thetas[i][r,c] = dropped_out_thetas[i]
    dropout_indices_each(selected_indices, task)
    return thetas

def create_initial_thetas(layer_sizes, epsilon):
    thetas = []
    for i in range(0,len(layer_sizes) - 1):
        thetas.append(rand_init_theta(layer_sizes[i], layer_sizes[i + 1], epsilon))
    return thetas

def gradient_decent(X, y, 
                    hidden_layer_sz = 2, 
                    iter = 1000, 
                    wd_coef = 0.0,
                    learning_rate = 0.35, 
                    momentum_multiplier = 0.9,
                    rand_init_epsilon = 0.12,
                    do_early_stopping = False,
                    do_dropout = False,
                    do_learning_adapt = False,
                    dropout_percentage = 0.9,
                    thetas = [],
                    X_val = [], y_val = []):
    
    # one hidden layer
    input_layer_sz = len(X[0])
    output_layer_sz = len(y[0])
    #print 'y shape', y.shape
    sizes = [input_layer_sz, hidden_layer_sz, output_layer_sz]
    if len(thetas) == 0:
        thetas = create_initial_thetas(sizes, rand_init_epsilon)
    #for t in thetas:
    #    print t.shape

    momentum_speeds = map(lambda x: x * 0, thetas)
    costs = []
    val_costs = []
    if do_early_stopping:
      best_so_far = {'thetas': [], 'validation_loss': 100000, 'after_n_iters': 0}
    selected_indices = []
    orig_learning_rate = learning_rate
    for i in range(iter):
        if do_learning_adapt:
            learning_rate = orig_learning_rate * 1.1 * np.log(iter - i) / np.log(iter) 
        # set up thetas for dropout
        if do_dropout:
            selected_indices = create_dropout_indices_new(thetas, dropout_percentage)
            in_use_thetas, selected_indices = dropout_thetas(thetas, selected_indices)
            in_use_momentum_speeds = dropout_indices_each(selected_indices, lambda i,r,c: momentum_speeds[i][r,c])
            #print selected_indices
        else:
            in_use_thetas = thetas
            in_use_momentum_speeds = momentum_speeds
        h_x, a = forward_prop(X, in_use_thetas)
        cost = cross_entropy_loss_with_wd(h_x, y, in_use_thetas, wd_coef)
        costs.append(cost)
        if X_val.any():
            # lets get the validation cost for the whole model at first
            vthetas = thetas
            if do_dropout:
                vthetas = map(lambda th: th * dropout_percentage, thetas)
            vh_x, va = forward_prop(X_val, vthetas)
            vcost = cross_entropy_loss_with_wd(vh_x, y_val, vthetas, wd_coef)
            val_costs.append(vcost)
        if X_val.any() and do_early_stopping and (vcost < best_so_far['validation_loss']):
            best_so_far['thetas'] = map(lambda x: x.copy(), thetas)
            best_so_far['validation_loss'] = vcost
            best_so_far['after_n_iters'] = i
        orig_thetas = thetas    
        orig_momentum_speeds = copy.deepcopy(momentum_speeds)
        # update thetas
        gradients = backprop(a, y, in_use_thetas, wd_coef)
        for ix in range(len(in_use_thetas)):
            in_use_momentum_speeds[ix] = in_use_momentum_speeds[ix] * momentum_multiplier - gradients[ix]
            in_use_thetas[ix] = in_use_thetas[ix] + learning_rate * in_use_momentum_speeds[ix]
        if do_dropout:
            thetas = recover_dropped_out_thetas(thetas, in_use_thetas, selected_indices)
            momentum_speeds = recover_dropped_out_thetas(momentum_speeds, in_use_momentum_speeds, selected_indices)
    if do_early_stopping and (len(best_so_far['thetas']) > 0):
        thetas = best_so_far['thetas']
        print 'Early stopping: validation loss was lowest after ', best_so_far['after_n_iters'], ' iterations. We chose the model that we had then.\n'
    if do_dropout:
        thetas = map(lambda th: th * dropout_percentage, thetas)
    return thetas, costs, val_costs

def gradient_check(X, y, thetas, cost_func):
    epsilon = 0.0001
    num_thetas = len(thetas)
    gradients = [0] * num_thetas
    for t in range(num_thetas):
        gradients[t] = np.zeros(thetas[t].shape)
        num_rows, num_columns = thetas[t].shape
        for i in range(num_rows):
            for j in range(num_columns):
                up_theta = thetas[t].copy() 
                up_theta[i,j] += epsilon 
                down_theta = thetas[t].copy() 
                down_theta[i,j] -= epsilon
                up_thetas = copy.copy(thetas)
                up_thetas[t] = up_theta
                down_thetas = copy.copy(thetas)
                down_thetas[t] = down_theta
                cost_up = cost_func(X, y, up_thetas, 0)
                cost_down = cost_func(X, y, down_thetas, 0)
                gradients[t][i,j] = (cost_up - cost_down) / (2 * epsilon)
    return gradients
    
