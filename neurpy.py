import numpy as np
import copy

def cross_validation_sets(X,y):
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
    return X, y, X_val, y_val, X_test, y_test

def rand_init_theta(input_size, output_size, epsilon = 0.12):
    return np.random.rand(output_size, input_size + 1) * 2 * epsilon - epsilon

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(z * -1))

def softmax(inputs):
    res = np.max(inputs, axis=1)
    res = res.repeat(len(inputs[0, :]))
    res.shape = len(inputs), len(inputs[0, :])
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

    a[num_thetas] = sigmoid(z[num_thetas])
    out = a[num_thetas]
    return out, a

def logistic_squared_distance(h_x, y):
    m = h_x.shape[0]
    return -1 * (y * np.log(h_x) + (1 - y) * np.log(1 - h_x)).sum() / m

def cost_function_weight_decay(m, thetas, lamb):
    theta_squared_sum = 0
    for i in range(len(thetas)):
      theta_squared_sum += (thetas[i][:, 1:thetas[i].shape[1]] ** 2).sum() 
    return (lamb/(2.0 * m)) * theta_squared_sum;

def logistic_squared_distance_with_wd(h_x, y, thetas, lamb):
    m = h_x.shape[0]
    return logistic_squared_distance(h_x, y) + cost_function_weight_decay(m, thetas, lamb)

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
    gradients[t_layer] = theta_derivatives[t_layer] + (lamb / m) * theta_reg

  return gradients

def create_dropout_indices(thetas):
    expanded_indices = []
    hid_layer_size = len(thetas[0])
    hid_layer_indices = map(int,np.random.rand(np.ceil(hid_layer_size / 2)) * hid_layer_size)
    expanded_indices.append([hid_layer_indices, range(thetas[0].shape[1])])
    row = False
    for i in range(1,len(thetas)):
        if row:
            hid_layer_size = len(thetas[i])
            hid_layer_indices = map(int,np.random.rand(np.ceil(hid_layer_size / 2)) * hid_layer_size)
            expanded_indices.append([hid_layer_indices, range(thetas[i].shape[1])])
        else:
            column_indices = [0] + map(lambda x: x + 1, hid_layer_indices) # have to account for bias column
            expanded_indices.append([range(thetas[i].shape[0]), column_indices])
        row = not row
    return expanded_indices

def dropout_thetas(thetas):
    # now there are 2 thetas
    # take rows out of the first theta
    # take columns out of second theta
    selected_indices = []
    new_thetas = []
    hid_layer_size = len(thetas[0])
    hid_layer_indices = map(int,np.random.rand(np.ceil(hid_layer_size / 2)) * hid_layer_size)
    selected_indices.append(hid_layer_indices)
    new_thetas.append(thetas[0][hid_layer_indices,:])
    row = False

    for i in range(1,len(thetas)):
        if row:
            hid_layer_size = len(thetas[i])
            hid_layer_indices = map(int,np.random.rand(np.ceil(hid_layer_size / 2)) * hid_layer_size)
            selected_indices.append(hid_layer_indices)          
            new_thetas.append(thetas[i][hid_layer_indices,:])
        else:
            column_indices = [0] + map(lambda x: x + 1, hid_layer_indices) # have to account for bias column
            new_thetas.append(thetas[i][:, column_indices])
        row = not row

    return new_thetas, selected_indices

def recover_dropped_out_thetas(thetas, dropped_out_thetas, selected_indices):
    thetas[0][selected_indices[0], :] = dropped_out_thetas[0] 
    row = False
    for i in range(1,len(thetas)):
      if row:
          thetas[i][selected_indices[i - 1], :] = dropped_out_thetas[i] 
      else:
          column_indices = [0] + map(lambda x: x + 1, selected_indices[i - 1]) # have to account for bias column
          thetas[i][:, column_indices] = dropped_out_thetas[i]
      row = not row
    return thetas



def mini_batch_gradient_decent(X, y, 
                               hidden_layer_sz = 2, 
                               iter = 1000, 
                               wd_coef = 0.0,
                               learning_rate = 0.35, 
                               momentum_multiplier = 0.9,
                               rand_init_epsilon = 0.12,
                               do_early_stopping = False,
                               do_dropout = False,
                               X_val = [], y_val = []):
    # one hidden layer
    input_layer_sz = len(X[0])
    output_layer_sz = len(y[0])
    sizes = [input_layer_sz, hidden_layer_sz, output_layer_sz]
    theta1 = rand_init_theta(input_layer_sz, hidden_layer_sz, rand_init_epsilon)
    theta2 = rand_init_theta(hidden_layer_sz, output_layer_sz, rand_init_epsilon)
    thetas = [theta1, theta2]
    momentum_speeds = map(lambda x: x * 0, thetas)
    costs = []
    val_costs = []
    if do_early_stopping:
      best_so_far = {'thetas': [], 'validation_loss': 100000, 'after_n_iters': 0}
    selected_indices = []
    for i in range(iter):
        # set up thetas for dropout
        in_use_thetas, selected_indices = dropout_thetas(thetas, selected_indices)
        print 'selected', selected_indices
        h_x, a = forward_prop(X, in_use_thetas)
        cost = logistic_squared_distance_with_wd(h_x, y, in_use_thetas, wd_coef)
        costs.append(cost)
        if X_val.any():
            # lets get the validation cost for the whole model at first
            vh_x, va = forward_prop(X_val, thetas)
            vcost = logistic_squared_distance_with_wd(vh_x, y_val, thetas, wd_coef)
            val_costs.append(vcost)
        if X_val.any() and do_early_stopping and (vcost < best_so_far['validation_loss']):
            best_so_far['thetas'] = map(lambda x: x.copy(), thetas)
            best_so_far['validation_loss'] = vcost
            best_so_far['after_n_iters'] = i
            
        # update thetas
        gradients = backprop(a, y, in_use_thetas, wd_coef)
        for ix in range(len(in_use_thetas)):
            momentum_speeds[ix] = momentum_speeds[ix] * momentum_multiplier - gradients[ix]
            in_use_thetas[ix] = in_use_thetas[ix] + learning_rate * momentum_speeds[ix]
        print "thetas before recover", thetas[0].shape, thetas[1].shape
        thetas = recover_dropped_out_thetas(thetas, in_use_thetas, selected_indices)
    if do_early_stopping and (len(best_so_far['thetas']) > 0):
      thetas = best_so_far['thetas']
      print 'Early stopping: validation loss was lowest after ', best_so_far['after_n_iters'], ' iterations. We chose the model that we had then.\n'

    return thetas, costs, val_costs


def gradient_decent(X, y, X_val = [], y_val = []):
    # one hidden layer
    lamb = 0.01
    input_layer_sz = len(X[0])
    hidden_layer_sz = 2
    output_layer_sz = len(y[0])
    sizes = [input_layer_sz, hidden_layer_sz, output_layer_sz]
    theta1 = rand_init_theta(input_layer_sz, hidden_layer_sz, 0.12)
    theta2 = rand_init_theta(hidden_layer_sz, output_layer_sz, 0.12)
    thetas = [theta1, theta2]
    learning_rate = 0.05
    momentum_speeds = map(lambda x: x * 0, thetas)
    momentum_multiplier = 0.9
    costs = []
    val_costs = []

    for i in range(1000):
        h_x, a = forward_prop(X, thetas)
        h_x = softmax(h_x)
        cost = logistic_squared_distance_with_wd(h_x, y, thetas, lamb)
        costs.append(cost)
        if X_val.any():
          vh_x, va = forward_prop(X_val, thetas)
          vh_x = softmax(vh_x)
          vcost = logistic_squared_distance_with_wd(vh_x, y_val, thetas, lamb)
          val_costs.append(vcost)
        gradients = backprop(a, y, thetas, lamb)
        for i in range(len(thetas)):
            momentum_speeds[i] = momentum_speeds[i] * momentum_multiplier - gradients[i]
            thetas[i] = thetas[i] + learning_rate * momentum_speeds[i]
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
    
