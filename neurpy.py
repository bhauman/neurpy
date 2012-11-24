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

def gradient_decent(X, y):
    # one hidden layer
    lamb = 0.01
    input_layer_sz = len(X[0])
    hidden_layer_sz = 2
    output_layer_sz = len(y[0])
    sizes = [input_layer_sz, hidden_layer_sz, output_layer_sz]
    theta1 = rand_init_theta(input_layer_sz, hidden_layer_sz, 0.12)
    theta2 = rand_init_theta(hidden_layer_sz, output_layer_sz, 0.12)
    thetas = [theta1, theta2]
    learning_rate = 0.35
    momentum_speeds = map(lambda x: x * 0, thetas)
    momentum_multiplier = 0.9

    for i in range(1000):
        h_x, a = forward_prop(X, thetas)
        cost = logistic_squared_distance_with_wd(h_x, y, thetas, lamb)
        print cost
        gradients = backprop(a, y, thetas, lamb)
        for i in range(len(thetas)):
            momentum_speeds[i] = momentum_speeds[i] * momentum_multiplier - gradients[i]
            thetas[i] = thetas[i] + learning_rate * momentum_speeds[i]


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
    
