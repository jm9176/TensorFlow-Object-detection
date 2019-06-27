# Plotting the behavior of different activation functions
# Reference: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return [1/float(1 + np.exp(-var)) for var in x]

# Rectified Linear Unit (ReLU)
def relu(x):
    return [float(var) if var>0 else 0 for var in x]

# Leaky ReLU
def leak_relu(x):
    return [float(var) if var>0 else 0.01*var for var in x]

# Parametric ReLU
def param_relu(x, a):
    if a>1:
        return [float(var) if var>0 else a*var for var in x]
    else:
        return [float(max(var, a*var)) for var in x]

# Softplus
def softplus(x):
    return [float(np.log(1 + np.exp(var))) for var in x]

# Exponential Linear Unit (ELU)
def elu(x, a):
    return [float(var) if var > 0 else a*var for var in x]

# Defining the input values ranging form -5 to 5
x_input = range(-5,5)

# Calculating the output values using sigmoid function
y_sig = sigmoid(x_input)

# Calculating output values using ReLU function
y_relu =  relu(x_input)

# Calculating output values using Leaky-ReLU function
y_leaky = leak_relu(x_input)

# Calculating output values using Parametric ReLU function
# The function takes an additional parameter 'a'
y_param_less = param_relu(x_input, 0.5)
y_param_gtr = param_relu(x_input, 1.5)

# Calculating output values using Softplus function
y_softplus = softplus(x_input)

# Calculating output values using ELU function
# This function also takes an additional paramete 'a' which
# si tuned accordingly
y_elu = elu(x_input, 0.5)

# Plotting the curves using the input and the function output
fig, sub_plt = plt.subplots(3,2)
fig.add_subplot(111, frameon = False)
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
plt.xlabel('x input')
plt.ylabel('function output')
fig.suptitle('Activation functions')

# Sigmoid function
sub_plt[0,0].plot(x_input, y_sig, color = 'k', label = 'Sigmoid')
sub_plt[0,0].grid(color = 'grey', linestyle = '-', linewidth = 0.5)
sub_plt[0,0].legend(loc = 'upper left')

# Leaky ReLU function
sub_plt[0,1].plot(x_input, y_leaky, color = 'k', label = 'Leaky ReLU')
sub_plt[0,1].grid(color = 'grey', linestyle = '-', linewidth = 0.5)
sub_plt[0,1].legend(loc = 'upper left')

# ReLU function
sub_plt[1,0].plot(x_input, y_relu, color = 'k', label = 'ReLU')
sub_plt[1,0].grid(color = 'grey', linestyle = '-', linewidth = 0.5)
sub_plt[1,0].legend(loc = 'upper left')

# Parametric ReLU function
sub_plt[1,1].plot(x_input, y_param_less, '-o', markersize = 3, color = 'k',label = 'Param ReLU a=<1')
sub_plt[1,1].plot(x_input, y_param_gtr, color = 'r',label = 'Param ReLU a>1')
sub_plt[1,1].legend(loc = 'lower right')
sub_plt[1,1].grid(color = 'grey', linestyle = '-', linewidth = 0.5)

# ELU function
sub_plt[2,0].plot(x_input, y_elu, color = 'k',label = 'ELUs')
sub_plt[2,0].legend(loc = 'best')
sub_plt[2,0].grid(color = 'grey', linestyle = '-', linewidth = 0.5)

# Softplus
sub_plt[2,1].plot(x_input, y_softplus, color = 'k',label = 'Softplus')
sub_plt[2,1].legend(loc = 'best')
sub_plt[2,1].grid(color = 'grey', linestyle = '-', linewidth = 0.5)

plt.show()
