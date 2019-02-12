#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), '01_intro'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# <img src="static/pytorch-vs-numpy-logo.png" align="center"/>
#%% [markdown]
# # PyTorch vs. NumPy
#%% [markdown]
# `NumPy` is the fundamental package for **scientific computing** in Python. It provides a **multidimensional array** object (and other) for fast operations on arrays, including mathematical, logical, shape manipulation, etc. We hope you're familiar with NumPy. If you feel some gaps in knowledge check out [this tutorial](https://docs.scipy.org/doc/numpy-1.15.0/user/quickstart.html).
# 
# 
# Today we'll explore **PyTorch** library. PyTorch is also a scientific computing package (like NumPy), but it has two awesome features:
# - Automatic differentiation (very useful for deep learning)
# - GPU support for calculations
# 
# 
# Actually PyTorch has great features. Later on we'll use them to build neural networks, but for now let's consider it just like a scientific computing package. Let's start with installation.
#%% [markdown]
# ## Installation
#%% [markdown]
# ### PyTorch
#%% [markdown]
# PyTorch is pretty easy to install:
# 1. Go to [pytorch.org](https://pytorch.org) and scroll down to **"Quick start locally"** section.
# 2. Choose options suitable for you. E.g.:
#     - PyTorch Build: **Stable (1.0)**
#     - Your OS: **Mac**
#     - Package: **Pip**
#     - Language: **Python 3.7** (check *your* python version via terminal `$ python3 --version`)
#     - CUDA: **None** (if you don't have GPU)
# 3. Just run given command from termial. E.g.: `$ pip3 install torch torchvision`.
# 4. PyTorch is quite big (especially CUDA versions), so you can have a cup of coffee while it's downloading.
#%% [markdown]
# **Lifehack**: you can run termial commands directly from notebooks providing `!` in the beginning of the command:

#%%
# !pip3 install torch torchvision

#%% [markdown]
# Verify proper installation:

#%%
import torch
print(torch.__version__)

#%% [markdown]
# ### NumPy
#%% [markdown]
# NumPy installation is much easier:
# 1. Run `$ pip3 install numpy`
# 2. That's it

#%%
# !pip3 install numpy

#%% [markdown]
# Verify proper installation:

#%%
import numpy as np
print(np.__version__)

#%% [markdown]
# ## Learning PyTorch like robots
#%% [markdown]
# We all love **machine learning**, so let's try to study PyTorch in a *supervised learning* manner. Here're some pairs of examples for NumPy and corresponding PyTorch:
#%% [markdown]
# ### Creation
#%% [markdown]
# NumPy:

#%%
x_numpy = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])

print("x_numpy = \n{}\n".format(x_numpy))

#%% [markdown]
# PyTorch:

#%%
x_torch = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])

print("x_torch = \n{}\n".format(x_torch))

#%% [markdown]
# ### Indexing:
#%% [markdown]
# NumPy:

#%%
print("x_numpy[0, 0] = \n{}\n".format(x_numpy[0, 0]))
print("x_numpy[:, 0] = \n{}\n".format(x_numpy[:, 0]))
print("x_numpy[:2, 1:3] = \n{}\n".format(x_numpy[:2, 1:3]))

#%% [markdown]
# PyTorch:

#%%
print("x_torch[0, 0] = \n{}\n".format(x_torch[0, 0]))
print("x_torch[:, 0] = \n{}\n".format(x_torch[:, 0]))
print("x_torch[:2, 1:3] = \n{}\n".format(x_torch[:2, 1:3]))

#%% [markdown]
# ### Operations:
#%% [markdown]
# NumPy:

#%%
print("x_numpy ** 2 = \n{}\n".format(x_numpy ** 2))
print("np.cos(x_numpy) = \n{}\n".format(np.cos(x_numpy)))
print("x_numpy.mean(axis=0) = \n{}\n".format(x_numpy.mean(axis=0)))
print("x_numpy.T = \n{}\n".format(x_numpy.T))
print("x_numpy.reshape(1, -1) = \n{}\n".format(x_numpy.reshape(1, -1)))
print("x_numpy.flatten() = \n{}\n".format(x_numpy.flatten()))

#%% [markdown]
# PyTorch:

#%%
print("x_torch ** 2 = \n{}\n".format(x_torch ** 2))
print("np.cos(x_torch) = \n{}\n".format(torch.cos(x_torch)))
print("x_torch.mean(dim=0) = \n{}\n".format(x_torch.mean(dim=0)))
print("x_torch.t() = \n{}\n".format(x_torch.t()))
print("x_torch.reshape(1, -1) = \n{}\n".format(x_torch.reshape(1, -1)))
print("x_torch.flatten() = \n{}\n".format(x_torch.flatten()))

#%% [markdown]
# ### Construction
#%% [markdown]
# NumPy:

#%%
print("np.arange(3) = \n{}\n".format(np.arange(3)))
print("np.linspace(0.0, 1.0, num=9).reshape(3, 3) = \n{}\n".format(np.linspace(0.0, 1.0, num=9).reshape(3, 3)))
print("np.ones() = \n{}\n".format(np.ones((2, 5))))
print("np.random.rand(3, 2) = \n{}\n".format(np.random.rand(3, 2)))

#%% [markdown]
# PyTorch:

#%%
print("torch.arange(3) = \n{}\n".format(torch.arange(3)))
print("torch.linspace(0.0, 1.0, steps=9).reshape(3, 3) = \n{}\n".format(torch.linspace(0.0, 1.0, steps=9).reshape(3, 3)))
print("torch.ones() = \n{}\n".format(torch.ones((2, 5))))
print("torch.random.rand(3, 2) = \n{}\n".format(torch.rand(3, 2)))

#%% [markdown]
# ### 2 arrays/tensors
#%% [markdown]
# NumPy:

#%%
x_numpy = np.arange(0, 9).reshape(3, 3)
y_numpy = np.linspace(0.0, 1.0, num=9).reshape(3, 3)

print("x = \n{}\n".format(x_numpy))
print("y = \n{}\n".format(y_numpy))
print("x * y = \n{}\n".format(x_numpy * y_numpy))
print("x.dot(y) = \n{}\n".format(x_numpy.dot(y_numpy)))
print("np.concatenate([x, y], axis=1) = \n{}\n".format(np.concatenate([x_numpy, y_numpy], axis=1)))

#%% [markdown]
# PyTorch:

#%%
x_torch = torch.arange(0, 9).reshape(3, 3).type(torch.float)
y_torch = torch.linspace(0.0, 1.0, steps=9).reshape(3, 3)

print("x = \n{}\n".format(x_torch))
print("y = \n{}\n".format(y_torch))
print("x * y = \n{}\n".format(x_torch * y_torch))
print("x.mm(y) = \n{}\n".format(x_torch.mm(y_torch)))
print("torch.cat([x, y], axis=1) = \n{}\n".format(torch.cat([x_torch, y_torch], dim=1)))
# print(" = \n{}\n".format())

#%% [markdown]
# ## What are differences?
# <img src="static/spiderman-meme.jpg" width=500px align="center"/>
# 
# On the first sight PyTorch is about replacing `np.` with `torch.`. In most cases it's really true, but there are some more *key* differences:
# 
# - by default NumPy creates float arrays with float64 dtype; PyTorch's default dtype is float32
# - `axis` in NumPy [`x.mean(axis=0)`] vs. `dim` in PyTorch [`x.mean(dim=0)`]
# - some functions' naming [e.g. `np.concatenate` vs. `torch.cat`]
# 
# All these differences can be easily googled or found in awesome [PyTorch docs](https://pytorch.org/docs/stable/index.html). Checkout [this repository](https://github.com/wkentaro/pytorch-for-numpy-users) where all differences are listed in a table. If you still have questions, you can ask them on [PyTorch forum](https://discuss.pytorch.org) (it's alive and questions are frequently answered).
# 
# So now you can ask: **why the hell should I use PyTorch instead of NumPy?** Actually PyTorch has some wonderful features, but we'll discuss them a bit later.
#%% [markdown]
# ## A bit more examples:
#%% [markdown]
# NumPy -> PyTorch:

#%%
x_numpy = np.arange(9).reshape(3, 3)
x_torch = torch.from_numpy(x_numpy)

print("x_torch = \n{}\n".format(x_torch))

#%% [markdown]
# PyTorch -> NumPy:

#%%
x_torch = torch.arange(9).reshape(3, 3)
x_numpy = x_torch.numpy()

print("x_numpy = \n{}\n".format(x_numpy))

#%% [markdown]
# Inplace operations. In PyTorch you can peform inplace operations (no copying). Many tensor methods have their inplace twins with `_` symbol in the end:

#%%
x = torch.arange(5).type(torch.float)
print("[before] x = {}".format(x))

x.sqrt_()
print("[after]  x = {}".format(x))


#%%
x = torch.arange(5).type(torch.float)
print("[before] x = {}".format(x))

x.zero_()
print("[after]  x = {}".format(x))

#%% [markdown]
# ## **Task 1 (1 point).** Drawing with PyTorch
# 
# Implement this function and plot it using matplotlib:
# 
# $$
# \begin{cases}
# x = 16 \sin^3(t) \\
# y = 13\cos(t) - 5\cos(2t) - 2\cos(3t) - \cos(4t) \\
# \end{cases}
# ,~ t \in [0, 2\pi]
# $$

#%%
from matplotlib import pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
t = torch.linspace(0.0, 2 * np.pi, 100)

x = 16 * (torch.sin(t) ** 3)
y = 13 * torch.cos(t) - 5 * torch.cos(2 * t) - 2* torch.cos(3* t) - torch.cos(4 * t)


#%%
plt.plot(x.numpy(), y.numpy(), c='red');

#%% [markdown]
# ## Automatic differentiation
#%% [markdown]
# The most important feature of PyTorch is that it can **differentiate (almost) any expression written** in PyTorch.
# 
# For example you have function $f(x) = x^2$. You want to calculate partial detivative $\frac{\partial f}{\partial x}$. PyTorch allows you to do it in 3 lines! Look:

#%%
x = torch.tensor(2.0, requires_grad=True)  # tells PyTorch that we'll need gradient of this tensor
f_x = x ** 2  # run our function
f_x.backward()  # calculate gradient

print("f_x = {}".format(x))
print("df/dx = {}".format(x.grad))

#%% [markdown]
# The salt is that $f(x)$ can be any (almost) any function you want (e.g. your neural network). It totally looks like magic. More on PyTorch automatic differentiation read [here](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html).
#%% [markdown]
# ## Simple linear regression.
#%% [markdown]
# Load data:

#%%
from sklearn.datasets import load_boston
boston = load_boston()

plt.scatter(boston.data[:, -1], boston.target)
plt.xlabel("x")
plt.ylabel("y")

#%% [markdown]
# Convert data to torch tensors:

#%%
x = torch.from_numpy(boston.data[:, -1]).type(torch.float)
x = (x - x.mean()) / x.std()  # normalization

y = torch.from_numpy(boston.target).type(torch.float)

#%% [markdown]
# 1-dimensional linear regression is formulated like this:
# $$\normalsize y^{pred} = wx + b$$
# where $x$ - is a feature, $w, b$ - model parameters (weight and bias), and $y^{pred}$ - model's prediction.
# 
# As a loss function we'll use **Mean Square Error** (MSE):
# $$MSE(y, y^{pred}) = \frac{1}{N}\sum_{i=0}^{N-1}(y_{i} - y_{i}^{pred}) ^ 2$$
# where $N$ - is a length of training set.
# 
# To train our model, we'll use **Gradient Descent** (GD):
# $$\normalsize w_{i} = w_{i - 1} - \eta \frac{\partial loss}{\partial w_{i - 1}}$$
# 
# $$\normalsize b_{i} = b_{i - 1} - \eta \frac{\partial loss}{\partial b_{i - 1}}$$
# where $\eta$ - is a learning rate.
# 
# But we're not going to calculate this partial derivative by hand. PyTorch will do it for us!
#%% [markdown]
# Declare model's parameters:

#%%
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

#%% [markdown]
# Train loop:

#%%
from IPython.display import clear_output

lr = 0.01
n_iters = 200

losses = []

for i in range(n_iters):
    # forward pass
    y_pred = w * x + b
    
    # calculate loss
    loss = torch.mean((y - y_pred) ** 2)
    losses.append(loss.item())
    
    # calculate gradients
    loss.backward()
    
    # make gradient step
    w.data = w.data - lr * w.grad.detach()  # detaching tensor from computational graph
    b.data = b.data - lr * b.grad.detach()  #
    
    # zero gradients (otherwise they'll accumulate)
    w.grad.zero_()
    b.grad.zero_()

    # visualization
    if i % 10 == 0:
        clear_output(True)
        
        print("loss: {:.04}".format(loss.item()))
        
        # training set
        plt.scatter(x.numpy(), y.numpy())
        
        # our predictions
        plt.scatter(x.numpy(), y_pred.detach().numpy(), color='red')
        
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    
        
plt.plot(losses)
plt.xlabel("iters")
plt.ylabel("loss");

#%% [markdown]
# ## Task 2 (1 point). Linear regression with polynomial features
# 
# As you can see above simple linear regression doesn't fit data well. Let's add polynomial features:
# $$\normalsize y^{pred} = \sum_{i=0}^{D-1}w_{i}x^{i}$$
# 
# To get total score for this task:
#  - choose any $D$ you want and make loss **< 30.0**.
#  - answer the question: "why don't we have bias (b) here"

#%%
# your code here

#%% [markdown]
# ## Task 3 (2 points). Earthquake
#%% [markdown]
# Big earthquake happened! Seismic sensors fixed heatmap of underground activity. You can find this map on the path `./data/earthquake-heatmap.npy`.
# Read earthquake heatmap and draw it with matplotlib:

#%%
earthquake_heatmap = # your code here (use np.load)

shape = earthquake_heatmap.shape
print("shape: {}".format(shape))

plt.imshow(earthquake_heatmap)
plt.colorbar()

#%% [markdown]
# Your task is to find coordinates of **earthquake epicenter** and **radius of seismic activity**.
# 
# Possibly, your first thoughts will be: "Hmm, epicenter I'll find by applying argmax for both dimensions. For radius I can write for-loop and... blablabla". Such methods are okay, but we'll apply more **SCIENTIFIC** (actually not) method here. Scheme of proposed method is below.
# 
# 
# <img src="static/earthquake-method.png" align="center"/>
# 
# We'll fit 2D gaussian to our earthquake heatmap.
# 
# Overall algorithm is like this:
# 1. generate 2D gaussian heatmap for current $\mu$ and $\sigma$ 
# 2. calculate per-pixel MSE of generated heatmap and earthquake heatmap
# 3. calculate partial derivatives $\frac{\partial{loss}}{\partial{\mu}}$, $\frac{\partial{loss}}{\partial{\sigma}}$ and make gradient descent step to minimize loss
# 4. go to step (1)
#%% [markdown]
# To generate heatmap we'll need Probability Dense Function (PDF) of independent 2D normal distribution:
# 
# $$p(x_1, x_2) = \frac{1}{2\pi \sigma_1 \sigma_2}\exp\left[{-\frac{(x_1 - \mu_1)^2}{2\sigma_1^2} - \frac{(x_2 - \mu_2)^2}{2\sigma_2^2}}\right]$$
#%% [markdown]
# Implement missing parts:

#%%
def gaussian_2d_pdf(coords, mu, sigma):
    """Independent 2D normal distribution PDF
    
    Args:
        coords (tensor of shape (N, 2)): coordinates, where to calculate function
        mu (tensor of shape (2,)): mu values
        sigma (tensor of shape (2,)): sigma values
        
    """
    normalization = # your code here
    exp = # your code here
    return exp / normalization


def draw_heatmap(mu, sigma, shape):
    xx, yy = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]))
    grid = torch.stack([xx, yy], dim=-1).type(torch.float32)
    grid = grid.reshape((-1, 2))
    
    heatmap = gaussian_2d_pdf(grid, mu, sigma)
    heatmap = heatmap.reshape(*shape)
    
    heatmap = (heatmap - heatmap.mean()) / heatmap.std()

    return heatmap

#%% [markdown]
# Generate heatmap with any $\mu$ and $\sigma$; shape same as earthquake heatmap: 

#%%
mu = # your code here
sigma = # your code here

heatmap = # your code here

plt.imshow(heatmap.numpy())
plt.colorbar()

#%% [markdown]
# Define parameters of your model (initialize them with reasonable values):

#%%
mu = # your code here
sigma = # your code here

#%% [markdown]
# Build optimization loop and fit `mu` and `sigma`:

#%%
# your code here

#%% [markdown]
# ### Tips:
# - Plot your map and loss curve every n-th iteration
# - Tune learning rate
# - Try different initializations of $\mu$ and $\sigma$
# - Play with different number of iterations
# - Try Mean Absolute Error (MAE)
#%% [markdown]
# ---

