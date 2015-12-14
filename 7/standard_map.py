
# coding: utf-8

# In[150]:

get_ipython().magic(u'matplotlib inline')
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import ode
import time
import matplotlib.pylab as pylab
import scipy.constants as spcon
pylab.rcParams['figure.figsize'] = 16, 12  # that's default image size for this 


# In[119]:

K = 1.
eigenvec = np.array([[1.,K/2.+np.sqrt(K*(4.+K))], [1.,K/2.+np.sqrt(K*(4.+K))]])
closeness = 1e-7


# In[120]:

def standard_map(xn,pn):
    x = np.mod(xn + pn,1)
    p = np.mod(pn + float(K)/(2*np.pi)*np.sin(2*np.pi*x), 1)
    return x, p
standard_map = np.vectorize(standard_map)


# In[121]:

def inv_standard_map(xn, pn):
    p = np.mod(pn - float(K)/(2*np.pi)*np.sin(2*np.pi*xn),1)
    x = np.mod(xn-p,1)
    return x, p 
inv_standard_map = np.vectorize(inv_standard_map)


# In[122]:

#Phasespace Potrait
def potrait():
    x0 = np.random.random(100)
    p0 = np.random.random(100)
    X_ret,P_ret = [],[]
    x,p = x0,p0
    for i in range(1000):
        x, p = standard_map(x,p)
        X_ret.append(x)
        P_ret.append(p)
    return np.array(X_ret), np.array(P_ret)


# In[193]:

def orbit(x0,p0, t, func = standard_map):
    X_ret ,P_ret = [],[]
    x, p = x0, p0
    i=0
    while i < t:
        x , p = func(x, p)
        X_ret.append(x)
        P_ret.append(p)
        i+=1
    return np.array(X_ret), np.array(P_ret)


# In[136]:

for K in np.linspace(0.,1.5,10):
    #Phasespace
    X,P = potrait()
    #unstable Manifold
    x0, p0 = [np.mod(-1*eigenvec[0, 0]*closeness,1), -1*eigenvec[0, 0]*closeness], [np.mod(eigenvec[0, 1]*closeness,1), eigenvec[0, 1]*closeness]
    X_unstable, P_unstable = orbit(x0,p0, 1000, standard_map)
    #stable Manifold
    x0, p0 = [np.mod(-1*eigenvec[1, 0]*closeness,1), -1*eigenvec[1, 0]*closeness], [np.mod(eigenvec[1, 1]*closeness,1), eigenvec[1, 1]*closeness]
    X_stable, P_stable = orbit(x0,p0, 1000, inv_standard_map)
    #Plot the phasespace
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('$x_n$')
    ax.set_ylabel('$p_n$')
    ax.set_title('K = {0}'.format(K))
    ax.scatter(X.reshape(1,-1), P.reshape(1,-1), s = 0.3, color='black')
    ax.scatter(X_unstable.reshape(1,-1), P_unstable.reshape(1,-1), s = 5., color = 'red')
    ax.scatter(X_stable.reshape(1,-1), P_stable.reshape(1,-1), s = 5., color = 'darkgreen')


# In[138]:

def std_map_nonmod(xn,pn):
    x = xn + pn
    p = np.mod(pn + float(K)/(2*np.pi)*np.sin(2*np.pi*x), 1)
    return x, p
std_map_nonmod = np.vectorize(std_map_nonmod)


# In[216]:

def search():
    K = 0.9716354
    p = np.linspace(0.59,0.61,10000)
    p0 = p
    x = np.zeros_like(p0)
    x_l, p_l = x,p
    tmax = 1000
    for i in range(tmax):
        x_l, p_l = x,p
        x , p = std_map_nonmod(x,p)
    w_no = x/float(tmax)
    return p0, w_no, x, x_l, p_l
    


# In[217]:

p0 , w_no,x, x_l, p_l = search()


# In[218]:

p_range = (0.59,0.61)
f = plt.figure()
ax = f.add_subplot(111)
ax.set_xlim(p_range)
ax.set_ylabel('Winding Number')
ax.set_xlabel('$p_0$')
ax.scatter(p0, w_no, s = 0.3, color='black')
ax.plot(np.arange(0,1,0.01), spcon.golden*np.ones_like(np.arange(0,1,0.01)) , color='red')
    
f = plt.figure()
ax = f.add_subplot(111)
ax.set_xlim(p_range)
ax.set_ylabel('Winding Number close to Golden Ratio')
ax.set_xlabel('$p_0$')
ax.grid(True)
#ax.plot(p0[(np.absolute(w_no-spcon.golden)<1e-2)][1]*np.ones(10), np.linspace(1.6,1.65, 10) , color='darkgreen')
ax.scatter(p0[(np.absolute(w_no-spcon.golden)<1e-2)], w_no[(np.absolute(w_no-spcon.golden)<1e-2)], s =5, color='black')
ax.plot(np.arange(0,1,0.01), spcon.golden*np.ones_like(np.arange(0,1,0.01)) , color='red')

f = plt.figure()
ax = f.add_subplot(111)
ax.set_xlim(p_range)
ax.set_ylabel('Difference n-1 th Winding number')
ax.set_xlabel('$p_0$')
ax.grid(True)
#ax.plot(p0[(np.absolute(w_no-spcon.golden)<1e-2)][1]*np.ones(10), np.linspace(0,0.02, 10) , color='darkgreen')
ax.scatter(p0[(np.absolute(w_no-spcon.golden)<1e-2)], ((x-x_l)/1000.)[(np.absolute(w_no-spcon.golden)<1e-2)], s = 5, color='black')

#print p0[(np.absolute(w_no-spcon.golden)<1e-2)][1]


# In[197]:

K = 0.9716354
X, P = potrait()
X_torus, P_torus = orbit(0., 0.02800280028, 1000)


# In[198]:

f = plt.figure()
ax = f.add_subplot(111)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xlabel('$x_n$')
ax.set_ylabel('$p_n$')
ax.set_title('K = {0}'.format(K))
ax.scatter(X.reshape(1,-1), P.reshape(1,-1), s = 0.3, color='black')
ax.scatter(X_torus.reshape(1,-1), P_torus.reshape(1,-1), s = 5., color = 'red')


# In[ ]:



