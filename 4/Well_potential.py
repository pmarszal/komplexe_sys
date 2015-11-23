
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import ode
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 16, 12  # that's default image size for this interactive session


# In[2]:

#Param
g = 1.
#Grid
bound = 10.
rng = np.arange(-bound,bound,0.1)
grng = np.arange(0,bound,0.05)
X,Y = np.meshgrid(rng,rng)


# In[3]:

def well(t,x):
    return [x[1],-g*x[1]+x[0]-x[0]**3.]


# In[9]:

def cycle(t,x):
    return [x[1], -g*(x[1]*(x[0]**2.+x[1]**2.-1.))-x[0]]


# In[5]:

def default_criterion(y,t):
    return True


# In[6]:

def time_check(t, t_end, dt):
    if dt < 0. and t_end < dt:
        return t+dt>t_end
    elif dt > 0. and t_end > dt:
        return t+dt<t_end
    else :
        print "Error"
    


# In[7]:

def integrate(x0, t_end=10, dt=0.1, criterion = default_criterion, option='endval', well=well):
    
    if option == 'endval':
        r = ode(well).set_integrator('dopri5')
        r.set_initial_value(x0,0)
        while r.successful() and time_check(r.t,t_end,dt) and criterion(r.y,r.t):
            r.integrate(r.t+dt)
        return r.y
    elif option == 'orbit':
        x_final = []
        r = ode(well).set_integrator('dopri5')
        r.set_initial_value(x0,0)
        while r.successful() and time_check(r.t,t_end,dt) and criterion(r.y,r.t):
            r.integrate(r.t+dt)
            x_final.append(r.y)
        return x_final


# In[40]:

x_init = [1.0,1.0]
x_orbit = integrate(x_init,t_end = 8, option='orbit', well=cycle)
x_orbit = np.array(x_orbit)
x_init2 = [0.00001,0.001]
x_orbit2 = integrate(x_init2,t_end=20, option='orbit', well=cycle)
x_orbit2 = np.array(x_orbit2)
x = np.arange(-1,1,0.001)
y = np.sqrt(1-x**2)


# In[47]:

plt.plot(x_orbit[:,0],x_orbit[:,1], lw = 3)
plt.plot(x_orbit2[:,0],x_orbit2[:,1], lw=3)
plt.plot(x,y, color = 'r')
plt.plot(x,-y, color = 'r')
plt.scatter(x_orbit[0,0],x_orbit[0,1], color = 'b', lw = 4)
plt.scatter(x_orbit2[0,0],x_orbit2[0,1], color = 'g', lw=4)


# In[20]:

vectors = []
inits = np.arange(-1.5,1.5,0.1)
for i in inits:
    for j in inits:
        k= cycle(0,[i,j])
        vectors.append([i,j,k[0],k[1]])
vectors = np.array(vectors)


# In[49]:

plt.quiver(vectors[:,0],vectors[:,1],vectors[:,2],vectors[:,3])
plt.plot(x_orbit[:,0],x_orbit[:,1], lw = 3)
plt.plot(x_orbit2[:,0],x_orbit2[:,1], lw=3)
plt.plot(x,y, color = 'r')
plt.plot(x,-y, color = 'r')
plt.scatter(x_orbit[0,0],x_orbit[0,1], color = 'b', lw = 4)
plt.scatter(x_orbit2[0,0],x_orbit2[0,1], color = 'g', lw=4)


# In[ ]:



