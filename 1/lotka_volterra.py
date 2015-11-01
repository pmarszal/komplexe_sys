from matplotlib import pyplot as plt
import numpy as np
#import scipy as sc
from scipy.integrate import ode

a = 1.
b = 1.
c = 2.
d = 2.
fixpoints = [[0,0],[1,0],[b/d,a/c-b/(c*d)]]



x0, t0=np.array([0.5,0.7]), 0

def lotka_volterra(t, x):
	return [a*(1-x[0])*x[0]-c*x[0]*x[1], -b*x[1]+d*x[0]*x[1]]

'''def integration(x0):
	f = []
	r = ode(lotka_volterra).set_integrator('dopri5')
	r.set_initial_value(x0,t0)
	t1 = 10
	dt = 0.1
	while r.successful() and r.t+dt < t1:
		f.append(r.integrate(r.t+dt))
		print(r.t, f[-1])
	return f
'''
def loop_over_space():
	vectors = []
	for x in np.arange(0,1, 0.05):
		for y in np.arange(0,1, 0.05):
			vectors.append([[x,y],lotka_volterra(0,[x,y])])
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.set_xlabel('Prey')
	ax.set_ylabel('Predator')
	ax.set_title('a,b,c,d = [%1.1f, %1.1f, %1.1f, %1.1f ]' %(a,b,c,d))
	vectors = np.array(vectors)

	ax.quiver(vectors[:,0, 0],vectors[:,0,1],vectors[:,1,0],vectors[:,1,1])
	for fix in fixpoints:
		print fix
		ax.scatter(fix[0],fix[1], color='r', marker='o')
	plt.savefig('lotka_volterra.pdf')
	plt.close('all')

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.set_xlabel('Prey')
	ax.set_ylabel('Predator')
	ax.set_title('a,b,c,d = [%1.1f, %1.1f, %1.1f, %1.1f ]' %(a,b,c,d))
	ax.streamplot(vectors[:,0, 0],vectors[:,0,1],vectors[:,1,0],vectors[:,1,1])
	for fix in fixpoints:
		print fix
		ax.scatter(fix[0],fix[1], color='r', marker='o')
	plt.show()


loop_over_space()
