import casadi_kin_dyn.casadi_kin_dyn as cas_kin_dyn
from robot_descriptions.a1_description import URDF_PATH
import numpy as np
import casadi as ca
import pinocchio as pin
from urdfpy import URDF, Inertial
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

def loadData2(csv_path):
	return np.genfromtxt(csv_path, delimiter=',')

def derivative(t, x):
	dt = np.diff(t)
	dx = np.diff(x,1,0)/0.01
	dx = np.insert(dx,0,np.zeros((1,dx.shape[1])),axis=0)
	return dx

def loadURDF(urdf_path):
	urdf = open(urdf_path, "r").read()
	model = cas_kin_dyn.CasadiKinDyn(
	    urdf, root_joint=cas_kin_dyn.CasadiKinDyn.JointType.OMIT,
	    )

	return model

def getRegressor(model,data,urdf_path):

	#Ki = [14.87,13.26, 11.13, 10.62, 11.03, 11.47]
	#Ki = [50, 1, 1, 100, 100, 100]

	torqueReg = model.jointTorqueRegressor()
	modelpin = pin.buildModelFromUrdf(urdf_path)
	 

	nv = model.nv()
	nd = 13*nv
	N = len(data)

	dx = derivative(data[:,0],data[:,1:nv+1])
	ddx = derivative(data[:,0],dx)

	reg = np.zeros((0, nd))
	tor = np.zeros((0, 1))
	ones = np.identity(nv)
	#build regressor data
	for i in range(0,N):
	    row = data[i]
	    arr = torqueReg(row[1:nv+1],dx[i,:],ddx[i,:]).full()
	    #arr = pin.computeJointTorqueRegressor(modelpin, pin.Data(modelpin),row[1:nv+1],dx[i,:],ddx[i,:])
	    fric = dx[i,:]*ones;
	    fric2 =  np.sign(dx[i,:])*ones;
	    add3 = ones;
	    arr = np.append(arr, fric, axis=1)
	    arr = np.append(arr, fric2, axis=1)
	    arr = np.append(arr, add3, axis=1)
	    reg = np.append(reg, arr, axis=0)
	    tor = np.append(tor,  row[nv+1:2*nv+1])


	#tor[6:-1:7] = 100.0*tor[6:-1:7]
	return reg, tor


def fitParameters(reg, tor, nv):

	#pi = np.zeros((10*nv,0))
	pinv_A = np.linalg.pinv(reg)
	pi = np.matmul(pinv_A,tor)

	return pi

def fitParametersOpt(reg, tor, m_nom, nv):

	#pi = np.zeros((10*nv,0))
	pinv_A = np.linalg.pinv(reg)
	pi = np.matmul(pinv_A,tor)

	def f(x):
		y = np.dot(reg, x) - tor
		return np.dot(y, y)

	cons = ({'type': 'ineq', 'fun':lambda x:x[0]},)
	#cons = cons + ({'type': 'ineq', 'fun':lambda x:x[0]-1.0},)
	#print((cons, {'type': 'ineq', 'fun':lambda x:x[1*nv]-1}))
	
	for i in range(0,nv):
		cons = cons + ({'type': 'ineq', 'fun':lambda x,i=i,m_nom=m_nom:x[int(i*10)]-0.9*m_nom[i]},)
		cons = cons + ({'type': 'ineq', 'fun':lambda x,i=i,m_nom=m_nom:1.1*m_nom[i] - x[int(i*10)]},)
		cons = cons + ({'type': 'ineq', 'fun':lambda x,i=i,nv=nv,:x[int(nv*10+i)]},)
		cons = cons + ({'type': 'ineq', 'fun':lambda x,i=i,nv=nv,:x[int(nv*11+i)]},)
	
	res = optimize.minimize(f, np.zeros(13*nv), method='SLSQP', constraints=cons, 
	                    options={'disp': False})
	pi = res['x']

	return pi


def plotFit(pi, reg, tor, nv):

	fig, axs = plt.subplots(nv, 1)

	for i in range(0,nv):
		ax = axs[i]
		ax.plot(tor[i:-1:nv])
		ax.plot(np.matmul(pi,reg[i:-1:nv].T))
	
	plt.show()

def getNominalMass(urdf_path):
	modelpin = pin.buildModelFromUrdf(urdf_path)

	mass =  np.zeros(modelpin.nv)
	for i in range(1,modelpin.nv+1):
		mass[i-1]=modelpin.inertias[i].mass


	return mass

def showParameters(urdf_path, model, pi):

	modelpin = pin.buildModelFromUrdf(urdf_path)
	labels = ['m','cx','cy','cz','ixx','iyy','izz','ixy','ixz','iyz']
	print(labels)
	for i in range(0,model.nv()):
		param = modelpin.inertias[i+1]
		inertia = param.inertia
		param_array = [param.mass,param.lever[0],param.lever[1],param.lever[2],inertia[0,0],inertia[0,1],inertia[0,2],inertia[1,1], inertia[1,2],inertia[2,2]]	
		print(np.array_str(np.array(param_array), precision=1))
	print(labels)
	for i in range(0,model.nv()):
		modelpin.inertias[i+1]=modelpin.inertias[i+1].FromDynamicParameters(pi[10*i:(10*i+10)])
		param = modelpin.inertias[i+1]
		inertia = param.inertia
		param_array = [param.mass,param.lever[0],param.lever[1],param.lever[2],inertia[0,0],inertia[0,1],inertia[0,2],inertia[1,1], inertia[1,2],inertia[2,2]]
		print(np.array_str(np.array(param_array), precision=1))
		

def getURDFParameters(urdf_path):

	modelpin = pin.buildModelFromUrdf(urdf_path)
	pi = np.zeros(0)

	for i in range(0,modelpin.nv):
		pi = np.append(pi, modelpin.inertias[i+1].toDynamicParameters())
		
	zeros = np.zeros((modelpin.nv,1))
	pi = np.append(pi, zeros)
	pi = np.append(pi, zeros)

	return pi

def invDynamics(model,data,urdf_path):

	Ki = [14.87,13.26, 11.13, 10.62, 11.03, 11.47]
	rnea = model.rnea()

	nv = model.nv()
	nd = 10*nv
	N = len(data)

	tor = np.zeros((0, 1))

	modelpin = pin.buildModelFromUrdf(urdf_path)
	datapin = modelpin.createData()
	#build regressor data
	for i in range(0,N):
	    row = data[i]
	    #arr = rnea(row[0:nv],row[nv:2*nv],row[2*nv:3*nv]).full()
	    arr = pin.rnea(modelpin, datapin, row[0:nv],row[nv:2*nv],row[2*nv:3*nv])
	    tor = np.append(tor,  arr)

	return tor


def plotTor(tor, tor_meas, nv):

	fig, axs = plt.subplots(nv, 1)

	Ki = [14.87,13.26, 11.13, 10.62, 11.03, 11.47]

	for i in range(0,nv):
		ax = axs[i]
		ax.plot(tor[i:-1:6])
		ax.plot(tor_meas[i:-1:6])
	
	plt.show()

def plotParameters(urdf_path, model, pi):

	modelpin = pin.buildModelFromUrdf(urdf_path)
	labels = ['m','cx','cy','cz','ixx','iyy','izz','ixy','ixz','iyz']

	for i in range(0,model.nv()):
		modelpin.inertias[i+1]=modelpin.inertias[i+1].FromDynamicParameters(pi[:,i])
		
	fig, axs = plt.subplots(model.nv(), 1)

	for i in range(0,model.nv()):
		ax = axs[i]
		param = modelpin.inertias[i+1]
		inertia = param.inertia
		ax.bar(labels,[param.mass,param.lever[0],param.lever[1],param.lever[2],inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1], inertia[0,2],inertia[1,2]])

	plt.show()


'''
cons = (
	{'type': 'ineq', 'fun':lambda x:x[0]-7.5},
	{'type': 'ineq', 'fun':lambda x:8-x[0]},
	{'type': 'ineq', 'fun':lambda x:x[10]-12},
	{'type': 'ineq', 'fun':lambda x:14-x[10]},
	{'type': 'ineq', 'fun':lambda x:x[20]-3},
	{'type': 'ineq', 'fun':lambda x:5-x[20]},
	{'type': 'ineq', 'fun':lambda x:x[30]-1.8},
	{'type': 'ineq', 'fun':lambda x:2.5-x[30]},
	{'type': 'ineq', 'fun':lambda x:x[40]-1.8},
	{'type': 'ineq', 'fun':lambda x:2.5-x[40]},
	{'type': 'ineq', 'fun':lambda x:x[50]},
	{'type': 'ineq', 'fun':lambda x:1-x[50]},
	{'type': 'ineq', 'fun':lambda x:x[60:-1]})
'''