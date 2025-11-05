from utilities import *

#urdf_path = "./lbr_iiwa7_r800.urdf"
urdf_path = "urdf/GEN3_URDF_V12_rev.urdf"
data = loadData2("data/kinova_cart_short2.csv")
model = loadURDF(urdf_path)

print(model.nq())
print(model.nv())
reg, tor = getRegressor(model,data,urdf_path)
print(reg.shape)

print(np.linalg.cond(reg))

pi2 = getURDFParameters(urdf_path)
#plotFit(pi2, reg, tor, model.nv())

'''
fig, axs = plt.subplots(7, 1)
for i in range(0,7):
	ax = axs[i]
	#ax.plot(reg[6:-1:7,60+i])
	ax.plot(data[:,i+1])
plt.show()





m_nom = getNominalMass(urdf_path)

pi = fitParametersOpt(reg, tor, m_nom, model.nv())

plotFit(pi, reg, tor, model.nv())
showParameters(urdf_path, model, pi)


#data = loadData2("data/ur-20_02_10-30sec_12harm.csv")
#reg, tor = getRegressor(model,data,urdf_path)

#plotFit(pi, reg, tor, model.nv())

#plotFit(pi, reg, tor, model.nv())
#pi2 = getURDFParameters(urdf_path)





model = loadURDF(urdf_path)
data = loadData2("ur-20_01_17-p6.csv")
data = data[:,1:]

reg, tor = getRegressor(model,data)

#pi = fitParametersOpt(reg, tor, model.nv())

plotFit(pi, reg, tor, model.nv())
#

#showParameters(urdf_path, model, pi)


#

#print(data[:,1])
#


q = ca.SX.sym("q", 6)   # symbolic generalized positions
v = ca.SX.sym("v", 6) 
a = ca.SX.sym("a", 6) 

print(torReg(q,v,a))


reg, tor = getRegressor(model,data)



tor2 = invDynamics(model,data,urdf_path)

plotTor(tor2,tor, 6)
'''
#plotFit(pi2, reg, tor, model.nv())

#

#plotFit(pi, reg, tor, model.nv())
#plotParameters(urdf_path, model, pi)
#showParameters(urdf_path, model, pi)
