import casadi_kin_dyn.casadi_kin_dyn as cas_kin_dyn
from robot_descriptions.a1_description import URDF_PATH
import numpy as np
import casadi as ca
import pinocchio as pin
from urdfpy import URDF, Inertial
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy.optimize as optimize

import pygad
import math

from utilities import *

def getQ(a,b,q0,w,tf):
	q = np.zeros(tf)
	dq = np.zeros(tf)
	ddq = np.zeros(tf)

	for t in range(0,tf):
		q[t] = (a/w)*math.sin(w*t/1000.0)-(b/w)*math.cos(w*t/1000.0) + q0
		dq[t] = a*math.cos(w*t/1000.0)+b*math.sin(w*t/1000.0)
		ddq[t] = w*(b*math.cos(w*t/1000.0)-a*math.sin(w*t/1000.0))

	return q, dq, ddq 

def getReg(q,dq,ddq,torqueReg,n):

	reg = np.zeros((0, n))
	for i in range(0,len(q)):
		arr = torqueReg(q[i],dq[i],ddq[i]).full()
		reg = np.append(reg, arr, axis=0)

	return reg


urdf_path = "urdf/ur10e.urdf"
model = loadURDF(urdf_path)
torqueReg = model.jointTorqueRegressor()
print(model.nq())
print(model.nv())

q,dq,ddq = getQ(1,1,1,10,100)
reg = getReg(q,dq,ddq,torqueReg,60)
print(LA.cond(reg))

qsx = ca.SX.sym("q", 6)   # symbolic generalized positions
vsx = ca.SX.sym("v", 6) 
asx = ca.SX.sym("a", 6) 

#print(torqueReg)
out1, out2 = ca.qr(torqueReg(qsx,vsx,asx).T)

f = ca.Function("my_function", [qsx,vsx,asx], [out2.T])

reg = getReg(q,dq,ddq,f,6)
#print(reg.shape)


'''

def fitness_func(ga_instance, solution, solution_idx):








    return fitness

fitness_function = fitness_func

num_generations = 50
num_parents_mating = 4

sol_per_pop = 8
num_genes = len(function_inputs)

init_range_low = -2
init_range_high = 5

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)


ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = numpy.sum(numpy.array(function_inputs)*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
'''