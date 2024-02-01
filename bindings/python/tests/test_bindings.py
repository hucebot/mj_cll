import unittest
import numpy as np
from pymj_cll import ClosedLinkageMjWrapper as mj_cll
from pymujokin import MujoKinWrapper as mjk
from pyIK import IKMj
import mujoco as mj
import random
import time
import unittest
import subprocess

from sympy.physics.paulialgebra import delta


def torad(x):
    return x * 3.1415/180.

xml_path = str(subprocess.Popen(["locate", "-l1", "kangaroo_mujoco"], stdout=subprocess.PIPE).communicate()[0])[2:-3] + "/kangaroo_fixed_base.xml" #hacky...
print("model_path: ", xml_path)

test = unittest.TestCase()

m = mj.MjModel.from_xml_path(xml_path)
d = mj.MjData(m)

mj.mj_fwdPosition(m, d)

print("python rows", d.ne)
print("python cols", m.nq)

active_dofs = list()
active_dofs.append("leg_left_1_motor")
active_dofs.append("leg_left_2_motor")
active_dofs.append("leg_left_3_motor")
active_dofs.append("leg_left_length_motor")
active_dofs.append("leg_left_4_motor")
active_dofs.append("leg_left_5_motor")
active_dofs.append("leg_right_1_motor")
active_dofs.append("leg_right_2_motor")
active_dofs.append("leg_right_3_motor")
active_dofs.append("leg_right_length_motor")
active_dofs.append("leg_right_4_motor")
active_dofs.append("leg_right_5_motor")

cl = mj_cll("kangaroo", active_dofs, m, d)
cl.print()
cl.update(compute_mapping_jacobian=False)
cl.computeMappingJacobian()  # not needed since

U = cl.getU()
print("U: ", U)

A = cl.getA()
print("A: ", A)

p = list()
p.append("leg_left_1_joint")
p.append("leg_left_2_joint")
p.append("leg_left_3_joint")
p.append("leg_left_knee_joint")
p.append("leg_left_4_joint")
p.append("leg_left_5_joint")
p.append("leg_right_1_joint")
p.append("leg_right_2_joint")
p.append("leg_right_3_joint")
p.append("leg_right_knee_joint")
p.append("leg_right_4_joint")
p.append("leg_right_5_joint")

cl.setP(p)
P = cl.getP()
print("P: ", P)
print("JointsP", cl.getJointsP())

print("ID: ", cl.getID())

print("Active DOFS: ", cl.getActiveDofs())
print("Passive DOFS: ", cl.getPassiveDofs())
print("Rows: ", cl.getRows())
print("NNZRows: ", cl.getNNZRows())
print("Cols: ", cl.getCols())
print("NNZRowsIndices: ", cl.getNNZRowsIndices())
print("ConstraintJacobian", cl.getConstraintJacobian())
print("PassivePartJacobian", cl.getPassivePartJacobian())
print("ActivePartJacobian", cl.getActivePartJacobian())
print("ConstraintError", cl.getConstraintError())
print("MappingJacobian", cl.getMappingJacobian())
print("MujocoJointMap", cl.getMujocoJointMap())

ik = IKMj(cl)
q_des = list()

q_des.append(torad(-30.))
q_des.append(torad(-20.))
q_des.append(torad(10.))
q_des.append(torad(90.))
q_des.append(torad(30.))
q_des.append(torad(10.))

q_des.append(torad(30.))
q_des.append(torad(-20.))
q_des.append(torad(10.))
q_des.append(torad(90.))
q_des.append(torad(30.))
q_des.append(torad(10.))

start_time = time.time()
test.assertTrue(ik.ikLoop(m, d, q_des, 1.0, 0.9, 1e-4, 1e-2))
q = np.matmul(cl.getP(), np.matmul(cl.getU().transpose(), d.qpos))
for qi, qdi in zip(q, q_des):
    test.assertAlmostEqual(qi, qdi, delta=1e-5)

end_time = time.time()
print("IKRESULT: ", ik.getIkResult())
print("#iterations: ", ik.getIterations())
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

h = mjk()
h.createScene(m)
f = lambda *args: None
h.loop(m, d, f, us=0, stop_in=5000)

qlims = m.jnt_range
qmin =  list()
qmax =  list()
print("Joint limits")
for qlim in qlims:
    qmin.append(qlim[0])
    qmax.append(qlim[1])
print("qlim: ", qmin, " : ", qmax)

for i in range(len(q_des)):
    q_des[i] = torad(random.uniform(-90, 90.0))

start_time = time.time()
ik.ikLoopQP(m, d, q_des, qmin, qmax, 1.0, 0.9, 1e-4, 1e-2)
for qmini, qmaxi, qi in zip(qmin, qmax, d.qpos):
    test.assertGreaterEqual(qmaxi+1e-6, qi)
    test.assertLessEqual(qmini-1e-6, qi)

end_time = time.time()
print("IKRESULT: ", ik.getIkResult())
print("#iterations: ", ik.getIterations())
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
h.createScene(m)
h.loop(m, d, f, us=0, stop_in=5000)
