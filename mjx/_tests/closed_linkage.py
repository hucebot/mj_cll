import IK
import subprocess
import mujoco as mj
from mujoco import mjx
import jax
import jax.numpy as jnp
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from pymj_cll import ClosedLinkageMjWrapper as mj_cll
from pyIK import IKMj
import unittest

def torad(x):
    return x * 3.1415/180.

A = jnp.array([[1., 2., 3.],
              [4., 5., 6.]])
b = jnp.array([10., -11.])
u = jnp.array([10., 10., 10.])
l = jnp.array([-10., -10., -10.])

sol = IK.solveQP(A, b, l, u, reg=1e-4, init=jnp.zeros(3))

print("sol: ", sol)
print("A*sol: ", jnp.matmul(A, sol))

xml_path = str(subprocess.Popen(["locate", "-l1", "kangaroo_mujoco"], stdout=subprocess.PIPE).communicate()[0])[2:-3] + "/kangaroo_fixed_base.xml" #hacky...
print("model_path: ", xml_path)
#
m = mj.MjModel.from_xml_path(xml_path)
d = mj.MjData(m)


mj.mj_fwdPosition(m, d)


print("python rows", d.ne)
print("python cols", m.nq)
print("d.efc_J shape: ", np.shape(d.efc_J))


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

U = cl.getU()
P = cl.getP()

PU_ = np.matmul(P, U.transpose())
PU = jnp.array(PU_)

q_des_ = list()

q_des_.append(torad(-30.))
q_des_.append(torad(-20.))
q_des_.append(torad(10.))
q_des_.append(torad(90.))
q_des_.append(torad(30.))
q_des_.append(torad(10.))

q_des_.append(torad(30.))
q_des_.append(torad(-20.))
q_des_.append(torad(10.))
q_des_.append(torad(90.))
q_des_.append(torad(30.))
q_des_.append(torad(10.))

qdes = jnp.array(q_des_)

qlims = m.jnt_range
qmin_ = list()
qmax_ = list()
for qlim in qlims:
    qmin_.append(qlim[0])
    qmax_.append(qlim[1])

qmin = jnp.array(qmin_)
qmax = jnp.array(qmax_)


mjx_m = mjx.device_put(m)
mjx_d = mjx.device_put(d)
mjx_d = mjx.make_data(mjx_m)


mjx_d = mjx.forward(mjx_m, mjx_d)

dt = 0.01
sol = IK.computeIFKVelocity(mjx_d,
                            PU, qdes,
                            error=jnp.zeros(d.ne),
                            qmin=qmin, qmax=qmax,
                            dt=dt, reg=1e-6, ne=d.ne, nv=m.nq)

ik = IKMj(cl)
qdot = ik.computeIFKVelocity(m, d, q_des_, error=np.zeros(d.ne), qmin=qmin_, qmax=qmax_, dt=dt, reg=1e-6)

print("mj_cll_jax sol: ", sol)
print("mj_cll sol: ", qdot)

test = unittest.TestCase()
for mj_cll_jax_sol, mj_cll_sol in zip(sol, qdot):
    print(mj_cll_sol, " : ", mj_cll_jax_sol)








