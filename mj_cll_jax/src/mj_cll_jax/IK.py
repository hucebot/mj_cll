import dataclasses
from typing import Tuple
from mujoco import mjx
from mujoco.mjx._src.dataclasses import PyTreeNode
import jax
import jax.numpy as jnp
import mujoco
from dataclasses import dataclass
from typing import List
from jaxopt import BoxCDQP
from functools import partial

@jax.jit
def solveQP(A: jax.numpy.array, b: jax.numpy.array,
            l: jax.numpy.array, u: jax.numpy.array,
            reg: jax.numpy.double, init: jax.numpy.array) -> jax.numpy.array:
    """
    Solve min square problems simply bounded using a QP

    :param A: matrix
    :param b: vector
    :param l: lower bound
    :param u: upper bound
    :param reg: regularization term
    :param init: initial variable state
    :return: solution
    """
    H1 = jnp.matmul(A.transpose(), A)
    H2 = jnp.eye(jnp.shape(H1)[0])
    H = H1 + reg * H2

    g = jnp.matmul(-A.transpose(), b)


    qp = BoxCDQP()
    return qp.run(init, params_obj=(H, g), params_ineq=(l, u)).params
@partial(jax.jit, static_argnames=['ne', 'nv'])
def createA(d: mjx.Data, PU: jax.numpy.array, ne: int, nv: int) -> jax.numpy.array:
    """
    Creates A matrix for optimization problem, it concatenates the selection matrix with the constraint (closed loops) matrix

    :param d: mujoco jax data
    :param PU: selection matrix for passive joints to control
    :param ne: number of equalities
    :param nv: number of dofs
    :return: A matrix
    """
    efc_J = jnp.array(d.efc_J[0:ne, 0:nv])
    return jnp.concatenate((PU, efc_J))
@jax.jit
def createb(q_error, constraint_error) -> jax.numpy.array:
    """
    Creates b vector for optimization problem, it concatenates the controlled joints error with the closed loops errors
    :param q_error: error associated to the joints to be controlled
    :param constraint_error: error associated to the closed loops
    :return: b vector
    """
    return jnp.concatenate((q_error, -constraint_error))
@jax.jit
def createJointLimitsConstraint(d: mjx.Data, qmin: jax.numpy.array, qmax: jax.numpy.array, dt: jax.numpy.double) -> Tuple[jax.numpy.array, jax.numpy.array]:
    """
    Creates lower and upper bound. The bounds correspond to joint limits constraints

    :param d: mujoco data
    :param qmin: inferior joint limits
    :param qmax: superior joint limits
    :param dt: loop time
    :return: lower and upper bounds
    """
    l = (qmin - d.qpos)/dt
    u = (qmax - d.qpos)/dt
    return l, u
@partial(jax.jit, static_argnames=['ne', 'nv'])
def createProblem(d: mjx.Data,
                  PU: jax.numpy.array,
                  qmin: jax.numpy.array, qmax: jax.numpy.array,
                  qdot_ref: jax.numpy.array,
                  error: jax.numpy.array,
                  dt: jax.numpy.double,
                  ne: int, nv: int) -> Tuple[jax.numpy.array, jax.numpy.array, jax.numpy.array, jax.numpy.array]:
    """
    Creates matrices and vectors to solve the problem

    :param d: mujoco data
    :param PU: selection matrix for passive joints to control
    :param qmin: inferior joint limits
    :param qmax: superior joint limits
    :param qdot_ref: reference velocity
    :param error: error associated to the closed loops
    :param dt: loop time
    :param ne: number of equalities
    :param nv: number of dofs
    :return: A matrix, b vector, lower and upper bounds
    """
    A = createA(d, PU, ne, nv)

    b = createb(qdot_ref, error)

    l, u = createJointLimitsConstraint(d, qmin, qmax, dt)

    return A, b, l, u

@partial(jax.jit, static_argnames=['ne', 'nv'])
def computeIFKVelocity(d: mjx.Data,
                       PU: jax.numpy.array, qdot_ref: jax.numpy.array,
                       error: jax.numpy.array,
                       qmin: jax.numpy.array, qmax: jax.numpy.array,
                       dt: jax.numpy.double,
                       reg: jax.numpy.double,
                       ne: int, nv: int,
                       init: jax.numpy.array = jax.numpy.zeros(0)) -> jax.numpy.array:
    """
    Solve the inverse kinematics problem associated to:
        given a desired velocity for a subset of passive dfos, find the velocity of the whole chain taking into account closed loops
        and joint limits.

    :param d: mujoco data
    :param PU: selection matrix for passive joints to control
    :param qdot_ref: reference velocity
    :param error: error associated to the closed loops
    :param qmin: inferior joint limits
    :param qmax: superior joint limits
    :param dt: loop time
    :param reg: regularization value for the QP
    :param ne: number of equality constraints
    :param nv: number of dofs
    :param init: initial variable state
    :return: solution
    """
    A, b, l, u = createProblem(d, PU, qmin, qmax, qdot_ref, error, dt, ne, nv)

    if jnp.shape(init)[0] == 0:
        init = jnp.zeros(jnp.shape(qmin)[0])

    return solveQP(A, b, l, u, reg=reg, init=init)


