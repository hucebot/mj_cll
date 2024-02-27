#!/usr/bin/env python

from distutils.core import setup

setup(name='mj_cll_jax',
      version='0.01',
      description='Functions for IK of closed chains in mujoco jax',
      author='Enrico Mingo Hoffman',
      author_email='enrico.mingo-hoffman@inria.fr',
      packages=["mj_cll_jax"],
      package_dir={"": "src"}
     )
