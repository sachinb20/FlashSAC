"""Simulator backends for the shared Go2 environment.

Nothing is imported here. Genesis and IsaacLab cannot coexist in one virtualenv, so each
backend module is only imported by ``sim_backend.make_backend`` once a backend has
actually been selected.
"""
