# ruff: noqa: F401, F821, F841
import jax.numpy as jnp
import jax.random as jr
import jaxtyping as jt
import equinox as eqx
from interpax import interp1d
from pathlib import Path
from jax.numpy import inf as oo
from jax.numpy import nan as nan

from amici.jax.model import JAXModel, safe_log, safe_div
from amici import _module_from_path

net1 = _module_from_path('net1', Path(__file__).parent / 'net1.py')


class JAXModel_lv(JAXModel):
    api_version = '0.0.4'

    def __init__(self):
        self.jax_py_file = Path(__file__).resolve()
        self.nns = {"net1": net1.net(jr.PRNGKey(0))}
        self.parameters = jnp.array([])
        super().__init__()

    def _xdot(self, t, x, args):
        p, tcl, h = args

        prey, predator,  = x
        _ = p
        _ = tcl
        _ = h
        alpha, delta, flux_r1, flux_r2,  = self._w(t, x, p, tcl, h)

        dpreydt = flux_r1
        dpredatordt = flux_r2

        return jnp.array([dpreydt, dpredatordt])

    def _w(self, t, x, p, tcl, h):
        prey, predator,  = x
        _ = p
        _ = tcl
        _ = h

        alpha = self.nns['net1'].forward(jnp.array([prey, predator]))[0]
        delta = self.nns['net1'].forward(jnp.array([prey, predator]))[1]
        flux_r1 = alpha
        flux_r2 = delta

        return jnp.array([alpha, delta, flux_r1, flux_r2])

    def _x0(self, t, p):
        _ = p

        x00 = 0.500000000000000
        x01 = 0.500000000000000

        return jnp.array([x00, x01])

    def _x_solver(self, x):
        prey, predator,  = x

        x_solver0 = prey
        x_solver1 = predator

        return jnp.array([x_solver0, x_solver1])

    def _x_rdata(self, x, tcl):
        prey, predator,  = x
        _ = tcl

        prey = prey
        predator = predator

        return jnp.array([prey, predator])

    def _tcl(self, x, p):
        prey, predator,  = x
        _ = p

        

        return jnp.array([])

    def _y(self, t, x, p, tcl, h, op):
        prey, predator,  = x
        _ = p
        alpha, delta, flux_r1, flux_r2,  = self._w(t, x, p, tcl, h)
        _ = op

        prey_o = prey
        predator_o = predator

        return jnp.array([prey_o, predator_o])

    def _sigmay(self, y, p, np):
        _ = p

        prey_o, predator_o,  = y
        _ = np

        sigma_prey_o = 0.0500000000000000
        sigma_predator_o = 0.0500000000000000

        return jnp.array([sigma_prey_o, sigma_predator_o])

    def _nllh(self, t, x, p, tcl, h, my, iy, op, np):
        y = self._y(t, x, p, tcl, h, op)
        if not y.size:
            return jnp.array(0.0)

        prey_o, predator_o,  = y
        sigma_prey_o, sigma_predator_o,  = self._sigmay(y, p, np)

        Jy0 = 0.5*safe_log(2*jnp.pi*sigma_prey_o**2) + safe_div(0.5*(-my + prey_o)**2, sigma_prey_o**2)
        Jy1 = 0.5*safe_log(2*jnp.pi*sigma_predator_o**2) + safe_div(0.5*(-my + predator_o)**2, sigma_predator_o**2)

        return jnp.array([Jy0, Jy1]).at[iy].get()

    def _known_discs(self, p):
        _ = p

        return jnp.array([])

    def _root_cond_fn(self, t, y, args, **_):
        p, tcl, h = args

        prey, predator,  = y
        _ = p
        _ = tcl
        _ = h
        alpha, delta, flux_r1, flux_r2,  = self._w(t, y, p, tcl, h)

        

        return jnp.array([])

    def _root_cond_fn_event(self, ie, t, y, args, **_):
        """
        Root condition function for a specific event index.
        """
        __, __, h = args
        rval = self._root_cond_fn(t, y, args, **_)
        # only allow root triggers where trigger function is negative (heaviside == 0)
        masked_rval = jnp.where(h == 0.0, rval, 1.0)
        return masked_rval.at[ie].get()

    def _root_cond_fns(self):
        """Return root condition functions for discontinuities."""
        return [
            eqx.Partial(self._root_cond_fn_event, ie)
            for ie in range(self.n_events)
        ]

    @property
    def n_events(self):
        return 0

    @property
    def observable_ids(self):
        return "prey_o", "predator_o", 

    @property
    def state_ids(self):
        return "prey", "predator", 

    @property
    def parameter_ids(self):
        return tuple()

    @property
    def expression_ids(self):
        return "alpha", "delta", "flux_r1", "flux_r2", 


Model = JAXModel_lv
