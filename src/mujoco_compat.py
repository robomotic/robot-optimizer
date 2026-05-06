import numpy as np
import mujoco
import jax.numpy as jnp
from types import SimpleNamespace
from typing import Any
import jax


class Model:
    """Lightweight MJX-like wrapper around a mujoco.MjModel."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self._model = model
        self.geom_size = jnp.array(np.copy(model.geom_size))
        self.body_mass = jnp.array(np.copy(model.body_mass))

    @classmethod
    def from_mujoco(cls, model: mujoco.MjModel) -> "Model":
        return cls(model)

    def replace(self, **kwargs: Any) -> "Model":
        geom_size = kwargs.get("geom_size", self.geom_size)
        body_mass = kwargs.get("body_mass", self.body_mass)

        wrapped = Model(self.model)
        wrapped.geom_size = geom_size
        wrapped.body_mass = body_mass
        return wrapped

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


class Contact:
    def __init__(self, dist: Any = None):
        self.dist = dist if dist is not None else jnp.zeros((0,))


class Data:
    """Lightweight MJX-like data wrapper around mujoco.MjData."""

    def __init__(self, model: Model):
        self.model = model
        self.raw = mujoco.MjData(model.model)
        self.qpos = jnp.array(np.copy(self.raw.qpos))
        self.qvel = jnp.array(np.copy(self.raw.qvel))
        self.ctrl = jnp.array(np.copy(self.raw.ctrl))
        self.sensordata = jnp.array(np.copy(self.raw.sensordata))
        self.ncon = int(getattr(self.raw, "ncon", 0))
        self.contact = Contact()

    def replace(self, **kwargs: Any) -> "Data":
        data = Data.__new__(Data)
        data.model = self.model
        data.raw = self.raw
        data.qpos = kwargs.get("qpos", self.qpos)
        data.qvel = kwargs.get("qvel", self.qvel)
        data.ctrl = kwargs.get("ctrl", self.ctrl)
        data.sensordata = kwargs.get("sensordata", self.sensordata)
        data.ncon = kwargs.get("ncon", self.ncon)
        data.contact = kwargs.get("contact", self.contact)
        return data


# Register Data as a JAX pytree
def _data_flatten(data):
    return (
        (data.qpos, data.qvel, data.ctrl, data.sensordata, data.ncon),
        (data.model, data.raw, data.contact)
    )

def _data_unflatten(aux, children):
    qpos, qvel, ctrl, sensordata, ncon = children
    model, raw, contact = aux
    data = Data.__new__(Data)
    data.model = model
    data.raw = raw
    data.qpos = qpos
    data.qvel = qvel
    data.ctrl = ctrl
    data.sensordata = sensordata
    data.ncon = ncon
    data.contact = contact
    return data

jax.tree_util.register_pytree_node(Data, _data_flatten, _data_unflatten)


def step(model: Model, data: Data) -> Data:
    raw = mujoco.MjData(model.model)
    # Convert JAX arrays to numpy for mujoco
    raw.qpos[:] = np.array(data.qpos)
    raw.qvel[:] = np.array(data.qvel)
    raw.ctrl[:] = np.array(data.ctrl)
    mujoco.mj_step(model.model, raw)
    return Data.from_raw(model, raw)


mjx = SimpleNamespace(Model=Model, Data=Data, step=step)
