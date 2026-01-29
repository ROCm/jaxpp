import jax
import jax.numpy as jnp
import jax.distributed as dist

from functools import partial
from jax import random
from flax import linen as nn
import numpy as np
import argparse, math

import jaxpp.api as jaxpp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax import value_and_grad
from jax.experimental.pjit import pjit
from jax.lax import with_sharding_constraint

def init_distributed():
  parser = argparse.ArgumentParser()
  parser.add_argument("--world_size", type=int, default=1)
  parser.add_argument("--rank", type=int, default=0)
  parser.add_argument("--coordination_service", type=str, default="")
  args, _ = parser.parse_known_args()

  dist.initialize(
      coordinator_address=args.coordination_service or "localhost:1234",
      num_processes=args.world_size,
      process_id=args.rank,
  )
  print(f"[rank {args.rank}] devices = {jax.local_devices()}")
  return (args.rank, args.world_size)

class MLP(nn.Module):
  hidden1: int = 256
  hidden2: int = 128
  out_dim: int = 64
  num_stages: int = 0
  
  def XDense(self, hidden, A, B):
    #k_init = nn.initializers.constant(const_val)
    def k_init(key, shape, dtype):
      K = jnp.linspace(A, B, math.prod(shape), dtype=dtype).reshape(shape)
      return K
    # nn.initializers.zeros
    def bias_init(key, shape, dtype):
      K = jnp.linspace(0, 1, math.prod(shape), dtype=dtype).reshape(shape)
      return jnp.cos(K)
      
    return nn.Dense(hidden) #, kernel_init=k_init, bias_init=bias_init)
    
  def setup(self):
    self.layers = [
        self.XDense(770, -0.5, 0.5),
        self.XDense(660, -0.1, 0.1),
        self.XDense(550, -0.1, 0.1),
        self.XDense(440, -0.1, 0.1),
        self.XDense(330, -1, 1),
     ]
    
  def __call__(self, x):
    num = self.num_stages
    for L in self.layers:
      x = L(x)
      x = nn.tanh(x)
      if num > 0:
         x = jaxpp.pipeline_enter_stage(x)
         num -= 1
    return x

def main():
  BATCH_SIZE=2048
  FEAT_SIZE=16
  # 4 ubatches
  #def_mpmd_idx 0 -> 1 shape (512, 770) --------
  #def_mpmd_idx 0 -> 1 shape (512, 770) --------
  # 16 ubatches
  #def_mpmd_idx 0 -> 1 shape (128, 770) --------
  #def_mpmd_idx 0 -> 1 shape (128, 770) --------
  
  N_UBATCHES=4      # number of microbatches
  LR=1e-3           # learning rate
  N_STEPS=400
  usePP=True
  jaxDistributed=True
  
  rank, world = init_distributed() if jaxDistributed else (0, 2)
  # pipeline parallelism - MPMD jaxpp
  # data parallelism - shard input data over GPUs
  # tensor parallelism - shard model weights over GPUs
  pp, dp, tp = (world, 1, 1) if usePP else (1, world, 1)
  batch_size = BATCH_SIZE // dp   # Local batch per rank

  print(f"pp {pp} dp {dp} tp {tp}")
  # Create a named global mesh "data"
  devices = mesh_utils.create_device_mesh((pp, dp, tp))
  mesh = Mesh(devices, axis_names=("stages", "data", "model"))
  print(f"[rank {rank}] mesh: {mesh}")
  if usePP:
    jaxpp_mesh = jaxpp.MpmdMesh(mesh, "stages")
    print(jaxpp_mesh)
  
  model = MLP(num_stages=pp if usePP else 0)
  param_sharding = NamedSharding(mesh, P())     # replicate parameters
  # HACK HACK
  data_sharding = NamedSharding(mesh, P()) #NamedSharding(mesh, P("data"))  # shard batch over processes
  replicate_sharding = NamedSharding(mesh, P())

  
  def loss_fn(params, xy):
    x, y = xy
    pred = model.apply(params, x)
    return (jnp.mean((pred - y) ** 2) / N_UBATCHES, pred)

  @jax.jit
  def gen_XY():
    #x = random.normal(key, (batch_size, FEAT_SIZE))
    x = jnp.linspace(-1, 1, batch_size * FEAT_SIZE)
    x = x.reshape([N_UBATCHES, -1, FEAT_SIZE])
    y = jnp.sum(x, axis=-1, keepdims=True)
    #y = jnp.exp(jnp.sin(y)+2)
    # Constrain local arrays to the proper sharding
    x = with_sharding_constraint(x, data_sharding)
    y = with_sharding_constraint(y, data_sharding)
    return (x, y)
  
  x_local, y_local = gen_XY()
  print(f"============= {x_local.shape}, {y_local.shape}", flush=True)
  
  
  def init_params(x):
      #key = jax.named_call(lambda: jax.random.key(9), name="rng_key")
      key = jax.random.key(9)
      return model.init(key, x)
  
  jit_init_params = jax.jit(
    init_params,
    in_shardings=data_sharding,
    out_shardings=param_sharding,
  )
  params = jit_init_params(x_local)
  
  def train_step(params, xy):
    ubatch_grad = partial(jax.value_and_grad(loss_fn, has_aux=True), params)
    
    if usePP:
      (losses, preds), grad = jaxpp.treduce(
            ubatch_grad, xy, schedule=jaxpp.Std1F1B(jaxpp_mesh.mpmd_dim))
      losses = jnp.array(losses)
      preds = jnp.array(preds)
    else:
      grad = None
      losses, preds = [], []
      for uidx in range(N_UBATCHES):
        uxy = jax.tree.map(lambda z: jax.numpy.take(z, uidx, axis=0), xy)
        (uloss, upred), ugrad = ubatch_grad(uxy)
        losses.append(uloss)
        preds.append(upred)
        grad = (jax.tree_util.tree_map(jnp.add, grad, ugrad)
                             if grad is not None else ugrad)

    new_params = jax.tree.map(lambda p, g: p - LR * g, params, grad)
    #loss = jnp.mean(jnp.asarray(losses))
    return new_params, jnp.asarray(losses), jnp.asarray(preds)
  
  if usePP:
    jit_train_step = jaxpp.mpmd_jit_with_loop(train_step, 
               mpmd_mesh=jaxpp_mesh)
  else:
    jit_train_step = jax.jit(
        train_step,
        in_shardings=(param_sharding, (data_sharding, data_sharding)),
        # (loss, grads) — replicated / small
        out_shardings=(replicate_sharding, replicate_sharding, replicate_sharding),
    )
  
  #jaxpr_fn = jax.make_jaxpr(value_and_grad_fn)
  #print(jaxpr_fn(val)) ---> prints jaxpr impression

  XY = (x_local, y_local)
  for step in range(N_STEPS):
    params, losses, preds = jit_train_step(params, XY)
    if (step + 1) % 10 == 0:
      if usePP and losses.is_partially_addressable:
        loss = losses.to_mpmd_local_array.sum() / N_UBATCHES
        print(f"[rank {rank}] step={step} part addr: {loss}")
      if not usePP:
        loss = losses.sum() / N_UBATCHES
        print(f"[rank {rank}] step={step} part addr: {loss}")

if __name__ == "__main__":
  main()
