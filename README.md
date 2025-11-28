# JaxPP

JaxPP is a JAX library enabling Multiple-Program Multiple-Data (MPMD)
pipeline parallelism through simple user annotations `pipeline_enter_stage(layer)`
and decorators `@mpmd_jit_with_loop`.

JaxPP automatically splits JAX computations into multiple SPMD modules that
are independently jitted and dispatched to different devices.


# Status
JaxPP is under active development, and its APIs are currently unstable and subject to change.

## Changelog

* [Aug 19, 2025] Users must now add a final `pipeline_enter_stage` to mark the last
  stage as well.

# Contacts

As project development is ongoing, we are not accepting Pull Requests to the GitHub repository.
Please contact the maintainers for any questions or concerns.

Issues and feature requests are welcome.




# Installation on ROCm-7.x

JaxPP dependencies and supported JAX versions are listed in [`pyproject.toml`](https://github.com/ROCm/jaxpp/-/blob/main/pyproject.toml).

## Installation of cupy
``` bash
git clone --recursive https://github.com/cupy/cupy.git
cd cupy
export HCC_AMDGPU_TARGET=gfx942  # This value should be changed based on your GPU
export __HIP_PLATFORM_HCC__
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/opt/rocm
export HIP_PATH=/opt/rocm
export HIPCC=/opt/rocm/bin/hipcc
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH}

pip install -v .
```
## Installation of JaxPP
```bash
git clone -b cj_rocm_jaxpp_dev https**://github.com/ROCm/jaxpp.git
cd jaxpp
pip install -e .
```

# Verification
You can verify the setup with [`examples/pp_vs_no_pp_gpt2_pp_tp_dp.py`](examples/pp_vs_no_pp_gpt2_pp_tp_dp.py) on a single-node with 8 GPUs.

## running with SPMD
```bash
python3 pp_vs_no_pp_gpt2_pp_tp_dp.py --mode=base --pp=1 --dp=1 --tp=1
```

```
(Config: PP=1, DP=1, TP=1, total_devices=1, train_steps=40
=== Running baseline (SPMD data parallel, no PP) ===
[baseline warmup] loss_sum=10.872209
[baseline 0005/0040] loss_sum=9.235132
[baseline 0010/0040] loss_sum=7.451328
2025-11-28 10:12:47.257395: starting profile (baseline)
2025-11-28 10:12:47.535898: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1764324767.560890   25327 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1764324767.566679   25327 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1764324767.605893   25327 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1764324767.605918   25327 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1764324767.605921   25327 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1764324767.605923   25327 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
[baseline 0015/0040] loss_sum=5.805967
[baseline 0020/0040] loss_sum=4.174860
2025-11-28 10:12:52.945652: blocking (baseline)
2025-11-28 10:12:52.949731: stopping profile (baseline)
[baseline 0025/0040] loss_sum=2.573964
[baseline 0030/0040] loss_sum=1.272751
[baseline 0035/0040] loss_sum=0.483864
[baseline 0040/0040] loss_sum=0.164586
[baseline] avg step time per step after profiling (steps 20..39): 276.524 ms)
```


##  running with PP, TP, DP
```bash
python3 pp_vs_no_pp_gpt2_pp_tp_dp.py --mode=pp --pp=2 --dp=1 --tp=4
```

```
Config: PP=2, DP=1, TP=4, total_devices=8, train_steps=40
=== Running pipeline-parallel (PP+DP+TP MPMD) ===
/pyenv/versions/3.12.10/lib/python3.12/site-packages/jax/_src/interpreters/mlir.py:1185: UserWarning: Some donated buffers were not usable: ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[32,8,1024,1]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[32,8,1024,1]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[1,1,512]), ShapedArray(float32[32,8,1024,1024]), ShapedArray(float32[32,8,1024,64]), ShapedArray(float32[32,8,1024,1024]), ShapedArray(float32[]), ShapedArray(float32[32,8,1024,64]), ShapedArray(float32[32,8,64,1024]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[1,1,512]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[1,1,512]), ShapedArray(float32[32,8,1024,1024]), ShapedArray(float32[32,8,1024,64]), ShapedArray(float32[32,8,1024,1024]), ShapedArray(float32[]), ShapedArray(float32[32,8,1024,64]), ShapedArray(float32[32,8,64,1024]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[1,1,512]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]).
See an explanation at https://docs.jax.dev/en/latest/faq.html#buffer-donation.
  warnings.warn("Some donated buffers were not usable:"
/pyenv/versions/3.12.10/lib/python3.12/site-packages/jax/_src/interpreters/mlir.py:1185: UserWarning: Some donated buffers were not usable: ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[1,1,512]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,8,1024,1024]), ShapedArray(float32[32,8,1024,64]), ShapedArray(float32[32,8,1024,1]), ShapedArray(float32[32,8,1024,1024]), ShapedArray(float32[32,8,1024,1]), ShapedArray(float32[]), ShapedArray(float32[32,8,1024,64]), ShapedArray(float32[32,8,64,1024]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[1,1,512]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,2048]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[1,1,512]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,8,1024,1024]), ShapedArray(float32[32,8,1024,64]), ShapedArray(float32[32,8,1024,1]), ShapedArray(float32[32,8,1024,1024]), ShapedArray(float32[32,8,1024,1]), ShapedArray(float32[]), ShapedArray(float32[32,8,1024,64]), ShapedArray(float32[32,8,64,1024]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,512]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[1,1,512]), ShapedArray(float32[32,1024,1]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024]), ShapedArray(float32[32,1024,512]), ShapedArray(int32[32,1024,1]).
See an explanation at https://docs.jax.dev/en/latest/faq.html#buffer-donation.
  warnings.warn("Some donated buffers were not usable:"
[pp warmup] loss_sum=10.873143
[pp 0005/0040] loss_sum=10.113508
[pp 0010/0040] loss_sum=9.279942
2025-11-28 10:14:24.503259: starting profile (PP)
2025-11-28 10:14:24.736599: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1764324864.745910   26108 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1764324864.748844   26108 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1764324864.757302   26108 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1764324864.757315   26108 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1764324864.757318   26108 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1764324864.757320   26108 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
[pp 0015/0040] loss_sum=7.984598
[pp 0020/0040] loss_sum=6.623155
2025-11-28 10:14:30.072466: blocking (PP)
2025-11-28 10:14:30.072779: stopping profile (PP)
[pp 0025/0040] loss_sum=5.287926
[pp 0030/0040] loss_sum=3.671954
[pp 0035/0040] loss_sum=2.230067
[pp 0040/0040] loss_sum=1.051485
[pp] avg step time per step after profiling (steps 20..39): 156.535 ms
```
The above loss comes down slower than that from SPMD at the moment.  

# Example

The example here shows the typical pattern used in a `flax` module to enable JaxPP.

```python
class ManualStagesModel(nn.Module):
    config: BertConfig
    pipeline_parallelism: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            FlaxBertLayer(
                self.config, name=f"flax_bert_layer_{i}", dtype=self.dtype
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(self, hidden_states):
        num_layers_per_stage = self.config.num_hidden_layers // self.pipeline_parallelism
        stage_id = 0
        for i, layer in enumerate(self.layers):
            outs = layer(hidden_states, None, None)
            hidden_states = outs[0]

            # Mark that we are entering a new stage
            if (i + 1) % num_layers_per_stage == 0:
                hidden_states = jaxpp.pipeline_enter_stage(hidden_states)
                stage_id += 1

        return hidden_states
```

And the code snippet below shows a typical train step function with JaxPP.
```python
def loss(pars, batch):
    res = model.apply(pars, batch)
    return jnp.mean((res - batch) ** 2) / num_mubatches, (res, 4)

# The `mpmd_jit_with_loop` transformation, with `treduce`,
# will make this function execute in mpmd_jit_with_loop fashion over 2 devices
# using the `Eager1F1B` schedule
@partial(jaxpp.mpmd_jit_with_loop, mpmd_mesh=mpmd_mesh)
def pp_train_step(opt_state, pars, batch):
    mubatch_grad = partial(jax.value_and_grad(loss_fn, has_aux=True), params)
    # Compute loss and gradients
    (losses, (pred, _)), grad = jaxpp.treduce(
        mubatch_grad, batch, schedule=jaxpp.Std1F1B(mpmd_mesh.mpmd_dim)
    )
    # Apply the optimizer as usual
    (updates, opt_state) = optimizer.update(grad, opt_state, pars)
    new_pars = optax.apply_updates(pars, updates)
    return opt_state, new_pars, losses, pred
```

To run the train step, we need to create a `MpmdMesh` object, which
is a wrapper of a standard Jax `Mesh` describing which dimension is the
mpmd one.

```python
devices = np.array(jax.devices()[0]).reshape(2, 1, 4)
jax_mesh = jax.sharding.Mesh(devices, ("mpmd", "data", "model"))
mpmd_mesh = jaxpp.MpmdMesh(jax_mesh, "mpmd")
print(mpmd_mesh.lowering_mesh().shape) # OrderedDict([('mpmd', 1), ('data', 1), ('model', 4)])
```

[examples/basic.py](examples/basic.py) provides a complete example.

# Building and Testing Docker Container

JaxPP provides Docker containers for development and testing. Currently it works on `rocm/jax-training:maxtext-v25.9`.



## Running Tests

The container includes several test suites that can be run:

1. **Unit Tests**:
```bash
docker run --gpus=all --shm-size=10.24gb --ulimit memlock=-1 --ulimit stack=67108864 \
  -e XLA_FLAGS='--xla_gpu_graph_level=0' --rm --workdir=/workdir/jaxpp jaxpp \
  "python /workdir/jaxpp/examples/basic.py --dtype=float32 && \
   python /workdir/jaxpp/examples/basic.py --dtype=float16"
```

2. **PyTest Suite**:
```bash
docker run --gpus=all --shm-size=10.24gb --ulimit memlock=-1 --ulimit stack=67108864 \
  -e XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  --rm --workdir=/workdir/jaxpp jaxpp "nvidia-smi && make install && pytest"
```

Note: The tests require GPU access and sufficient GPU memory.


# Multi-node setup
JaxPP needs to be installed on all nodes that are participating in the parallel
execution and the [installation instruction](#installation-instructions) needs
to be repeated on each node.
In addition, all packages that are needed for the execution of the workload
needs to be installed on all nodes.

# Benchmarks

JaxPP has been tested with several models from MaxText.
We have integrated JaxPP into a [fork of MaxText](https://github.com/NVIDIA/maxtext-jaxpp/blob/jaxpp/main/jaxpp.README.md) with minimal changes.


# Citing JaxPP

```
@misc{jaxpp,
      title={Scaling Deep Learning Training with MPMD Pipeline Parallelism}, 
      author={Anxhelo Xhebraj and Sean Lee and Hanfeng Chen and Vinod Grover},
      year={2024},
      eprint={2412.14374},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2412.14374}, 
}
```
