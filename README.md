## Dependencies

```bash
conda install -c conda-forge openblas lapack xtl xtensor xsimd nlohmann_json xtensor-blas
```

Set CMAKE prefix to look for the headers in the conda env:
```
```bash
-DCMAKE_PREFIX_PATH=/anaconda3/envs/env_name/lib/cmake
```

####Optional
To enable multi threaded loops, set `set(XTENSOR_USE_TBB 1)` in the root `CMakeLists.txt` and install the `TBB` dependency.
Will require setting the `TBB_INSTALL_DIR` env variable to point to the conda `lib` folder
```bash
conda install -c intel tbb  