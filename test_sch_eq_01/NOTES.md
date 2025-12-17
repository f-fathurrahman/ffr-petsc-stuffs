```
./main03.x -eps_nev 5 -eps_monitor_conv -eps_type jd
```

```
mpirun -np 4 ./main05.x \
  -eps_type jd \
  -eps_nev 5 \
  -eps_tol 1e-10 \
  -eps_monitor_conv \
  -st_type sinvert \
  -st_shift 0.0 \
  -st_ksp_type cg \
  -st_pc_type gamg
```