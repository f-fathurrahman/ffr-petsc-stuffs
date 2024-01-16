Tried on WSL2.

Need to avoid path with spaces.


# PETSc

Configure
```
./configure --prefix=/home/efefer/mysoftwares/petsc-3.20.3
```


At the end of `configure` process:
```
xxx=======================================================================================xxx
 Configure stage complete. Now build PETSc libraries with:
   make PETSC_DIR=/home/efefer/parallel_c/petsc-3.20.3 PETSC_ARCH=arch-linux-c-debug all
xxx=======================================================================================xxx
```

After calling `make` according to the prescription above:
```
=========================================
Now to install the libraries do:
make PETSC_DIR=/home/efefer/parallel_c/petsc-3.20.3 PETSC_ARCH=arch-linux-c-debug install
=========================================
```

Important notice after successful installation:
```
====================================
Install complete.
Now to check if the libraries are working do (in current directory):
make PETSC_DIR=/home/efefer/mysoftwares/petsc-3.20.3 PETSC_ARCH="" check
====================================
```


# SLEPc

Configuring SLEPc:
```
export PETSC_DIR=/home/efefer/mysoftwares/petsc-3.20.3
export PETSC_ARCH=""
./configure --prefix=/home/efefer/mysoftwares/slepc-3.20.1
```

Important notice after successful configuration:
```
================================================================================
SLEPc Configuration
================================================================================

SLEPc directory:
  /home/efefer/parallel_c/slepc-3.20.1
SLEPc prefix directory:
  /home/efefer/mysoftwares/slepc-3.20.1
PETSc directory:
  /home/efefer/mysoftwares/petsc-3.20.3
Prefix install with double precision real numbers

xxx==========================================================================xxx
 Configure stage complete. Now build the SLEPc library with:
   make SLEPC_DIR=/home/efefer/parallel_c/slepc-3.20.1 PETSC_DIR=/home/efefer/mysoftwares/petsc-3.20.3
xxx==========================================================================xxx
```

`Using double precision real numbers` setting is detected.


After successful compilation:
```
make SLEPC_DIR=/home/efefer/parallel_c/slepc-3.20.1 PETSC_DIR=/home/efefer/mysoftwares/petsc-3.20.3 install
```

After successful installation:
```
====================================
Install complete.
Now to check if the libraries are working do (in current directory):
make SLEPC_DIR=/home/efefer/mysoftwares/slepc-3.20.1 PETSC_DIR=/home/efefer/mysoftwares/petsc-3.20.3 PETSC_ARCH="" check
====================================
```

# Build

Set the environment variables:
```
export SLEPC_DIR=/home/efefer/mysoftwares/slepc-3.20.1
export PETSC_DIR=/home/efefer/mysoftwares/petsc-3.20.3
export PETSC_ARCH=""
```


