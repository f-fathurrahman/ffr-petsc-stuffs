# Preliminary

Tried on WSL2.

Need to avoid path with spaces.

# PETSc 3.5.4

```
python2 './configure' '--with-fc=0' '--download-mpich' '--with-scalar-type=complex'
```

```
make PETSC_DIR=/home/efefer/parallel_c/petsc-3.5.4 PETSC_ARCH=arch-linux2-c-debug test
```

```
Running test examples to verify correct installation
Using PETSC_DIR=/home/efefer/parallel_c/petsc-3.5.4 and PETSC_ARCH=arch-linux2-c-debug
C/C++ example src/snes/examples/tutorials/ex19 run successfully with 1 MPI process
C/C++ example src/snes/examples/tutorials/ex19 run successfully with 2 MPI processes
Completed test examples
=========================================
Now to evaluate the computer systems you plan use - do:
make PETSC_DIR=/home/efefer/parallel_c/petsc-3.5.4 PETSC_ARCH=arch-linux2-c-debug streams NPMAX=<number of MPI processes you intend to use>
```


# PETSc

Configure
```
./configure --prefix=/home/efefer/mysoftwares/petsc-3.20.3
```

Using scalar=complex
```
./configure --prefix=/home/efefer/mysoftwares/petsc-3.20.3-complex --with-scalar-type=complex
```
complex with int32


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

Complex int32
```
export PETSC_DIR=/home/efefer/mysoftwares/petsc-3.20.3-complex
export PETSC_ARCH=""
./configure --prefix=/home/efefer/mysoftwares/slepc-3.20.1-complex
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


