basnam=`basename $1 .c`

PETSC_HOME="/home/efefer/mysoftwares/petsc-3.20.3"
SLEPC_HOME="/home/efefer/mysoftwares/slepc-3.20.1"
# Probably it doesn't have to be this long
LIB_OTHER="-Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
-Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/9 -L/usr/lib/gcc/x86_64-linux-gnu/9 \
-lslepc -lpetsc -llapack -lblas -lm -lX11 -ldl -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi \
-lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lpthread -lstdc++ -lquadmath -ldl"

# Can be obtained by `make`-ing some tutorials:
mpicc -fPIC -Wall -Wwrite-strings -Wno-unknown-pragmas -Wno-lto-type-mismatch -fstack-protector \
-fvisibility=hidden -g3 -O0 \
-I${PETSC_HOME}/include \
-I${SLEPC_HOME}/include \
-Wl,-export-dynamic $1 \
-Wl,-rpath,${PETSC_HOME}/lib -L${PETSC_HOME}/lib \
-Wl,-rpath,${SLEPC_HOME}/lib -L${SLEPC_HOME}/lib \
${LIB_OTHER} \
-o $basnam.x
