set -x

basnam=`basename $1 .c`

PETSC_HOME="../install/petsc-3.24.2_complex"

#C_OPTS="-fPIC -Wall -Wwrite-strings -Wno-unknown-pragmas -Wno-lto-type-mismatch -fstack-protector -fvisibility=hidden -g3 -O0"

C_OPTS="-Wall -g3 -O0"

LIBS="-Wl,-rpath,${PETSC_HOME}/lib \
-L${PETSC_HOME}/lib \
-lpetsc \
-llapacke -llapack -lblas -lm"

mpicc ${C_OPTS} \
-I${PETSC_HOME}/include \
$1 -o $basnam.x \
$LIBS
