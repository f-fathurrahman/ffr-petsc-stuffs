basnam=`basename $1 .c`

# Can be obtained by `make`-ing some tutorials:
#mpicc -fPIC -Wall -Wwrite-strings -Wno-unknown-pragmas -Wno-lto-type-mismatch -fstack-protector \
#-fvisibility=hidden -g3 -O0  -I/home/efefer/mysoftwares/petsc-3.20.3-complex/include \
#-Wl,-export-dynamic $1

# Version 3.5.4
/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/bin/mpicc -c \
-fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -g3 -O0 \
-I/home/efefer/parallel_c/petsc-3.5.4/include \
-I/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/include $1

