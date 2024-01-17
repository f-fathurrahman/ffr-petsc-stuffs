basnam=`basename $1 .c`

# Can be obtained by `make`-ing some tutorials:
mpicc -fPIC -Wall -Wwrite-strings -Wno-unknown-pragmas -Wno-lto-type-mismatch -fstack-protector \
-fvisibility=hidden -g3 -O0  -I/home/efefer/mysoftwares/petsc-3.20.3-complex/include \
-Wl,-export-dynamic $1  \
-Wl,-rpath,/home/efefer/mysoftwares/petsc-3.20.3-complex/lib -L/home/efefer/mysoftwares/petsc-3.20.3-complex/lib \
-Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
-Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/9 -L/usr/lib/gcc/x86_64-linux-gnu/9 \
-lpetsc -llapack -lblas -lm -lX11 -ldl -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi \
-lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lpthread -lstdc++ -lquadmath -ldl -o $basnam.x
