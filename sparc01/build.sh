basnam=`basename $1 .c`

# Can be obtained by `make`-ing some tutorials:
#mpicc -fPIC -Wall -Wwrite-strings -Wno-unknown-pragmas -Wno-lto-type-mismatch -fstack-protector \
#-fvisibility=hidden -g3 -O0  -I/home/efefer/mysoftwares/petsc-3.20.3/include \
#-Wl,-export-dynamic $1  \
#-Wl,-rpath,/home/efefer/mysoftwares/petsc-3.20.3/lib -L/home/efefer/mysoftwares/petsc-3.20.3/lib \
#-Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
#-Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/9 -L/usr/lib/gcc/x86_64-linux-gnu/9 \
#-lpetsc -llapack -lblas -lm -lX11 -ldl -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi \
#-lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lpthread -lstdc++ -lquadmath -ldl -o $basnam.x


mpicc -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing \
-Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3  -fPIC -Wall -Wwrite-strings \
-Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3  \
-I/home/efefer/mysoftwares/petsc-3.12.5/include main.cc libsparc.a \
-Wl,-rpath,/home/efefer/mysoftwares/petsc-3.12.5/lib \
-L/home/efefer/mysoftwares/petsc-3.12.5/lib \
-Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib \
-L/usr/lib/x86_64-linux-gnu/openmpi/lib -Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/9 \
-L/usr/lib/gcc/x86_64-linux-gnu/9 -Wl,-rpath,/usr/lib/x86_64-linux-gnu \
-L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/lib/x86_64-linux-gnu \
-L/lib/x86_64-linux-gnu -lpetsc \
-llapacke -llapack -lblas -lm -lX11 -lstdc++ \
-ldl -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lgfortran \
-lm -lgfortran -lm -lgcc_s -lquadmath -lpthread -lquadmath -lstdc++ -ldl -o sparc01.x

