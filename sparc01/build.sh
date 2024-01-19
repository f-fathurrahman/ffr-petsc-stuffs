basnam=`basename $1 .cc`

#mpicc -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing \
#-Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3  -fPIC -Wall -Wwrite-strings \
#-Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden -g3  \
#-I/home/efefer/mysoftwares/petsc-3.12.5/include $1 \
#-Wl,-rpath,/home/efefer/mysoftwares/petsc-3.12.5/lib \
#-L/home/efefer/mysoftwares/petsc-3.12.5/lib \
#-Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib \
#-L/usr/lib/x86_64-linux-gnu/openmpi/lib -Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/9 \
#-L/usr/lib/gcc/x86_64-linux-gnu/9 -Wl,-rpath,/usr/lib/x86_64-linux-gnu \
#-L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/lib/x86_64-linux-gnu \
#-L/lib/x86_64-linux-gnu -lpetsc \
#-llapacke -llapack -lblas -lm -lX11 -lstdc++ \
#-ldl -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lgfortran \
#-lm -lgfortran -lm -lgcc_s -lquadmath -lpthread -lquadmath -lstdc++ -ldl -o $basnam.x

#/home/efefer/mysoftwares/petsc-3.12.5_mpich/bin/mpicc \
#-Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector -fvisibility=hidden \
#-g3  -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -fstack-protector \
#-fvisibility=hidden -g3  \
#-I/home/efefer/mysoftwares/petsc-3.12.5_mpich/include $1 \
#-Wl,-rpath,/home/efefer/mysoftwares/petsc-3.12.5_mpich/lib -L/home/efefer/mysoftwares/petsc-3.12.5_mpich/lib \
#-Wl,-rpath,/home/efefer/mysoftwares/petsc-3.12.5_mpich/lib -L/home/efefer/mysoftwares/petsc-3.12.5_mpich/lib \
#-Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/9 -L/usr/lib/gcc/x86_64-linux-gnu/9 \
#-Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu \
#-Wl,-rpath,/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu \
#-lpetsc -llapack -lblas -lpthread -lm -lX11 -lstdc++ -ldl \
#-lmpifort -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lstdc++ -ldl -o $basnam.x


/home/efefer/mysoftwares/petsc-3.12.5_mpich/bin/mpicc -Wall \
-I/home/efefer/mysoftwares/petsc-3.12.5_mpich/include $1 \
-Wl,-rpath,/home/efefer/mysoftwares/petsc-3.12.5_mpich/lib -L/home/efefer/mysoftwares/petsc-3.12.5_mpich/lib \
-Wl,-rpath,/home/efefer/mysoftwares/petsc-3.12.5_mpich/lib -L/home/efefer/mysoftwares/petsc-3.12.5_mpich/lib \
-Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/9 -L/usr/lib/gcc/x86_64-linux-gnu/9 \
-Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu \
-Wl,-rpath,/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu \
-lpetsc -llapack -lblas -lpthread -lm -lX11 -lstdc++ -ldl \
-lmpifort -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lstdc++ -ldl -o $basnam.x