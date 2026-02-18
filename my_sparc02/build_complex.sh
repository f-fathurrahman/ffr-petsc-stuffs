set -x

basnam=`basename $1 .c`

# Version 3.5.4
PETSC_HOME="../sources/petsc-3.5.4"
${PETSC_HOME}/arch-linux2-c-debug/bin/mpicc \
-fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -g3 -O0 \
-I${PETSC_HOME}/include \
-I${PETSC_HOME}/arch-linux2-c-debug/include $1 \
libmain.a cblas_LINUX.a \
-L${PETSC_HOME}/lib -lpetsc \
-llapacke -llapack -lblas -lm \
-o ${basnam}.x


#PETSC_HOME="/home/efefer/WORKS/my_github_repos/ffr-petsc-stuffs/sources/petsc-3.5.4/"
#PETSC_HOME="../install/petsc-3.12.5_complex"
#mpicc \
#-fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -g3 -O0 \
#-I${PETSC_HOME}/include \
#-I${PETSC_HOME}/arch-linux2-c-debug/include $1 \
#libmain.a cblas_LINUX.a \
#-L${PETSC_HOME}/lib -lpetsc \
#-llapacke -llapack -lblas -lm \
#-o ${basnam}.x

#PETSC_HOME="../install/petsc-3.24.2_complex"
#C_OPTS="-fPIC -Wall -Wwrite-strings -Wno-unknown-pragmas -Wno-lto-type-mismatch -fstack-protector -fvisibility=hidden -g3 -O0"
#
#mpicc -c ${C_OPTS} \
#-I${PETSC_HOME}/include \
#$1


#/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/bin/mpicc \
#-o main.o \
#-c -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas \
#-g3 -O0  \
#-I/home/efefer/parallel_c/petsc-3.5.4/include \
#-I/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/include \
#main.c


#/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/bin/mpicc \
#-fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas \
#-g3 -O0 main.o libmain.a \
#-Wl,-rpath,/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/lib \
#-L/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/lib  -lpetsc \
#-llapacke -llapack -lblas -lX11 -lpthread -lm \
#-Wl,-rpath,/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/lib \
#-Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/9 -L/usr/lib/gcc/x86_64-linux-gnu/9 \
#-Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu \
#-Wl,-rpath,/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu \
#-lmpichcxx -lstdc++ \
#-Wl,-rpath,/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/lib -L/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/lib \
#-Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/9 -L/usr/lib/gcc/x86_64-linux-gnu/9 \
#-Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu \
#-Wl,-rpath,/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu \
#-Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl \
#-Wl,-rpath,/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/lib \
#-lmpich -lopa -lmpl -lrt -lpthread -lgcc_s -ldl -o sparc2.x
