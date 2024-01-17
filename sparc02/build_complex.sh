/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/bin/mpicc \
-o main.o \
-c -fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas \
-g3 -O0  \
-I/home/efefer/parallel_c/petsc-3.5.4/include \
-I/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/include \
main.c


/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/bin/mpicc \
-fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas \
-g3 -O0 main.o libmain.a \
-Wl,-rpath,/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/lib \
-L/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/lib  -lpetsc \
-llapacke -llapack -lblas -lX11 -lpthread -lm \
-Wl,-rpath,/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/lib \
-Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/9 -L/usr/lib/gcc/x86_64-linux-gnu/9 \
-Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu \
-Wl,-rpath,/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu \
-lmpichcxx -lstdc++ \
-Wl,-rpath,/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/lib -L/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/lib \
-Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/9 -L/usr/lib/gcc/x86_64-linux-gnu/9 \
-Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu \
-Wl,-rpath,/lib/x86_64-linux-gnu -L/lib/x86_64-linux-gnu \
-Wl,-rpath,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -ldl \
-Wl,-rpath,/home/efefer/parallel_c/petsc-3.5.4/arch-linux2-c-debug/lib \
-lmpich -lopa -lmpl -lrt -lpthread -lgcc_s -ldl -o sparc2.x
