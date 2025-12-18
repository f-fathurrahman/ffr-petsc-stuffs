set -x

basnam=`basename $1 .c`

# Version 3.5.4
PETSC_HOME="../sources/petsc-3.5.4"
${PETSC_HOME}/arch-linux2-c-debug/bin/mpicc -c \
-fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -g3 -O0 \
-I${PETSC_HOME}/include \
-I${PETSC_HOME}/arch-linux2-c-debug/include $1


#PETSC_HOME="/home/efefer/WORKS/my_github_repos/ffr-petsc-stuffs/sources/petsc-3.5.4/"
#PETSC_HOME="../install/petsc-3.12.5_complex"
#mpicc -c \
#-fPIC -Wall -Wwrite-strings -Wno-strict-aliasing -Wno-unknown-pragmas -g3 -O0 \
#-I${PETSC_HOME}/include \
#-I${PETSC_HOME}/arch-linux2-c-debug/include $1

#PETSC_HOME="../install/petsc-3.24.2_complex"
#C_OPTS="-fPIC -Wall -Wwrite-strings -Wno-unknown-pragmas -Wno-lto-type-mismatch -fstack-protector -fvisibility=hidden -g3 -O0"
#
#mpicc -c ${C_OPTS} \
#-I${PETSC_HOME}/include \
#$1
