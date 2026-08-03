#FROM ubuntu:rolling
FROM ubuntu:20.04
#From: nvidia/cuda:11.6.0-devel-ubuntu20.04

#%files
#    /home/domain/data/silwer/gpcr_martini2/scripts/gmx /usr/local/bin/gmx

# Install python3
RUN apt-get update -y ;\
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 \
    python3-pip
# Install latest CMAKE
RUN  pip3 install "cmake<4"
# Install dependencies for gromacs
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        libblas-dev \
        liblapack-dev \
        ninja-build \
        perl \
        wget \
        liblapack64-dev \
        libblas64-dev \
        git xxd  unzip tmux vim \ 
        gfortran \
        libopenmpi-dev

ENV TMPDIR=/tmp

WORKDIR  ${TMPDIR}
RUN cd /opt ;  wget -q -nc --no-check-certificate -P /opt https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip ; ls -lh
RUN cd /opt ;  unzip  "libtorch-cxx11-abi-shared-with-deps-2.2.0+cpu.zip" ; rm -f "libtorch-cxx11-abi-shared-with-deps-2.2.0+cpu.zip"
RUN echo "export CPATH=/opt/libtorch/include/torch/csrc/api/include/:/opt/libtorch/include/:/opt/libtorch/include/torch:$CPATH ; \ 
export INCLUDE=/opt/libtorch/include/torch/csrc/api/include/:/opt/libtorch/include/:/opt/libtorch/include/torch:$INCLUDE ;\
export LIBRARY_PATH=/opt/libtorch/lib:$LIBRARY_PATH ; export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH "  > /opt/libtorch/sourceme.sh

ENV  PLUMED_VERSION=2.9.0
RUN  mkdir -p ${TMPDIR} && wget -q -nc --no-check-certificate -P ${TMPDIR} https://github.com/plumed/plumed2/releases/download/v${PLUMED_VERSION}/plumed-src-${PLUMED_VERSION}.tgz ;\
    mkdir -p ${TMPDIR} && tar -x -f ${TMPDIR}/plumed-src-${PLUMED_VERSION}.tgz -C ${TMPDIR} -z ;\
    cd ${TMPDIR}/plumed-${PLUMED_VERSION} ;\
    . /opt/libtorch/sourceme.sh ;\
    env ;\
    sed -i.back 's/-std=c++14/-std=c++17/' ./configure ;\
    ./configure --prefix=/opt/plumed2.9 --enable-libtorch --enable-modules=all ;\
    make -j4 ;\
    make install 



WORKDIR  ${TMPDIR}
RUN wget https://github.com/dftbplus/dftbplus/releases/download/21.2/dftbplus-21.2.tar.xz ;\
    tar xf dftbplus-21.2.tar.xz ;\
    cd ${TMPDIR}/dftbplus-21.2/ ;\
    cmake -DWITH_API=TRUE -DWITH_TBLITE=true -DWITH_SDFTD3=true   -DBLAS_LIBRARY=/usr/lib/x86_64-linux-gnu/libblas.a -DLAPACK_LIBRARY=/usr/lib/x86_64-linux-gnu/liblapack.a -B _build . ;\
    cmake --build _build -- -j ;\
    cmake --install _build  ;\
    cp _build/src/dftbp/libdftbplus.a _build/external/mudpack/libmudpack.a _build/external/*/*/*.a  /usr/lib/x86_64-linux-gnu/ 



WORKDIR  ${TMPDIR}
ENV  PLUMED_VERSION=2.7.4
RUN  mkdir -p ${TMPDIR} && wget -q -nc --no-check-certificate -P ${TMPDIR} https://github.com/plumed/plumed2/releases/download/v${PLUMED_VERSION}/plumed-src-${PLUMED_VERSION}.tgz ;\
    mkdir -p ${TMPDIR} && tar -x -f ${TMPDIR}/plumed-src-${PLUMED_VERSION}.tgz -C ${TMPDIR} -z ;\
    cd ${TMPDIR}/plumed-${PLUMED_VERSION} ;\
    ./configure --prefix=/opt/plumed ;\
    make -j4 ;\
    make install 

    

WORKDIR  ${TMPDIR}
RUN git clone https://github.com/Vetrov-Anton/gromacs-dftbplus.git ;\
    cd ${TMPDIR}/gromacs-dftbplus ;\
    ln -s /opt/plumed/include/plumed/wrapper/Plumed.h ;\
    printf 'set(PLUMED_LOAD  "/opt/plumed/lib/libplumed.so" -ldl  )\nset(PLUMED_DEPENDENCIES  "/opt/plumed/libplumed.so")\n' > Plumed.cmake ;\
    printf 'PLUMED_LOAD="/opt/plumed/lib/libplumed.so" -ldl\nPLUMED_DEPENDENCIES="/opt/plumed/libplumed.so"\n' > Plumed.in
RUN cd ${TMPDIR}/gromacs-dftbplus ; sed -i 's/libdftd3.a/libs-dftd3.a/' src/programs/CMakeLists.txt  

RUN cd ${TMPDIR}/gromacs-dftbplus ;cmake -DGMX_X11=OFF -DGMX_QMMM_PROGRAM=dftbplus -DGMX_DOUBLE=YES -DGMX_GPU=OFF -DGMX_DEFAULT_SUFFIX=OFF -DGMX_QMMM_DFTBPLUS_LIB=/usr/lib/x86_64-linux-gnu/  -DGMX_OPENMP=ON  -DGMX_PLUMED=ON -DGMX_BUILD_OWN_FFTW=ON -DGMX_MPI=ON  -DGMX_BUILD_MDRUN_ONLY=Off  -DGMXAPI=off -DGMX_INSTALL_NBLIB_API=OFF 

RUN cd ${TMPDIR}/gromacs-dftbplus ;echo "/usr/bin/c++  -O3 -DNDEBUG      CMakeFiles/gmx_objlib.dir/gmx.cpp.o CMakeFiles/gmx_objlib.dir/legacymodules.cpp.o CMakeFiles/mdrun_objlib.dir/mdrun/mdrun.cpp.o CMakeFiles/mdrun_objlib.dir/mdrun/nonbonded_bench.cpp.o CMakeFiles/view_objlib.dir/view/view.cpp.o  -o ../../bin/gmx  -Wl,-rpath,\"\$ORIGIN/../lib:/opt/plumed/lib\" ../../lib/libgromacs.so.7.0.0 \
/usr/lib/x86_64-linux-gnu/libdftbplus.a \
/usr/lib/x86_64-linux-gnu/libmudpack.a \
/lib/x86_64-linux-gnu/libtblite.a \
/lib/x86_64-linux-gnu/libs-dftd3.a \
/lib/x86_64-linux-gnu/libdftd4.a \
/lib/x86_64-linux-gnu/libmulticharge.a \
/lib/x86_64-linux-gnu/libtoml-f.a \
/lib/x86_64-linux-gnu/libmstore.a \
/lib/x86_64-linux-gnu/libmctc-lib.a \
/lib/x86_64-linux-gnu/liblapack.a /lib/x86_64-linux-gnu/libblas.a  /usr/lib/x86_64-linux-gnu/libgfortran.so.5 /usr/lib/x86_64-linux-gnu/libgomp.so.1 -lm /opt/plumed/lib/libplumed.so -ldl /lib/x86_64-linux-gnu/libmpi.so.40  /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so /usr/lib/x86_64-linux-gnu/libpthread.so " > src/programs/CMakeFiles/gmx.dir/link.txt 

RUN cd ${TMPDIR}/gromacs-dftbplus ;cmake --build ${TMPDIR}/gromacs-dftbplus --target all -- -j4 ;\
    cmake --build ${TMPDIR}/gromacs-dftbplus --target install -- -j4 
#    rm -rf ${TMPDIR}/gromacs-dftbplus ${TMPDIR}/gromacs-dftbplus.tar.gz 

ENV SHELL /bin/bash
ENV USER vetrov
ENV UID 1000
ENV HOME /home/$USER
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Create whitelab user with UID=1000 and in the 'users' group
RUN useradd -m -s /bin/bash -N -u $UID $USER
USER $USER
RUN mkdir /home/$USER/work
WORKDIR /home/$USER/work
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/plumed2.9/lib:/opt/libtorch/lib
ENV PATH=${PATH}:/usr/local/gromacs/bin


# Additional shell to run CHARMM-GUI stuff
#%post
#    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#        csh
#%environment
#    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/plumed/lib
#%runscript 
#    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/plumed/lib
#    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/plumed/lib /usr/local/gromacs/bin/gmx "$@"
