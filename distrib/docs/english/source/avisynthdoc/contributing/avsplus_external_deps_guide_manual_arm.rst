
Building AviSynth+'s external dependencies for Windows on ARM
=============================================================

.. toctree::
    :maxdepth: 3

.. contents:: Table of contents


Unlike for x86 or x64, where we support using MSVC to build the
AviSynth+ core, for Windows on ARM we only support using MinGW-based
compilers.  At present this is restricted to llvm-mingw, which
uses Clang.  Windows on ARM support has currently (as of February of 2025)
not appeared in a stable version of GCC, but whenever it does, that
would be acceptable as well.

Owing to this fact, the build process is different from the assumptions
made for x86/x64 with MSVC.  The easiest way is probably to cross-compile
from a Linux distro (that includes WSL2, especially for running the task
directly on a Windows on ARM device, since Snapdragon X support is still
in its early stages for most/all distros).

The instructions are going to assume Ubuntu as the distro in question,
particularly as it relates to things like the repositories.

Make sure you have a list of other necessary build system and source download
components.

    ::

        sudo apt-get install build-essential gcc-multilib g++-multilib \
        checkinstall nasm yasm cvs git gperf subversion mercurial automake* \
        autoconf* libtool* m4 bison flex p7zip-full lzip texinfo help2man \
        tofrodos texi2html docutils-common cmake pkgconf bzr autopoint meson \
        ninja-build gettext binfmt-support ruby doxygen gtk-doc-tools zlib1g-dev \
        python-is-python3 python3-setuptools


Any time \ is at the end of a line, it means the command spans multiple lines.
Make sure to copy the entire command.  To make this easier to see, such
commands have been indented.

The '&& \' at the end of each instruction is to allow running the
entire piece at once.  This is for convenience, and should just work.
If there are errors, run each instruction one at a time and adjust
accordingly.


Create staging areas and a space for constructed packages to be stored:

    ::

        mkdir -p ~/mingw-packages ~/mpv-build-deps ~/mingw_debs/aarch64



Cross-compilation Toolchain
---------------------------

LLVM/MinGW installation
^^^^^^^^^^^^^^^^^^^^^^^

.. Note:
    If the process fails with an error related to cc1plus
    being terminated, reduce the number of jobs by using
    the CMAKE_BUILD_PARALLEL_LEVEL environment variable.
..

    ::

        cd ~/mingw-packages && \
        mkdir llvm-mingw-build && \
        git clone https://github.com/mstorsjo/llvm-mingw && \
        cd llvm-mingw && \
        ./build-all.sh ../llvm-mingw-build


Packaging Preparation
^^^^^^^^^^^^^^^^^^^^^

Copy NASM and pkg-config into the bin directory of the toolchain so that they
can easily be found when a prefixed copy of these tools are needed.

    ::

        cd ../llvm-mingw-build && \
        cp /usr/bin/pkg-config bin/aarch64-w64-mingw32-pkg-config && \

A few of these pieces require using CMake to build them. This is not as
straight-forward as using autotools to cross-compile, and requires some setup:

    ::

        cd aarch64-w64-mingw32 && \
        wget https://fastapi.metacpan.org/source/TOKUHIROM/mRuby-0.06/vendor/mruby/cmake/Toolchain-Ubuntu-mingw32.cmake.sample -O toolchain-aarch64-w64-mingw32.cmake && \
        sed -i -e 's/ ~\/crossdev\/w32//g' -e 's/i686/aarch64/g' -e 's/usr/usr\/llvm-mingw/g' toolchain-aarch64-w64-mingw32.cmake

Setting up meson's cross-files:

    ::

        cd ../ && \
        mkdir -p share/meson/cross/ && \
        wget "https://raw.githubusercontent.com/mesonbuild/meson/master/cross/linux-mingw-w64-64bit.txt" -O share/meson/cross/aarch64-w64-mingw32 && \

        sed -i -e 's/usr\/bin\/x86_64/usr\/llvm-mingw\/bin\/aarch64/g' -e 's/x86_64/aarch64/g' \
        -e 's/wine64//g' -e 's/wine//g' \
        -e '13,16d' -e "s/\[properties\]/\[built-in options\]/" \
        share/meson/cross/aarch64-w64-mingw32

        sed -i -e "13ic_args = ['-I/usr/aarch64-w64-mingw32/include']\ncpp_args = ['-I/usr/aarch64-w64-mingw32/include']\nc_link_args = ['-L/usr/aarch64-w64-mingw32/lib']\ncpp_link_args = ['-L/usr/aarch64-w64-mingw32/lib']" \
        share/meson/cross/aarch64-w64-mingw32


Installing the toolchain
^^^^^^^^^^^^^^^^^^^^^^^^

    ::

        cd ~/mingw-packages && \
            sudo checkinstall --pkgname=llvm-mingw \
            --pkgversion="1:$(llvm-mingw-build/bin/x86_64-w64-mingw32-clang --version | head -1 | \
            cut -f3 -d ' ')" --backup=no --deldoc=yes --delspec=yes --deldesc=yes \
            --strip=yes --stripso=yes --addso=yes --fstrans=no --default cp -R llvm-mingw-build /usr/llvm-mingw && \
        mv *.deb ~/


Adding LLVM/MinGW to the PATH
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To try and avoid a collision between MinGW/GCC and LLVM/MinGW,
LLVM/MinGW should be added to the *end* of the $PATH.  Since it
will be the only thing providing -mingw32 prefixes for clang and aarch64,
and the only source of clang-cl, those will go through correctly,
otherwise, the PATH ordering will prefer real GCC to the symlinks
that LLVM/MinGW creates.

    ::

        echo "export PATH=\$PATH:/usr/llvm-mingw/bin" >> ~/.bashrc

In order to use checkinstall with llvm-mingw, you'll also need
to add the PATH epxort to the root user's .bashrc.

    ::

        echo "export PATH=\$PATH:/usr/llvm-mingw/bin" | sudo tee -a /root/.bashrc

Close and re-open the Terminal.  `source ~/.bashrc` might also work, but
whether it also works for the root user, I don't know.

There is an issue with attempting to use some of the aarch64-w64-mingw32
tools when checkinstall gets involved, and errors out saying that the aarch64
binaries can't be found, even though due to the two above commands,
they are on the $PATH.  This could be resolved by the user elevating
to root with `sudo su` and then running checkinstall, but a slightly
less cumbersome approach is to just simply symlink the tools into
/usr/bin.

    ::

        sudo ln -s /usr/llvm-mingw/bin/aarch64-w64-mingw32-ranlib /usr/bin/aarch64-w64-mingw32-ranlib && \
        sudo mkdir -p /usr/share/meson/cross && \
        sudo ln -s /usr/llvm-mingw/share/meson/cross/aarch64-w64-mingw32 /usr/share/meson/cross/aarch64-w64-mingw32


SoundTouch
----------

SoundTouch is used by the TimeStretch plugin.

Jump to the correct drive and source location:

    ::

        cd ~/mpv-build-deps

Clone the source using Git:

    ::

        git clone https://codeberg.org/soundtouch/soundtouch

Create the proper build directory:

    ::

        mkdir -p soundtouch/build

Enter the build directory:

    ::

        cd ~/mpv-build-deps/soundtouch/build && \

Configure the build:

    ::

        cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=/usr/aarch64-w64-mingw32 \
        -DCMAKE_TOOLCHAIN_FILE="/usr/llvm-mingw/aarch64-w64-mingw32/toolchain-aarch64-w64-mingw32.cmake" \
        -DCMAKE_EXE_LINKER_FLAGS="-municode" && \

Compile:

    ::

        ninja

Install:

    ::

        sudo checkinstall --pkgname=soundtouch-mingw-aarch64 --pkgversion="$(grep -r \
        Version soundtouch.pc | cut -f2 -d " ")-$(date --rfc-3339=date | \
        sed 's/-//g')-git" --backup=no --deldoc=yes --delspec=yes --deldesc=yes \
        --strip=yes --fstrans=no --default ninja install && \

Move the created package to the package cache:

    ::

        mv *.deb ~/mingw_debs/aarch64



Building DevIL's dependencies manually
--------------------------------------


.. WARNING::

    What follows is the dependency chain for building DevIL locally, along with
    all of its dependencies and their dependencies.  Consider this an exercise
    for the masochistic.  It is, however, laid out in a more or less linear
    fashion to make it easier to follow along.

    Unlike the SoundTouch steps above, the reason for each step won't be
    explained, simply because it would introduce a massive amount of
    identical text.  Every one of these are broken up into a group of steps
    to jump into the source download area, download the source, and create
    the build subdirectories.

    Unless there's something important to note about the options or something
    weird to account for, the description for those steps are exactly the same.

    If there are weird things to account for, they'll be noted.


zlib-ng
^^^^^^^

.. Note::
    -DPKGCONFIG_INSTALL_DIR is necessary because otherwise it will
    install to root.
..

    ::

        cd ~/mpv-build-deps && \
        git clone https://github.com/zlib-ng/zlib-ng && \
        mkdir -p zlib-ng/zlib-ng-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/zlib-ng/zlib-ng-build/aarch64 && \
            cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=/usr/aarch64-w64-mingw32 \
            -DCMAKE_TOOLCHAIN_FILE="/usr/llvm-mingw/aarch64-w64-mingw32/toolchain-aarch64-w64-mingw32.cmake" \
            -DBUILD_SHARED_LIBS:bool=off -DZLIB_COMPAT:bool=on -DZLIB_ENABLE_TESTS:bool=off \
            -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DPKGCONFIG_INSTALL_DIR=/usr/aarch64-w64-mingw32/lib/pkgconfig && \
        ninja && \
            sudo checkinstall --pkgname=zlib-mingw-aarch64 --pkgversion="$(git describe \
            --tags)-$(date --rfc-3339=date | sed 's/-//g')-git" --backup=no \
            --deldoc=yes --delspec=yes --deldesc=yes --strip=yes --fstrans=no --default \
            ninja install && \
        mv *.deb ~/mingw_debs/aarch64


xz-tools
^^^^^^^^

    ::

        cd ~/mpv-build-deps && \
        wget http://tukaani.org/xz/xz-5.6.3.tar.gz -O - | tar -xzvf - && \
        mkdir -p xz-5.6.3/xz-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/xz-5.6.3/xz-build/aarch64 && \
            ../../configure --prefix=/usr/aarch64-w64-mingw32 --disable-shared \
            --disable-nls --enable-silent-rules --host=aarch64-w64-mingw32 && \
        make -j$(nproc) && \
            sudo checkinstall --pkgname=xz-tools-mingw-aarch64 --pkgversion="$(grep Version \
            src/liblzma/liblzma.pc | sed 's/Version: //g')-$(date --rfc-3339=date | \
            sed 's/-//g')" --backup=no --deldoc=yes --delspec=yes --deldesc=yes \
            --strip=yes --fstrans=no --default && \
        mv *.deb ~/mingw_debs/aarch64


lcms2
^^^^^

    ::

        cd ~/mpv-build-deps && \
        git clone https://github.com/mm2/Little-CMS && \
        mkdir -p Little-CMS/littlecms-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/Little-CMS/littlecms-build/aarch64 && \
            ../../configure --prefix=/usr/aarch64-w64-mingw32 \
            --disable-shared --without-jpeg --without-tiff --enable-silent-rules \
            --host=aarch64-w64-mingw32 && \
        make -j$(nproc) && \
            sudo checkinstall --pkgname=lcms2-mingw-aarch64 --pkgversion="1:$(grep Version \
            lcms2.pc | sed 's/Version: //g')-$(date --rfc-3339=date | sed 's/-//g')-git" \
            --backup=no --deldoc=yes --delspec=yes --deldesc=yes --strip=yes \
            --fstrans=no --default && \
        mv *.deb ~/mingw_debs/aarch64


libjpeg-turbo
^^^^^^^^^^^^^

    ::

        cd ~/mpv-build-deps && \
        git clone https://github.com/libjpeg-turbo/libjpeg-turbo && \
        mkdir -p libjpeg-turbo/libjpegturbo-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/libjpeg-turbo/libjpegturbo-build/aarch64 && \
            cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=/usr/aarch64-w64-mingw32 \
            -DCMAKE_TOOLCHAIN_FILE="/usr/llvm-mingw/aarch64-w64-mingw32/toolchain-aarch64-w64-mingw32.cmake" \
            -DENABLE_SHARED:bool=off -DCMAKE_SYSTEM_PROCESSOR="aarch64" && \
        ninja && \
            sudo checkinstall --pkgname=libjpeg-turbo-mingw-aarch64 --pkgversion="$(grep \
            Version pkgscripts/libturbojpeg.pc | sed 's/Version: //g')-$(date --rfc-3339=date | \
            sed 's/-//g')" --backup=no --deldoc=yes --delspec=yes --deldesc=yes \
            --strip=yes --fstrans=no --default ninja install && \
        mv *.deb ~/mingw_debs/aarch64


libpng
^^^^^^

    ::

        cd ~/mpv-build-deps && \
        git clone https://git.code.sf.net/p/libpng/code libpng && \
        cd libpng && \
        git checkout libpng16 && \
        mkdir -p libpng-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/libpng/libpng-build/aarch64 && \
            cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=/usr/aarch64-w64-mingw32 \
            -DCMAKE_TOOLCHAIN_FILE="/usr/llvm-mingw/aarch64-w64-mingw32/toolchain-aarch64-w64-mingw32.cmake" \
            -DPNG_SHARED:bool=off -DPNG_TESTS:bool=off -DZLIB_INCLUDE_DIR=/usr/aarch64-w64-mingw32/include \
            -DZLIB_LIBRARY=/usr/aarch64-w64-mingw32/lib/libz.a -DCMAKE_C_FLAGS="-DPNG_ARM_NEON_OPT=0" && \
        ninja && \
            sudo checkinstall --pkgname=libpng-mingw-aarch64 --pkgversion="$(grep Version \
            libpng.pc | sed 's/Version: //g')-$(date --rfc-3339=date | sed 's/-//g')" \
            --backup=no --deldoc=yes --delspec=yes --deldesc=yes --strip=yes \
            --fstrans=no --default ninja install && \
        mv *.deb ~/mingw_debs/aarch64


jbigkit
^^^^^^^

    ::

        cd ~/mpv-build-deps && \
        git clone https://github.com/qyot27/jbigkit && \
        cd jbigkit && \
        autoreconf -fiv && \
        mkdir -p jbigkit-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/jbigkit/jbigkit-build/aarch64 && \
            ../../configure --prefix=/usr/aarch64-w64-mingw32 \
            --disable-shared --enable-silent-rules --host=aarch64-w64-mingw32 && \
        make -j$(nproc) && \
            sudo checkinstall --pkgname=libjbig-mingw-aarch64 --pkgversion="$(grep \
            JBG_VERSION ../../libjbig/jbig.h | sed 's/\"/\t/g' | cut -f2)-$(date \
            --rfc-3339=date | sed 's/-//g')-git" --backup=no --deldoc=yes \
            --delspec=yes --deldesc=yes --strip=yes --fstrans=no --default && \
        mv *.deb ~/mingw_debs/aarch64


deflate
^^^^^^^

    ::

        cd ~/mpv-build-deps && \
        git clone https://github.com/ebiggers/libdeflate && \
        mkdir -p libdeflate/libdeflate-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/libdeflate/libdeflate-build/aarch64 && \
            cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=/usr/aarch64-w64-mingw32 \
            -DCMAKE_TOOLCHAIN_FILE="/usr/llvm-mingw/aarch64-w64-mingw32/toolchain-aarch64-w64-mingw32.cmake" \
            -DLIBDEFLATE_BUILD_SHARED_LIB:bool=off && \
        ninja && \
            sudo checkinstall --pkgname=libdeflate-mingw-aarch64 --pkgversion="$(grep -r \
            Version libdeflate.pc | cut -f2 -d " ")-$(date --rfc-3339=date | \
            sed 's/-//g')-git" --backup=no --deldoc=yes --delspec=yes --deldesc=yes \
            --strip=yes --fstrans=no --default ninja install && \
        mv *.deb ~/mingw_debs/aarch64


lerc
^^^^

    ::

        cd ~/mpv-build-deps && \
        git clone https://github.com/esri/lerc && \
        mkdir -p lerc/lerc-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/lerc/lerc-build/aarch64 && \
            cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=/usr/aarch64-w64-mingw32 \
            -DCMAKE_TOOLCHAIN_FILE="/usr/llvm-mingw/aarch64-w64-mingw32/toolchain-aarch64-w64-mingw32.cmake" \
            -DBUILD_SHARED_LIBS:bool=off && \
        ninja && \
            sudo checkinstall --pkgname=lerc-mingw-aarch64 --pkgversion="$(grep -r \
            Version Lerc.pc | cut -f2 -d " ")-$(date --rfc-3339=date | \
            sed 's/-//g')-git" --backup=no --deldoc=yes --delspec=yes --deldesc=yes \
            --strip=yes --fstrans=no --default ninja install && \
        mv *.deb ~/mingw_debs/aarch64


zstd
^^^^

    ::

        cd ~/mpv-build-deps && \
        git clone https://github.com/facebook/zstd && \
        mkdir -p zstd/zstd-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/zstd/zstd-build/aarch64 && \
            cmake ../../build/cmake -G "Ninja" -DCMAKE_INSTALL_PREFIX=/usr/aarch64-w64-mingw32 \
            -DCMAKE_TOOLCHAIN_FILE="/usr/llvm-mingw/aarch64-w64-mingw32/toolchain-aarch64-w64-mingw32.cmake" \
            -DZSTD_BUILD_SHARED:bool=off && \
        ninja && \
            sudo checkinstall --pkgname=zstd-mingw-aarch64 --pkgversion="$(grep -r \
            Version lib/libzstd.pc | cut -f2 -d " ")-$(date --rfc-3339=date | \
            sed 's/-//g')-git" --backup=no --deldoc=yes --delspec=yes --deldesc=yes \
            --strip=yes --fstrans=no --default ninja install && \
        mv *.deb ~/mingw_debs/aarch64


libwebp
^^^^^^^

    ::

        cd ~/mpv-build-deps && \
        git clone https://chromium.googlesource.com/webm/libwebp && \
        cd libwebp && \
        autoreconf -fiv && \
        mkdir -p libwebp-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/libwebp/libwebp-build/aarch64 && \
            LIBPNG_CONFIG="/usr/aarch64-w64-mingw32/bin/libpng-config --static" \
            PKG_CONFIG_PATH=/usr/aarch64-w64-mingw32/lib/pkgconfig \
            ../../configure --prefix=/usr/aarch64-w64-mingw32 --disable-shared \
            --enable-swap-16bit-csp --disable-tiff --enable-libwebpmux \
            --enable-libwebpdemux --enable-libwebpdecoder --host=aarch64-w64-mingw32 && \
        make -j$(nproc) && \
            sudo checkinstall --pkgname=libwebp-mingw-aarch64 --pkgversion="$(grep Version \
            src/libwebp.pc | sed 's/Version: //g')-$(date --rfc-3339=date | \
            sed 's/-//g')-git" --backup=no --deldoc=yes --delspec=yes --deldesc=yes \
            --strip=yes --fstrans=no --default && \
        mv *.deb ~/mingw_debs/aarch64


libtiff
^^^^^^^

    ::

        cd ~/mpv-build-deps && \
        git clone https://gitlab.com/libtiff/libtiff.git && \
        mkdir -p libtiff/libtiff-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/libtiff/libtiff-build/aarch64 && \
            cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=/usr/aarch64-w64-mingw32 \
            -DCMAKE_TOOLCHAIN_FILE="/usr/llvm-mingw/aarch64-w64-mingw32/toolchain-aarch64-w64-mingw32.cmake" \
            -DCMAKE_STAGING_PREFIX=/usr/aarch64-w64-mingw32 -Dlerc:bool=off -DBUILD_SHARED_LIBS:bool=off && \
        ninja && \
        sed -i 's/Libs.private:  -ljbig/Libs.private: -ljbig -ljpeg -llzma/' libtiff-4.pc && \
            sudo checkinstall --pkgname=libtiff-mingw-aarch64 --pkgversion="$(grep Version \
            libtiff-4.pc | sed 's/Version: //g')-$(date --rfc-3339=date | \
            sed 's/-//g')-git" --backup=no --deldoc=yes --delspec=yes --deldesc=yes \
            --strip=yes --fstrans=no --default ninja install && \
        mv *.deb ~/mingw_debs/aarch64


libmng
^^^^^^

    ::

        cd ~/mpv-build-deps && \
        wget https://downloads.sourceforge.net/project/libmng/libmng-devel/2.0.3/libmng-2.0.3.tar.xz -O - | tar -xJvf - && \
        mkdir -p libmng-2.0.3/libmng-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/libmng-2.0.3/libmng-build/aarch64 && \
            cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=/usr/aarch64-w64-mingw32 \
            -DCMAKE_TOOLCHAIN_FILE="/usr/llvm-mingw/aarch64-w64-mingw32/toolchain-aarch64-w64-mingw32.cmake" \
            -DBUILD_SHARED_LIBS:bool=off -DCMAKE_STAGING_PREFIX=/usr/aarch64-w64-mingw32 \
            -DCMAKE_PREFIX_PATH=/usr/aarch64-w64-mingw32 && \
        ninja && \
            sudo checkinstall --pkgname=libmng-mingw-aarch64 --pkgversion="$(grep -w VERSION \
            config.h | cut -f2 -d '"')-$(date --rfc-3339=date | sed 's/-//g')" --backup=no \
            --deldoc=yes --delspec=yes --deldesc=yes --strip=yes --fstrans=no \
            --default ninja install && \
        mv *.deb ~/mingw_debs/aarch64


libsquish
^^^^^^^^^

.. Note::
    The libsquish tarball is actually a tarbomb,
    so we need to create a directory for it first.
..

    ::

        cd ~/mpv-build-deps && \
        mkdir libsquish && cd libsquish && \
        wget https://downloads.sourceforge.net/project/libsquish/libsquish-1.15.tgz -O - | tar -xzvf - && \
        mkdir -p libsquish-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/libsquish/libsquish-build/aarch64 && \
            cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=/usr/aarch64-w64-mingw32 \
            -DCMAKE_TOOLCHAIN_FILE="/usr/llvm-mingw/aarch64-w64-mingw32/toolchain-aarch64-w64-mingw32.cmake" \
            -DBUILD_SHARED_LIBS:bool=off && \
        ninja && \
            sudo checkinstall --pkgname=libsquish-mingw-aarch64 --pkgversion="$(grep -w \
            "VER =" ../../Makefile | cut -f3 -d ' ')-$(date --rfc-3339=date | \
            sed 's/-//g')-git" --backup=no --deldoc=yes --delspec=yes --deldesc=yes \
            --strip=yes --fstrans=no --default ninja install && \
        mv *.deb ~/mingw_debs/aarch64


JasPer
^^^^^^

    ::

        cd ~/mpv-build-deps && \
        git clone https://github.com/jasper-software/jasper.git && \
        mkdir -p jasper/jasper-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/jasper/jasper-build/aarch64 && \
            cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=/usr/aarch64-w64-mingw32 \
            -DCMAKE_TOOLCHAIN_FILE="/usr/llvm-mingw/aarch64-w64-mingw32/toolchain-aarch64-w64-mingw32.cmake" \
            -DCMAKE_PREFIX_PATH=/usr/aarch64-w64-mingw32 -DJAS_ENABLE_SHARED:bool=off \
            -DJAS_ENABLE_OPENGL:bool=off -DJAS_ENABLE_DOC:bool=off \
            -DJAS_ENABLE_PROGRAMS:bool=off -DALLOW_IN_SOURCE_BUILD:bool=on \
            -DJAS_CROSSCOMPILING:bool=on -DJAS_STDC_VERSION=0 && \
        ninja && \
            sudo checkinstall --pkgname=jasper-mingw-aarch64 --pkgversion="$(grep -r \
            Version build/pkgconfig/jasper.pc | cut -f2 -d " ")-$(date --rfc-3339=date | \
            sed 's/-//g')-git" --backup=no --deldoc=yes --delspec=yes --deldesc=yes \
            --strip=yes --fstrans=no --default ninja install && \
        mv *.deb ~/mingw_debs/aarch64


OpenEXR
^^^^^^^

    ::

        cd ~/mpv-build-deps && \
        git clone https://github.com/AcademySoftwareFoundation/openexr && \
        mkdir -p openexr/openexr-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/openexr/openexr-build/aarch64 && \
            cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=/usr/aarch64-w64-mingw32 \
            -DCMAKE_TOOLCHAIN_FILE="/usr/llvm-mingw/aarch64-w64-mingw32/toolchain-aarch64-w64-mingw32.cmake" \
            -DBUILD_SHARED_LIBS:bool=off && \
        ninja && \
            sudo checkinstall --pkgname=openexr-mingw-aarch64 --pkgversion="$(grep -r \
            Version cmake/OpenEXR.pc | cut -f2 -d " ")-$(date --rfc-3339=date | \
            sed 's/-//g')-git" --backup=no --deldoc=yes --delspec=yes --deldesc=yes \
            --strip=yes --fstrans=no --default ninja install && \
        mv *.deb ~/mingw_debs/aarch64


DevIL
-----

    ::

        cd ~/mpv-build-deps && \
        git clone https://github.com/DentonW/DevIL.git && \
        mkdir -p DevIL/DevIL/devil-build/{i686,amd64,aarch64} && \

Comment out ILUT subdirectory in CMakeLists.txt

    ::

        sed -i '9d' DevIL/DevIL/CMakeLists.txt && \

Remove SHARED definition from src-ILU CMakeLists.txt to force static ILU
and avoid weird dll pointing in avsplus step

    ::

        sed -i '/ILU SHARED/ s/SHARED //' DevIL/DevIL/src-ILU/CMakeLists.txt && \

Apply patch to use newer versions of JasPer:

    ::

        cd DevIL && \
        wget https://gist.githubusercontent.com/qyot27/b362b3e3834485c3e7b7e33e3b8d5049/raw/4fdcfa2b5b516f47d8ce1e967d70877f63c85497/0001-jasper-git.patch && \
        git am 0001-jasper-git.patch

Convert .h files in src-ILU/include/ilu-error from
ISO-8859-1 to UTF-8 to avoid build errors

    ::

        for n in DevIL/src-ILU/include/ilu_error/ilu_err-{french,german,italian,spanish}.h ; do iconv -f ISO-8859-1 -t UTF-8 "$n" > "$n-utf8" && mv "$n-utf8" "$n" ; done

    ::

        cd ~/mpv-build-deps/DevIL/DevIL/devil-build/aarch64 && \
            cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=/usr/aarch64-w64-mingw32 \
            -DCMAKE_TOOLCHAIN_FILE="/usr/llvm-mingw/aarch64-w64-mingw32/toolchain-aarch64-w64-mingw32.cmake" \
            -DBUILD_SHARED_LIBS:bool=off -DCMAKE_PREFIX_PATH=/usr/aarch64-w64-mingw32 \
            -DCMAKE_CXX_STANDARD=14 && \
        ninja && \
            sudo checkinstall --pkgname=devil-mingw-aarch64 --pkgversion="$(git describe --tags | \
            sed 's/^v//')-$(date --rfc-3339=date | sed 's/-//g')-git" --backup=no --deldoc=yes \
            --delspec=yes --deldesc=yes --strip=yes --fstrans=no --default ninja install && \
        mv *.deb ~/mingw_debs/aarch64


AviSynth+
---------

    ::

        cd ~/mpv-build-deps && \
        git clone https://github.com/AviSynth/AviSynthPlus && \
        mkdir -p AviSynthPlus/avisynth-build/{i686,amd64,aarch64} && \

    ::

        cd ~/mpv-build-deps/AviSynthPlus/avisynth-build/aarch64 && \
            cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=/usr/aarch64-w64-mingw32 \
            -DCMAKE_TOOLCHAIN_FILE="/usr/llvm-mingw/aarch64-w64-mingw32/toolchain-aarch64-w64-mingw32.cmake" \
            -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_PREFIX_PATH=/usr/aarch64-w64-mingw32 \
            -DCMAKE_STAGING_PREFIX=/usr/aarch64-w64-mingw32 -DCMAKE_CXX_FLAGS="-DIL_STATIC_LIB" \
            -DIL_LIBRARIES="/usr/aarch64-w64-mingw32/lib/libIL.a;/usr/aarch64-w64-mingw32/lib/libturbojpeg.a;\
        /usr/aarch64-w64-mingw32/lib/libpng16.a;/usr/aarch64-w64-mingw32/lib/libtiff.a;\
        /usr/aarch64-w64-mingw32/lib/libsquish.a;/usr/aarch64-w64-mingw32/lib/libjasper.a;\
        /usr/aarch64-w64-mingw32/lib/libz.a;/usr/aarch64-w64-mingw32/lib/liblzma.a;\
        /usr/aarch64-w64-mingw32/lib/libjbig.a;/usr/aarch64-w64-mingw32/lib/libLerc.a" \
            -DILU_LIBRARIES=/usr/aarch64-w64-mingw32/lib/libILU.a && \
        ninja && \
            sudo checkinstall --pkgname=avisynthplus-mingw-aarch64 --pkgversion="$(grep -r \
            Version avs_core/avisynth.pc | cut -f2 -d " ")-$(date --rfc-3339=date | \
            sed 's/-//g')-git" --backup=no --deldoc=yes --delspec=yes --deldesc=yes \
            --strip=yes --fstrans=no --default ninja install && \
        mv *.deb ~/mingw_debs/aarch64

Back to the :doc:`main page <../../index>`

$ Date: 2025-02-08 17:38:02-05:00 $
