
Building AviSynth+'s external dependencies with Visual Studio
=============================================================

.. toctree::
    :maxdepth: 3

.. contents:: Table of contents


Starting with 3.7.4, we no longer ship copies of DevIL and SoundTouch inside the
AviSynth+ git repository.  There are a couple reasons for this:

1. The internal copy of DevIL was only ever used on Windows and shipped the
   binary DevIL.dll itself rather than building it as a submodule. All of the
   other OSes could already build against a newer version of the library or
   build ImageSeq with DevIL linked in statically (the macOS builds in the
   Releases are an example).

2. The embedded SoundTouch code was out of date, and since it was fully possible
   to build it against the current version (and on \*nix, use the existing
   package, exactly like the DevIL case), there was no harm in moving it to the
   same model.

So those building AviSynth+ from source have much more flexibility in terms of
making sure both SoundTouch and DevIL (and DevIL's dependencies) are actually
up-to-date, and can configure the dependencies as static or shared according to
their own preferences.

For building SoundTouch and DevIL on Windows using MSVC, we'll take advantage of
the fact that CMake's internal install processes mimic the install directories
on Linux, et al., which simplifies the ability to point at that singular spot
and tell it to bring in the libraries we've built.

Make sure that pkg-config (or pkgconf), cmake, git, 7-zip, wget, sed, bash,
autotools, patch, and meson are somewhere on the Windows **%PATH%**.  An existing
MSys2 MinGW64 installation that's been added to the **%PATH%** will suffice.
pkg-config is necessary for SoundTouch to be found when configuring AviSynth+;
we find DevIL using CMake's own **FindDevIL.cmake** script.

Even though we do need to piggy-back on some of MSys2's tools, the intention is
that all - or at least, most - of this guide can be followed in the regular old
Windows Terminal.

We'll use a central install spot for the libraries, by using a bit of environment
trickery.  Set the following variable as the root of one of your drives.

::

    set AVS_DEPS_BUILD_HOME=E:

.. Note::
    If you want to use a drive other than E:, change it here.  You can also set
    this to a subdirectory on the drive, but for simplicity's sake, the guide is
    just going with the root of the drive.

Then make that setting permanent, so you can close and re-open the Command
Prompt and not have to worry about the variable getting erased.

::

    setx AVS_DEPS_BUILD_HOME %AVS_DEPS_BUILD_HOME%

And make a directory to house the downloaded sources for SoundTouch, DevIL, and
its dependencies:

::

    mkdir %AVS_DEPS_BUILD_HOME%\avsplus-build-deps

To run the build process with as many cores as you have available, set the *-j #*
parameter when invoking *cmake --build*.  My i5-9400 has 6 cores, so I use *-j 6*.
While Windows does have a CLI command to get this information, it can't be
injected into the CMake process the way something like this can under Bash.

Launching the VS Command Prompt can be done from an existing cmd session by
invoking one of the vcvarsall scripts in

::

    C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build

| The ones we're interested in on a 64-bit install of Windows 10 are
| **vcvars64.bat**
| **vcvars32.bat**
| **vcvarsamd64_arm64.bat**

I have a directory at ``E:\Programs\ScriptTools`` on my **%PATH%**, so I create
chained launchers to these launchers there, as they aren't otherwise available
on the **%PATH%**.

::

    echo @ECHO OFF > E:\Programs\ScriptTools\msvc64.bat
    echo "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >> E:\Programs\ScriptTools\msvc64.bat

    echo @ECHO OFF > E:\Programs\ScriptTools\msvc32.bat
    echo "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars32.bat" >> E:\Programs\ScriptTools\msvc32.bat

    echo @ECHO OFF > E:\Programs\ScriptTools\msvcarm.bat
    echo "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsamd64_arm64.bat" >> E:\Programs\ScriptTools\msvcarm.bat

.. Note::
    Creating symlinks on Windows 10 requires Administrator privileges (Open
    Elevated Command Prompt).

    Apparently Windows 11 doesn't require this.


SoundTouch
----------

SoundTouch is used by the TimeStretch plugin.

Jump to the correct drive and source location:

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps

Clone the source using Git:

    ::

        git clone https://codeberg.org/soundtouch/soundtouch

Enter the directory and create the proper build directories:

    ::

        cd soundtouch && ^
        mkdir build && ^
        cd build && ^
        mkdir x64 x86 arm64

.. Note::
    As demonstrated above, multiple commands can be chained by using &&:

        cd soundtouch && mkdir build && cd build && mkdir x64 x86 arm64

    The caret (^) is used to break to a new line in cmd.exe, the same way that
    bash uses \\:

        | $AVS_DEPS_BUILD_HOME && \\
        | cd $AVS_DEPS_BUILD_HOME\avsplus-build-deps

    Be sure to copy the entire command wherever carets appear.



x86-64
^^^^^^

Enter the build directory for x64:

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\soundtouch\build\x64

Configure the build:

    ::

        cmake ../ -G "Visual Studio 16 2019" -A "x64" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64

Compile and install in one step:

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
^^^^^

Enter the build directory for ARM64:

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\soundtouch\build\arm64

Configure the build:

    ::

        cmake ../ -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64

Compile and install in one step:

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
^^^^^^

Enter the build directory for x86:

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\soundtouch\build\x86

Configure the build:

    ::

        cmake ../ -G "Visual Studio 16 2019" -A "Win32" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32

Compile and install in one step:

    ::

        cmake --build . --config Release -j 6 --target install


DevIL (using prebuilt SDK)
--------------------------

Jump to the correct drive and source location:

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps

Download the DevIL SDK zipfile:

    ::

        wget http://downloads.sourceforge.net/openil/DevIL-Windows-SDK-1.8.0.zip

Unpack it to the build_dep destination area:

    ::

        7z x -o%AVS_DEPS_BUILD_HOME%\avsplus_build_deps DevIL-Windows-SDK-1.8.0.zip

And then jump to the last section to build :ref:`AviSynth+`.


Building DevIL's dependencies manually
--------------------------------------


.. WARNING::

    What follows is the dependency chain for building DevIL locally, along with
    all of its dependencies and their dependencies.  Consider this an exercise
    for the masochistic.  It is, however, laid out in a more or less linear
    fashion to make it easier to follow along.

    Unlike the SoundTouch and DevIL SDK steps above, the reason for each step
    won't be explained, simply because it would introduce a massive amount of
    identical text.  Every one of these are broken up into a group of steps to
    jump into the source download area, download the source, and create the
    build subdirectories.  And then it breaks down the actual build steps under
    headers for x64, x86, and ARM64.

    Unless there's something important to note about the options or something
    weird to account for, the description for those steps are exactly the same.

    If there are weird things to account for, they'll be noted.


zlib-ng
^^^^^^^

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        git clone https://github.com/zlib-ng/zlib-ng && ^
        cd zlib-ng && ^
        mkdir build && ^
        cd build && ^
        mkdir x64 x86 arm64


x86-64
++++++

.. Note::
    -DPKGCONFIG_INSTALL_DIR is necessary because otherwise the .pc file will install to root.
..

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\zlib-ng\build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DBUILD_SHARED_LIBS:bool=off -DZLIB_COMPAT:bool=on -DZLIB_ENABLE_TESTS:bool=off ^
        -DPKGCONFIG_INSTALL_DIR=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\pkgconfig

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
+++++

.. Note::
    -DPKGCONFIG_INSTALL_DIR is necessary because otherwise the .pc file will install to root.
..

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\zlib-ng\build\arm64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DBUILD_SHARED_LIBS:bool=off -DZLIB_COMPAT:bool=on -DZLIB_ENABLE_TESTS:bool=off ^
        -DPKGCONFIG_INSTALL_DIR=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib\pkgconfig ^
        -DCMAKE_SYSTEM_PROCESSOR=aarch64

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

.. Note::
    -DPKGCONFIG_INSTALL_DIR is necessary because otherwise the .pc file will install to root.

.. Note::
    **-DWITH_SSE2:bool=off** exists here to disable SSE2, as the guide assumes
    a very broad install base.  Most users don't need to worry about that option
    and can safely remove it.
..

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\zlib-ng\build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -A "Win32" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DBUILD_SHARED_LIBS:bool=off -DZLIB_COMPAT:bool=on -DZLIB_ENABLE_TESTS:bool=off ^
        -DWITH_SSE2:bool=off -DCMAKE_RC_FLAGS="--target=pe-i386" ^
        -DPKGCONFIG_INSTALL_DIR=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\pkgconfig

    ::

        cmake --build . --config Release -j 6 --target install


xz-tools
^^^^^^^^

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        wget http://tukaani.org/xz/xz-5.4.2.tar.gz -O - | tar -xzvf - && ^
        cd xz-5.4.2 && ^
        mkdir build && ^
        cd build && ^
        mkdir x64 x86 arm64


x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\xz-5.4.2\build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DBUILD_SHARED_LIBS:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
+++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\xz-5.4.2\build\arm64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DBUILD_SHARED_LIBS:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\xz-5.4.2\build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" -A Win32 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DBUILD_SHARED_LIBS:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


LCMS2
^^^^^

Using meson with MSVC requires launching the VS Command Prompt.

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        git clone https://github.com/mm2/Little-CMS && ^
        cd Little-CMS && ^
        mkdir build && ^
        cd build && ^
        mkdir x64 x86 arm64


x86-64
++++++

    ::

        msvc64

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\Little-CMS\build\x64

    ::

        meson setup ../../ --prefix=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        --libdir=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib ^
        --default-library static --backend vs

    ::

        meson compile -C .

    ::

        meson install --strip


ARM64
+++++

Unfortunately, there are issues trying to use the MSVC cross environment for ARM
with meson, because it attempts to test the binary during the configure phase.
In the event this gets fixed, these would be the approximate steps required.

    ::

        msvcarm

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\Little-CMS\build\arm64

    ::

        meson setup ../../ --prefix=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        --libdir=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib ^
        --default-library static --backend vs

    ::

        meson compile -C .

    ::

        meson install --strip

    ::

        exit

Instead, we can use the autotools chain through bash to get around cross-compiling
error. You can instantiate the full MSys2 environment if you prefer, but since
everything else here is trying to stick to running in the native Windows terminal,
we'll also do that here.


    ::

        msvcarm

    ::

        bash ../../configure --prefix=E:/avsplus_build_deps/arm64 --disable-shared ^
        --without-jpeg --without-tiff --enable-silent-rules --host aarch64-w64-mingw32 ^
        CC=cl CXX=cl

    ::

        make -j6

    ::

        make install

    ::

        exit

.. Note::
    *make* and *make install* will fail on compiling the tools, but the library
    and headers will still be built and installed.


x86-32
++++++

    ::

        msvc32

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\Little-CMS\build\x86

    ::

        meson setup ../../ --prefix=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        --libdir=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib ^
        --default-library static --backend vs

    ::

        meson compile -C .

    ::

        meson install --strip

    ::

        exit

.. Note::
    Remember to exit the VS Command Prompt before continuing


libjpeg-turbo
^^^^^^^^^^^^^

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        git clone https://github.com/libjpeg-turbo/libjpeg-turbo && ^
        cd libjpeg-turbo && ^
        mkdir build && ^
        cd build && ^
        mkdir x64 x86 arm64


x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libjpeg-turbo\build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DENABLE_SHARED:bool=off -DCMAKE_SYSTEM_PROCESSOR="x86_64"

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
+++++

.. Note::
    Turning off the simd is necessary to avoid linkage error
..

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libjpeg-turbo\build\arm64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DENABLE_SHARED:bool=off -DCMAKE_SYSTEM_PROCESSOR="aarch64" ^
        -DWITH_SIMD:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libjpeg-turbo\build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" -A Win32 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DENABLE_SHARED:bool=off -DCMAKE_SYSTEM_PROCESSOR="i686"

    ::

        cmake --build . --config Release -j 6 --target install


libpng
^^^^^^

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        git clone git://git.code.sf.net/p/libpng/code libpng && ^
        cd libpng && ^
        git checkout libpng16 && ^
        mkdir build && ^
        cd build && ^
        mkdir x64 x86 arm64


x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libpng\build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DPNG_SHARED:bool=off -DPNG_TESTS:bool=off ^
        -DZLIB_INCLUDE_DIR=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\include ^
        -DZLIB_LIBRARY=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\zlibstatic.lib

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
+++++

.. Note::
    -DPNG_ARM_NEON_OPT=0 disables neon, because the functions aren't public
    and cause configure check errors in FFmpeg for libbluray, libfontconfig,
    and libfreetype due to undefined symbols in libpng16.a

    Is this necessary for our purposes with MSVC on Windows? Maybe not.
..

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libpng\build\arm64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DPNG_SHARED:bool=off -DPNG_TESTS:bool=off ^
        -DZLIB_INCLUDE_DIR=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\include ^
        -DZLIB_LIBRARY=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib\zlibstatic.lib ^
        -DCMAKE_C_FLAGS="-DPNG_ARM_NEON_OPT=0"

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libpng\build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" -A Win32 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DPNG_SHARED:bool=off -DPNG_TESTS:bool=off ^
        -DZLIB_INCLUDE_DIR=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\include ^
        -DZLIB_LIBRARY=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\zlibstatic.lib

    ::

        cmake --build . --config Release -j 6 --target install


jbigkit
^^^^^^^

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        git clone https://github.com/qyot27/jbigkit && ^
        cd jbigkit && ^
        git checkout mingw-w64 && ^
        mkdir build && ^
        cd build && ^
        mkdir x64 x86 arm64


x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\jbigkit\build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
+++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\jbigkit\build\arm64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\jbigkit\build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" -A Win32 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32

    ::

        cmake --build . --config Release -j 6 --target install


deflate
^^^^^^^

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        git clone https://github.com/ebiggers/libdeflate && ^
        cd libdeflate && ^
        mkdir build && ^
        cd build && ^
        mkdir x64 x86 arm64


x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libdeflate\build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DLIBDEFLATE_BUILD_SHARED_LIB:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
+++++

.. Note::
    Builds with ClangCL but not standard MSVC 2019, maybe 2022 can, or maybe not.
..

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libdeflate\build\arm64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -A ARM64 -T ClangCL ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DLIBDEFLATE_BUILD_SHARED_LIB:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libdeflate\build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" -A Win32 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DLIBDEFLATE_BUILD_SHARED_LIB:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


lerc
^^^^

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        git clone https://github.com/esri/lerc/ && ^
        cd lerc\build && ^
        mkdir x64 x86 arm64


x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\lerc\build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DBUILD_SHARED_LIBS:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
+++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\lerc\build\arm64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DBUILD_SHARED_LIBS:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\lerc\build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" -A Win32 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DBUILD_SHARED_LIBS:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


zstd
^^^^

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        git clone https://github.com/facebook/zstd && ^
        cd zstd && ^
        mkdir zstd-build && ^
        cd zstd-build && ^
        mkdir x64 x86 arm64


x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\zstd\zstd-build\x64

    ::

        cmake ../../build/cmake -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DZSTD_BUILD_SHARED:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
+++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\zstd\zstd-build\arm64

    ::

        cmake ../../build/cmake -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DZSTD_BUILD_SHARED:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\zstd\zstd-build\x86

    ::

        cmake ../../build/cmake -G "Visual Studio 16 2019" -T "v141_xp" -A Win32 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DZSTD_BUILD_SHARED:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


libwebp
^^^^^^^

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        git clone https://chromium.googlesource.com/webm/libwebp && ^
        cd libwebp && ^
        mkdir build && ^
        cd build && ^
        mkdir x64 x86 arm64


x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libwebp\build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DBUILD_SHARED_LIBS:bool=off -DWEBP_ENABLE_SWAP_16BIT_CSP:bool=on

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
+++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libwebp\build\arm64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DBUILD_SHARED_LIBS:bool=off -DWEBP_ENABLE_SWAP_16BIT_CSP:bool=on

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libwebp\build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" -A Win32 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DBUILD_SHARED_LIBS:bool=off -DWEBP_ENABLE_SWAP_16BIT_CSP:bool=on

    ::

        cmake --build . --config Release -j 6 --target install


libtiff
^^^^^^^

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        git clone https://gitlab.com/libtiff/libtiff.git && ^
        cd libtiff && ^
        mkdir libtiff-build && ^
        cd libtiff-build && ^
        mkdir x64 x86 arm64

.. Note::
    Seemingly won't link to the static webp we just built.

x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libtiff\libtiff-build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DBUILD_SHARED_LIBS:bool=off -Dtiff-docs:bool=off -Dtiff-tools:bool=off ^
        -Dtiff-tests:bool=off -DCMAKE_C_FLAGS="-DLZMA_API_STATIC"

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
+++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libtiff\libtiff-build\arm64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DBUILD_SHARED_LIBS:bool=off -Dtiff-docs:bool=off -Dtiff-tools:bool=off ^
        -Dtiff-tests:bool=off -DCMAKE_C_FLAGS="-DLZMA_API_STATIC"

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libtiff\libtiff-build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" -A Win32 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DBUILD_SHARED_LIBS:bool=off -Dtiff-docs:bool=off -Dtiff-tools:bool=off ^
        -Dtiff-tests:bool=off -DCMAKE_C_FLAGS="-DLZMA_API_STATIC"

    ::

        cmake --build . --config Release -j 6 --target install


libmng
^^^^^^

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        curl -Lo libmng-2.0.3.tar.xz https://sourceforge.net/projects/libmng/files/libmng-devel/2.0.3/libmng-2.0.3.tar.xz/download && ^
        tar -xJvf libmng-2.0.3.tar.xz && ^
        cd libmng-2.0.3/build && ^
        mkdir x64 x86 arm64


x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libmng-2.0.3\build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DBUILD_SHARED_LIBS:bool=off ^
        -DCMAKE_STAGING_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DLCMS2_LIBRARY=%AVS_DEPS_BUILD_HOME%\x86-64\lib\liblcms2.a

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
+++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libmng-2.0.3\build\arm64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DBUILD_SHARED_LIBS:bool=off ^
        -DCMAKE_STAGING_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libmng-2.0.3\build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" -A Win32 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DBUILD_SHARED_LIBS:bool=off ^
        -DCMAKE_STAGING_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DLCMS2_LIBRARY=%AVS_DEPS_BUILD_HOME%\x86-32\lib\liblcms2.a

    ::

        cmake --build . --config Release -j 6 --target install


libsquish
^^^^^^^^^

.. WARNING::
    The libsquish tarball is actually a tarbomb, so we need to create a
    directory for it first.
..

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        mkdir libsquish && ^
        cd libsquish && ^
        curl -Lo libsquish-1.15.tar.gz https://sourceforge.net/projects/libsquish/files/libsquish-1.15.tgz/download && ^
        tar -xzvf libsquish-1.15.tar.gz && ^
        mkdir build && ^
        cd build && ^
        mkdir x64 x86 arm64


x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libsquish\build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DBUILD_SHARED_LIBS:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
+++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libsquish\build\arm64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DBUILD_SHARED_LIBS:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libsquish\build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" -A Win32 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DBUILD_SHARED_LIBS:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


JasPer
^^^^^^

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        git clone https://github.com/jasper-software/jasper.git && ^
        cd jasper && ^
        mkdir jasper-build && ^
        cd jasper-build && ^
        mkdir x64 x86 arm64


ARM64
+++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\jasper\jasper-build\arm64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DJAS_ENABLE_SHARED:bool=off -DJAS_ENABLE_OPENGL:bool=off ^
        -DJAS_ENABLE_DOC:bool=off -DJAS_ENABLE_PROGRAMS:bool=off ^
        -DALLOW_IN_SOURCE_BUILD:bool=on -DJAS_CROSSCOMPILING:bool=on ^
        -DJAS_STDC_VERSION=0

    ::

        cmake --build . --config Release -j 6 --target install

.. Note::
    JasPer HEAD not compatible with v141_xp due to missing sysinfoapi.h header
    and threading library.  Doesn't affect ARM64, and because it requires
    changing the source, that's why the ARM64 instructions came first here.
..

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\jasper
        sed -e '97s/^^/\/\//' -e '659,669s/^^/\/\//' src/libjasper/base/jas_malloc.c


x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\jasper\jasper-build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DJAS_ENABLE_SHARED:bool=off -DJAS_ENABLE_OPENGL:bool=off ^
        -DJAS_ENABLE_DOC:bool=off -DJAS_ENABLE_PROGRAMS:bool=off ^
        -DALLOW_IN_SOURCE_BUILD:bool=on -DJAS_ENABLE_MULTITHREADING_SUPPORT:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\jasper\jasper-build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" -A Win32 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DJAS_ENABLE_SHARED:bool=off -DJAS_ENABLE_OPENGL:bool=off ^
        -DJAS_ENABLE_DOC:bool=off -DJAS_ENABLE_PROGRAMS:bool=off ^
        -DALLOW_IN_SOURCE_BUILD:bool=on -DJAS_ENABLE_MULTITHREADING_SUPPORT:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


OpenEXR
^^^^^^^

.. Note::
    Not compatible with XP.
..

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        git clone https://github.com/AcademySoftwareFoundation/openexr && ^
        cd openexr && ^
        mkdir build && ^
        cd build && ^
        mkdir x64 x86 arm64


x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\openexr\build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DBUILD_SHARED_LIBS:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
+++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\openexr\build\arm64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DBUILD_SHARED_LIBS:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\openexr\build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" -A Win32 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DBUILD_SHARED_LIBS:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


DevIL
-----

    ::

        %AVS_DEPS_BUILD_HOME% && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps && ^
        git clone https://github.com/DentonW/DevIL.git

Comment out or delete the reference to ILUT subdirectory in CMakeLists.txt:

    ::

        sed -i '9d' DevIL/DevIL/CMakeLists.txt

Remove SHARED definition from src-ILU CMakeLists.txt to force static ILU:

    ::

        sed -i '46s/SHARED //' DevIL\DevIL\src-ILU\CMakeLists.txt

Apply patch to use newer versions of JasPer:

    ::

        cd DevIL && ^
        wget https://gist.githubusercontent.com/qyot27/b362b3e3834485c3e7b7e33e3b8d5049/raw/4fdcfa2b5b516f47d8ce1e967d70877f63c85497/0001-jasper-git.patch && ^
        git am 0001-jasper-git.patch

Convert .h files in ``src-ILU/include/ilu-error`` from ISO-8859-1 to UTF-8 to avoid
build errors:

    ::

        cd DevIL\DevIL\src-ILU\include\ilu_error && ^
        iconv -f ISO-8859-1 -t UTF-8 ilu_err-french.h | tee ilu_err-french.h && ^
        iconv -f ISO-8859-1 -t UTF-8 ilu_err-german.h | tee ilu_err-german.h && ^
        iconv -f ISO-8859-1 -t UTF-8 ilu_err-italian.h | tee ilu_err-italian.h && ^
        iconv -f ISO-8859-1 -t UTF-8 ilu_err-spanish.h | tee ilu_err-spanish.h && ^
        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps

    ::

        cd DevIL/DevIL && ^
        mkdir devil-build && ^
        cd devil-build && ^
        mkdir x64 x86 arm64

.. Note::
    DevIL looks for LCMS2 as lcms2.lib; use a symlink to fix that.


x86-64
^^^^^^

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\DevIL\DevIL\devil-build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DBUILD_SHARED_LIBS:bool=off ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
^^^^^

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\DevIL\DevIL\devil-build\arm64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DBUILD_SHARED_LIBS:bool=off ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
^^^^^^

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\DevIL\DevIL\devil-build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" -A Win32 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32 ^
        -DBUILD_SHARED_LIBS:bool=off ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32

    ::

        cmake --build . --config Release -j 6 --target install


.. _avisynth+:

AviSynth+
---------

::

    %AVS_DEPS_BUILD_HOME%
    cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps
    git clone https://github.com/AviSynth/AviSynthPlus.git
    cd AviSynthPlus
    mkdir build && cd build
    mkdir x64 x86 arm64


Using prebuilt DevIL SDK
^^^^^^^^^^^^^^^^^^^^^^^^

x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\AviSynthPlus\build\x64

    ::

        cmake ../../  -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avisynth_build\x86-64 ^
        -DWINXP_SUPPORT:bool=on ^
        -DIL_LIBRARIES="E:\avsplus_build_deps\DevIL Windows SDK\lib\x64\Release\DevIL.lib" ^
        -DILU_LIBRARIES="E:\avsplus_build_deps\DevIL Windows SDK\lib\x64\Release\ILU.lib" ^
        -DCMAKE_PREFIX_PATH="E:\avsplus_build_deps\x86-64;E:\avsplus_build_deps\DevIL Windows SDK"

    ::

        cmake --build . --config Release -j 6 --target install

Copy 64-bit **DevIL.dll** from the SDK into the bin directory of the AviSynth+
install, using the Windows-native copy command (although if you have MSys2's
tools on the **%PATH%**, *cp* would be usable as well):

    ::

        copy "%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\DevIL Windows SDK\lib\x64\Release\DevIL.dll" ^
        %AVS_DEPS_BUILD_HOME%\avisynth_build\x86-64\bin


ARM64
+++++

.. Note::
    Cannot use prebuilt DevIL SDK because it only provides x86(-64) binaries.
    It was published in 2017, after all.


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\AviSynthPlus\build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" -A "Win32" ^
        -DCMAKE_INSTALL_PREFIX=E:/avisynth_build/x86-32 -DMSVC_CPU_ARCH="SSE" ^
        -DWINXP_SUPPORT:bool=on ^
        -DIL_LIBRARIES="E:\avsplus_build_deps\DevIL Windows SDK\lib\x86\Release\DevIL.lib" ^
        -DILU_LIBRARIES="E:\avsplus_build_deps\DevIL Windows SDK\lib\x86\Release\ILU.lib" ^
        -DCMAKE_PREFIX_PATH="E:\avsplus_build_deps\x86-32;E:\avsplus_build_deps\DevIL Windows SDK"

    ::

        cmake --build . --config Release -j 6 --target install

Copy 32-bit **DevIL.dll** from the SDK into the bin directory of the AviSynth+
install, using the Windows-native copy command (although if you have MSys2's
tools on the **%PATH%**, *cp* would be usable as well):

    ::

        copy "%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\DevIL Windows SDK\lib\x86\Release\DevIL.dll" %AVS_DEPS_BUILD_HOME%\avisynth_build\x86-32\bin



Using manually-built static DevIL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\AviSynthPlus\build\x64

    ::

        cmake ../../  -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avisynth_build\x86-64 ^
        -DWINXP_SUPPORT:bool=on ^
        -DIL_LIBRARIES="%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\DevIL.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\jpeg-static.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\libpng16_static.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\tiff.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\squish.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\jasper.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\zlibstatic.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\liblzma.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\jbig.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\deflatestatic.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\Lerc.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\zstd_static.lib" ^
        -DILU_LIBRARIES=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\ILU.lib ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DCMAKE_CXX_FLAGS="-DIL_STATIC_LIB"

    ::

        cmake --build . --config Release -j 6 --target install


ARM64
+++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\AviSynthPlus\build\arm64

    ::

        cmake ../../  -G "Visual Studio 16 2019" -A ARM64 ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avisynth_build\arm64 ^
        -DIL_LIBRARIES="%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib\DevIL.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib\jpeg-static.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib\libpng16_static.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib\tiff.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib\squish.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib\jasper.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib\zlibstatic.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib\liblzma.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib\jbig.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib\deflatestatic.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib\Lerc.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib\zstd_static.lib" ^
        -DILU_LIBRARIES=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64\lib\ILU.lib ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\arm64 ^
        -DCMAKE_CXX_FLAGS="-DIL_STATIC_LIB" -DENABLE_INTEL_SIMD:bool=off ^
        -DBUILD_VDUBFILTER:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\AviSynthPlus\build\x86

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" -A "Win32" ^
        -DCMAKE_INSTALL_PREFIX=E:/avisynth_build/x86-32 -DMSVC_CPU_ARCH="SSE" ^
        -DWINXP_SUPPORT:bool=on ^
        -DIL_LIBRARIES="%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\DevIL.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\jpeg-static.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\libpng16_static.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\tiff.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\squish.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\jasper.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\zlibstatic.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\liblzma.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\jbig.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\deflatestatic.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\Lerc.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\zstd_static.lib" ^
        -DILU_LIBRARIES=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\ILU.lib ^
        -DCMAKE_PREFIX_PATH="E:\avsplus_build_deps\x86-32" ^
        -DCMAKE_CXX_FLAGS="-DIL_STATIC_LIB"

    ::

        cmake --build . --config Release -j 6 --target install

Back to the :doc:`main page <../../index>`

$ Date: 2024-02-25 16:52:26-05:00 $
