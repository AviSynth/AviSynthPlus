
Building AviSynth+'s external dependencies with Visual Studio
=============================================================

.. toctree::
    :maxdepth: 3

Related documents:

* :doc:`compiling_avsplus` — the general build guide that uses the
  libraries built here.
* :doc:`avsplus_external_deps_guide_manual_arm` — the same dependencies for
  Windows on ARM (llvm-mingw, cross-compiled from Linux/WSL2).
* :doc:`release_build_automation` — the automated Windows dev build
  (``build_release.bat``), including a helper script for SoundTouch.
* :doc:`posix` — building AviSynth+ natively on Linux, macOS and BSD, where
  DevIL and SoundTouch come from the system packages.
* :doc:`contributing_with_git` — git primer for contributors.

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

I have a directory at ``E:\Programs\ScriptTools`` on my **%PATH%**, so I create
chained launchers to these launchers there, as they aren't otherwise available
on the **%PATH%**.

::

    echo @ECHO OFF > E:\Programs\ScriptTools\msvc64.bat
    echo "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >> E:\Programs\ScriptTools\msvc64.bat

    echo @ECHO OFF > E:\Programs\ScriptTools\msvc32.bat
    echo "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars32.bat" >> E:\Programs\ScriptTools\msvc32.bat

.. Note::
    Creating symlinks on Windows 10 and 11 requires Administrator privileges (Open
    Elevated Command Prompt).

    On Windows 11, this can be done from a non-Admin prompt by using sudo as one
    would on Linux or OS X.  sudo on Windows 11 needs to be enabled by the user
    through Developer Settings.


pkgconf
-------

pkg-config is a tool used to detect and add headers and libraries from the filesystem.
pkgconf is compatible with pkg-config while both being slimmer on dependencies and adding
a couple of useful features.

On Windows, MSVC can be used to build pkgconf, but meson is required.  The quickest way
to do this is through pip.

Anyone wanting to build the AviSynth+ docs already has Python, hence they also already
have pip, and can install meson through there.  Or you have Python installed for other
reasons anyway.  During Python installation it should have been added to the PATH.

Open the Visual Studio Command Prompt and install meson:

    ::

        pip install meson

Clone the source using Git:

    ::

        git clone https://github.com/pkgconf/pkgconf

Create the proper build directory and move into it:

    ::

        mkdir pkgconf\build && ^
        cd pkgconf\build

Configure the pkgconf build:

    ::

        meson setup ../ -Dtests=disabled -Dprefix=C:\pkgconf_for_windows

Build the source:

    ::

        meson compile

Install pkgconf:

    ::

        meson install

Create a symbolic link so pkg-config can be used as an alias for pkgconf (Windows 11 instructions shown):

    ::

        sudo mklink C:\pkgconf_for_windows\bin\pkg-config.exe C:\pkgconf_for_windows\bin\pkgconf.exe

Add the folder containing pkgconf to the PATH:

    ::

        setx PATH "%PATH%;C:\pkgconf_for_windows\bin"

Close and re-open the Command Prompt.


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
        mkdir x64 x86

.. Note::
    As demonstrated above, multiple commands can be chained by using &&:

        cd soundtouch && mkdir build && cd build && mkdir x64 x86

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

        cmake ../../ -G "Visual Studio 16 2019" -A "x64" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64

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

        cmake ../../ -G "Visual Studio 16 2019" -A "Win32" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32

Compile and install in one step:

    ::

        cmake --build . --config Release -j 6 --target install


.. _devil_prebuilt_sdk_section1:

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
    headers for x64 and x86.

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
        mkdir x64 x86


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
        wget http://tukaani.org/xz/xz-5.6.4.tar.gz -O - | tar -xzvf - && ^
        cd xz-5.6.4 && ^
        mkdir build && ^
        cd build && ^
        mkdir x64 x86


x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\xz-5.6.4\build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DBUILD_SHARED_LIBS:bool=off

    ::

        cmake --build . --config Release -j 6 --target install


x86-32
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\xz-5.6.4\build\x86

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
        mkdir x64 x86


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

    ::

        sudo mklink %AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\lcms2.lib %AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\liblcms2.a


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

        sudo mklink %AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\lcms2.lib %AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\liblcms2.a

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
        mkdir x64 x86


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
        mkdir x64 x86


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
        mkdir x64 x86


x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\jbigkit\build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64

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
        mkdir x64 x86


.. Note::
    (2026, ``make_release.bat``) libdeflate 1.26 and newer require a C11 compiler, which for
    Visual Studio means 2019 v16.8 or newer: with ``v141_xp`` (the VS 2017 compiler) the build stops
    at ``#error "MSVC versions older than Visual Studio 2019 v16.8 are no longer supported"``.
    For XP use libdeflate 1.25 (VS 2015 or newer): ``git checkout v1.25`` after the clone.
    ``helpers\makedevil.ps1`` does this for ``-Toolset v141_xp``.
..


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
        mkdir x64 x86


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
        mkdir x64 x86


.. Note::
    (2026, ``make_release.bat``) For XP add ``-DZSTD_MULTITHREAD_SUPPORT:bool=off`` to the cmake
    lines below. zstd's multithreading uses Windows condition variables
    (``InitializeConditionVariable`` etc.), which exist only from Vista on. The build succeeds with
    ``v141_xp`` without any message, but a DLL linking it (ImageSeq) does not load on XP
    ("entry point not found"). libtiff needs only the single-threaded zstd.
..


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
        mkdir x64 x86


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
        mkdir x64 x86

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
        curl -Lo libmng-2.0.3.tar.gz https://sourceforge.net/projects/libmng/files/libmng-devel/2.0.3/libmng-2.0.3.tar.gz/download && ^
        tar -xzvf libmng-2.0.3.tar.gz && ^
        cd libmng-2.0.3/build && ^
        mkdir x64 x86


x86-64
++++++

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\libmng-2.0.3\build\x64

    ::

        cmake ../../ -G "Visual Studio 16 2019" -T "v141_xp" ^
        -DCMAKE_INSTALL_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DBUILD_SHARED_LIBS:bool=off ^
        -DCMAKE_STAGING_PREFIX=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64

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
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32

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
        mkdir x64 x86


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
        mkdir x64 x86


.. Note::
    JasPer HEAD not compatible with v141_xp due to missing sysinfoapi.h header
    and threading library, but this can be kludged.
..

    ::

        cd %AVS_DEPS_BUILD_HOME%\avsplus-build-deps\jasper
        sed -e '97s/^^/\/\//' -e '659,669s/^^/\/\//' src/libjasper/base/jas_malloc.c


.. Note::
    (2026, ``make_release.bat``, JasPer 4.2.9) The kludge above comments out the include of
    ``sysinfoapi.h`` and the Windows branch of ``jas_get_total_mem_size()``, which calls
    ``GetPhysicallyInstalledSystemMemory`` (Vista SP1 and newer). The function then returns 0
    (memory size unknown), and JasPer additionally warns on stderr, on every jp2 read or write, that
    its memory limit "exceeds system memory size (... > 0)". ``helpers\makedevil.ps1``
    (``-Toolset v141_xp``) instead removes only the include and
    replaces the call with ``GlobalMemoryStatusEx`` (available on XP), so JasPer still knows the
    memory size::

        MEMORYSTATUSEX ms;
        ms.dwLength = sizeof(ms);
        if (!GlobalMemoryStatusEx(&ms)) {
            return 0;
        }
        return (ms.ullTotalPhys < SIZE_MAX) ? JAS_CAST(size_t, ms.ullTotalPhys) : SIZE_MAX;

    The line numbers in the ``sed`` command are of the JasPer version of that time; check them for
    a newer one (in 4.2.9 the same lines: 97 and 659-669).
..


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
        mkdir x64 x86


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
        mkdir x64 x86

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
    mkdir x64 x86



.. _devil_prebuilt_sdk_section2:

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
        -DIL_LIBRARIES="%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\DevIL.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\jpeg-static.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\libpng16_static.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\tiff.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\squish.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\jasper.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\zlibstatic.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\lzma.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\jbig.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\deflatestatic.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\Lerc.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\zstd_static.lib" ^
        -DILU_LIBRARIES=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64\lib\ILU.lib ^
        -DCMAKE_PREFIX_PATH=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-64 ^
        -DCMAKE_CXX_FLAGS="-DIL_STATIC_LIB"

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
        -DIL_LIBRARIES="%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\DevIL.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\jpeg-static.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\libpng16_static.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\tiff.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\squish.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\jasper.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\zlibstatic.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\lzma.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\jbig.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\deflatestatic.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\Lerc.lib;%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\zstd_static.lib" ^
        -DILU_LIBRARIES=%AVS_DEPS_BUILD_HOME%\avsplus_build_deps\x86-32\lib\ILU.lib ^
        -DCMAKE_PREFIX_PATH="E:\avsplus_build_deps\x86-32" ^
        -DCMAKE_CXX_FLAGS="-DIL_STATIC_LIB"

    ::

        cmake --build . --config Release -j 6 --target install

Back to the :doc:`main page <../../index>`

$ Date: 2025-03-08 21:34:07-05:00 $
