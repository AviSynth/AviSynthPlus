
Compiling AviSynth+
===================

This guide uses a command line-based compilation methodology, because
it's easier to provide direct instructions for this that can just be copy/pasted.

Later on some other compiling method (CMake GUI, Visual Studio solution) is shown as well with different compilers.

`MSys2 <https://msys2.github.io/>`_ and `7zip <http://www.7-zip.org/>`_ should
already be installed, and msys2's bin directory should have been added to Windows'
%PATH% variable.

.. toctree::
    :maxdepth: 3

.. contents:: Table of contents



AviSynth+ prerequisites
-----------------------

Note that AviSynth+ is not restricted to Windows.

AviSynth+ can be built by a few different compilers:

* Visual Studio 2019 or higher. (May work for VS2017)
  - native msvc or clang-cl
* Clang 7.0.1 or higher.
* GCC 7 or higher.
* Intel C++ Compiler 2021 (ICX: LLVM based NextGen)
* Intel C++ Compiler 19.2 (ICL: classic)


| Download and install Visual Studio Community:
| `<https://visualstudio.microsoft.com/downloads/>`_

| Install the latest version of CMake:
| `<http://www.cmake.org/cmake/resources/software.html>`_

After installing MSys2, make sure to enable some convenience functions in MSys2's config files.

In msys.ini:
::

    CHERE_INVOKING=1
    MSYS2_PATH_TYPE=inherit
    MSYSTEM=MSYS

In mingw64.ini:
::

    CHERE_INVOKING=1
    MSYS2_PATH_TYPE=inherit
    MSYSTEM=MINGW64

In mingw32.ini:
::

    CHERE_INVOKING=1
    MSYS2_PATH_TYPE=inherit
    MSYSTEM=MINGW32

Add CMake's bin directory to the system %PATH% manually if the installer won't.
Also add 7zip and upx to the %PATH%.


Building with Visual Studio
---------------------------

For ease of use, we'll also be making use of MSys2 to streamline the build process,
even with the VC++ compiler.


DirectShowSource Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DirectShowSource requires extra setup that building the AviSynth+ core does not.
DirectShowSource is not a requirement for a working AviSynth+ setup, especially
with the options of using either FFmpegSource2 or LSMASHSource, but the guide
wouldn't be complete otherwise.


C++ Base Classes library
........................

DirectShowSource requires strmbase.lib, the C++ Base Classes library, which for some
reason isn't included in a standard install of Visual Studio.  The source code for
the library is provided with the Windows SDK, and requires the user to build it first.

| Download the Windows SDK 7.1:
| `<http://www.microsoft.com/en-US/download/details.aspx?Id=8442>`_

| Download the following ISO for 32-bit Windows installations:
| GRMSDK_EN_DVD.iso

| Download the following ISO for 64-bit Windows installations:
| GRMSDKX_EN_DVD.iso

The ISO you download is based on the version of Windows you're actually running,
*not* on the Windows installs you're targetting.  Both ISOs include the correct
tools to build for either 32-bit or 64-bit targets.

| Verify the 32-bit ISO against CRC32 or SHA1:
| CRC#: 0xBD8F1237
| SHA1: 0xCDE254E83677C34C8FD509D6B733C32002FE3572

| Verify the 64-bit ISO against CRC32 or SHA1:
| CRC#: 0x04F59E55
| SHA1: 0x9203529F5F70D556A60C37F118A95214E6D10B5A

For convenience (and on computers without an optical drive), you can use either Pismo
File Mount (if you've already got it installed for AVFS) or Windows 10's own Mount option
to mount the ISO to a virtual drive. Then just launch setup.exe and follow the wizard.

Install only the Samples, uncheck everything else.

| Open Visual Studio, and open the .sln file in the 7.1 SDK, at
| ``C:\Program Files\Microsoft SDKs\Windows\v7.1\Samples\multimedia\directshow\baseclasses``

Allow Visual Studio to convert the project, switch the configuration to ``Release``,
and enter the project Properties by right-clicking on the solution name and selecting
``Properties``.

Select the ``Visual Studio 17 - Windows XP (v141_xp)`` option on the main Properties page
under ``Toolset``, and on the ``C/C++ Code Generation`` page select *Disabled* or *SSE*
from the ``Enhanced Instruction Set`` option (IMO, it's safer to disable it for system
support libraries like strmbase.lib), and finally, exit back to the main screen.

Now select ``Build``. That's it.

For 64-bit, change to ``Release x64`` and ``Build``. The SSE2 note isn't relevant here, since
64-bit CPUs are required to have SSE2 support.


Miscellaneous
.............

To make the AviSynth+ build instructions more concise, we'll set a couple of environment
variables.  After starting msys2, open the file /etc/profile in Wordpad:
::

    write /etc/profile

and copy the following three lines into it somewhere:
::

    export STRMBASELIB="C:/Program Files/Microsoft SDKs/Windows/v7.1/Samples/multimedia/directshow/baseclasses/Release/strmbase.lib"
    export STRMBASELIB64="C:/Program Files/Microsoft SDKs/Windows/v7.1/Samples/multimedia/directshow/baseclasses/x64/Release/strmbase.lib"

(64-bit Windows users should use ``Program Files (x86)``, but you probably already knew that ;P)

Thankfully, all of this setup only needs to be done once.


Building AviSynth+
~~~~~~~~~~~~~~~~~~

Start the Visual Studio x86 Native Command Prompt.

You can use Visual Studio's compilers from MSys2 by launching MSys2 from the Visual Studio
Command Prompt. So type 'msys' and hit Enter.

Note: in the instructions below, the ``\`` character means the command spans more than
one line.  Make sure to copy/paste all of the lines in the command.

Download the AviSynth+ source:
::

    git clone https://github.com/AviSynth/AviSynthPlus && \
    cd AviSynthPlus

Set up the packaging directory for later:
::

    AVSDIRNAME=avisynth+_r$(git rev-list --count HEAD)-g$(git rev-parse --short HEAD)-$(date --rfc-3339=date | sed 's/-//g') && \
    cd .. && \
    mkdir -p avisynth_build $AVSDIRNAME/32bit/dev $AVSDIRNAME/64bit/dev && \
    cd avisynth_build

Now, we can build AviSynth+.


Using MSBuild
.............

Note: depending on your Visual Studio 2019 or 2022 version, choose only one cmake build block.

For 32-bit (no XP, SSE2):
::

    cmake ../AviSynthPlus -G "Visual Studio 17 2022" -A Win32 -DMSVC_CPU_ARCH:string="SSE2" -DBUILD_DIRECTSHOWSOURCE:bool=on && \
    cmake --build . --config Release -j $(nproc)

    or

    cmake ../AviSynthPlus -G "Visual Studio 16 2019" -A Win32 -DMSVC_CPU_ARCH:string="SSE2" -DBUILD_DIRECTSHOWSOURCE:bool=on && \
    cmake --build . --config Release -j $(nproc)


For 32-bit (XP, SSE):
::

    cmake ../AviSynthPlus -G "Visual Studio 17 2022" -A Win32 -T "v141_xp" -DMSVC_CPU_ARCH:string="SSE" -DWINXP_SUPPORT:bool=on -DBUILD_DIRECTSHOWSOURCE:bool=on && \
    cmake --build . --config Release -j $(nproc)

    or

    cmake ../AviSynthPlus -G "Visual Studio 16 2019" -A Win32 -T "v141_xp" -DMSVC_CPU_ARCH:string="SSE" -DWINXP_SUPPORT:bool=on -DBUILD_DIRECTSHOWSOURCE:bool=on && \
    cmake --build . --config Release -j $(nproc)


Copy the .dlls to the packaging directory:
::

    cp Output/AviSynth.dll Output/system/DevIL.dll Output/plugins/* ../$AVSDIRNAME/32bit

Copy the .libs to the packaging directory:
::

    cp avs_core/Release/AviSynth.lib plugins/DirectShowSource/Release/*.lib \
    ../AviSynthPlus/plugins/ImageSeq/lib/DevIL_x86/DevIL.lib plugins/ImageSeq/Release/ImageSeq.lib \
    plugins/Shibatch/PFC/Release/PFC.lib plugins/Shibatch/Release/Shibatch.lib \
    plugins/TimeStretch/Release/TimeStretch.lib plugins/TimeStretch/SoundTouch/Release/SoundTouch.lib \
    plugins/VDubFilter/Release/VDubFilter.lib ../$AVSDIRNAME/32bit/dev


Undo the upx packing on the 32-bit copy of DevIL.dll:
::

    upx -d ../$AVSDIRNAME/32bit/DevIL.dll


For 64-bit (no XP):
::

    cmake ../AviSynthPlus -G "Visual Studio 17 2022" -A x64 -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on && \
    cmake --build . --config Release -j $(nproc)

    or

    cmake ../AviSynthPlus -G "Visual Studio 16 2019" -A x64 -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on && \
    cmake --build . --config Release -j $(nproc)



For 64-bit (XP):
::

    cmake ../AviSynthPlus -G "Visual Studio 17 2022" -A x64 -T "v141_xp" -DWINXP_SUPPORT:bool=on -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on && \
    cmake --build . --config Release -j $(nproc)

    or

    cmake ../AviSynthPlus -G "Visual Studio 16 2019" -A x64 -T "v141_xp" -DWINXP_SUPPORT:bool=on -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on && \
    cmake --build . --config Release -j $(nproc)


Copy the .dlls to the packaging directory:
::

    cp Output/AviSynth.dll Output/system/DevIL.dll Output/plugins/* ../$AVSDIRNAME/64bit

Copy the .libs to the packaging directory:
::

    cp avs_core/Release/AviSynth.lib plugins/DirectShowSource/Release/*.lib \
    ../AviSynthPlus/plugins/ImageSeq/lib/DevIL_x64/DevIL.lib plugins/ImageSeq/Release/ImageSeq.lib \
    plugins/Shibatch/PFC/Release/PFC.lib plugins/Shibatch/Release/Shibatch.lib \
    plugins/TimeStretch/Release/TimeStretch.lib plugins/TimeStretch/SoundTouch/Release/SoundTouch.lib \
    plugins/VDubFilter/Release/VDubFilter.lib ../$AVSDIRNAME/64bit/dev


Finishing up
~~~~~~~~~~~~

Packaging up everything can be quickly done with 7-zip:
::

    cd ..
    7z a -mx9 $AVSDIRNAME.7z $AVSDIRNAME


Building with Microsoft C++ (cmake command line)
------------------------------------------------

From CMake GUI:
~~~~~~~~~~~~~~~

1. Delete Cache
2. ``Where is source code`` and ``Where to build binaries``: git project folder e.g. C:/Github/AviSynthPlus
3. Press Configure
4. Choose an available generator:

   - `Visual Studio 17 2022` (solution will be generated for VS2022)
   - `Visual Studio 16 2019` (solution will be generated for VS2019)
5. Choose optional platform generator: default is `x64` when left empty, `Win32` is another option
6. When you want XP compatible build, set ``Optional toolset to use (-T option)``:

  - `v141_xp`

  (note: for XP this is only the half of the prerequisites)

7. Fill options, Generate
8. Open the generated solution with Visual Studio GUI, build/debug

Note: you can't have a solution file containing both x86 and x64 configuration at a time.

Command line
~~~~~~~~~~~~

Examples (assuming we are in ``avisynth-build`` folder)
Config (--config parameter) can be Debug, Release, RelWithDebInfo.

**Visual Studio 2022**


``msvc_2022_win64_cleanfirst.bat``

::

      @rem cd avisynth-build
      del .\CMakeCache.txt
      cmake .. -G "Visual Studio 17 2022" -A x64 -DWINXP_SUPPORT:bool=off -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on -DENABLE_INTEL_SIMD:bool=ON
      cmake --build . --config Release --clean-first


``msvc_2022_win64_cuda_plugins_allowed_cleanfirst.bat``

::

      @rem cd avisynth-build
      del .\CMakeCache.txt
      cmake .. -G "Visual Studio 17 2022" -A x64 -DENABLE_CUDA:bool=on -DWINXP_SUPPORT:bool=off -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on -DENABLE_INTEL_SIMD:bool=ON
      cmake --build . --config Release --clean-first

``msvc_2022_win32_xp_sse_cleanfirst.bat`` 

::

      @rem cd avisynth-build
      del .\CMakeCache.txt
      cmake .. -G "Visual Studio 17 2022" -A Win32 -T "v141_xp" -DMSVC_CPU_ARCH:string="SSE" -DWINXP_SUPPORT:bool=on -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on -DENABLE_INTEL_SIMD:bool=ON
      cmake --build . --config Release --clean-first


``msvc_2022_win64_xp_cleanfirst.bat``

::

      @rem cd avisynth-build
      del .\CMakeCache.txt
      cmake .. -G "Visual Studio 17 2022" -A x64 -T "v141_xp" -DWINXP_SUPPORT:bool=on -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on -DENABLE_INTEL_SIMD:bool=ON
      cmake --build . --config Release --clean-first


**Visual Studio 2019**

``msvc_win64_cleanfirst.bat``

::

      @rem cd avisynth-build
      del .\CMakeCache.txt
      cmake .. -G "Visual Studio 16 2019" -A x64 -DENABLE_CUDA:bool=on -DWINXP_SUPPORT:bool=off -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on -DENABLE_INTEL_SIMD:bool=ON
      cmake --build . --config Release --clean-first


``msvc_win32_xp_sse_cleanfirst.bat`` 

::

      @rem cd avisynth-build
      del .\CMakeCache.txt
      cmake .. -G "Visual Studio 16 2019" -A Win32 -T "v141_xp" -DMSVC_CPU_ARCH:string="SSE" -DWINXP_SUPPORT:bool=on -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on -DENABLE_INTEL_SIMD:bool=ON
      cmake --build . --config Release --clean-first

``msvc_win64_xp_cleanfirst.bat``

::

      @rem cd avisynth-build
      del .\CMakeCache.txt
      cmake .. -G "Visual Studio 16 2019" -A x64 -T "v141_xp" -DWINXP_SUPPORT:bool=on -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on -DENABLE_INTEL_SIMD:bool=ON
      cmake --build . --config Release --clean-first


``msvc_win32_xp_nointel_cleanfirst.bat``

::

    @rem cd avisynth-build
    del .\CMakeCache.txt
    cmake .. -G "Visual Studio 16 2019" -A Win32 -T "v141_xp" -DMSVC_CPU_ARCH:string="SSE" -DWINXP_SUPPORT:bool=on -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on -DENABLE_INTEL_SIMD:bool=OFF
    cmake --build . --config Release --clean-first





Building with Intel C++ Compiler ICX or ICL (Windows)
-----------------------------------------------------

Prerequisites:
~~~~~~~~~~~~~~

Useful link:

https://www.intel.com/content/www/us/en/developer/articles/news/free-intel-software-developer-tools.html

We need Intel oneAPI Base Kit and optionally oneAPI HPC Toolkit

- Download Intel® oneAPI DPC++/C++ Compiler

  - https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit
  - Includes C++ 2021.4 (we need Intel C++ 2021; DPC++ is not suitable for Avisynth+)
  - Save disk space: No Python, No Math kernel Library, No Video Processing, No Deep Neural

- Download component for C++

  - https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#hpc-kit
  - Intel® oneAPI HPC Toolkit for Windows*
  - Why: Intel® C++ Compiler Classic
  - Choose Custom Installation (Fortran support not needed)

Howto: https://www.intel.com/content/www/us/en/developer/articles/technical/using-oneapi-compilers-with-cmake-in-visual-studio.html

Choose "Intel(R) oneAPI DPC++ Compiler", there are two flavours (DPCPP is not compatible with Avisynth)

- Intel® NextGen Compiler (in base kit, LLVM based)

  TOOLSET = "Intel C++ Compiler 2021", COMPILER EXE NAME = icx.exe

- Intel® Classic Compiler (in extra HPC kit)

  TOOLSET = "Intel C++ Compiler 19.2", COMPILER EXE NAME = icl.exe

Once installed first one or both, check some files.

There are CMake support files (info from: c:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\cmake\\SYCL\\)

Copy

  c:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\cmake\\SYCL\\FindIntelDPCPP.cmake

to

  c:\\Program Files\\CMake\\share\\cmake-3.20\\Modules\\

Note: Intel C++ Compilers need Cmake 3.20 as a minimum.

From CMake GUI:
~~~~~~~~~~~~~~~

1. Delete Cache
2. ``Where is source code`` and ``Where to build binaries``: git project folder e.g. C:/Github/AviSynthPlus
3. Press Configure
4. Choose an available generator:
   - `Visual Studio 17 2022` (solution will be generated for VS2022)
   - `Visual Studio 16 2019` (solution will be generated for VS2019)
5. Choose optional platform generator: default is `x64` when left empty, `Win32` is another option
6. Set ``Optional toolset to use (-T option)``:

  - For LLVM based icx: `Intel C++ Compiler 2021`
  - For classic 19.2 icl: `Intel C++ Compiler 19.2`

7. Specify native compilers (checkbox): browse for the appropriate compiler executable path.

  - icx: C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\bin\\icx.exe
  - icl: C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\bin\\intel64\\icl.exe

If you have errors like ``xilink: : error : Assertion failed (shared/driver/drvutils.c, line 312`` then
as a workaround you must copy clang.exe (by default it is located in C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\bin)
to the folder beside xilink (for x64 configuration it is in C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\bin\\intel64).

Successful log looks like:

::

      The CXX compiler identification is IntelLLVM 2021.4.0 with MSVC-like command-line
      Check for working CXX compiler: C:/Program Files (x86)/Intel/oneAPI/compiler/2021.4.0/windows/bin/icx.exe

or

::

      The CXX compiler identification is Intel 2021.4.0.20210910
      Check for working CXX compiler: C:/Program Files (x86)/Intel/oneAPI/compiler/2021.4.0/windows/bin/intel64/icl.exe

8. Fill options, Generate
9. Open the generated solution with Visual Studio GUI, build/debug


Command line
~~~~~~~~~~~~

Examples (assuming we are in ``avisynth-build`` folder). Config can be Debug, Release, RelWithDebInfo.

``x_icl_cleanfirst.bat`` 

::

      @rem cd avisynth-build
      del .\CMakeCache.txt
      C:\Program Files (x86)\Intel\oneAPI\setvars.bat
      cmake ../ -T "Intel C++ Compiler 19.2" -DCMAKE_CXX_COMPILER="icl.exe" -DBUILD_DIRECTSHOWSOURCE:bool=off -DENABLE_PLUGINS:bool=on -DENABLE_INTEL_SIMD:bool=ON
      cmake --build . --config Debug --clean-first

``x_icx_cleanfirst.bat``

::

      @rem cd avisynth-build
      del .\CMakeCache.txt
      C:\Program Files (x86)\Intel\oneAPI\setvars.bat
      cmake ../ -T "Intel C++ Compiler 2021" -DCMAKE_CXX_COMPILER="icx.exe" -DBUILD_DIRECTSHOWSOURCE:bool=off -DENABLE_PLUGINS:bool=on -DENABLE_INTEL_SIMD:bool=ON
      cmake --build . --config Debug --clean-first

``x_icx_cleanfirst_no_simd.bat``
This one will build only Avisynth.dll, no external plugins, plain C code (no SIMD)

::

      @rem cd avisynth-build
      del .\CMakeCache.txt
      C:\Program Files (x86)\Intel\oneAPI\setvars.bat
      cmake ../ -T "Intel C++ Compiler 2021" -DCMAKE_CXX_COMPILER="icx.exe" -DBUILD_DIRECTSHOWSOURCE:bool=off -DENABLE_PLUGINS:bool=OFF -DENABLE_INTEL_SIMD:bool=OFF
      cmake --build . --config Debug --clean-first


Building with Clang
-------------------

Command line: todo

Using Cmake GUI:
~~~~~~~~~~~~~~~~

1. Delete Cache
2. ``Where is source code`` and ``Where to build binaries``: git project folder e.g. C:/Github/AviSynthPlus
3. Press Configure
4. Choose generator:

   - `Visual Studio 17 2022` (solution will be generated for VS2022)
   - `Visual Studio 16 2019` (solution will be generated for VS2019)
5. Choose optional platform generator: default is `x64` when left empty, `Win32` is another option
6. Set ``Optional toolset to use (-T option)``:

  Type ``llvm`` or ``clangcl``

  clangcl (Clang-cl) comes with Visual Studio.
  
  for native LLVM you may need to specify native compilers (checkbox): browse for the appropriate compiler executable path.

Hint: How to install Clang-cl in Visual Studio: as it appears in VS2019/2022:

    Tools|Get Tools and Features|Add Individual Components|Compilers, build tools, and runtimes
    
        [X] C++ Clang compiler for Windows
        [X] C++ Clang-cl for v142/v143 build tools (x64/x86)

7. Fill options, Generate
8. Open the generated solution with Visual Studio GUI, build/debug


Building with GCC
-----------------

AviSynth+ can be built with GCC two different ways: using MSys2 as a native toolchain,
or cross-compiled under another OS such as a Linux distribution.

Building with GCC in MSys2
~~~~~~~~~~~~~~~~~~~~~~~~~~

Launch MSys2 and install GCC and Ninja:
::

    pacman -S mingw64/mingw-w64-x86_64-gcc gcc mingw64/ninja mingw32/ninja mingw32/mingw-w64-i686-gcc

Grab the AviSynth+ source code:
::

    cd $HOME && \
    git clone https://github.com/AviSynth/AviSynthPlus && \
    cd AviSynthPlus && \
    mkdir -p avisynth-build/i686 avisynth-build/amd64

If you were in the MSys2 MSYS prompt, open the MinGW32 prompt, then navigate into
the build directory, build AviSynth+, and install it:
::

    cd $HOME/AviSynthPlus/avisynth-build/i686 && \
        cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=$HOME/avisynth+_build/32bit \
        -DBUILD_SHIBATCH:bool=off && \
    ninja && \
    ninja install

(The Shibatch plugin currently has issues on GCC, so disable it for now.
DirectShowSource also has issues, but it doesn't get built by default.)

Open the MinGW64 prompt now, navigate into the build directory, build AviSynth+, and install it:
::

    cd $HOME/AviSynthPlus && \
    AVSDIRNAME=avisynth+_r$(git rev-list --count HEAD)-g$(git rev-parse --short HEAD)-$(date --rfc-3339=date | sed 's/-//g') && \
    cd avisynth-build/amd64 && \
        cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=$HOME/avisynth+_build/64bit \
        -DBUILD_SHIBATCH:bool=off && \
    ninja && \
    ninja install

(The Shibatch plugin currently has issues on GCC, so disable it for now.
DirectShowSource also has issues, but it doesn't get built by default.)


Finishing up
............

Now, without leaving the MinGW64 prompt, package the binaries up in a 7zip archive:
::

    mv $HOME/avisynth+_build $HOME/$AVSDIRNAME && \
    7za a -mx9 ~/$AVSDIRNAME.7z ~/$AVSDIRNAME


Cross-compiling with GCC
~~~~~~~~~~~~~~~~~~~~~~~~

For ease of explanation, we'll assume Ubuntu Linux.  The method to cross-compile under
most distributions is largely the same, so don't worry about that.

Ubuntu's repositories lag behind upstream GCC releases, and my current build
instructions are built around a most-recent-stable version of GCC and MinGW.
The full instructions for that are contained in the first section of
`<https://github.com/qyot27/mpv/blob/extra-new/DOCS/crosscompile-mingw-tedious.txt>`_

Download the source code and prepare the build directories:
::

    git clone https://github.com/AviSynth/AviSynthPlus && \
    cd AviSynthPlus && \
    mkdir -p avisynth-build/i686 avisynth-build/amd64 && \
    AVSDIRNAME=avisynth+-gcc_r$(git rev-list --count HEAD)-g$(git rev-parse --short HEAD)-$(date --rfc-3339=date | sed 's/-//g') && \

32-bit:
::

    cd avisynth-build/i686 && \
        cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=$HOME/avisynth+_build/32bit \
        -DCMAKE_TOOLCHAIN_FILE="/usr/x86_64-w64-mingw32/toolchain-x86_64-w64-mingw32.cmake" \
        -DCMAKE_C_FLAGS="-m32" -DCMAKE_CXX_FLAGS="-m32" -DCMAKE_RC_FLAGS="-F pe-i386" \
        -DBUILD_SHIBATCH:bool=off && \
    ninja && \
    ninja install

64-bit:
::

    cd ../amd64 && \
        cmake ../../ -G "Ninja" -DCMAKE_INSTALL_PREFIX=$HOME/avisynth+_build/64bit \
        -DCMAKE_TOOLCHAIN_FILE="/usr/x86_64-w64-mingw32/toolchain-x86_64-w64-mingw32.cmake" \
        -DBUILD_SHIBATCH:bool=off && \
    ninja && \
    ninja install


Finishing up
............

Packaging:
::

    mv $HOME/avisynth+_build $HOME/$AVSDIRNAME
    7za a -mx9 ~/$AVSDIRNAME.7z ~/$AVSDIRNAME


Back to the :doc:`main page <../../index>`

$ Date: 2021-12-08 12:12:00 +01:00 $
