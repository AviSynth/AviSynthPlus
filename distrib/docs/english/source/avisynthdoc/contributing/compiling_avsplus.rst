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

Related documents:

* :doc:`avsplus_external_deps_guide_manual` — building DevIL and SoundTouch
  with Visual Studio.
* :doc:`avsplus_external_deps_guide_manual_arm` — the Windows on ARM
  dependencies and toolchain (llvm-mingw, cross-compiled from Linux/WSL2).
* :doc:`release_build_automation` — the automated multi-configuration
  Windows dev build (``build_release.bat``).
* :doc:`visual_studio_cmake_presets` — ``CMakePresets.json``: x64 MSVC and
  ARM64 llvm-mingw builds from the Visual Studio IDE (Open Folder).
* :doc:`posix` — building and using AviSynth+ on Linux, macOS and BSD.
* :doc:`contributing_with_git` — git primer for contributors.

.. contents:: Table of contents



AviSynth+ prerequisites
-----------------------

Note that AviSynth+ is not restricted to Windows.

AviSynth+ can be built by a few different compilers:

* Visual Studio 2019, 2022, 2026 or higher. (May work for VS2017)
  - native msvc or clang-cl
* Clang 7.0.1 or higher.
* GCC 7 or higher.
* Intel C++ Compiler (2021-) (ICX: LLVM based NextGen)
* Intel C++ Compiler 19.2 (ICL: classic) - discontinued by Intel


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


Visual Studio 2026 Support
--------------------------

Visual Studio 2026 support was introduced in November 2025. To use it, you must have **CMake 4.2** or higher installed.

The generator name is ``Visual Studio 18 2026``.

**Solution Format (.slnx):**
The VS 2026 generator creates a **.slnx** file instead of the traditional .sln file. This is a new XML-based solution format. You should open the **.slnx** file in Visual Studio.

**Windows XP Support:**
Visual Studio 2026 still supports the ``v141_xp`` toolset. If you have the toolset installed, you can generate XP-compatible builds using the new generator.


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

| Download the Windows SDK 7.1 (Windows SDK for Windows 7 and .NET Framework 4, June 2010):
| `<https://developer.microsoft.com/en-us/windows/downloads/sdk-archive/>`_

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

Allow Visual Studio to convert the project, switch the configuration to ``Release_MBCS``,
and enter the project Properties by right-clicking on the solution name and selecting
``Properties``.

Select the ``Visual Studio 17 - Windows XP (v141_xp)`` option on the main Properties page
under ``Toolset``, and on the ``C/C++ Code Generation`` page select *Disabled* or *SSE*
from the ``Enhanced Instruction Set`` option (IMO, it's safer to disable it for system
support libraries like strmbase.lib), and finally, exit back to the main screen.

Now select ``Build``. That's it.

For 64-bit, change to ``Release_MBCS x64`` and ``Build``. The SSE2 note isn't relevant here, since
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

Note: depending on your Visual Studio 2019, 2022 or 2026 version, choose only one cmake build block.
**Reminder:** You need CMake 4.2+ for Visual Studio 2026.

For 32-bit (no XP, SSE2):
::

    cmake ../AviSynthPlus -G "Visual Studio 18 2026" -A Win32 -DMSVC_CPU_ARCH:string="SSE2" -DBUILD_DIRECTSHOWSOURCE:bool=on && \
    cmake --build . --config Release -j $(nproc)

    or

    cmake ../AviSynthPlus -G "Visual Studio 17 2022" -A Win32 -DMSVC_CPU_ARCH:string="SSE2" -DBUILD_DIRECTSHOWSOURCE:bool=on && \
    cmake --build . --config Release -j $(nproc)

    or

    cmake ../AviSynthPlus -G "Visual Studio 16 2019" -A Win32 -DMSVC_CPU_ARCH:string="SSE2" -DBUILD_DIRECTSHOWSOURCE:bool=on && \
    cmake --build . --config Release -j $(nproc)


For 32-bit (XP, SSE):
::

    cmake ../AviSynthPlus -G "Visual Studio 18 2026" -A Win32 -T "v141_xp" -DMSVC_CPU_ARCH:string="SSE" -DWINXP_SUPPORT:bool=on -DBUILD_DIRECTSHOWSOURCE:bool=on && \
    cmake --build . --config Release -j $(nproc)

    or

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

    cmake ../AviSynthPlus -G "Visual Studio 18 2026" -A x64 -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on && \
    cmake --build . --config Release -j $(nproc)

    or

    cmake ../AviSynthPlus -G "Visual Studio 17 2022" -A x64 -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on && \
    cmake --build . --config Release -j $(nproc)

    or

    cmake ../AviSynthPlus -G "Visual Studio 16 2019" -A x64 -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on && \
    cmake --build . --config Release -j $(nproc)



For 64-bit (XP):
::

    cmake ../AviSynthPlus -G "Visual Studio 18 2026" -A x64 -T "v141_xp" -DWINXP_SUPPORT:bool=on -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on && \
    cmake --build . --config Release -j $(nproc)

    or

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

   - `Visual Studio 18 2026` (requires CMake 4.2+; generates **.slnx** file)
   - `Visual Studio 17 2022` (solution will be generated for VS2022)
   - `Visual Studio 16 2019` (solution will be generated for VS2019)
5. Choose optional platform generator: default is `x64` when left empty, `Win32` is another option
6. When you want XP compatible build, set ``Optional toolset to use (-T option)``:

  - `v141_xp`

  (note: for XP this is only the half of the prerequisites. Tested and working with VS 2026)

7. Fill options

   - ImageSeq.DLL

     This plugin has external dependencies: DevIL SDK headers and libraries 
     are no longer included as a copy in Avisynth repo since 2024.

     In order to be able to debug them we'd need to download and set some things manually.
     
     Download and extract DevIL SDK into a folder. In our example it is ``C:/avsplus_build_deps/DevIL Windows SDK/``.
     See :ref:`devil_prebuilt_sdk_section1` and :ref:`devil_prebuilt_sdk_section2` for more details.

     Manually edit CMAKE GUI options (x64 example): 

     * BUILD_IMAGESEQ [X]
     * ``ILU_LIBRARIES`` ``C:/avsplus_build_deps/DevIL Windows SDK/lib/x64/Release/ILU.lib``
     * ``IL_INCLUDE_DIR`` ``c:\avsplus_build_deps\DevIL Windows SDK\include\IL``
     * ``IL_LIBRARIES`` ``C:/avsplus_build_deps/DevIL Windows SDK/lib/x64/Release/DevIL.lib``
     
   - TimeStretch.DLL
   
     This plugin has external dependencies: SoundTouch project. Its source snapshot is 
     no longer included as a copy in Avisynth repo since 2024.
     
     See the :doc:`avsplus_external_deps_guide_manual` for more details.

     * BUILD_TIMESTRETCH [X]
     * ``SOUNDTOUCH`` ``... to be filled ...``
     * ``SOUNDTOUCH_DIR`` ``... to be filled ...``

8. Generate

9. Open the generated solution (**slnx for VS2026**, sln for others) with Visual Studio GUI, build/debug

Note: you can't have a solution file containing both x86 and x64 configuration at a time.

Command line
~~~~~~~~~~~~

Examples (assuming we are in ``avisynth-build`` folder)
Config (--config parameter) can be Debug, Release, RelWithDebInfo.

**Visual Studio 2026**

``msvc_2026_win64_cleanfirst.bat``

::

      @rem cd avisynth-build
      del .\CMakeCache.txt
      del ..\CMakeCache.txt
      cmake .. -G "Visual Studio 18 2026" -A x64 -DWINXP_SUPPORT:bool=off -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on -DENABLE_INTEL_SIMD:bool=ON
      cmake --build . --config Release --clean-first

``msvc_2026_win32_xp_sse_cleanfirst.bat`` 

::

      @rem cd avisynth-build
      del .\CMakeCache.txt
      del ..\CMakeCache.txt
      cmake .. -G "Visual Studio 18 2026" -A Win32 -T "v141_xp" -DMSVC_CPU_ARCH:string="SSE" -DWINXP_SUPPORT:bool=on -DBUILD_DIRECTSHOWSOURCE:bool=on -DENABLE_PLUGINS:bool=on -DENABLE_INTEL_SIMD:bool=ON
      cmake --build . --config Release --clean-first


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





Building with Intel C++ Compiler ICX (IntelLLVM) or ICL (Windows)
-----------------------------------------------------------------

Prerequisites:
~~~~~~~~~~~~~~

Useful link:

`<https://www.intel.com/content/www/us/en/developer/articles/news/free-intel-software-developer-tools.html>`_

We need Intel oneAPI Base Kit for LLVM based compiler and optionally oneAPI HPC Toolkit for the classic C++ compiler.

- Download Intel® oneAPI DPC++/C++ Compiler

  - `<https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit>`_

    Download the base kit.

    `<https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_

    Choose online or offline installer.

    DPC++/C++ is compulsory (we need then only Intel C++; DPC++ is not suitable for Avisynth)

    Save disk space: No Math kernel Library, No Video Processing, No Deep Neural

  - Choose IDE Integration: Visual Studio 2022 (or 2019)

- Optionally: download component for C++

  - `<https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#hpc-kit>`_
  - Intel® oneAPI HPC Toolkit for Windows
  - Why: Intel® C++ Compiler Classic 19.2 (will be discontinued)
  - Choose Custom Installation (Fortran support not needed)

Howto: `<https://www.intel.com/content/www/us/en/developer/articles/technical/using-oneapi-compilers-with-cmake-in-visual-studio.html>`_

There are two main flavours which we can use (DPC++ is not compatible with Avisynth)

- Intel® NextGen Compiler (in base kit, LLVM based)

  - TOOLSET = "Intel C++ Compiler 2025", COMPILER EXE NAME = icx.exe
  - TOOLSET = "Intel C++ Compiler 2024", COMPILER EXE NAME = icx.exe
  - TOOLSET = "Intel C++ Compiler 2023", COMPILER EXE NAME = icx.exe
  - TOOLSET = "Intel C++ Compiler 2022", COMPILER EXE NAME = icx.exe
  - TOOLSET = "Intel C++ Compiler 2021", COMPILER EXE NAME = icx.exe

- Intel® Classic Compiler (in extra HPC kit)

  - TOOLSET = "Intel C++ Compiler 19.2", COMPILER EXE NAME = icl.exe

  Note that this classic compiler will get discontinued, as of late 2023.

Once installed first one or both, check some files.

CMake integration and support files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. For Intel C++ Compiler 2025:

   Info from: c:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\lib\\cmake\\IntelDPCPP\\ReadMeDPCPP.txt

   Copy

     c:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\lib\\cmake\\IntelDPCPP\\IntelDPCPPConfig.cmake

   and

     c:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\lib\\cmake\\IntelDPCPP\\IntelDPCPPConfigVersion.cmake

   to

    c:\\Program Files\\CMake\\share\\cmake-3.25\\Modules\\IntelDPCPP\\

2. For Intel C++ Compiler 2024:

   Info from: c:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\lib\\cmake\\IntelDPCPP\\ReadMeDPCPP.txt

   Copy

     c:\\Program Files (x86)\\Intel\oneAPI\\compiler\\latest\\windows\\IntelDPCPP\\IntelDPCPPConfig.cmake

   and

     c:\\Program Files (x86)\\Intel\oneAPI\\compiler\\latest\\windows\\IntelDPCPP\\IntelDPCPPConfigVersion.cmake

   to

    c:\\Program Files\\CMake\\share\\cmake-3.25\\Modules\\


3. For Intel C++ Compiler 2023:

   Info from: c:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\IntelDPCPP\\ReadMe.txt

   Copy

     c:\\Program Files (x86)\\Intel\oneAPI\\compiler\\latest\\windows\\IntelDPCPP\\IntelDPCPPConfig.cmake

   to

    c:\\Program Files\\CMake\\share\\cmake-3.25\\Modules\\


4. For Intel C++ Compiler 2021:

   Info from: c:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\cmake\\SYCL\\

   Copy

     c:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\cmake\\SYCL\\FindIntelDPCPP.cmake

   to

     c:\\Program Files\\CMake\\share\\cmake-3.20\\Modules\\

Note: Intel C++ Compilers need Cmake 3.22.3 (Windows) or 3.22.1 (Linux) as a minimum (as of Intel 2024 or 2025)


From CMake GUI:
~~~~~~~~~~~~~~~

1. Delete Cache
2. ``Where is source code`` and ``Where to build binaries``: git project folder e.g. C:/Github/AviSynthPlus
3. Press Configure
4. Choose an available generator:

  - `Visual Studio 18 2026` (requires CMake 4.2+, solution generated for VS2026)
  - `Visual Studio 17 2022` (solution will be generated for VS2022)
  - `Visual Studio 16 2019` (solution will be generated for VS2019)

5. Choose optional platform generator: default is `x64` when left empty, `Win32` is another option
6. Set ``Optional toolset to use (-T option)``:

  - For LLVM based icx:
   
    - `Intel C++ Compiler 2025` or
    - `Intel C++ Compiler 2024` or
    - `Intel C++ Compiler 2023` or
    - `Intel C++ Compiler 2022` or
    - `Intel C++ Compiler 2021`

  - For classic icl:

    - `Intel C++ Compiler 19.2`

7. Specify native compilers (checkbox): browse for the appropriate compiler executable path.

  - Intel C++ Compiler 2025:

    - icx: c:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\bin\\icx.exe

  - Intel C++ Compiler 2024:

    - icx: C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\bin\\icx.exe
    - icl: C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\bin\\intel64\\icl.exe

If you have errors like ``xilink: : error : Assertion failed (shared/driver/drvutils.c, line 312`` then
as a workaround you must copy clang.exe (by default it is located in C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\bin)
to the folder beside xilink (for x64 configuration it is in C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\bin\\intel64).

Successful log looks like:

(Note: If CXX compiler is not the Intel one, then you probably missed step #7.)

::

      The CXX compiler identification is IntelLLVM 2025.0.0 with MSVC-like command-line
      Check for working CXX compiler: C:/Program Files (x86)/Intel/oneAPI/compiler/2025.0/bin/icx.exe

or

::

      The CXX compiler identification is IntelLLVM 2023.0.0 with MSVC-like command-line
      Check for working CXX compiler: C:/Program Files (x86)/Intel/oneAPI/compiler/2023.0.0/windows/bin/icx.exe

or

::

      The CXX compiler identification is IntelLLVM 2021.4.0 with MSVC-like command-line
      Check for working CXX compiler: C:/Program Files (x86)/Intel/oneAPI/compiler/2021.4.0/windows/bin/icx.exe

or (classic 19.2 version downloaded in 2023)

::

      The CXX compiler identification is Intel 2021.8.0.20221119
      Check for working CXX compiler: C:/Program Files (x86)/Intel/oneAPI/compiler/2023.0.0/windows/bin/intel64/icl.exe - skipped

or (classic 19.2 version downloaded in 2021)

::

      The CXX compiler identification is Intel 2021.4.0.20210910
      Check for working CXX compiler: C:/Program Files (x86)/Intel/oneAPI/compiler/2021.4.0/windows/bin/intel64/icl.exe

8. Fill options, Generate
9. Open the generated solution with Visual Studio GUI, build/debug
10. Note that the built program would need Intel redistributable components or else you may face errors that dependencies
    could not be loaded: svml_dispmd.dll and libmmd.dll. Check Intel OneAPI Redistributable package at:

    `<https://www.intel.com/content/www/us/en/developer/articles/tool/compilers-redistributable-libraries-by-version.html>`_


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
      cmake ../ -T "Intel C++ Compiler 2025" -DCMAKE_CXX_COMPILER="icx.exe" -DBUILD_DIRECTSHOWSOURCE:bool=off -DENABLE_PLUGINS:bool=on -DENABLE_INTEL_SIMD:bool=ON
      cmake --build . --config Debug --clean-first

``x_icx_cleanfirst_no_simd.bat``
This one will build only Avisynth.dll, no external plugins, plain C code (no SIMD)

::

      @rem cd avisynth-build
      del .\CMakeCache.txt
      C:\Program Files (x86)\Intel\oneAPI\setvars.bat
      cmake ../ -T "Intel C++ Compiler 2025" -DCMAKE_CXX_COMPILER="icx.exe" -DBUILD_DIRECTSHOWSOURCE:bool=off -DENABLE_PLUGINS:bool=OFF -DENABLE_INTEL_SIMD:bool=OFF
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

   - `Visual Studio 18 2026` (solution will be generated for VS2026)
   - `Visual Studio 17 2022` (solution will be generated for VS2022)
   - `Visual Studio 16 2019` (solution will be generated for VS2019)

5. Choose optional platform generator: default is `x64` when left empty, `Win32` is another option
6. Set ``Optional toolset to use (-T option)``:

  Type ``llvm`` or ``clangcl``

  clangcl (Clang-cl) comes with Visual Studio.

  for native LLVM you may need to specify native compilers (checkbox): browse for the appropriate compiler executable path.

  Hint: How to install Clang-cl in Visual Studio:

  Tools|Get Tools and Features|Add Individual Components|Compilers, build tools, and runtimes

  For VS2026:
    - [X] C++ Clang compiler for Windows
    - [X] MSBuild support for LLVM (clang-cl) toolset

  For VS2019/2022:
    - [X] C++ Clang compiler for Windows
    - [X] C++ Clang-cl for v142/v143 build tools (x64/x86)

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

.. _compiling_avsplus_crosscompiling1:

Cross-compiling with GCC (Version #1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. _compiling_avsplus_crosscompiling2:

Cross-compiling with GCC (Version #2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**AviSynth+ cross-compilation Guide (On Ubuntu to create Windows, MinGW-w64)**

**Using POSIX-threaded MinGW toolchains (x86 and x64)**

This document describes how to cross‑compile **AviSynth+** for **Windows 32-bit (x86)** and 
**Windows 64-bit (x64)** from a Linux (Ubuntu) system using the **MinGW-w64 GCC** toolchain.

The guide includes fixes for:

*   `std::mutex` / POSIX thread model issues
*   CMake/Ninja RPATH errors
*   32-bit vs. 64-bit toolchain selection
*   SSE2/MMX intrinsic mismatches
*   Packaging and dependency notes

This guide is verified against Ubuntu 22.04 and MinGW-w64 POSIX compilers.


1. Overview
...........

AviSynth+ uses C++11 threading primitives (e.g. `std::mutex`, `std::thread`), which require
the **POSIX** flavor of MinGW-w64. 
Ubuntu provides separate cross-compilers for:

*    i686-w64-mingw32        => 32-bit Windows target
*    x86_64-w64-mingw32      => 64-bit Windows target

Ubuntu MinGW packages are not multilib; the 64-bit compiler cannot build 32-bit code.
You must use **two toolchains**.

2. Prerequisites
................

Install all needed packages:

.. code-block:: bash

    sudo apt update \
    sudo apt install git cmake ninja-build mingw-w64 p7zip-full

Ninja is used for fast builds; p7zip is recommended for the final package.

3. About the POSIX Threading Model
..................................

MinGW-w64 provides two threading backends:

*   **win32** (default on older distros)
*   **posix** (required for `std::mutex` and modern C++ threading)

Even on modern systems, *you must explicitly use the POSIX compiler variants*:

*    i686-w64-mingw32-g++-posix
*    x86_64-w64-mingw32-g++-posix

You may *optionally* configure update-alternatives:

.. code-block:: bash

    sudo update-alternatives --set i686-w64-mingw32-gcc /usr/bin/i686-w64-mingw32-gcc-posix \
    sudo update-alternatives --set i686-w64-mingw32-g++ /usr/bin/i686-w64-mingw32-g++-posix \
    sudo update-alternatives --set x86\_64-w64-mingw32-gcc /usr/bin/x86\_64-w64-mingw32-gcc-posix \
    sudo update-alternatives --set x86\_64-w64-mingw32-g++ /usr/bin/x86\_64-w64-mingw32-g++-posix

.. note::
   **This is optional.**
   The CMake commands below explicitly select the correct compilers, so you do **not** need to modify system defaults.

4. Get the Source
.................

And create a reproducible output directory name.

.. code-block:: bash

    git clone https://github.com/AviSynth/AviSynthPlus \
    cd AviSynthPlus \
    mkdir -p avisynth-build/i686 avisynth-build/amd64 \
    AVSDIRNAME=avisynth+-gcc\_r$(git rev-list --count HEAD)-g$(git rev-parse --short HEAD)-$(date +%Y%m%d)

5. Build for 32-bit Windows (i686)
..................................

AviSynth+ internally applies SIMD and intrinsic flags based on `CMAKE_SYSTEM_PROCESSOR`.  
To avoid “target-specific option mismatch” errors, explicitly set it to `i686`.

Ubuntu MinGW is not multilib, so you **must use** the dedicated 32-bit toolchain.

.. code-block:: bash

    cd avisynth-build/i686

    cmake ../../ -G Ninja \
    -DCMAKE\_SYSTEM\_NAME=Windows \
    -DCMAKE\_SYSTEM\_PROCESSOR=i686 \
    -DCMAKE\_BUILD\_TYPE=Release \
    -DCMAKE\_SKIP\_RPATH=TRUE \
    -DCMAKE\_INSTALL\_PREFIX="$HOME/avisynth+\_build/32bit" \
    -DCMAKE\_C\_COMPILER=i686-w64-mingw32-gcc-posix \
    -DCMAKE\_CXX\_COMPILER=i686-w64-mingw32-g++-posix \
    -DBUILD\_SHIBATCH:BOOL=off

    ninja
    ninja install

Notes
^^^^^

*   `CMAKE_SKIP_RPATH=TRUE` eliminates CMake/Ninja install-RPATH relinking failures.
*   SSE2 is always enabled via AviSynth+ internal logic for 32-bit builds.

6. Build for 64-bit Windows (x86\_64)
.....................................

.. code-block:: bash

    cd ../amd64

    cmake ../../ -G Ninja \
      -DCMAKE_SYSTEM_NAME=Windows \
      -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_SKIP_RPATH=TRUE \
      -DCMAKE_INSTALL_PREFIX="$HOME/avisynth+_build/64bit" \
      -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc-posix \
      -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++-posix \
      -DBUILD_SHIBATCH:BOOL=off

    ninja
    ninja install


Notes
^^^^^

*   All x86-64 CPUs support SSE2, so no special flags are required.
*   POSIX compiler ensures correct C++11 threading.

7. Packaging
............

.. code-block:: bash

    cd \~ \
    mv avisynth+\_build "$AVSDIRNAME" \
    7za a -mx9 "$AVSDIRNAME.7z" "$AVSDIRNAME"

You now have:

    <name>.7z
        /32bit  → aviynth.dll + includes + plugins
        /64bit  → same for x64

8. Runtime Dependencies
.......................

POSIX thread model dependency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Builds produced with the POSIX toolchain require:

    libwinpthread-1.dll

This is expected and normal.

Place this DLL next to `avisynth.dll` or ship it with your application.

Static linking
^^^^^^^^^^^^^^

To produce a DLL with no external MinGW dependencies, you can try forcing the linker to use static archives. 
Note that this may require specific library files to be present in your Ubuntu environment.

.. code-block:: bash

   # Add these to your CMake command:
   -DCMAKE_CXX_FLAGS="-static-libgcc -static-libstdc++" \
   -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-Bstatic -lstdc++ -lwinpthread -Wl,-Bdynamic"

**This is advanced.** Fully static POSIX builds are fragile on Ubuntu. If the linker cannot 
find ``libwinpthread.a``, the build will fail. Additionally, static linking in a DLL can cause 
issues with memory management if plugins also statically link their runtimes.

Shipping libwinpthread-1.dll alongside avisynth.dll is the recommended and most stable approach.

9. Known Pitfalls and Fixes
...........................

9.1 Ninja RPATH errors
^^^^^^^^^^^^^^^^^^^^^^


If you see:

    The install of <target> requires changing an RPATH ...
    not supported with Ninja ...

Always ensure:

    -DCMAKE_SKIP_RPATH=TRUE

This avoids relinking steps which do not apply to Windows binaries.

9.2 Intrinsic "Target specific option mismatch" errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Occurs when the CPU type is not set properly.

Fix: set

    -DCMAKE_SYSTEM_PROCESSOR=i686

for 32-bit builds.

9.3 Mixing win32 and posix toolchains
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Never mix:

*    i686-w64-mingw32-g++   (win32 threading)
*    i686-w64-mingw32-g++-posix  (POSIX threading)

This causes:

*   Missing `std::mutex`
*   ABI mismatches
*   Linker errors involving `dllimport` or winpthreads

9.4 Redefinition of CRT functions (`_strlwr`, `_strupr`, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MinGW-w64 already provides these.
When you encounter this, MINGW was not properly detected by CMake, for example
due to a missing `-DCMAKE_SYSTEM_NAME=Windows`. Normally, the directory of 
compatibility string util helpers is omitted.

If still building AviSynth+ with a compatibility layer, guard the definitions using:

.. code-block::

    #if !defined(MINGW32) && !defined(MINGW64)
    /* custom implementations */
    #endif

10. **Understanding "Install" During Cross-Compilation**
........................................................

When building AviSynth+ for Windows using a Linux host system (Ubuntu), `ninja install` may 
appear confusing at first - after all, **Ubuntu cannot load or execute Windows DLLs**.
However, the *install* step in a cross-compilation context **does not mean installing software 
on Ubuntu**. Instead, it serves a very different and critically important purpose.

This section explains **what the install step really does**, **why it is needed**, and **how it fits into the cross-compile workflow**.

10.1. What “Install” *Does NOT* Mean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It does **not** mean any of the following:

*   Installing AviSynth+ into Ubuntu’s system libraries
*   Making Ubuntu able to load or use AviSynth.dll
*   Registering anything system-wide
*   Enabling AviSynth+ on Linux

None of that happens.

Ubuntu cannot run or link Windows binaries; the Windows DLLs are **not** for the host system at all.

10.2. What “Install” *Actually Means* in Cross‑Compilation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a cross-build scenario, the install step simply:

**Creates a Windows-style filesystem layout inside a directory on Linux**

This includes folders like:

*    bin/         → Windows executables and DLLs  
*    lib/         → import libraries, pkgconfig files  
*    include/     → headers for Windows development  
*    lib/avisynth/ → internal plugins  

**Collects all produced build artifacts in one clean prefix**

Instead of having binaries scattered through CMake’s intermediate directories, the install step:

*   copies the DLL
*   copies import libraries (`.dll.a`)
*   copies headers
*   copies built‑in plugins
*   creates `pkgconfig` files

into the exact structure expected by Windows build systems.

**Produces the folder you will later package or ship to Windows**

Your install prefix:

    ``$HOME/avisynth+_build/64bit/``

is not a Linux installation - it is a **staging directory** representing what would be a real 
installation on a Windows filesystem.

You can zip or 7-zip this folder and distribute it.

10.3. Why "Install" Is Required
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**(A) To build a proper Windows release**

AviSynth+, FFmpeg, Qt, LLVM, and almost all cross-built libraries **use the install phase** 
to generate their final, distributable layout.

This ensures your release archive has:

*   the correct include paths
*   the correct lib paths
*   the correct bin path
*   consistent plugin locations
*   accurate metadata (`pkgconfig`, `version.h`, etc.)

**(B) To separate build artifacts from install artifacts**

The **build tree** contains:

*   temporary object files
*   CMake intermediate files
*   Ninja rules
*   dependency databases
*   incremental build caches

This tree is **not a clean or portable output**.
  
The install tree is.

**(C) For dependent projects to find AviSynth+**

If someone cross-compiles a project that depends on AviSynth+, it can use:

.. code-block:: bash

   pkg-config --cflags --libs avisynth``

This only works because the install step generates:

    ``lib/pkgconfig/avisynth.pc``

containing the correct Windows paths.

**(D) For plugin builds**

Plugins expect:
::

    include/avisynth/...
    lib/AviSynth.dll.a

These are only collected during *install*.

10.4. Example: Installed Output Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After:

.. code-block:: bash

   ninja install

You get something like (in your home directory ~):
::

    avisynth+_build/64bit/
    ├── bin/
    │   └── AviSynth.dll
    ├── lib/
    │   ├── AviSynth.dll.a
    │   └── pkgconfig/
    │       └── avisynth.pc
    ├── include/
    │   └── avisynth/
    │       ├── avisynth.h
    │       ├── avs/
    │       └── ...
    └── lib/avisynth/
        ├── ConvertStacked.dll
        └── ConvertStacked.dll.a

And similarly, for 32bit.

Everything here is intended to be **copied to Windows**, not used on Ubuntu.

10.5. What Happens If You Skip `ninja install`?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You would be left with:

*   scattered build outputs
*   missing import libraries
*   missing pkgconfig file
*   missing plugin installation
*   headers buried in the source tree

Your build would not be package‑ready.

In short: *you could not produce a proper Windows release.*

10.6. Summary
^^^^^^^^^^^^^

The install step is **not installation on Ubuntu**.  
It is:

**“Exporting the Windows build into a clean, structured staging directory.”**

This directory is then:

*   packaged (e.g., `.7z` or `.zip`)
*   used by plugin developers
*   consumed by downstream Windows projects
*   deployed to real Windows systems

It is an essential part of cross‑compiling AviSynth+.

11. Summary
...........

This guide provides a **fully correct, stable, reproducible** method to build AviSynth+ for Windows on Ubuntu.  
It works reliably for:

*   Modern C++17 build environments
*   Both 32-bit and 64-bit Windows
*   Ninja + CMake workflows
*   POSIX-threaded MinGW toolchains

.. warning::

    **Note on ABI compatibility**: This build uses the MinGW-w64 (GCC) ABI. While it is fully compatible with the 
    AviSynth C-API and most existing plugins, it is not binary-compatible with C++ plugins compiled 
    specifically for the MSVC ABI.

Finishing up
............

Packaging:
::

    mv $HOME/avisynth+_build $HOME/$AVSDIRNAME
    7za a -mx9 ~/$AVSDIRNAME.7z ~/$AVSDIRNAME


Back to the :doc:`main page <../../index>`

$ Date: 2026-01-25 21:00:00 +01:00 $