Automated Dev Build (Visual Studio 2026)
=========================================

This document describes the automated multi-configuration **developer build**
process for AviSynth+ on Windows using Visual Studio 2022/2026 and MinGW.
The batch script ``avisynth-build-win\build_release.bat`` replaces the manual steps 
of opening CMake GUI, configuring each toolset, switching to the IDE, and copying
outputs by hand.

.. note::

   This is a **dev/test build**, not an official release.  An official release
   additionally requires building the installer and compiling the HTML
   documentation — processes described in :doc:`compiling_avsplus`.  The dev
   build produced here is suitable for distribution to beta testers as a plain
   folder tree (or 7-Zip archive).

Related documents:

* :doc:`compiling_avsplus` — the general build guide: compilers, CMake
  options, the installer and the HTML documentation (official release).
* :doc:`avsplus_external_deps_guide_manual` — building DevIL and SoundTouch
  with Visual Studio.
* :doc:`avsplus_external_deps_guide_manual_arm` — the Linux/WSL2 llvm-mingw
  route for Windows on ARM.
* :doc:`visual_studio_cmake_presets` — single builds (x64 MSVC, ARM64
  llvm-mingw) from the Visual Studio IDE with ``CMakePresets.json``.
* :doc:`posix` — building and using AviSynth+ on Linux, macOS and BSD.
* :doc:`contributing_with_git` — git primer for contributors.

.. contents:: Table of contents
   :depth: 3


Overview
--------

A dev build contains nine binary variants produced in a single script run:

+----------------------------+-------+------------------------------------------------+
| Destination subfolder      | Arch  | Toolset / notes                                |
+============================+=======+================================================+
| x86-xp                     | Win32 | v141_xp, SSE2, WINXP_SUPPORT=on                |
+----------------------------+-------+------------------------------------------------+
| x64-xp                     | x64   | v141_xp, WINXP_SUPPORT=on                      |
+----------------------------+-------+------------------------------------------------+
| x86-VER-clangcl-llvm-VER   | Win32 | ClangCL (Clang/LLVM front-end)                 |
+----------------------------+-------+------------------------------------------------+
| x64-VER-clangcl-llvm-VER   | x64   | ClangCL (Clang/LLVM front-end)                 |
+----------------------------+-------+------------------------------------------------+
| x86-VER                    | Win32 | v145 — current VS 2026 native MSVC             |
+----------------------------+-------+------------------------------------------------+
| x64-VER                    | x64   | v145 — current VS 2026 native MSVC             |
+----------------------------+-------+------------------------------------------------+
| x64-win7-VER-VER           | x64   | v143/19.44 — highest toolset that runs on Win7 |
+----------------------------+-------+------------------------------------------------+
| x86-win7-VER-VER           | Win32 | v143/19.44 — Win7-compatible 32-bit            |
+----------------------------+-------+------------------------------------------------+
| arm64-mingw-clang-VER      | ARM64 | llvm-mingw — MinGW ABI, cross-compiled         |
+----------------------------+-------+------------------------------------------------+

*VER* placeholders are filled from the configuration variables at the top of
the script (e.g. ``x64-19.50``, ``x64-win7-19.44-17.14``).  The ARM64
variant's label carries the LLVM version of the llvm-mingw toolchain
(e.g. ``arm64-mingw-clang-23.1.4``).

Each variant contains:

* ``AviSynth.dll`` — the core library
* ``c_api\AviSynth.lib`` — import library for the C API
  (on ARM64: ``c_api\AviSynth.dll.a``, the MinGW import library)
* ``plugins\`` — built and/or copied plugin DLLs (see per-variant notes)

The build root also contains the shared ``include\`` header tree, license
files, and ``readme_history.txt``.

The version, the build date and the revision number of the build are **not**
configured by hand — they are detected by the script (see `Build
identification`_).


Prerequisites
-------------

#. **Visual Studio 2026** with the following individual components installed:

   * MSVC v145 build tools (x64/x86) — the VS 2026 native toolset
   * MSVC v143 build tools (x64/x86) — the VS 2022 toolset, for the win7 variant
   * Clang compiler for Windows (ClangCL)

   The **XP variants** additionally need **Visual Studio 2022** with the
   *MSVC v141 XP build tools* component (*Individual components* →
   "MSVC v141 - VS 2017 C++ build tools (v14.16)").  VS 2026 no longer
   offers the ``v141_xp`` toolset, so the XP builds are configured with the
   VS 2022 generator (``-G "Visual Studio 17 2022" -T v141_xp``).  When no
   VS 2022 installation is present, the XP variants are skipped with
   ``BUILD_XP=0``.

#. **CMake 4.2 or higher** — required for the ``Visual Studio 18 2026``
   generator.  Add CMake's bin directory to the system ``%PATH%``.

#. **7-Zip** — for archive creation.  Default assumed path:
   ``C:\Program Files\7-Zip\7z.exe``.

#. **SoundTouch static libraries** — built once from source before the first
   run of ``build_release.bat``.  See `Building SoundTouch`_ below.

#. **DevIL Windows SDK 1.8.0** (prebuilt x86/x64 DLL SDK) — needed by
   ImageSeq in all x86/x64 variants.  Unpack it to
   ``C:\avsplus_build_deps\DevIL Windows SDK`` (the folder name has spaces;
   the ``DEVIL_SDK`` variable of the script).  ``DevIL.dll`` is copied next
   to ``ImageSeq.dll`` in ``plugins\``.  Building DevIL yourself is described
   in :doc:`avsplus_external_deps_guide_manual`.

#. **llvm-mingw** — only needed for the ARM64 variant (build 9).  Download the
   *UCRT* build of the ``x86_64`` host package from
   https://github.com/mstorsjo/llvm-mingw/releases and unpack it to
   ``C:\avsplus_build_deps\llvm-mingw-<tag>-ucrt-x86_64``.  The UCRT flavour
   is the one to use: it links against ``ucrtbase.dll`` and the
   ``api-ms-win-crt-*`` forwarders, i.e. the same C runtime as the MSVC
   builds, while still using the MinGW C++ ABI.  The same toolchain built
   and used from Linux/WSL2 is described in
   :doc:`avsplus_external_deps_guide_manual_arm`.

#. **Ninja** — only needed for the ARM64 variant (build 9), because the Visual
   Studio generator cannot cross-compile with a different compiler.  Any
   ``ninja.exe`` on ``%PATH%`` works (e.g. from ``pip install ninja`` or from
   the Meson distribution).

#. A populated ``avisynth-build-win\release-sample\`` folder — see
   `Origin folder (release-sample)`_ below.


Building SoundTouch
-------------------

The TimeStretch plugin links against a static SoundTouch library that must be
built once before running ``build_release.bat``.  (The manual procedure, also
covering DevIL, is in :doc:`avsplus_external_deps_guide_manual`.)  A helper
script is provided:

.. code-block:: bat

   cd /d c:\Github\AviSynthPlus\avisynth-build-win\helpers
   makesoundtouch.bat

The script clones SoundTouch from ``https://codeberg.org/soundtouch/soundtouch``
into ``C:\avsplus-build-deps\soundtouch\`` (skipping the clone if the directory
already exists) and builds five toolset variants.  Note the two spellings:
the hyphenated ``avsplus-build-deps`` holds the SoundTouch source and its
build tree, the underscored ``avsplus_build_deps`` holds the installed
dependencies (and the DevIL SDK and llvm-mingw).  The difference is
deliberate.

+--------------------------------------+-------+---------+-----------------------------+
| Install path                         | Arch  | Toolset | Used by                     |
+======================================+=======+=========+=============================+
| ``C:\avsplus_build_deps\xp-x86-32``  | Win32 | v141_xp | Build 1 (v141_xp Win32)     |
+--------------------------------------+-------+---------+-----------------------------+
| ``C:\avsplus_build_deps\xp-x86-64``  | x64   | v141_xp | Build 2 (v141_xp x64)       |
+--------------------------------------+-------+---------+-----------------------------+
| ``C:\avsplus_build_deps\v143-x86-32``| Win32 | v143    | Builds 3, 5 (non-XP Win32)  |
+--------------------------------------+-------+---------+-----------------------------+
| ``C:\avsplus_build_deps\v143-x86-64``| x64   | v143    | Builds 4, 6, 7 (non-XP x64) |
+--------------------------------------+-------+---------+-----------------------------+
| ``C:\avsplus_build_deps\mingw-arm64``| ARM64 | llvm-   | Build 9 (ARM64)             |
|                                      |       | mingw   |                             |
+--------------------------------------+-------+---------+-----------------------------+

``build_release.bat`` picks up these paths automatically via its ``DEPS_XP_*``,
``DEPS_V143_*`` and ``DEPS_ARM64`` variables — no manual step is needed between
the two scripts.  When the aarch64 SoundTouch library is missing, the ARM64
variant is built without the TimeStretch plugin.

.. note::

   ``makesoundtouch.bat`` sets two switches that SoundTouch needs with the
   newer tools:

   * ``-DCMAKE_POLICY_VERSION_MINIMUM=3.5`` — SoundTouch's ``CMakeLists.txt``
     still says ``cmake_minimum_required(VERSION 3.1)``, and CMake 4.x
     refuses that without the switch.  The script passes it for all five
     variants, not only for aarch64.
   * ``-DSOUNDSTRETCH=OFF`` — the *soundstretch* command line utility is
     built as a GUI application for MinGW (it wants ``WinMain``) and does
     not link; AviSynth+ needs only the static library.  This switch is
     needed for the aarch64 variant only.

Re-running ``makesoundtouch.bat`` is only necessary when the SoundTouch source
is updated or the installed toolchains change.


Origin folder (release-sample)
-------------------------------

``avisynth-build-win\release-sample\`` holds DLLs that are **not rebuilt** by the
script but appear in the dev build package.  Populate it once from the
previous dev build folder and refresh only when those DLLs actually change::

    release-sample\
      x86\
        plugins\
          DirectShowSource.dll        (32-bit, copy-only)
          imageseq-xp-legacy\
            DevIL.dll
            ImageSeq.dll              (legacy bundle with DevIL dependency)
      x64\
        plugins\
          DirectShowSource.dll        (64-bit, copy-only)
          imageseq-xp-legacy\
            DevIL.dll
            ImageSeq.dll

.. note::

   ``DirectShowSource`` is a copy-only item because its build process requires
   additional Windows SDK setup (``strmbase.lib``) documented separately in
   :doc:`compiling_avsplus`.

   ``ImageSeq.dll`` and ``TimeStretch.dll`` in the *main* ``plugins\`` folders
   **are** freshly built by the script (``ENABLE_PLUGINS=on``).  The
   ``imageseq-xp-legacy\`` subfolder is a separate, older build bundling a
   legacy DevIL version and is included only in the XP variants.


Running the script
------------------

1. Open a **plain Command Prompt** (no Developer Prompt required — the VS
   generator handles the compiler environment internally).

2. Edit the **CONFIGURATION block** near the top of
   ``avisynth-build-win\build_release.bat``::

       rem  Build identification
       set BUILD_TAG=tst

       rem  Which variants to build (set a switch to 0 to skip that group)
       set BUILD_XP=0        (the two v141_xp XP builds)
       set BUILD_CLANG=1     (the two ClangCL builds)
       set BUILD_V145=1      (the two v145 builds)
       set BUILD_WIN7=1      (the two v143 win7 builds)
       set BUILD_ARM64=1     (the MinGW ARM64 build)

       rem  Compiler version labels (cosmetic, appear in subfolder names)
       set WIN7_MSVC_VER=19.44   (v143 win7-compat build)
       set WIN7_TV=17.14

       set BUILD_JOBS=8

   The version number, the build date and the revision number are **detected**
   (see `Build identification`_ below); only the tag and the two pinned
   Win7 labels are manual.  The MSVC/Clang version labels are auto-detected
   from the installed toolchain as well.

   The variant switches make it cheap to build a single variant — the XP
   build alone takes almost an hour, so iterating on the ARM64 build does
   not need the whole set.  They can also be set from the environment
   instead of editing the script::

       set BUILD_CLANG=0 && set BUILD_V145=0 && set BUILD_WIN7=0
       build_release.bat

   Only the variants that are switched on get a destination folder, a
   dependency check and a build slot.

   To find the current version strings of the pinned toolset::

       cl /?              (first line shows MSVC version, e.g. "19.50.xxxxx")

3. Run the script::

       cd /d c:\Github\AviSynthPlus\avisynth-build-win
       build_release.bat

   A tag may be given on the command line instead of editing the script::

       build_release.bat beta

   The script will:

   a. Create ``DEST_ROOT\AVS_VER_DATE_TAG_rNUM\`` with all subfolders
      (``DEST_ROOT`` is ``avisynth-build-win\current-dev-release\``).
   b. Copy license files and ``readme_history.txt`` to the build root.
   c. Build each of the nine configurations into its own isolated build
      directory (``bld_xp_x86\``, ``bld_clang_x64\``, ``bld_lm_arm64``, etc.).
   d. Copy freshly built DLLs and copy-only DLLs from origin into each
      variant subfolder.
   e. Copy the shared ``include\`` headers (identical across all configs).
   f. Create a ``Avisynth_VERSION.7z`` archive inside the build folder.

4. The ``bld_*`` directories are left in place for post-build inspection.
   They are deleted and recreated at the start of the next run.


Build identification
--------------------

The version number, the build date and the revision number that make up the
name of the release folder are detected by the script, so they cannot go
stale:

.. code-block:: bat

   set BUILD_NAME=%AVS_VER%_%BUILD_DATE%_%BUILD_TAG%_r%BUILD_RNUM%

=========================  =====================================================
Value                      Source
=========================  =====================================================
``AVS_VER``                ``avs_core\core\version.h.in`` — the same file the
                           top-level ``CMakeLists.txt`` reads for
                           ``AVS_MAJOR_VER``/``AVS_MINOR_VER``/
                           ``AVS_BUGFIX_VER``, so the folder name can never
                           disagree with the version reported by the DLLs
``BUILD_DATE``             the day the build runs (YYYYMMDD, local time)
``BUILD_RNUM``             ``git rev-list --count HEAD`` — the same revision
                           counter the built DLLs report as ``AVS_SEQREV``
``COMMIT_DATE``            the day of the last commit (informational)
``GIT_BRANCH``             the branch, and ``GIT_DESCRIBE`` the tag
                           description (informational)
=========================  =====================================================

The detection is done by ``avisynth-build-win\build_info.cmake``, which writes
``avisynth-build-win\build_info.cmd`` — a few plain ``set`` commands that the
batch script calls.  It needs CMake and git; if either is missing the script
stops with an error instead of producing a folder with a wrong version.

Any of the values can be overridden from the environment, which is useful for
a one-off build with a different label::

    set BUILD_TAG=beta
    build_release.bat

or, for the version and the revision number::

    set AVS_VER=3.7.6
    set BUILD_RNUM=4900
    build_release.bat


What each configuration builds
-------------------------------

v141_xp  Win32 / x64
~~~~~~~~~~~~~~~~~~~~~

* Toolset: ``v141_xp`` — the last toolset to support Windows XP and Vista.
  It is only shipped with **Visual Studio 2022** (as an optional component);
  VS 2026 dropped it, so these two variants are configured with the VS 2022
  generator (``-G "Visual Studio 17 2022" -T v141_xp``) and need a VS 2022
  installation to be present.
* ``WINXP_SUPPORT=on`` — adds ``/Zc:threadSafeInit-`` and restricts the C++
  runtime to XP-compatible API calls.
* 32-bit uses ``MSVC_CPU_ARCH=SSE2`` (Pentium 4 / Athlon 64 baseline; matches
  the CMakeLists.txt default for 32-bit builds).
* Plugins built: ConvertStacked, ImageSeq, Shibatch, TimeStretch, VDubFilter.
* Also copied from origin: DirectShowSource, ``imageseq-xp-legacy\``.

ClangCL  Win32 / x64
~~~~~~~~~~~~~~~~~~~~~

* Toolset: ``ClangCL`` — Clang front-end with MSVC ABI, included with
  VS 2022/2026.
* No XP support (Clang dropped Windows XP support in 2016).
* Plugins built: ConvertStacked, ImageSeq, Shibatch, TimeStretch, VDubFilter.
* Also copied from origin: DirectShowSource.

v145  Win32 / x64
~~~~~~~~~~~~~~~~~~

* Toolset: ``v145`` — the native VS 2026 MSVC toolset (MSVC 19.50+).
* Standard modern build; no backwards-compatibility flags.
* Plugins built: ConvertStacked, ImageSeq, Shibatch, TimeStretch, VDubFilter.
* Also copied from origin: DirectShowSource.

v143  x64 / Win32  (Windows 7 compatibility)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Toolset: ``v143`` — the VS 2022 toolset (MSVC 19.44), invoked from VS 2026
  via the ``-T v143`` switch.
* **Why v143 and not v145?**  The v145 toolset (MSVC 19.50+) links the CRT
  against ``CreateFile2``, a kernel32 entry point introduced in Windows 8
  that does not exist in Windows 7.  Binaries built with v145 crash on
  Windows 7 at startup with a missing entry point error.  MSVC 19.44
  (the latest v143 revision) uses ``CreateFileW`` instead and runs correctly.
* Plugins built: same full set as the other modern builds — ConvertStacked,
  ImageSeq, Shibatch, TimeStretch, VDubFilter.
* Also copied from origin: DirectShowSource.

.. note::

   The ``x86-win7`` variant provides a 32-bit Win7-compatible build without
   requiring the older ``v141_xp`` toolset.  For users who only need Win7
   (not XP), ``x86-win7`` is the recommended 32-bit choice.  Users requiring
   XP support should use ``x86-xp`` instead.

MinGW  ARM64  (Windows on ARM64)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Toolset: **llvm-mingw** — https://github.com/mstorsjo/llvm-mingw.  The
  distribution is hosted on x86_64 and targets ``aarch64-w64-mingw32``, so the
  variant is cross-compiled on the build machine: neither ARM64 hardware nor
  ARM64 Windows is needed to produce it.  The build uses the Ninja generator
  (the Visual Studio generator cannot cross-compile with a different
  compiler), driven by ``avisynth-build-win\toolchains\llvm-mingw-aarch64.cmake``.

* **Why MinGW and not MSVC?**  The C++ ABI of an MSVC-built core and a
  MinGW-built plugin differ, so an ARM64 build that mixes the two toolchains
  would be broken.  The whole ARM64 variant is therefore one toolchain
  family.  llvm-mingw gives the MinGW-w64 ABI (the same calling conventions,
  name mangling and standard library behaviour as the x86_64 MinGW builds) and
  still compiles the NEON intrinsics of the aarch64 code paths natively.

* **Runtimes.**  The C runtime is the UCRT (``ucrtbase.dll`` plus the
  ``api-ms-win-crt-*`` forwarders), which is part of Windows 10/11 — nothing
  has to be shipped for it.  The C++ runtime is ``libc++`` and the unwinder
  is ``libunwind``; both are shared DLLs that are **not** part of Windows, so
  ``libc++.dll`` and ``libunwind.dll`` are copied next to ``AviSynth.dll`` and
  into ``plugins\``.

* Plugins built: ConvertStacked, Shibatch, and TimeStretch when the aarch64
  SoundTouch library is available.  **ImageSeq is not part of this variant**:
  the prebuilt DevIL SDK has no ARM64 binaries.

.. note::

   A second, *GNU GCC* based ARM64 variant is not built.  llvm-mingw also
   ships an ``aarch64-w64-mingw32-g++.exe``, but that is the very same clang
   in its GCC-compatible driver mode, so a build with it would produce the
   same code — building it twice would only cost time.  A real GNU GCC for
   ``aarch64-w64-mingw32`` has to be compiled from source (binutils + gcc +
   the mingw-w64 runtime), which is done in WSL on Linux; that toolchain can
   be added as an extra variant later by pointing a second toolchain file at
   it.


Options for an aarch64 build from an x86_64 Windows machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building for Windows on ARM64 from an x86_64 Windows machine means using a
*cross* toolchain: the compiler runs on x86_64 and produces aarch64 PE/COFF
binaries.  These are the ways to get there, and what each of them is worth:

#. **MSYS2's ``clangarm64`` environment** — the packages exist
   (``mingw-w64-clang-aarch64-*``), but every executable in that prefix is an
   **ARM64** program.  It is an environment to *run* on Windows on ARM64, not
   a cross compiler: on an x86_64 machine its ``clang.exe`` cannot be
   executed at all.  It becomes usable only when the build machine itself is
   an ARM64 Windows machine.

#. **A GNU GCC cross compiler for ``aarch64-w64-mingw32``** — does not exist
   as a ready-made Windows-hosted package.  mingw-w64 upstream supports
   aarch64 as a *target*, but nobody ships a prebuilt Windows-hosted
   cross-toolchain for it, and the Linux distributions only package the
   ``x86_64-w64-mingw32`` and ``i686-w64-mingw32`` flavours (the
   ``aarch64-linux-gnu`` cross compilers that they do ship target Linux, not
   Windows, so they have a different ABI and are useless here).

#. **Building such a GCC in WSL** — possible, but it means building
   binutils, gcc and the mingw-w64 runtime (crt, headers, libstdc++,
   libgcc) from source for the ``aarch64-w64-mingw32`` target: a multi-hour,
   fragile endeavour that has to be redone for every toolchain update.  It is
   the only way to get a *real* GNU GCC for this target, and it is the way it
   would be done "always on WSL" — but it is not worth it as long as
   llvm-mingw produces the required ABI.

#. **llvm-mingw** — `https://github.com/mstorsjo/llvm-mingw
   <https://github.com/mstorsjo/llvm-mingw>`_ — exists, and is what this
   script uses.  It is an x86_64-hosted distribution of clang/LLVM with the
   mingw-w64 runtime for the ``aarch64-w64-mingw32`` target.  The produced
   binaries are ordinary MinGW/GCC-ABI PE/COFF objects — the same calling
   conventions, name mangling and standard library behaviour as an x86_64
   MinGW build — so a plugin compiled this way is ABI-compatible with other
   MinGW-built plugins, and the NEON intrinsics are compiled natively by
   clang, exactly like GCC would do it.

   The distribution also contains ``aarch64-w64-mingw32-g++.exe`` and
   ``-gcc.exe``: these are **not** GNU GCC, they are the very same clang in
   its GCC-compatible *driver mode* (different default flags and driver
   behaviour, same compiler and same code generation).  They exist so that
   build systems expecting a ``gcc`` executable keep working.

#. **Cross-compiling from Linux / WSL2** — this is the route the project
   documents in :doc:`avsplus_external_deps_guide_manual_arm`
   ("Building AviSynth+'s external dependencies for Windows on ARM").  That
   document describes building llvm-mingw from source on a Linux distro
   (``./build-all.sh``) and using it to cross-compile the dependencies — and
   the core — for ``aarch64-w64-mingw32``.  WSL2 is the interesting case
   there, especially for running the task directly on a Windows on ARM
   device.  The compiler is the same one as in option 4, only built and run
   on a different host; the ABI is the same MinGW ABI.

So the short answer for "Windows + Intel host, want a Windows/aarch64 DLL with
the GCC ABI": use **llvm-mingw**.  It is the only readily available option,
and the GCC ABI is exactly what it produces.


Build directory isolation
--------------------------

Each configuration is configured and built in its own subdirectory::

    bld_xp_x86\
    bld_xp_x64\
    bld_clang_x86\
    bld_clang_x64\
    bld_v145_x86\
    bld_v145_x64\
    bld_win7_x64\
    bld_win7_x86\
    bld_lm_arm64\

Because CMake's ``Output\`` path is relative to ``CMAKE_BINARY_DIR``, each
build has an independent output tree (e.g. ``bld_v145_x64\Output\``) with no
risk of one configuration overwriting another's artefacts.  The ARM64 build is
the exception: it uses the Ninja generator, which has no ``Output\`` tree —
its artefacts are directly in ``bld_lm_arm64\avs_core\`` and
``bld_lm_arm64\plugins\<plugin>\``.


Troubleshooting
---------------

*"release-sample folder not found"*
    Create ``avisynth-build-win\release-sample\`` and populate it as described
    above.  A ``README.txt`` inside that folder lists the expected layout.

*CMake configure fails for v141_xp*
    The ``v141_xp`` toolset is not installed by default, and VS 2026 does
    not offer it at all any more.  Install it in **Visual Studio 2022**:
    go to **Individual components** and install
    "MSVC v141 - VS 2017 C++ build tools (v14.16)".  The XP variants are
    configured with the VS 2022 generator, so a VS 2022 installation has to
    be present; without it, set ``BUILD_XP=0``.

*CMake configure fails for ClangCL*
    Install "Clang compiler for Windows" from VS 2026 Individual components.

*"Visual Studio 18 2026" generator not found*
    CMake 4.2 or higher is required.  Download from https://cmake.org and
    update ``%PATH%``.

*v145 toolset not found*
    VS 2026 ships v145 as the default toolset; it should be present if VS 2026
    is installed.  Verify in the VS Installer that "MSVC v145 build tools" is
    checked.

*"llvm-mingw aarch64 toolchain not found"*
    The ARM64 variant needs the llvm-mingw UCRT build.  Download it from
    https://github.com/mstorsjo/llvm-mingw/releases and update the
    ``LLVM_MINGW_ROOT`` variable in ``build_release.bat``, or set
    ``BUILD_ARM64=0`` to skip the variant.

*"ninja.exe not found on PATH"*
    Only the ARM64 variant needs Ninja.  Install it (e.g.
    ``pip install ninja``) or set ``BUILD_ARM64=0``.

*"the build identification could not be determined"*
    The version/date/revision detection needs git and a git working copy of
    the repository.  Check that git is installed and that
    ``avisynth-build-win\build_info.cmake`` can be run by hand::

       cmake -D REPO=<repo> -D SRC=<repo>\avs_core\core\version.h.in ^
             -D OUT=<repo>\avisynth-build-win\build_info.cmd ^
             -P avisynth-build-win\build_info.cmake


Back to the :doc:`main page <../../index>`
