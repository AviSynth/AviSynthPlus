Visual Studio and CMake GUI: CMake Presets
===========================================

The repository root contains a ``CMakePresets.json``. It is used in two ways:

* **CMake GUI** (3.19 and later): pick a preset in the **Preset** drop-down,
  then Configure, Generate and **Open Project**. The Visual Studio presets
  produce a solution (``AviSynth+.slnx`` with VS 2026, ``AviSynth+.sln`` with
  VS 2022) in ``out\build\<preset>``. This replaces choosing generator,
  platform and toolset by hand in the Configure dialog.
* **Visual Studio Open Folder** (**File > Open > Folder...**; VS Code with the
  CMake Tools extension as well): the Ninja presets appear in the configuration
  drop-down of the toolbar. This is the CMake mode of the IDE, not a solution.
  It is the only way to build the Windows on ARM64 variant from the IDE: the
  Visual Studio generator can drive the MSVC toolsets only, it cannot
  cross-compile with llvm-mingw.

.. warning::

   Do not configure into the source folder itself (CMake GUI "Where to build
   the binaries" = the repository root). An in-source ``CMakeCache.txt`` in the
   root breaks Open Folder: Visual Studio passes the root to CMake, which then
   takes it for an existing build tree and fails with *"generator: Ninja does
   not match the generator used previously: Visual Studio 18 2026"*. Fix: delete
   ``CMakeCache.txt`` and ``CMakeFiles\`` in the root. With presets every
   variant has its own folder under ``out\build``, so this does not happen.

Related documents:

* :doc:`compiling_avsplus` — the general build guide.
* :doc:`release_build_automation` — ``build_release.bat``, all nine release
  variants in one run. The presets use the same dependency paths and the same
  ARM64 options.
* :doc:`avsplus_external_deps_guide_manual` — building DevIL and SoundTouch.
* :doc:`avsplus_external_deps_guide_manual_arm` — llvm-mingw and the aarch64
  SoundTouch build.

.. contents:: Table of contents
   :depth: 2


The presets
-----------

Visual Studio generator presets (a solution; CMake GUI, or ``cmake --preset``):

+---------------------------+----------------------------------------------------+
| Preset                    | Generator, platform, toolset                       |
+===========================+====================================================+
| ``vs-v145-x64``           | VS 18 2026, x64 / Win32, ``v145``                  |
| ``vs-v145-win32``         |                                                    |
+---------------------------+----------------------------------------------------+
| ``vs-v143-x64``           | VS 18 2026, x64 / Win32, ``v143`` (Windows 7       |
| ``vs-v143-win32``         | compatible runtime)                                |
+---------------------------+----------------------------------------------------+
| ``vs-clangcl-x64``        | VS 18 2026, x64 / Win32, ``ClangCL``               |
| ``vs-clangcl-win32``      |                                                    |
+---------------------------+----------------------------------------------------+
| ``vs-intel-x64``          | VS 18 2026, x64, ``Intel C++ Compiler 2025``       |
|                           | (icx). Needs the oneAPI environment, see below.    |
+---------------------------+----------------------------------------------------+
| ``vs2022-v141xp-x64``     | VS 17 2022, x64 / Win32, ``v141_xp``,              |
| ``vs2022-v141xp-win32``   | ``WINXP_SUPPORT=ON``, Win32 also                   |
|                           | ``MSVC_CPU_ARCH=SSE2``. VS 2026 has no ``v141_xp``.|
+---------------------------+----------------------------------------------------+

Ninja presets (VS Open Folder, or the command line):

+------------------------------+------------------------------------------------+
| Preset                       | What it builds                                 |
+==============================+================================================+
| ``x64-msvc-debug``           | x64, MSVC (``cl.exe`` of the Visual Studio     |
| ``x64-msvc-release``         | that opened the folder), Ninja.                |
+------------------------------+------------------------------------------------+
| ``arm64-llvm-mingw-debug``   | Windows on ARM64 (aarch64), cross-compiled     |
| ``arm64-llvm-mingw-release`` | with llvm-mingw: MinGW ABI, UCRT. The Debug    |
|                              | preset also writes ``.pdb`` files (see below). |
+------------------------------+------------------------------------------------+

Linux, macOS and BSD: ``posix-debug`` and ``posix-release`` (default generator
and compiler, default options, like a plain ``cmake -S . -B build``). They are
there for IDEs which switch to presets when a ``CMakePresets.json`` exists
(e.g. VS Code CMake Tools); the Windows presets are hidden on these hosts and
vice versa. Command line builds without ``--preset`` do not read the file at
all, see :doc:`posix`.

Output goes to ``out\build\<preset>`` (installs to ``out\install\<preset>``),
which is ignored by git and is separate from ``avisynth-build`` and the
``build_release.bat`` folders.

All presets use the options of the matching ``build_release.bat`` variant
(plugins on, DirectShowSource off, DevIL and SoundTouch paths). The ARM64
presets: no Intel SIMD, no ImageSeq and VDubFilter, TimeStretch with the
aarch64 SoundTouch library.


CMake GUI
---------

1. Start ``cmake-gui``. For the Intel preset start it from the oneAPI
   environment (``setvars-vcvarsall.bat`` of oneAPI first, then
   ``cmake-gui.exe``), otherwise configure fails with
   ``CMAKE_LINKER-NOTFOUND``.
2. "Where is the source code": the repository root.
3. **Preset**: choose a variant. "Where to build the binaries" is set by the
   preset (``out\build\<preset>``), leave it.
4. **File > Delete Cache** for a clean configure (it deletes the cache of that
   preset's folder only), then **Configure** and **Generate**.
5. **Open Project** opens the solution of that variant in Visual Studio.

Each variant keeps its own folder and solution, several can exist side by
side; switching variants is choosing another preset.


Prerequisites
-------------

* Visual Studio 2022 or 2026 with the **C++ CMake tools for Windows**
  component (it brings CMake and Ninja).
* The dependencies at the paths used by ``build_release.bat``:

  * x64: ``C:\avsplus_build_deps\DevIL Windows SDK`` and
    ``C:\avsplus_build_deps\v143-x86-64`` (SoundTouch).
  * ARM64: ``C:\avsplus_build_deps\llvm-mingw-20261020-ucrt-x86_64`` and
    ``C:\avsplus_build_deps\mingw-arm64`` (aarch64 SoundTouch).

Different paths: do not edit ``CMakePresets.json``, put a
``CMakeUserPresets.json`` (it is not tracked by git) next to it, which inherits a
preset and overrides the cache variables:

.. code-block:: json

    {
      "version": 3,
      "configurePresets": [
        {
          "name": "arm64-mine",
          "displayName": "ARM64 llvm-mingw Release (my paths)",
          "inherits": "arm64-llvm-mingw-release",
          "cacheVariables": {
            "AVS_LLVM_MINGW_ROOT": "D:/tools/llvm-mingw-ucrt-x86_64",
            "SOUNDTOUCH_LINK_LIBRARIES": "D:/deps/arm64/lib/libSoundTouch.a",
            "SOUNDTOUCH_INCLUDE_DIRS": "D:/deps/arm64/include/soundtouch"
          }
        }
      ]
    }

Without the aarch64 SoundTouch, set ``"BUILD_TIMESTRETCH": "OFF"`` there.


Command line
------------

The same presets work without the IDE. The Visual Studio generator presets
need no special prompt (except ``vs-intel-x64``: oneAPI environment). For the
x64 MSVC Ninja ones run them from a *x64 Native Tools Command Prompt* (they use
the compiler found in that environment); the ARM64 ones need only
``ninja.exe`` on ``PATH``:

.. code-block:: bat

    cmake --preset arm64-llvm-mingw-release
    cmake --build --preset arm64-llvm-mingw-release

``cmake --list-presets`` lists them.


What works for ARM64 in the IDE, and what does not
---------------------------------------------------

* Works: configure, build, the error list, IntelliSense with the aarch64
  view of the sources (``NEON_INTRINSICS`` defined, ``arm_neon.h``, the
  ``filters\aarch64`` files active).
* Does not work on an x64 PC: running and debugging. ARM64 binaries do not run
  on x64 Windows (only the opposite direction is emulated).
* Debugging on an ARM64 Windows device: install the *Remote Tools for Visual
  Studio* (``msvsmon.exe``) there and attach remotely. The Visual Studio
  debugger needs PDB files; llvm-mingw writes DWARF by default, so the Debug
  preset adds ``-gcodeview`` (CodeView debug info) and ``-Wl,--pdb=``
  (the linker writes ``AviSynth.pdb`` etc. next to the DLLs).

.. note::

   The root ``CMakeLists.txt`` sets ``CMAKE_SHARED_LINKER_FLAGS`` for MinGW
   builds, replacing any value given on the command line. That is why the
   Debug preset passes the PDB option through
   ``CMAKE_SHARED_LINKER_FLAGS_DEBUG``.

.. note::

   The ARM64 DLLs import ``libc++.dll`` and ``libunwind.dll``, llvm-mingw's
   C++ runtime, which is not part of Windows. To use a preset build on a
   device, copy both from ``<llvm-mingw>\aarch64-w64-mingw32\bin`` next to
   ``AviSynth.dll`` and into the plugin folder (``build_release.bat`` does
   this for the release variant).
