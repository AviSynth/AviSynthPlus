Full Windows Release Build (make_release.bat)
==============================================

``avisynth-build-win\make_release.bat`` builds a complete Windows release of AviSynth+ in one run, from
source and with all third-party dependencies built from pinned sources as well: the files-only pack,
the installers, the PDB files and a SHA-256 list. Its output follows the official 3.7.6 release (whose
packs were assembled by hand), with four changes: the 3.7.7 SDK additions, a separate Windows 7 build
(``_win7``, the 14.44 toolset that 3.7.6 used for its main build), the default ``x86``/``x64`` build
now made with the current toolset for Windows 10 and newer, and the XP ImageSeq with DevIL linked in
statically like the others (no separate ``DevIL.dll``).

It exists so that a release can be made again quickly when a fix is needed in a dependency rather than
in AviSynth+ itself. Example: `issue #532 <https://github.com/AviSynth/AviSynthPlus/issues/532>`_, where
the x64 ``ImageSeq.dll`` of 3.7.6 crashed with an illegal instruction on CPUs with AVX-VNNI but without
AVX-512, because of a zlib-ng bug (`zlib-ng #2485 <https://github.com/zlib-ng/zlib-ng/issues/2485>`_)
in the statically linked image libraries. With this script such a fix is a pin change and a re-run.

Related documents:

* :doc:`release_build_automation` — ``build_release.bat``, the dev/test build (nine variants incl.
  ClangCL, prebuilt DevIL SDK). It is independent of this script and unchanged by it.
* :doc:`avsplus_external_deps_guide_manual` — the manual build of DevIL and its dependencies, the recipe
  that ``helpers\makedevil.ps1`` automates.
* :doc:`avsplus_external_deps_guide_manual_arm` — the same for Windows on ARM (llvm-mingw).
* :doc:`compiling_avsplus` — the general build guide.

.. contents:: Table of contents
   :depth: 2


What it produces
----------------

Which download for which Windows (the names users see; the toolsets are internal):

+------------------------------+-------------------------+--------------------------+--------------------+
| Windows                      | Installer               | Files-only folders       | Built with         |
+==============================+=========================+==========================+====================+
| 10 / 11 (most users)         | ``<name>.exe``          | ``x86\``, ``x64\``       | v145, latest VC++  |
|                              |                         |                          | runtime            |
+------------------------------+-------------------------+--------------------------+--------------------+
| 7 SP1, 8, 8.1                | ``<name>_win7.exe``     | ``x86-win7\``,           | v143 (14.44), the  |
|                              |                         | ``x64-win7\``            | 14.44 runtime      |
+------------------------------+-------------------------+--------------------------+--------------------+
| XP SP3, Vista                | ``<name>_xp.exe``       | ``x86-xp\``, ``x64-xp\`` | v141_xp, the 14.27 |
|                              |                         |                          | runtime            |
+------------------------------+-------------------------+--------------------------+--------------------+
| 10 / 11 on ARM64             | ``<name>_arm64.exe``    | ``arm64\``               | llvm-mingw, UCRT   |
+------------------------------+-------------------------+--------------------------+--------------------+

Each x86/x64 installer installs both the 32-bit and the 64-bit AviSynth+ (see `Installers`_).

Everything goes to ``avisynth-build-win\release-out\``; the name is
``AviSynthPlus_<version>_<date>[_<tag>]``, the version from ``avs_core\core\version.h.in`` and the date
from the build (``build_info.cmake``, the same as ``build_release.bat``).

+-----------------------------------+-------------------------------------------------------------------+
| Asset                             | Contents                                                          |
+===================================+===================================================================+
| ``<name>.exe``                    | Installer, x86 + x64, v145 (current MSVC), Windows 10 and newer,  |
|                                   | the latest VC++ redistributable; the default for most users       |
+-----------------------------------+-------------------------------------------------------------------+
| ``<name>_win7.exe``               | Installer, x86 + x64, v143 (MSVC 14.44), Windows 7 SP1 and 8.x,   |
|                                   | with the 14.44 VC++ redistributable (the last one for Windows 7)  |
+-----------------------------------+-------------------------------------------------------------------+
| ``<name>_xp.exe``                 | Installer, x86 + x64, v141_xp, Windows XP SP3 and newer, with the |
|                                   | 14.27 VC++ redistributable (the last one for XP)                  |
+-----------------------------------+-------------------------------------------------------------------+
| ``<name>_arm64.exe``              | Installer, Windows on ARM64 (llvm-mingw), Windows 10 1709 and     |
|                                   | newer, no redistributable needed (UCRT, static C++ runtime)       |
+-----------------------------------+-------------------------------------------------------------------+
| ``<name>-filesonly.7z``           | All variants without installer, see the layout below              |
+-----------------------------------+-------------------------------------------------------------------+
| ``<name>-pdb.7z``                 | The PDB files of all DLLs, same layout as the binaries            |
+-----------------------------------+-------------------------------------------------------------------+
| ``<name>-SHA256SUMS.txt``         | SHA-256 of the assets above, for the release notes                |
+-----------------------------------+-------------------------------------------------------------------+

The files-only pack::

    <name>-filesonly\
      gpl.txt  lgpl_for_used_libs.txt  OFL.TXT
      html\                  the HTML documentation (sphinx, incl. _sources\ with the .rst files)
      sdk\c_loader\          the C API loader samples (distrib\FilterSDK\c_loader)
      x86\  x64\             Windows 10+: AviSynth.dll, c_api\ (AviSynth.lib, .exp), include\, plugins\,
                             system\ (empty)
      x86-win7\  x64-win7\   Windows 7 SP1 / 8.x: the same
      x86-xp\  x64-xp\       Windows XP: the same (3.7.6 had system\DevIL.dll here, now static)
      arm64\                 the cmake --install tree: bin\AviSynth.dll, lib\AviSynth.dll.a,
                             lib\avisynth\*.dll, lib\pkgconfig\avisynth.pc, include\avisynth\,
                             share\doc\avisynth\examples\c_loader\

Plugins: ConvertStacked, DirectShowSource, ImageSeq, Shibatch, TimeStretch, VDubFilter (x86/x64);
ConvertStacked, ImageSeq, Shibatch, TimeStretch (ARM64).

Users see the Windows version in the names, not the toolset. Inside the script the variants are named by
toolset (``-Variants``, build folders, log):

+-----------+-----------------------------+--------------+---------------------------------------------+
| Variant   | Toolset                     | Min. Windows | ImageSeq                                    |
+===========+=============================+==============+=============================================+
| x86-v145, | v145, VS 2026               | 10           | DevIL and its libraries linked in statically|
| x64-v145  | (folders ``x86``, ``x64``)  |              | (the v143 libraries link into v145)         |
+-----------+-----------------------------+--------------+---------------------------------------------+
| x86-v143, | v143 (14.44), VS 2026       | 7 SP1        | the same                                    |
| x64-v143  | generator ``-T v143``       |              |                                             |
|           | (``x86-win7``, ``x64-win7``)|              |                                             |
+-----------+-----------------------------+--------------+---------------------------------------------+
| x86-xp,   | v141_xp, VS 2022 generator  | XP SP3       | static chain built with v141_xp (3.7.6:     |
| x64-xp    |                             |              | ``DevIL.dll`` of the DevIL SDK 1.8.0)       |
+-----------+-----------------------------+--------------+---------------------------------------------+
| arm64     | llvm-mingw (UCRT), Ninja    | 10 on ARM    | static chain, built with llvm-mingw         |
+-----------+-----------------------------+--------------+---------------------------------------------+
| x86-      | ClangCL (clang-cl, VS 2026) | 10           | as v145                                     |
| clangcl,  | only with ``-ClangCL``      |              |                                             |
| x64-      |                             |              |                                             |
| clangcl   |                             |              |                                             |
+-----------+-----------------------------+--------------+---------------------------------------------+

**Two modern builds** (``-ClangCL``): clang-cl then builds the default Windows 10+ release (``x86\``,
``x64\``, ``<name>.exe``), and the MSVC v145 build is shipped beside it under an ``msvc`` suffix:
folders ``x86-msvc\``, ``x64-msvc\``, installer ``<name>_msvc.exe`` (same ``MinVersion`` 10.0 and
redistributable). Without ``-ClangCL`` MSVC v145 is the default, as in the tables above. Both use the
MSVC runtime and the same v143 static libraries; clang-cl runs inside the VS 2026 generator
(``-T ClangCL``).

The v145 toolset and its runtime need Windows 10: the 14.50+ STL calls Windows 8+ functions. 14.44 (v143)
is the last toolset and runtime for Windows 7 (see ``distrib\Prerequisites\keep.me``).


Prerequisites
-------------

On the release machine:

* Visual Studio 2026 with the MSVC v143 and v145 build tools (and ClangCL, if used); Visual Studio 2022
  with "MSVC v141 - VS 2017 C++ build tools" (the v141_xp toolset; VS 2026 does not offer it).
* CMake 4.x, Ninja, git, curl (Windows 10/11 has it), 7-Zip (``C:\Program Files\7-Zip``).
* llvm-mingw, UCRT flavour, in ``C:\avsplus_build_deps\llvm-mingw-<tag>-ucrt-x86_64`` (see
  :doc:`release_build_automation`).
* Python with Sphinx (``pip install sphinx``), for the HTML documentation.
* Inno Setup 5 (5.5.9 or 5.6.1 unicode, 5.6.1 is the last version that supports XP) and Inno Setup 6
  (6.7.x), installed side by side in their default folders. See `Installers`_.
* The Windows SDK 7.1 samples (``C:\Program Files\Microsoft SDKs\Windows\v7.1\Samples\multimedia\
  directshow\baseclasses``): the source of the DirectShow base classes for DirectShowSource.
* The VC++ redistributables in ``distrib\Prerequisites`` (see ``keep.me`` there): the frozen
  ``VC_redist_14.44.35211.0.x86/x64.exe`` (Windows 7 installer) and ``VC_redist_14.27.29114.0.x86/x64.exe``
  (XP installer). The rolling ``VC_redist.x86/x64/arm64.exe`` (default installer) is downloaded by the
  installer stage before every release.
* Internet access for the first run (the dependency sources are cloned and downloaded).

All paths are set in the CONFIGURATION block at the top of ``make_release.ps1``.


Running it
----------

From a plain Command Prompt (no Developer Prompt needed)::

    cd /d c:\Github\AviSynthPlus\avisynth-build-win
    make_release.bat                                all default stages: deps build check docs pack
    make_release.bat -Stages installer              only the installers (after a build)
    make_release.bat -Stages build,check -Variants x64-v145,arm64
    make_release.bat -Tag test                      name suffix, e.g. AviSynthPlus_3.7.7_20261122_test
    make_release.bat -ClangCL                       also build x86/x64 with clang-cl (the default then,
                                                    MSVC v145 becomes _msvc)

``make_release.bat`` calls ``make_release.ps1`` (PowerShell) next to it, which does the work. Options
can also be given in the environment:

* ``DEPS_BUILD=auto`` (default): a dependency is rebuilt only when its prefix is missing or was built
  from other pinned sources; ``all``: everything from scratch; ``off``: use the prefixes as they are.

A full run (all variants, docs, pack, installers, without the dependencies) takes about 15 minutes.
A log of every command is written to ``avisynth-build-win\release-build\make_release_<time>.log``; the
build folders are ``release-build\bld_rel_<variant>``. Neither touches the folders of
``build_release.bat``.

Stages, in this order:

+-------------+----------------------------------------------------------------------------------------+
| Stage       | What it does                                                                           |
+=============+========================================================================================+
| deps        | Builds the missing or outdated dependencies (see `Dependencies`_)                      |
+-------------+----------------------------------------------------------------------------------------+
| build       | Configures and builds the variants, each in its own folder                             |
+-------------+----------------------------------------------------------------------------------------+
| check       | Binary checks, see `Checks`_; a failure stops the release                              |
+-------------+----------------------------------------------------------------------------------------+
| docs        | ``python -m sphinx`` into ``distrib\docs\english\build\html``                          |
+-------------+----------------------------------------------------------------------------------------+
| pack        | The files-only tree, the PDB tree, both as ``.7z``, and the SHA-256 list               |
+-------------+----------------------------------------------------------------------------------------+
| installer   | Stages the binaries as the Inno Setup scripts expect them and compiles the installers  |
+-------------+----------------------------------------------------------------------------------------+

See `Version bump and tagging`_ for the order of a real release.


Version bump and tagging
------------------------

Where the version number lives. Make sure every place says the new version (e.g. 3.7.6 -> 3.7.7)
before the release commit:

+------------------------------------------------+-----------------------------------------------------+
| File                                           | What                                                |
+================================================+=====================================================+
| ``avs_core/core/version.h.in``                 | ``AVS_BUGFIX_VER 6`` -> ``7`` (or MAJOR/MINOR). The |
|                                                | only place in the code: CMake reads it, so the DLL  |
|                                                | version resource, ``avs/version.h`` and the asset   |
|                                                | names follow                                        |
+------------------------------------------------+-----------------------------------------------------+
| ``distrib/Readme/readme_history.txt``          | the head entry ``<date> 3.7.6.rXXXX (pre 3.7.7)``   |
|                                                | becomes ``<date> 3.7.7.r<number>``                  |
+------------------------------------------------+-----------------------------------------------------+
| ``.../avisynthdoc/changelist377.rst``          | review against what ships; linked from              |
|                                                | ``overview.rst``                                    |
+------------------------------------------------+-----------------------------------------------------+
| ``distrib/FilterSDK/c_loader/``                | the interface version list and the                  |
| ``avisynth_c_loader.h`` / ``.hpp``             | ``has_at_least_*`` flags: a new interface version   |
|                                                | is marked ``WIP`` until its release; remove it then |
+------------------------------------------------+-----------------------------------------------------+
| ``avs_core/include/avisynth.h``,               | the interface version comments name the release     |
| ``avisynth_c.h``                               | (``V14.1``: 3.7.7)                                  |
+------------------------------------------------+-----------------------------------------------------+
| ``distrib/docs/english/source/conf.py``        | ``version`` / ``release`` (optional, now ``'3.7'``) |
+------------------------------------------------+-----------------------------------------------------+

The git tag is used at build time as well, not only on GitHub:

* the installers take their version from git (``update_git_rev.ps1``, run by the ``.iss`` scripts):
  ``Version`` from the nearest tag, ``IsRelease`` only when HEAD itself is tagged. Without the new tag an
  installer calls itself "AviSynth+ 3.7.6 r5014" (the previous version plus the revision);
* the DLL's version string (``Version()``, ``AVS_FULLVERSION``) comes from ``avs/version.h``, generated
  from ``git describe``: ``AVS_DEV_GITHASH`` (``v3.7.7`` on the tag, ``v3.7.6-105-g544dc4c1`` without),
  ``AVS_DEVNEXT_REV`` (commits after the tag, 0 on it), ``AVS_SEQREV`` (revision), ``AVS_BRANCH``
  (the current branch: the 3.7.6 DLLs say ``3.7``, they were built on a ``3.7`` release branch);
* the other platforms (Linux packages, macOS, distributions) build from the pushed tag: it must point
  to exactly the commit the Windows assets were built from.

So the tag goes on before the final build, locally, and is pushed only after the tests:

1. **Prepare**: step the version in the files above, commit. Check that the GitHub CI build of that
   commit (Windows, Ubuntu, macOS) passes.
2. **Dry run, no tag yet**: ``make_release.bat -Tag rc``. Tests the build, the checks, the packs and the
   installers; only the version texts still name the previous release.
3. **Tag locally**: ``git tag -a v3.7.7 -m "AviSynth+ 3.7.7"`` on that commit, not pushed. Optionally
   build on a release branch (``git switch -c 3.7``) to get ``AVS_BRANCH 3.7`` as in 3.7.6.
4. **Final build**: ``make_release.bat`` (no ``-Tag``), then install the installers on Windows 10/11,
   Windows 7 SP1, XP and Windows on Arm (virtual machines).
5. **A fix needed?** ``git tag -d v3.7.7``, fix, commit, tag again, build again. Harmless while the tag
   is local.
6. **Publish**: push the commit and the tag, create the GitHub release with the assets and the SHA-256
   list, tell those who build the other platforms that the tag is out.

Never move a pushed tag: packagers may have fetched it already, and GitHub's source archives of the tag
would change. A problem found after the push gets a new version (3.7.7.1 or 3.7.8).


Dependencies
------------

The ``deps`` stage calls three helper scripts in ``avisynth-build-win\helpers``; each can also be run
alone. Sources go to ``C:\avsplus-build-deps`` (hyphen), the built libraries to
``C:\avsplus_build_deps`` (underscore), as for SoundTouch before.

``makestrmbase.bat``
    The DirectShow base classes (``strmbase.lib``) for DirectShowSource, built from the Windows SDK 7.1
    sample project with the v141_xp toolset (Visual Studio 2022 MSBuild), into
    ``C:\avsplus_build_deps\strmbase\{x86,x64,include}``. One v141_xp build serves the XP and the v143
    DirectShowSource: a static library of an older MSVC toolset links into a newer one.

``makesoundtouch.bat``
    SoundTouch for TimeStretch, unchanged (see :doc:`release_build_automation`).

``makedevil.ps1 -Target msvc-x64 | msvc-x86 | mingw-arm64``
    DevIL and the libraries it reads images with, static, from pinned sources: zlib-ng, xz (lzma),
    libjpeg-turbo, libpng, jbigkit, libdeflate, lerc, zstd, libtiff, libsquish, JasPer, DevIL. This is
    the set of the 3.7.6 ImageSeq (OpenEXR, lcms2, libwebp and libmng are not part of it). The MSVC
    chains are built with v143 into ``C:\avsplus_build_deps\devil-msvc-<arch>-v143`` (``-Toolset
    v141_xp``: ``devil-msvc-<arch>-v141_xp``, for XP), the ARM64 chain with llvm-mingw into
    ``C:\avsplus_build_deps\devil-mingw-arm64``. A ``deps_stamp.txt`` in each prefix
    records the commits it was built from; ``DEPS_BUILD=auto`` compares it with the pins.

    The pins (repository + tag or commit) are at the top of the script. zlib-ng is pinned to a commit of
    its ``develop`` branch that contains the fix of zlib-ng #2485 (no zlib-ng release had it yet).

    What the script changes compared with the manual recipe, per target (XP and the others), why, and
    what 3.7.6 did: see `DevIL chain changes`_.

    ImageSeq is linked with all these libraries (``IL_LIBRARIES``, plus ``-DIL_STATIC_LIB``), the list
    is in ``make_release.ps1``.


DevIL chain changes
-------------------

What ``helpers\makedevil.ps1`` does differently from the manual recipe
(:doc:`avsplus_external_deps_guide_manual`), per target. "XP" is ``-Toolset v141_xp`` (the VS 2017
compiler); "others" are v143 (linked by the v143, v145 and ClangCL builds) and llvm-mingw (ARM64).

.. list-table::
   :header-rows: 1
   :widths: 14 17 17 26 26

   * - What
     - XP
     - Others
     - Why
     - Caveat / in 3.7.6
   * - How ImageSeq gets DevIL
     - static, chain built with ``v141_xp``
     - static, chain built with v143 / llvm-mingw
     - one way for all variants, current library versions everywhere
     - 3.7.6 XP: ``DevIL.dll`` of the DevIL Windows SDK 1.8.0 (2017, built for Vista, image libraries
       of 2016) in ``system\``, installed to ``{sys}``. Now no separate DLL
   * - zlib-ng
     - pinned ``develop`` commit
     - the same
     - contains the fix of zlib-ng #2485: 3.7.6's x64 ImageSeq crashed on CPUs with AVX-VNNI but no
       AVX-512 (issue #532)
     - not a zlib-ng release (none had the fix when pinned): move the pin to a release when one has it.
       The 3.7.6 XP ImageSeq (SDK DLL, no zlib-ng) was not affected
   * - libdeflate
     - 1.25
     - 1.26
     - 1.26 requires C11, i.e. Visual Studio 2019 16.8 (``#error`` with ``v141_xp``)
     - the XP build stays one version behind; check libdeflate's release notes when updating the
       pin. qyot27's recipe used libdeflate HEAD of its time (before 1.26)
   * - JasPer memory size query
     - ``GetPhysicallyInstalledSystemMemory`` replaced with ``GlobalMemoryStatusEx``, include of
       ``sysinfoapi.h`` removed
     - unchanged
     - the call exists from Vista SP1 on, the header is missing in the XP SDK
     - the guide's kludge comments the code out: memory size 0, an extra "exceeds system memory
       size" warning on stderr for every jp2 read/write. The patch is a text replacement, the script
       stops if ``jas_malloc.c`` changes so that it no longer applies
   * - JasPer threading
     - off
     - off
     - the XP SDK has no thread library for it; not needed by DevIL
     - as in the guide
   * - JasPer C version probe
     - —
     - ARM64: ``JAS_STDC_VERSION=201710L``
     - a cross build cannot run JasPer's probe program
     - —
   * - zstd multithreading
     - off (``ZSTD_MULTITHREAD_SUPPORT=OFF``)
     - on (zstd default)
     - uses Windows condition variables (Vista+)
     - silent: builds without a message, but ImageSeq then does not load on XP ("entry point not
       found"). Not in the guide. The check stage now rejects Vista+ imports in XP DLLs. libtiff uses
       zstd single-threaded anyway
   * - C runtime of the libraries
     - ``/MD``; additionally ``CMAKE_MSVC_RUNTIME_LIBRARY`` with policy CMP0091 ``NEW``
     - ``/MD`` (libjpeg-turbo ``WITH_CRT_DLL=ON``, libdeflate Release flags with ``/MD``)
     - ImageSeq uses the DLL runtime; a static-runtime library mixes two C runtimes in one DLL
       (LNK4098). With ``v141_xp`` projects requiring an old CMake (libdeflate 1.25) silently got
       the static runtime
     - a check stops the script if any library refers to ``LIBCMT``
   * - libtiff CMake package
     - removed after install
     - removed after install
     - it does not import its dependencies (``Deflate::Deflate`` etc.); DevIL then uses the plain
       ``tiff.lib``
     - —
   * - xz tests
     - built
     - ARM64: off
     - their resource file does not compile with llvm-rc
     - —
   * - DevIL source edits
     - ILUT removed, ILU static, qyot27's JasPer patch, ILU error headers converted to UTF-8
     - the same
     - from the guide; UTF-8: the headers are ISO-8859-1, which newer compilers warn about
     - —
   * - JasPer messages on stderr
     - deprecation of ``jas_init``, default memory limit
     - the same
     - DevIL calls JasPer's deprecated ``jas_init`` without setting a memory limit
     - every jp2 read/write in all builds; harmless, visible only in console hosts. A fix belongs in
       DevIL's JasPer code (not done)


Checks
------

The ``check`` stage stops the release on any of these:

* a variant without all its plugins;
* x86/x64 DLLs: the Windows version in the PE header is not 5.01/5.02 (XP) or 6.00 (main);
* ARM64 DLLs importing ``libc++.dll``, ``libunwind.dll`` or other MinGW runtime DLLs (they must import
  only Windows and the UCRT);
* a DLL whose PDB reference (``llvm-readobj --coff-debug-directory``) is not exactly ``<dll name>.pdb``:
  missing, with a folder of the build machine, or with an escaped ``%_PDB%`` (see `PDB files`_);
* an XP DLL importing a function that exists only from Vista on (condition variables, SRW locks,
  ``GetTickCount64``, ``Fls*`` and the like; a list of the usual suspects in ``make_release.ps1``).
  Such a DLL builds without any message but does not load on XP; zstd's multithreading was such a case.

The pack stage logs the compression methods of each ``.7z`` and the oldest 7-Zip that extracts it, for
the release notes. 7-Zip 23.00 and newer apply an ARM64 filter (method ``0A``) to ARM64 DLLs on their own;
7-Zip before 23.00 (and older 7z-based tools) then report "Unsupported method" for the ``arm64\`` files
(the 3.7.6 pack was like this). This is not a Windows version limit: the current 7-Zip still runs on
Windows XP and 2000, so XP users only need a current 7-Zip. Packed with 7-Zip 22 or older, the pack
uses only LZMA2, LZMA and BCJ2 (7-Zip 9.20 and newer). ``-mf=off`` would avoid the ARM64 filter, but
also turns off BCJ2 for the x86/x64 DLLs (a bigger pack).

For a bug like #532, ``dumpbin /disasm ImageSeq.dll`` shows the encoding of an instruction: the
``vpdpbusd`` of the AVX-VNNI path must be VEX-encoded (bytes start with ``C4``); the 3.7.6 ImageSeq had
only the EVEX (AVX-512, ``62``) form. This is not a standing check, it catches only that one pattern.


Installers
----------

The installer scripts are ``distrib\WinInstaller\avisynth+.iss`` (x86 + x64) and
``avisynth+_arm64.iss``. The script stages the binaries under ``release-build\installer-stage`` and
passes the build folders and options with ``/D`` defines, so the scripts need no editing per release.

**One installer for 32 and 64 bit.** Every Inno Setup installer is a 32-bit (x86) program. A single
x86/x64 installer carries both builds and installs both: the 32-bit AviSynth+ (component
``main\avs32``, to ``SysWOW64`` on 64-bit Windows) and, on 64-bit Windows, the 64-bit one as well
(``main\avs64``, ``Check: IsWin64``, ``Flags: 64bit`` to the native ``System32``, ``Root: HKLM64``).
So there is no separate 32-bit and 64-bit installer per Windows version; 32-bit hosts (e.g. old
VirtualDub) and 64-bit hosts both find their AviSynth+.

+--------------+---------------+-----------------+-----------------------+---------------------------------+
| Installer    | Inno Setup    | ``MinVersion``  | VC++ redistributable  | Notes                           |
+==============+===============+=================+=======================+=================================+
| default      | 5.5.9 / 5.6.1 | 10.0            | latest (rolling),     | v145 binaries; clang-cl ones    |
|              |               |                 | ``WITH_VC_REDIST``    | with ``-ClangCL``               |
+--------------+---------------+-----------------+-----------------------+---------------------------------+
| ``_msvc``    | 5.5.9 / 5.6.1 | 10.0            | as the default        | v145 binaries, only with        |
|              |               |                 |                       | ``-ClangCL``                    |
+--------------+---------------+-----------------+-----------------------+---------------------------------+
| ``_win7``    | 5.5.9 / 5.6.1 | 6.1sp1          | 14.44.35211.0,        | v143 14.44 binaries             |
|              |               |                 | ``WITH_VC_REDIST``    |                                 |
+--------------+---------------+-----------------+-----------------------+---------------------------------+
| ``_xp``      | 5.5.9 / 5.6.1 | 5.1sp3          | 14.27.29114.0         | v141_xp binaries (3.7.6 also    |
|              |               |                 | (script default)      | put ``DevIL.dll`` to ``{sys}``) |
+--------------+---------------+-----------------+-----------------------+---------------------------------+
| ``_arm64``   | 6.7.x         | 10.0.16299      | none needed (UCRT,    | docs and SDK components on      |
|              |               |                 | static C++ runtime)   |                                 |
+--------------+---------------+-----------------+-----------------------+---------------------------------+

Inno Setup versions, what they support, and why these are used:

+----------------+--------------------------------+--------------------------------------------------------+
| Inno Setup     | Generated setups run on        | For us                                                 |
+================+================================+========================================================+
| 5.6.1          | Windows 2000 / XP / Server     | The last version for XP: needed for ``_xp``; also used |
|                | 2003 and newer                 | for the default and ``_win7`` installers, as 3.7.6     |
|                |                                | (``MinVersion`` 5.1sp3 compiles only with 5.x)         |
+----------------+--------------------------------+--------------------------------------------------------+
| 6.0 to 6.2.2   | Vista / Server 2008 and newer  | 6.2.2 is the last for Vista                            |
+----------------+--------------------------------+--------------------------------------------------------+
| 6.3 and newer  | Windows 7 / Server 2008 R2 and | ``_arm64`` (3.7.6 used 6.3.0); ``MinVersion`` must be  |
| (6.7.4, 7.x)   | newer (``MinVersion`` >= 6.1)  | 6.1 or higher. Adds setup.exe hardening (DLL preload   |
|                |                                | protection 6.3, random temp names 6.6,                 |
|                |                                | RedirectionGuard 6.7)                                  |
+----------------+--------------------------------+--------------------------------------------------------+
| 7.x            | the same                       | Removed ``EnableFsRedirection`` (used by               |
|                |                                | ``avisynth+.iss``): not used until that is rewritten   |
+----------------+--------------------------------+--------------------------------------------------------+

The version in an installer's ``Inno Setup Setup Data (x.y.z)`` header string is the data format, not
the compiler release: 5.5.9 and 5.6.1 write ``5.5.7``, 6.7.4 writes ``6.7.0``. To list what an installer
contains, use innounp 2.x (github.com/jrathlev/InnoUnpacker-Windows-GUI; the old 0.50 cannot read the
6.3+ format): ``innounp -v <installer>.exe``.

* Inno Setup 6.0 dropped Windows XP, so the XP installer needs 5.x (5.6.1 is the last that supports
  XP). The main installer uses 5.x as well, as 3.7.6 did; any later version would support Windows 7
  (the current versions still accept ``MinVersion`` 6.1). Without Inno Setup 5 the script falls back
  to Inno Setup 6 with a warning, and that XP installer then does not run on XP.
* Inno Setup 7 removed ``EnableFsRedirection``, which ``avisynth+.iss`` still uses: moving to 7 needs
  that part rewritten (``{sysnative}`` or ``ExecWithNativeSysDir``).
* The ARM64 installer is an x86 program, as every Inno Setup installer is: it runs under the x86
  emulation of Windows on Arm. It installs the ARM64 files to the native ``System32`` with
  ``Flags: 64bit`` and writes the registry with ``Root: HKLM64``; it does not need Inno's 64-bit install
  mode.
* The redistributable is installed as ``VC_redist.x86/x64.exe`` (``DestName``) and run silently. The
  file names in ``distrib\Prerequisites`` are given with ``/DVcRedistX86=`` and ``/DVcRedistX64=``.
* Before compiling, the stage downloads the latest ``VC_redist.x86/x64/arm64.exe`` from
  ``https://aka.ms/vc14/`` into ``distrib\Prerequisites``, checks the Microsoft signature and logs the
  version (``-RefreshRedist $false`` skips it, e.g. offline). The 14.44 and 14.27 files are never touched.
* A Windows 7 user who starts the default installer by mistake gets the installer's own refusal
  (``MinVersion`` 10.0); the release notes should name ``_win7`` for Windows 7 SP1 and 8.x.


PDB files
---------

The DLLs are Release builds (``/O2 /Ob2``) with debug information in separate PDB files: compiled with
``/Zi``, linked with ``/DEBUG /OPT:REF /OPT:ICF`` (unused code is still removed and identical functions
merged, which ``/DEBUG`` alone would turn off), and ``/PDBALTPATH:%_PDB%``, so a DLL records only the
name of its PDB, not the folder of the build machine. The ARM64 build uses ``-g -gcodeview`` and the
linker's ``--pdb=``.

``/PDBALTPATH:%_PDB%`` is passed in the ``LINK`` environment variable during the build (link.exe and
lld-link both read it), not through CMake: the Visual Studio generator escapes the ``%``, and the DLL
then records ``%AviSynth.pdb%`` (link.exe) or ``%%_PDB%%`` (lld-link, clang-cl), names a debugger does
not find.

The PDBs are published as ``<name>-pdb.7z`` (``<variant>\AviSynth.pdb``,
``<variant>\plugins\<plugin>.pdb``). With them a crash dump of a user can be read, and plugin developers
get readable call stacks in AviSynth+. 3.7.6 was built as RelWithDebInfo instead (about twice the DLL
size: less inlining, and no removal or merging of functions by the linker), and its PDBs were not
published.


Making a hotfix release for a dependency
-----------------------------------------

1. Change the pin of the dependency in ``helpers\makedevil.ps1`` (or the SoundTouch/strmbase helper).
2. Run ``make_release.bat``; ``DEPS_BUILD=auto`` rebuilds the chains whose pins changed, then everything
   else follows.
3. Check the log and the ``check`` stage, test the installers, publish the assets and the SHA-256 list.
