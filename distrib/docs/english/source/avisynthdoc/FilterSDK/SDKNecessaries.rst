
SDKNecessaries
==============

You must have some compatible development tool
----------------------------------------------

- Historically there were Microsoft Visual C/C++ 9 (2008), 10 (2010),
  11 (2012), 12 (2013) and 14 (2017) versions.
- Actual source requires c++ 17. (or c++1z for gcc 7.0)
- Microsoft compilers:

  As of December 2021 the latest one is Microsoft Visual C++ 2022 (v16) and 
  Microsoft Visual C++ 2019 (v15)

  Community Edition is free for open source development.
  (`free download <http://www.visualstudio.com/en-us/downloads/>`__
  (web version is fine); )
  
  Both 2019 and 2022 versions still support Windows XP when you include Visual Studio 2017 XP (v141_xp) platform toolset in the customized installation.
  (Though user must not update their VC++ redistributables after a specific version point, when XP support has been stopped)
  
- Intel ICX NextGen 2021.4 or ICL 2021.4 (19.2 Classic)

  You need either or both Intel C++ compiler engine:
  
  - Intel C++ 2021 (ICX: LLVM based NextGen) in Base Kit
  
    Intel oneAPI Base Kit `download <https://www.intel.com/content/www/us/en/developer/articles/news/free-intel-software-developer-tools.html>`__

    We need it for Intel® oneAPI DPC++/C++ Compiler. It includes C++ 2021.4 but we don't need DPC++ - it is not suitable for Avisynth+)

  - Intel C++ Classic (ICL 19.2) addon in HPC toolkit

    From Intel oneAPI toolkit explorer homepage choose Intel® oneAPI HPC Toolkit
    <https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#hpc-kit>
    
    The oneAPI HPC Toolkit is an addon (complements) to the base toolkit.
    For less disk space and omit unnecassary components, choose Custom Installation (Fortran support not needed)

  During installation they integrate with Visual Studio IDE (2017, 2019 and 2022).
  Note: when you install VS2022 after installing base toolkit,
  you'll need to install the toolkit again to have VS2022 integration.

- LLVM 9 and up, or use LLVM (clang-cl) platform toolset integrated into Visual Studio
- Gnu C++ 9.0 (7.0 with special flags is working as well)

Notes

- AviSynth+ is no longer restricted to Windows and Intel processors.
  It runs under several platforms: Linux, Haiku, MacOS.
  Either gcc or clang compilers are used for them.

  See also:

  :doc:`Contributing with git <../contributing/contributing_with_git>`,
  :doc:`Compiling AviSynth+ <../contributing/compiling_avsplus>` and
  :doc:`posix <../contributing/posix>`.

- Free registration is mandatory to use Community Edition versions
- Targeting Windows XP requires v141_xp (Visual Studio 2017 - XP) platform toolset
  along with a special compiler flag. Moreover XP requires that running system must not have the latest VC++ redistributable but the last XP compatible one.

Our aim is that sources can be built with all these compilers.
Though AviSynth+ itself requires c++17, plugins are not obeyed to use it.


You also need in Microsoft Platform SDK (if it is not included with compiler)
-----------------------------------------------------------------------------

For MS VC++ 2008 or more recent it is included in the compiler (?) (and
it is called Windows PSK).

For some very special plugins (GPU) you might need the DirectX SDK
(when compiling AviSynth itself you will need it).


Finally, you must include the main header file 'avisynth.h'
------------------------------------------------------------

Avisynth+ source code can be found on github.
`github repository <https://github.com/AviSynth/AviSynthPlus>`__

For plugin development you can get the header with this FilterSDK, or download 
with AviSynth source code, or take from some plugin source package.
There are several versions of this header file from various AviSynth versions.

Header file avisynth.h from v1.0.x to v2.0.x have
``AVISYNTH_INTERFACE_VERSION = 1.`` Plugins compiled with them will not be
(natively) compatible with AviSynth 2.5.x.

Header file avisynth.h from v2.5.0 to v2.5.5 have
``AVISYNTH_INTERFACE_VERSION = 2.`` Plugins compiled with them will
(generally) work in AviSynth v2.5.0 to v2.5.7 (and above). But avisynth.h
files from versions v2.5.0 - v2.5.5 (and betas) are not identical. We
recommend to use avisynth.h from versions 2.5.5 or later. Previous versions
of avisynth.h are obsolete and have some bugs.

Header file avisynth.h from v2.5.6 to v2.5.8 are almost identical and have
``AVISYNTH_INTERFACE_VERSION = 3.`` Plugins compliled with them will work in
v2.5.6 and up, and v2.5.5 and below if you do not use new
interface features and do not call ``env->CheckVersion`` function.

AviSynth version 2.6 and Avisynth+ versions up to 3.5.1 are using
currently ``AVISYNTH_INTERFACE_VERSION = 6.`` Plugins compiled with
AviSynth v2.5.x header will work in AviSynth 2.6.x and specific
(without script array support) Avisynth+ versions.

Note: since Avisynth+ ``avisynth.h`` alone is not enough, some other helper headers
exist under ``avs`` folder.

Avisynth+ version (though a bit late) changed version to
``AVISYNTH_INTERFACE_VERSION = 7.``

From AviSynth+ version 3.6.0 new interface functions arrived
along with frame properties support
``AVISYNTH_INTERFACE_VERSION = 8.``

From AviSynth+ version 3.7.1.test.32 new interface functions arrived
``AVISYNTH_INTERFACE_VERSION = 8`` ``AVISYNTH_INTERFACE_BUGFIX = 1``
or ``AVISYNTH_INTERFACE_VERSION = 9``

Generally good start is to take some similar plugin source code as a draft
for improving or own development. Attention: there are many old plugins
source code packages with older avisynth.h included. Simply replace it by new one.



Compiling options
-----------------

On Windows plugin CPP source code must be compiled as Win32 or x64 DLL (multi-threaded (MT) or
multi-threaded DLL (MD)) without MFC. Latter is recommended in general and
requires the actual Microsoft Visual C++ redistributables.

In Visual Studio Windows XP builds require v141_xp platform toolset 
(Visual Studio 2019's default is v142) along 
with compiler option /Zc:threadSafeInit-

Note that GCC and the other builds cannot be mixed due to the different ABI.
This affects "only" C++ (and not C) interface but since 99% of plugins are using C++ 
interfaces we can say that GCC Avisynth host and non-GCC user plugins are fully incompatible.

Of course, use Release build with optimization.

Use CMake make environment, for MSVC it generates the solution file as well
(note: you cannot have both x86 and x64 configured in the solution at the same time)

See step by step :doc:`compiling instructions. <CompilingAvisynthPlugins>`


Other compilers
---------------
note from 2021: this section maybe a bit outdated.

Since v2.5.7, AviSynth includes an updated version of Kevin Atkinson's
AviSynth C API you can use to create "C-Plugins" with compilers such as
GNU C++, Visual Basic and Delphi.

You can NOT use the C++ API with compilers like GNU C++ to create
plugins, because of :doc:`binary incompatibilities <CompilingAvisynthPlugins>`.

There is also `Pascal conversion of avisynth_c.h`_ by Myrsloik

Some info about `Using in Visual Basic`_

`PureBasic port of the Avisynth C Interface`_ by Inc

There is also `AvsFilterNet`_ wrapper for Avisynth in .NET (any .NET
language) by SAPikachu, see `discussion`_

----

Back to :doc:`FilterSDK <FilterSDK>`

$Date: 2020/04/22 06:08:10 $

.. _[1]:
   http://www.google.nl/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&ved=0CCoQFjAA&url=http://go.microsoft.com/?linkid=7729279&ei=HfWhUuTjL8Og0wW7wYDwBw&usg=AFQjCNEulTGchEeozkLGRH8LZELiTKlC5A&sig2=Mi7Rwn_jNL5Qffi7LiGS3w&bvm=bv.57752919,d.d2k
.. _[5]: http://www.visualstudio.com/en-us/downloads/
.. _[7]: http://www.microsoft.com/en-us/download/details.aspx?id=15656
.. _LLVM / clang: https://releases.llvm.org/download.html
.. _CodeBlocks: http://www.codeblocks.org
.. _Microsoft site: http://www.microsoft.com/downloads/details.aspx?familyid=EBA0128F-A770-45F1-86F3-7AB010B398A3&displaylang=en
.. _Pascal conversion of avisynth_c.h:
    http://forum.doom9.org/showthread.php?t=98327
.. _Using in Visual Basic: http://forum.doom9.org/showthread.php?t=125370
.. _PureBasic port of the Avisynth C Interface:
    http://forum.doom9.org/showthread.php?t=126530
.. _AvsFilterNet: http://www.codeplex.com/AvsFilterNet
.. _discussion: http://forum.doom9.org/showthread.php?t=144663
.. _direct link: http://go.microsoft.com/?linkid=9709949
