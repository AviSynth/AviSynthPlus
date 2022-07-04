
AviSynth+
=========


New Video Features
------------------

-   High bit depth support 10, 12, 14 and 16 bits and 32 bit float.
-   Planar RGB
-   Planar Alpha-plane formats
-   Frame properties

New Audio Features
------------------

-   none


New Internal Features
---------------------

-   64 bit with no compromises
-   Source on github
-   CMake build environment
-   No internal assembler (only a few is left in some special windows 32 bit modes), SIMD intrinsics everywhere.
-   Source supports C++ only: supports non-Intel x86/x64 platforms.
-   Support for various OS's: not Windows-only
-   Rewritten multithreading support
-   New language elements: function objects, arrays
-   A bunch of new and extended internal filters and accepted formats
-   Experimental host of direct CUDA-plugins in specific builds.
    (but note that core does not use any CUDA though, just provides a framework through IScriptEnvironment)

$Date: 2009/09/12 20:57:20 $
