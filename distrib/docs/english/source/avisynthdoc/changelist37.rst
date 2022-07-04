
Changes.
========


Changes from 3.6 to 3.7 (20210111)
----------------------------------

Additions
~~~~~~~~~
- Resizers: throw error on too small dimensions vs. taps
- Add ShowCRC32 debug filter. Parameters are the same as in ShowFrameNumber
- Overlay: allow 4:1:1 input
- RemoveAlphaPlane: do nothing on YUY2 instead of throwing an error message
- AviSource: support non-printing characters in fourCC code: allow [number] style, e.g. G3[0][16]
- AviSource: add Y410 (YUVA444P10) format support. Allow 'Y410' pixel_type hints.
- AviSource: decode b64a, b48r, v210, P210, P010, P016, P216, v410, Y416, r210, R10k, v308, v408, Y410 fourCCs natively.
- Add: AverageA
- New: Average...: allow YUY2, RGB24/32/48/64 inputs
- support for Win10 long file path option
- internally refactored ConvertAudio
- New: Histogram("color2") to support 10+ bits.
  Allow bits=x (x=8,9,10,11,12) parameter for this kind of histogram as well.
- pass V3 (2.5) IScriptEnvironment for CPP plugins which directly load avisynth.dll and
  request CreateScriptEnvironment with version <= 3 
- Histogram("levels") to allow greyscale


Build environment, Interface, Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Haiku support
- PowerPC support
- Support for building the core as a static library (mcmtroffaes)
- Fixes for MinGW-w64 compilation
- Shibatch, TimeStretch, and ImageSeq GCC build support
- Shibatch, TimeStretch, and ImageSeq non-Windows support
- project: Improve inclusion of the ghc filesystem helper library
- project: Add a GitHub action workflow
- Make frame property related constants to not collide with VapourSynth's definitions. (favour for multi-target plugins)


Bugfixes
~~~~~~~~
- Fix: AddBorders did not pass frame properties
- Fix: propSet, propDelete and propClearAll not to ruin visibility of variables (property read functions are still kept being runtime only)
- Fix: Average...: check for valid colorspace (e.g. no AverageB for a YUV clip)
- Fix: Overlay: Actual frame of a mask clip would be freed up too early in MT environment
- Fix: ConvertBits to ignore dither parameter instead of throwing error, in a 8 to 8 bit case
- Fix: GeneralConvolution missing internal rounding on 8-16 bit formats
- posix: fix crash when autoloading imports
- ConvertBits(8): fix dither=1 (floyd) for RGB48/RGB64
- Fix: Blur right side garbage: 16 bit+AVX2+non mod32 width
- Fix: check fn signature with implicite "last" first (3.6 regression)
- Fix: function parameters provided as arrays (e.g. GrunT callback of WriteFileIf)
- Fix: ConvertBits (YUV): proper rounding when bit depth is reduced and origin is 10-16 bits (added rounder before bit-shift)
- Fix: proper handling of autoload directories precedence
- Fix: ScriptClip + Runtime function object (which are new in 3.6) under heavy multithreading
- fix: GeneralConvolution: incorrect parse of negative integer coefficient (+1) regression since r2772
- fix: GeneralConvolution: possible crash when chroma=true for 420 and 422 formats
- Fix 3.6 regressions
  - when explicit "return last" was needed when followed by legacy function definition.
  - Windows XP is supported again (thread local storage workaround)
  - Stabilize CPP 2.5 plugins
  - allow forced named arrays usage again from plugins (MP_PipeLine)


Optimizations
~~~~~~~~~~~~~
- Overlay: may work quicker, most input/overlay/mask/output clip format conversions moved to filter constructor



Please report bugs at `github AviSynthPlus page`_ - or - `Doom9's AviSynth+
forum`_

$Date: 2021/12/07 13:36:0 $

.. _github AviSynthPlus page:
    https://github.com/AviSynth/AviSynthPlus
.. _Doom9's AviSynth+ forum:
    https://forum.doom9.org/showthread.php?t=181351

Useful links:
-------------

- Source (from 3.4): https://github.com/AviSynth/AviSynthPlus
- Source (before 3.4): https://github.com/pinterf/AviSynthPlus/tree/MT
- Forum: https://forum.doom9.org/showthread.php?t=168856
- Forum on some avs+ filters: https://forum.doom9.org/showthread.php?t=169832
- Avisynth+ info page: http://avisynth.nl/index.php/AviSynth%2B
- Info on Avisynth+ new color spaces: https://forum.doom9.org/showthread.php?p=1783714#post1783714
- Avisynth Universal Installer by Groucho2004: https://forum.doom9.org/showthread.php?t=172124

