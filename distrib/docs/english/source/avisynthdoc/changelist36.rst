
Changes.
========


Changes from 3.5 to 3.6 (20200520)
----------------------------------

Additions
~~~~~~~~~
- User defined functions add array parameter types:
  "array" or "val_array", "bool_array" "int_array", "float_array", "string_array", "clip_array", "func_array"
- New: Format function
- Add frame property getter and setter functions
- Add SetMaxCPU to limit reported CPU level (useful for debugging)
- Allow multiple prefetchers (MT) (from Neo fork)
- Add argument to Prefetch to change # of prefetch frames without changing # of threads (from Neo fork)
- Caching enhancements SetCacheMode (from Neo fork)
- ScriptClip and variable stability in multithreading, add 'local' parameter to
  ConditionalSelect, ConditionalFilter, ScriptClip, ConditionalReader, FrameEvaluate, WriteFile, WriteFileIf, WriteFileStart, WriteFileEnd
- UseVar, special filter, opens a clean variable environment in which only the variables in the parameter list can be seen (from Neo fork)
- "escaped" string constants: with e prefix right before the quotation mask (from Neo fork)
- Introduce function objects into scripts (from Neo fork)
- Filter graph to e.g. svg. SetGraphAnalysis and DumpFilterGraph. (from Neo fork)
- POSIX: better behaviour under non-Windows because of having multiple sized fixed fonts
- Text filter (the only way to "SubTitle" for non-Windows)
- Info() filter: when parameter "size" < 0, font is automatically enlarged over 640x480
- SIL OPEN FONT LICENSE added because of usage of Terminus fonts
- Exist() to have bool utf8 parameter (file existance check)

Build environment, Interface, Internal source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- frame properties framework, IScriptEnvironment extension, Interface V8
- re-enable and fix script arrays (NEW_AVSVALUE define at the moment)
- Big Neo Merge
- Added predefined macros for ARM processors. Tested on Raspberry Pi 4B with the aarch64 image of Ubuntu 20.04.
- Added support for disabling the Intel SIMD intrinsics. Gets automatically disabled on non-x86 targets.
- Added submodule to allow macOS 10.13 and 10.14 to build AviSynth+ with the native Clang compiler
- Support for various processors and OSs
- IScriptEnvironment new additions
  Ex-IScriptEnvironment2: no-throw version of Invoke name change to -> InvokeTry
  Ex-INeo Invoke versions to -> Invoke2Try, Invoke3, Invoke3Try
  New (was not implemented in any former Interface): Invoke2 (Exception Thrower version of Invoke2Try)


Bugfixes
~~~~~~~~
- Fix: broken Exist for directories (regression appeared in 3.5.0)
- Fix: ColorYUV: really disable variable search when parameter "conditional" is false
- Fix: ReplaceStr when the pattern string to be replaced is empty
- Fix: BuildPixelType: chroma subsampling of sample clip was ignored.
- Fix: Mix/Max Runtime function 32bit float chroma: return -0.5..0.5 range (was: 0..1 range)
- Multithreading and deadlock fixes for ScriptClip (Neo fork)


Optimizations
~~~~~~~~~~~~~
- Enhanced: Planar RGB to YUV 444 10-14 bits: more precision (32 bit float internally)
- Enhanced: Planar RGB to YUV 444 10-16 bits: AVX2 (speed improvement)


Please report bugs at `github AviSynthPlus page`_ - or - `Doom9's AviSynth+
forum`_

$Date: 2021/12/07 13:36:0 $

.. _github AviSynthPlus page:
    https://github.com/AviSynth/AviSynthPlus
.. _Doom9's AviSynth+ forum:
    https://forum.doom9.org/showthread.php?t=181351
