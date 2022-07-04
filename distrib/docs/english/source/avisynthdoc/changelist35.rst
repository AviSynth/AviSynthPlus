
Changes.
========


Changes from to 3.4 to 3.5 (20200302)
-------------------------------------


Additions
~~~~~~~~~
- Layer: support RGB24 and RGB48 (internally processed as Planar RGB - lossless pre and post conversion)
- New bool IsVersionOrGreater(int majorVersion, int minorVersion [,int bugfixVersion]) function
- Overlay: show error when Overlay is fed with clips with different bit depths
- New: AddBorders, LetterBox: new color_yuv parameter like in BlankClip
- WavSource: really use "utf8" parameter, fix some debug asserts
- TimeStrech: pass internal errors as Avisynth exception text (e.g. proper "Excessive sample rate!" instead of "unhandled C++ error")
- "Expr" helper constants "yrange_min", "yrange_half", "yrange_max"
- "Expr" new parameter: bool clamp_float_UV
- "Expr" "clamp_float" is not ignored (and set to true) when parameter "scale_inputs" auto-scales 32 bit float type pixels
- "Expr" "yscalef" and "yscaleb" keywords similar to "scalef" and "scaleb" but scaling is forced to use rules for Y (non-UV) planes
- "Expr" new allowed value "floatUV" for scale_inputs supporting special 32 bit chroma case

Build environment, Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Native Linux, macOS, and BSD support.
- Build system: Cmake: use platform toolset "ClangCL" for using built-in Clang support of Visual Studio (Since VS2019 v16.4 there is LLVM 9.0 support)

Bugfixes
~~~~~~~~
- Fix: ConvertBits 32->8 for extremely out of range float pixel values.
  When pixel value in a 32 bit float format video was way out of range and greater than 128 (e.g. instead of 0 to 1.0 for Y plane) then the ConvertBits(8) had artifacts.
- Fix potential crash on exit or cache shrink (linux/gcc only?)
- Fix: RGBP to 444 8-14bit right side artifacts at specific widths
- Fix: "scalef" and "scaleb" for 32 bit input, when scale_inputs="floatf" produced wrong result
- Fix: missing rounder in V channel calculation of PlanarRGB->YUV 8-14bits SSE2 code
- Fix: TemporalSoften possible access violation after SeparateFields (in general: after filters that only change frame pitch)
- Fix: Shibatch.DLL Access Violation crash when exit when target rate is the same as vi.audio_samples_per_second or audio_samples_per_second is 0
- Fix: Resizers to really resize alpha channel (YUVA, RGBPA)
- Fix: crash when outputting VfW (e.g. VirtualDub) for YUV444P16, other fixes for r210 and R10k formats


Optimizations
~~~~~~~~~~~~~



Please report bugs at `github AviSynthPlus page`_ - or - `Doom9's AviSynth+
forum`_

$Date: 2021/12/07 13:36:0 $

.. _github AviSynthPlus page:
    https://github.com/AviSynth/AviSynthPlus
.. _Doom9's AviSynth+ forum:
    https://forum.doom9.org/showthread.php?t=181351
