
Changes.
========


Changes from 3.7.1 to 3.7.2
---------------------------

Additions, changes
~~~~~~~~~~~~~~~~~~
- Bump copyright year to 2022
- Allow top_left (2) and bottom_left (4) chroma placements for 422 in colorspace conversions, they act as "left" (0, "mpeg2")
- ShowRed/Green/Blue/Alpha/Y/U/V
  - support YUY2 input
  - support YV411 output
  - Copy alpha from source when target is alpha-capable
  - Fill alpha with maximum pixel value when target is alpha-capable but source ha no alpha component
  - Delete _Matrix and _ChromaLocation frame properties when needed.
  - More consistent behaviour for YUV and planar RGB sources, adaptive default pixel_type (YUV, planar or packed rgb)
- Histogram "Luma": support 10-16 and 32 bits
- Histogram: give parameter name "factor" and type 'float' for Histogram's unnamed optional parameter used in "Level" mode.
- Histogram Levels: stop using shades of grey on top of bars.
- Histogram Levels: use bar color 255 for RGB instead of Y's 235. (and scaled eqivivalents)
- PropCopy: new string parameter "props": list of property names to copy (or ignore)
- PropCopy: new bool parameter "exclude": whether property list is positive (copy) or negative (do not copy; blacklist)
- PropDelete: accept an array string parameter as list of property names to remove
- MergeRGB, MergeARGB
  - add MergeARGB parameter "pixel_type", similar to MergeRGB
  - accept pixel_type other than packed RGB formats, plus a special one is "rgb"
  - output format can be planar rgb(a)
  - Accept planar RGB clip in place of input clips and the appropriate color plane is copied from them
  - Fill alpha channel with zero when MergeRGB output pixel_type format is specified to have an alpha plane
  - frame property source is the R clip; _Matrix and _ChromaLocation are removed if R is not an RGB clip
- "FadeX" filter family new parameters: int 'color_yuv' and array of float 'colors' similar to BlankClip
- BlankClip: allow 'colors' with array size more than the number of actual planes.
- BlankClip, AddBorders, LetterBox: no "alpha part must be zero" check for non-YUVA
- Version (#261): New optional parameters int length, int width, int height, string pixel_type, clip c
- Trim, AudioTrim: (#274) bool 'cache' (default true) parameter. Lower memory consumption but may be slower

Build environment, Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Fix (#276): C interface Win32 access: add V8 interface function names to avisynth.def to have undecorated function names

Bugfixes
~~~~~~~~
- Fix deadlock in GetFrame and Invoke (AvsPMod use case)
- Fix Histogram AudioLevels half character upshift (regression since v3.6)
- Fix (#255) Overlay "blend": using accurate formula with internal float calculation. (Issue #255)
- Fix (#263). Escaping double-quotes in 'e' string results in error
- Fix: Expr LUT operation Access Violation on x86 + AVX2 due to an unaligned internal buffer
- Fix: Chroma full scale as ITU Rec H.273 (e.g +/-127.5 and not +/-127) in internal converters, ColorYUV and Histogram
- Fix (#257): GreyScale to not convert to limited range when input is RGB. (regression in 3.7.1)
  GreyScale accepts only matrix names of limited range as it is put in the documentation.
- Fix (#256): ColorYUV(analyse=true) to not set _ColorRange property to "full" if input has no such property and range cannot be 100% surely established.
- Fix: Histogram "color" may crash on certain dimensions for subsampled formats. Regression since 20180301 r2632.
- Fix: Histogram "color" and "color2" mode check and give error on Planar RGB
- Fix: missing Histogram "color2" CCIR rectangle top and bottom line (black on black). Regression since 3.6.2-test1
- Fix: Compare support 10-14 bits 
- Fix: Compare 'channels' parameter default to "Y" when input is greyscale;
- Histogram "Audiolevels" and StereoOverlay to deny planar RGB
- Fix: Histogram "Levels": prevent crash when factor=0.0
- Fix: Histogram "Levels": fix regression incorrect "factor" applied for U/V part drawing when format was subsampled (non-444). Regression since 20160916 r2666
- Expr: "scale_inputs" values to case insensitive and add "floatUV" to error message as an allowed value.

Optimizations
~~~~~~~~~~~~~
- ConvertBits: no compulsory get frame #0 in constructor for frame properties if 'fulls' is directly specified


Please report bugs at `github AviSynthPlus page`_ - or - `Doom9's AviSynth+
forum`_

$Date: 2022/03/17 0:0:0 $

.. _github AviSynthPlus page:
    https://github.com/AviSynth/AviSynthPlus
.. _Doom9's AviSynth+ forum:
    https://forum.doom9.org/showthread.php?t=181351
