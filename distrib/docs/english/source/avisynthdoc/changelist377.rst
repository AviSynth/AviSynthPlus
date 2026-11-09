Changes from 3.7.6 to 3.7.7
---------------------------

Additions, changes
~~~~~~~~~~~~~~~~~~
- Overlay, Layer, Subtitle, Text: chroma of subsampled formats: low-pass filter kernels instead of point samples.
  ``top_left`` on 4:2:0 (3x3), 4:2:2 and 4:4:0: ``[1 2 1]``. 4:1:1 and 4:1:0: ``[1 2 3 4 3 2 1]/16`` (co-sited) and
  ``[1 3 5 7 7 5 3 1]/32`` (centred) per axis, also instead of the centred box; 4:1:0 ``left``, ``bottom_left``,
  ``DV`` get a new 7x8 kernel, ``top`` is centred. Mask edges are clamped. Output changes for these
  placements; see :doc:`Overlay <corefilters/overlay>` ``placement``.
- Text and Subtitle: all chroma samples reached by the kernel are drawn, also those
  beside the chroma blocks of the text.
- Frame properties: ``_Range`` is added, the number follows ITU-T H.273, as in VapourSynth: 1 is full, 0 is limited.
  ``_ColorRange`` stays reserved and it is its older name with 0 and 1 swapped: the two are one property, the frame holds it
  as ``_Range``. Plugins and scripts of any version reach it by both names. Key lists give ``_Range`` only, ``propGetAll``
  and ``propShow`` both. See :doc:`frame properties <syntax/syntax_internal_functions_frame_properties>`.

Build environment, Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Fix: GCC build with LTO (link-time optimization) failed with "inlining failed in call to 'always_inline'
  ... target specific option mismatch" in the SSE4.1 Overlay/Layer mask row preparation (x86/x64).
  (`issue #531 <https://github.com/AviSynth/AviSynthPlus/issues/531>`_)

Bugfixes
~~~~~~~~

Most of these bugs were found by AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan) builds, or by comparing
the output of the different CPU paths, then analysed and fixed with AI tools. Some were reported by users.

Bugs in normal use
^^^^^^^^^^^^^^^^^^
- Fix #462 (part 2): autoload of ``.avsi`` files with non-ASCII names (e.g. Hungarian or Japanese characters) failed
  with "Import: couldn't open" on a non-UTF-8 ANSI code page. Regression since 3.7.6; 3.7.5 crashed on names
  outside the ANSI code page. (`issue #462 <https://github.com/AviSynth/AviSynthPlus/issues/462>`_)
- Fix: non-Intel builds (aarch64): crash in YUV with alpha to RGB when the bit depth changes, e.g. ``ConvertToRGB32`` from
  10-16 bit or 32 bit float YUVA, or ``ConvertToPlanarRGBA(bits=8)``. Existed since 3.7.6.
- Fix: YUV to planar RGB (and RGB to YUV/Y) conversion from 8-16 bit to 32 bit float, AVX2: memory
  overwrite past the end of the rows (and the last row past the plane) on narrow clips, corrupting the
  first row of the next plane. Existed since 3.7.6.
- Fix: Histogram ``color2``, 10-16 bit limited range, ``bits`` below the clip bit depth: U or V at the top of the range
  (e.g. 1022 and 1023 at 10 bit, ``bits=8``) was plotted outside the panel, overwriting another plane or memory past
  the frame. Existed since 3.7.6.
- Fix: Subtitle (``gdi=true``), 10-16 bit, RGB as well: overflow on a bright background. At 16 bit the edge of white
  text on white wrapped around to dark pixels. Existed since the 10-16 bit support of text overlay (2016).
- Fix: 32 bit float horizontal resize (all resizers), AVX2 and AVX-512: near the right edge of each row the
  source was read past the row end; multiplied by zero padding coefficients, a NaN found there gave NaN
  output pixels, at random (depending on memory content). Existed since 3.7.6.
- Fix: YUV to planar RGB (and RGB to YUV/Y) conversion to 10-14 bit, AVX2: values above 32767 before the
  final clamp were not clamped (e.g. 14 bit target, B channel at high Y and near-maximum U gave ~32800
  instead of 16383). Existed since 3.7.6.
- Fix: ConvertToYUV444/ConvertToY (and the other planar RGB to YUV/Y conversions) from 32 bit float RGB
  to 8-16 bit: the output range scaling depended on uninitialized memory, giving full or limited range
  scaling at random (e.g. 10 bit full range 910 instead of 913), different from run to run.
  Existed since 3.7.6.
- Fix: ConvertToXXXX with ``interlaced=true``, 4:2:0 or 4:1:0 involved: heights which do not give an integral number
  of chroma rows per field (e.g. 720x486 to 4:2:0) were accepted, and the last chroma row was read from outside the
  source frame. Now an error: height must be mod4, or mod8 if 4:1:0 is involved; see
  :doc:`Convert <corefilters/convert>`. Existed since AviSynth 2.6.
  (`AviSynthPlus-UT issue #5 <https://github.com/HomeOfAviSynthPlusEvolution/AviSynthPlus-UT/issues/5>`_)
- Fix: Overlay and Layer, 32 bit float, AVX2, ``left``-sited mask: 4:2:2 read wrong mask samples in the vector loop,
  4:2:0 read mask samples 4 positions off in the scalar tail. Existed since 3.7.6.
- Fix: Text and Subtitle(gdi=false) on 4:1:0: chroma was misplaced if ``y`` was not divisible by 4. Existed since 3.7.6.
- Fix: Histogram ``levels``, 32 bit float, limited range: luma bars were 16 levels too high, levels above 239 piled up
  in the last bar, which shrank the others to 2 pixels. Existed since 3.7.1.
- Fix: Histogram ``levels`` on Y+Alpha clips gave an error, now works as on greyscale. ``color`` and ``color2`` give the
  greyscale error message.
- Fix: Expr, YA clips: a single expression string was applied on the alpha plane too. Now alpha is copied, as on YUVA and
  planar RGBA.
- Fix: Expr: ``x._ColorRange`` on a clip without this frame property read as 0, which means full range, also on YUV.
  Now it is the default of the clip's format: 0 for RGB, 1 (limited) for the others, as in ``ConvertBits``.
  The same for the new name ``x._Range``, with its own numbers: 1 (full) for RGB, 0 (limited) for the others.
- Fix: Layer(op="Subtract") with YUVA clips: the blend weight is the original overlay alpha, not the inverted
  one. Regression since 3.7.6 (test build r4589)
- Fix: ConvertToYUY2 from sources with alpha (planar RGBA, YUVA, YA) gave
  "ConvertYV16ToYUY2: Only YV16 is allowed as input". Alpha is now dropped, as with RGB32/RGB64.
  (`issue #529 <https://github.com/AviSynth/AviSynthPlus/issues/529>`_)
- Fix: AddAlphaPlane refused greyscale (Y) clips; it now gives a YA clip. RemoveAlphaPlane turns
  YA back into Y. Existed since 3.7.6 (new YA formats).
  (`issue #530 <https://github.com/AviSynth/AviSynthPlus/issues/530>`_)

Accuracy: rounding, limited range levels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Output changes by a rounding step; the values now match ConvertBits and the other paths.

- Fix: Tweak: integer results are rounded on all paths. ``realcalc=true``, chroma at 12-16 bit and chroma from the lookup
  table without ``dither`` truncated: chroma was biased low, ``realcalc=true`` luma could differ from the lookup table.
  Output changes by up to 1 at these paths, see :doc:`Tweak <corefilters/tweak>`.
- Fix: Expr ``cmin``, ``scale_inputs="floatUV"``, 8-16 bit clips: it was half a step too high and rounded up, 17 instead of
  16 at 8 bit, 65 instead of 64 at 10 bit. Existed since 3.5.0 (2020).
- Fix: Histogram ``classic``, 32 bit float, limited range, ``bits`` above 8: bars were not at the same position as at
  integer bit depths.
- Fix: Subtitle (``gdi=true``), 10-16 bit YUV: levels were level * 257 instead of a limited range shift: white Y was
  60393 instead of 60160, neutral U 32895 instead of 32768 at 16 bit. 8 bit output is the same as before, see
  :doc:`Subtitle <corefilters/subtitle>`. Existed since the 10-16 bit support of text overlay (2016).
- Fix: Subtitle (``gdi=true``), 32 bit float: chroma was off by half an 8 bit step, neutral chroma was 0.0019 instead
  of 0.0; levels now reach level / 255, white in RGB is 1.0. Existed since the float support of text overlay.
- Fix: Tweak, 32 bit float: ``dither`` was not centred: luma was brighter by about half an 8 bit step, chroma moved
  as well. Existed since AviSynth+ r2290.
  (`AviSynthPlus-UT issue #53 <https://github.com/HomeOfAviSynthPlusEvolution/AviSynthPlus-UT/issues/53>`_)
- Fix: Tweak, 32 bit float: ``coring`` limits and ``bright`` used a 256 scale: white 235 became 234.08, black 16
  became 15.94 by the limit, ``bright=128`` gave half an 8 bit step less. Existed since AviSynth+ r2290.
- Fix: Limiter, Levels, MaskHS, Expr ``ymax``, ``cmin``, ``cmax``, 32 bit float: limits differed from ConvertBits by a
  rounding step. A limit value from ConvertBits was found over the limit: e.g. Limiter(show="chroma") marked U or V of
  16 or 240 as out of range.
- Fix: YToUV without ``clipY``, 32 bit float: the grey Y plane was 126/256, now 126/255, the same as 8 bit limited
  range 126 converted by ConvertBits. Existed since AviSynth+ r2487.

Same output on every CPU, OS and compiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The C, the x86 SIMD and the aarch64 NEON paths, or two runs of the same script, gave different output. Found by
comparing them, and by the sanitizer builds (planes left unset kept the content of a reused frame buffer).

- Fix: Expr: ``pow`` with a negative base, x86 JIT: it gave 2.4e38 on any power which was not a constant 2, 3 or 4. Now it follows
  IEEE 754 as the C paths do: an integer power has a real result (``-2 ^ 3`` is -8), a fractional power is NaN. Also
  ``0 ^ 0.25`` was 3.3e-10, now 0. Results of a positive base are not changed.
- Fix: Expr: ``sqrt`` of a negative number gave NaN instead of 0 when the expression did not run in JIT (aarch64, ``acos``,
  ``asin``, ``atan`` in it, ``SetMaxCPU("none")``) or was constant, while x86 JIT gave 0. Now it is 0 on all of them.
- Fix: Limiter with (Degenerated case) min > max (e.g. ``Limiter(200, 100, 180, 50, paramscale=true)``), which is outside the
  documented [min, max] range use: the x86 SIMD paths gave ``max`` for every pixel, the C and the aarch64 NEON
  give ``min`` below ``min`` and ``max`` otherwise. On x86 this case now goes to C as well; min <= max is unchanged.
  Existed since classic AviSynth (8 bit) and since the 10-16 bit Limiter (2016).
- Fix: Expr: a plane with an empty expression (or the alpha plane without an expression) was left unset, its content was
  random, when ``format`` changed the bit depth. Now it is converted to the new bit depth as in ConvertBits: Y, U, V as
  limited range, R, G, B and alpha as full range.
- Fix: ``UToY``, ``VToY``, clips with alpha: the alpha plane of the output was left unset, its content was random.
  Now it is opaque.
- Fix: Histogram ``classic``, ``levels``, ``color``, clips with alpha: the alpha plane of the histogram area was left
  unset, its content was random. Now it is opaque.
- Histogram ``color2``: the alpha plane of the histogram area is opaque, it was 0 in 3.7.6.
- Fix: YToUV with YUVA ``clipU`` and no ``clipY``: the alpha plane of the output was left unset, now it is fully opaque.
  Existed since AviSynth+ r2487.

Crash or hang on unusual parameters or input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Found by AddressSanitizer (ASan) builds: memory access outside the frame or the buffer, division by zero, endless
loop. Most need an unusual parameter or input.

- Fix: RGBAdjust, ColorYUV ``analyze=true``: stack buffer overflow in the text of the statistics on very large float pixel
  values or on only NaN pixels.
- Fix: RGBAdjust ``analyze=true``, planar RGB 32 bit float: crash on NaN pixels, which can come from ``Expr`` or from a NaN parameter.
- Fix: ShowFrameNumber, ShowCRC32 ``scroll=true``: division by zero when ``size`` is larger than the clip height, which is also
  the default ``size`` on a clip less than 24 pixels high.
- Fix: ShowFrameNumber, ShowCRC32: ``size`` of 0 or below, NaN or too large never finished. Now an error: it must be between 1
  and 1000000.
- Fix: YToUV accepted clips which do not fit the output frame, and wrote past the plane rows: different bit depth,
  packed or planar RGB for V or Y; RGB24, RGB48, planar RGB without alpha or YUY2 for A. These are errors now.
  RGB32 and RGB64 as ``clipA`` give their alpha plane; see :doc:`Swap filters <corefilters/swap>`.
  Existed since AviSynth+ r2487.
  (`AviSynthPlus-UT issue #8 <https://github.com/HomeOfAviSynthPlusEvolution/AviSynthPlus-UT/issues/8>`_)
- Fix: 8 and 16 bit horizontal resize, AVX-512: some kernels could read up to ~120 bytes past the end of the
  source frame buffer on the last row (possible access violation; the output was not affected).
  Existed since 3.7.6.

Undefined behaviour, NaN parameters, cosmetics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Undefined behaviour (UB): an operation whose result the C++ standard leaves open, e.g. a NaN converted to an
integer or a shift by 32 bits; the compiler may do anything. Here the result was not used or was the same on the
tested platforms. Found by UndefinedBehaviorSanitizer (UBSan) builds.

- Float parameters of the built-in filters which are not a number (NaN) or infinite give an error, when the filter uses them.
- Fix: resizers, Bob, and ``param1``, ``param2``, ``param3`` of the ``ConvertTo...`` filters, AddBorders and LetterBox: NaN
  and Inf were accepted, the result was undefined and differed by platform. Now an error, as are ``src_left``, ``src_top``,
  ``src_width``, ``src_height`` too large for the resizer.
- Fix: NaN passed the range check of Tweak (``startHue``, ``endHue``, ``minSat``, ``maxSat``, ``interp``), MaskHS (same
  without ``interp``), Histogram ``factor``, AddAlphaPlane (``opacity``, ``mask``), ShowSMPTE ``fps``. Now an error.
- Fix: ``Format``, ``propShow`` and the other users of the shortest exact number form gave one digit
  more than needed for a few powers of two: 37 of the 2098 double and 3 of the 277 float powers of two,
  e.g. 2^-24 gave ``5.9604644775390625e-08`` (17 digits) instead of ``5.960464477539063e-08``. The value
  read back exactly; only the last digit was superfluous. Existed since 3.7.6.

Optimizations
~~~~~~~~~~~~~
- Optimization: Text and Subtitle(gdi=false): luma loops skip empty bitmap bytes, chroma visits only the set bits
  of the bitmap. About 20-35% quicker (19 lines of text, frame copies excluded) without halo and fading.
- Optimization: x86 SSE2 (no SSE4.1) emulation of unsigned 16 bit min/max (Limiter, Blur/Sharpen, 10-14 bit
  resizers) and the pack + clamp-to-limit in the MSVC SSE2 build of the C resizer: fewer instructions.

Documentation
~~~~~~~~~~~~~

Please report bugs at `github AviSynthPlus page`_ - or - `Doom9's AviSynth+
forum`_

$Date: 2026/11/09 11:00:00 $

.. _github AviSynthPlus page:
    https://github.com/AviSynth/AviSynthPlus
.. _Doom9's AviSynth+ forum:
    https://forum.doom9.org/showthread.php?t=181351
