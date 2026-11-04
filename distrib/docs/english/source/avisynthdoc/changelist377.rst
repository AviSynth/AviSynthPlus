Changes from 3.7.6 to 3.7.7
---------------------------

Additions, changes
~~~~~~~~~~~~~~~~~~

Build environment, Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Fix: GCC build with LTO (link-time optimization) failed with "inlining failed in call to 'always_inline'
  ... target specific option mismatch" in the SSE4.1 Overlay/Layer mask row preparation (x86/x64).
  (`issue #531 <https://github.com/AviSynth/AviSynthPlus/issues/531>`_)

Bugfixes
~~~~~~~~
- Fix: 8 and 16 bit horizontal resize, AVX-512: some kernels could read up to ~120 bytes past the end of the
  source frame buffer on the last row (possible access violation; the output was not affected).
  Existed since 3.7.6.
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
- Fix: YUV to planar RGB (and RGB to YUV/Y) conversion from 8-16 bit to 32 bit float, AVX2: memory
  overwrite past the end of the rows (and the last row past the plane) on narrow clips, corrupting the
  first row of the next plane. Existed since 3.7.6.
- Fix: Limiter with (Degenerated case) min > max (e.g. ``Limiter(200, 100, 180, 50, paramscale=true)``), which is outside the
  documented [min, max] range use: the x86 SIMD paths gave ``max`` for every pixel, the C and the aarch64 NEON
  give ``min`` below ``min`` and ``max`` otherwise. On x86 this case now goes to C as well; min <= max is unchanged.
  Existed since classic AviSynth (8 bit) and since the 10-16 bit Limiter (2016).
- Fix: Layer(op="Subtract") with YUVA clips: the blend weight is the original overlay alpha, not the inverted
  one. Regression since 3.7.6 (test build r4589)
- Fix: ConvertToYUY2 from sources with alpha (planar RGBA, YUVA, YA) gave
  "ConvertYV16ToYUY2: Only YV16 is allowed as input". Alpha is now dropped, as with RGB32/RGB64.
  (`issue #529 <https://github.com/AviSynth/AviSynthPlus/issues/529>`_)
- Fix: AddAlphaPlane refused greyscale (Y) clips; it now gives a YA clip. RemoveAlphaPlane turns
  YA back into Y. Existed since 3.7.6 (new YA formats).
  (`issue #530 <https://github.com/AviSynth/AviSynthPlus/issues/530>`_)
- Fix: ``Format``, ``propShow`` and the other users of the shortest exact number form gave one digit
  more than needed for a few powers of two: 37 of the 2098 double and 3 of the 277 float powers of two,
  e.g. 2^-24 gave ``5.9604644775390625e-08`` (17 digits) instead of ``5.960464477539063e-08``. The value
  read back exactly; only the last digit was superfluous. Existed since 3.7.6.

Optimizations
~~~~~~~~~~~~~
- Optimization: x86 SSE2 (no SSE4.1) emulation of unsigned 16 bit min/max (Limiter, Blur/Sharpen, 10-14 bit
  resizers) and the pack + clamp-to-limit in the MSVC SSE2 build of the C resizer: fewer instructions.

Documentation
~~~~~~~~~~~~~

Please report bugs at `github AviSynthPlus page`_ - or - `Doom9's AviSynth+
forum`_

$Date: 2026/10/30 21:00:00 $

.. _github AviSynthPlus page:
    https://github.com/AviSynth/AviSynthPlus
.. _Doom9's AviSynth+ forum:
    https://forum.doom9.org/showthread.php?t=181351
