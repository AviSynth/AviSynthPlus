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

Documentation
~~~~~~~~~~~~~

Please report bugs at `github AviSynthPlus page`_ - or - `Doom9's AviSynth+
forum`_

$Date: 2026/10/30 21:00:00 $

.. _github AviSynthPlus page:
    https://github.com/AviSynth/AviSynthPlus
.. _Doom9's AviSynth+ forum:
    https://forum.doom9.org/showthread.php?t=181351
