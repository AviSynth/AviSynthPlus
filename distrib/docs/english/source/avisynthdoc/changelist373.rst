
Changes.
========


Changes from 3.7.2 to 3.7.3
---------------------------

Additions, changes
~~~~~~~~~~~~~~~~~~
- New: add a sixth array element to PlaneMinMaxStats: average. Defines variable "PlaneStats_average" as well if setting variables is required.

Build environment, Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Fix: C interface crash when using avs_new_video_frame_p(_a)

Bugfixes
~~~~~~~~
- Fix (#283): broken runtime functions Min/Max/MinMaxDifference when threshold is not 0 (returned -1). Regression in 3.7.2
- Fix (#282): ConvertToRGB
  - do check for exact 8 or 16 bit input, because packed RGB formats exist only for 8 and 16 bits
  - keep alpha for RGBA planar - convert RGBAP8/16 to RGB32/64, while RGBP8/16 is still RGB24/48

Optimizations
~~~~~~~~~~~~~


Please report bugs at `github AviSynthPlus page`_ - or - `Doom9's AviSynth+
forum`_

$Date: 2022/06/27 0:0:0 $

.. _github AviSynthPlus page:
    https://github.com/AviSynth/AviSynthPlus
.. _Doom9's AviSynth+ forum:
    https://forum.doom9.org/showthread.php?t=181351
