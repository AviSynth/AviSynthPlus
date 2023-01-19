
Changes.
========


Changes from 3.7.2 to 3.7.3
---------------------------

Additions, changes
~~~~~~~~~~~~~~~~~~
- MIPS build support
- New: add a sixth array element to PlaneMinMaxStats: average. Defines variable "PlaneStats_average" as well if setting variables is required.
- "Text" ``halo_color`` allows to have both halo and shaded background
- "Text" (#308) much nicer rendering of subsampled formats
- "TimeStretch" (#278) add TimeStretch overload with rational pair arguments and update SoundTouch library to v2.3.1.

Build environment, Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Address Issue #305: Support for non-decorated 'avisynth_c_plugin_init' in C-plugins
- Fix: C API undefined behavior when upstream throw runtime error
- CMakeLists.txt: fix clang-cl/intel with ninja generator
- Fix: C interface crash when using avs_new_video_frame_p(_a)
- Fix: C interface avs_prop_get_data behave like C++ counterpart.
- Bump Interface version bugfix part. Interface version is 9.2
  AVISYNTH_INTERFACE_VERSION = 9,
  AVISYNTHPLUS_INTERFACE_BUGFIX_VERSION = 2

Bugfixes
~~~~~~~~
- Fix: "Text" filter negative x or y coordinates (e.g. 0 instead of -1)
- Fix: "Text" filter would omit last character when x<0
- Fix: "Text" halo_color needs only MSB=$FF and not the exact $FF000000 constant for fade
- Fix: (#304) "ColorYUV" analyze=true was displaying wrong min-max values for YUY2
- Fix: (#293) "Text" to throw proper error message if the specified font name (e.g. Arial) is not found among internal bitmap fonts.
- Fix: (#293) "Subtitle" and "Text" filter to respect the explicitely given coorditanes for y=-1 or x=-1, 
  instead of applying vertical/horizontal center alignment.
- Fix (#283): broken runtime functions Min/Max/MinMaxDifference when threshold is not 0 (returned -1). Regression in 3.7.2
- Fix (#282): ConvertToRGB
  - do check for exact 8 or 16 bit input, because packed RGB formats exist only for 8 and 16 bits
  - keep alpha for RGBA planar - convert RGBAP8/16 to RGB32/64, while RGBP8/16 is still RGB24/48

Optimizations
~~~~~~~~~~~~~

Documentation
~~~~~~~~~~~~~
- Internal plugins, syntax, ...: almost fully revised and made up-to-date. Big thanks to Reel-Deal!

Please report bugs at `github AviSynthPlus page`_ - or - `Doom9's AviSynth+
forum`_

$Date: 2023/01/19 13:52:00 $

.. _github AviSynthPlus page:
    https://github.com/AviSynth/AviSynthPlus
.. _Doom9's AviSynth+ forum:
    https://forum.doom9.org/showthread.php?t=181351
