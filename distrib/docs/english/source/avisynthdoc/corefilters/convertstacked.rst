==============
ConvertStacked
==============

These compatibility conversion filters are to allow filters that `use 16-bit
video in a pseudo-8-bit colorspace`_ aka stacked (or lsb) and interleaved format
to interact with `AviSyth+ high bit depth formats`_ aka HBD.

* :ref:`ConvertToStacked` converts a native 16-bit clip to the stacked or double
  width format.
* :ref:`ConvertFromStacked` converts a stacked or double width clip back to a
  native high bit depth format.

**These filters are intended to serve as a stopgap for plugins that are not
supporting true high bit depth yet so you shouldn't use them nowadays unless
with some rare filters that work only with these High-bit-depth hacks.**

**Notes**

* In `Stack16`_ (Stacked 16-bit) aka Double Height, the picture is made of two
  parts: one containing the highest 8 bits (MSB) for each pixel, stacked on top
  of another containing the lowest 8 bits (LSB), hence the name "Double Height".
  It's used in `Dither tools`_ and many others. (Sometimes in filters is referred
  as "lsb" only).

* The Interleaved 16-bit aka Double Width is similar to Stack16, but the MSBs and
  LSBs are horizontally interleaved which is like Native HBD, and it's faster
  than lsb. It's used in [HDRCore] and it was also used in `flash3kyuu_deband`_,
  `LSMASHSource`_ and other filters before they were updated to support 16-bit
  planar (native HBD).

* Both can be between 10-16 bits not only 16-bit.

.. _ConvertToStacked:
.. _ConvertToDoubleWidth:

ConvertToStacked / ConvertToDoubleWidth
---------------------------------------

**ConvertToStacked** accepts a 16-bit (HBD) clip and returns 16-bit stacked
(MSB and LSB stacked one on top of the other in a fake double height "8-bit" stream).

**ConvertToDoubleWidth** accepts a 16-bit (HBD) clip and returns 16-bit interleaved
(MSB and LSB interleaved together in a fake double width "8-bit" stream).

.. rubric:: Syntax and Parameters

::

    ConvertToStacked (clip clip)
    ConvertToDoubleWidth (clip clip)

.. describe:: clip

    Source clip. Must be native 16-bit: YUV420P16, YUV422P16, YUV444P16, Y16.

.. _ConvertFromStacked:
.. _ConvertFromDoubleWidth:

ConvertFromStacked / ConvertFromDoubleWidth
-------------------------------------------

**ConvertFromStacked** accepts a stacked clip and returns a HBD clip.

**ConvertFromDoubleWidth** accepts a Double-Width clip and returns a HBD clip.

.. rubric:: Syntax and Parameters

::

    ConvertFromStacked (clip clip, int "bits")
    ConvertFromDoubleWidth (clip clip, int "bits")

.. describe:: clip

    Source clip. Must be YV12, YV16, YV24 or Y8.

.. describe:: bits

    Bit depth of returned clip. Must match the original bit depth.
    See the 1st examples of both Stacked and DoubleWidth below.

    Default: 16


Examples
--------

Stacked / LSB
^^^^^^^^^^^^^

1st example,

::

    ## Decoding 10bit, YUV 4:2:0 source
    ## this is not needed nowadays since LSMASHSource added HBD support
    ## so nowadays only LWLibavVideoSource(<path>) will did the job
    ## but just for example, keep in mind that lsb is the slowest
    LWLibavVideoSource(<path>, stacked=true, format="YUV420P10")
    ConvertFromStacked(bits=10)
    ## the line below is only if you need to down to 8bit
    ConvertBits(8, dither=0) # '0' means on, '-1' means off


2nd example,

::

    ## this not needed since Dfttest support HBD, but just for example
    ## some 16bit HBD clip here
    ConvertToStacked()
    Dfttest(lsb_in=true, lsb=true)
    ConvertFromStacked()
    ## Continue filtering with 16bit HBD


Double-Width / Interleaved Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1st example (F3KDB can work with lsb but Double-Width is faster than lsb),

::

    ## this not needed since neo_f3kdb support HBD, but just for example
    ## 10bit clip here
    ConvertBits(16, truerange=false) #convey 10bit clip on 16bit clip with truerange=false
    ConvertToDoubleWidth()
    f3kdb(input_mode=2, input_depth=10, output_mode=2, output_depth=10)
    ConvertFromDoubleWidth(bits=10)

2nd example,

::

    ## 16bit clip here
    ConvertToDoubleWidth()
    Hqdn3d16Y()
    ConvertFromDoubleWidth()


Changelog
---------

+-----------------+------------------------------------------------------------+
| Version         | Changes                                                    |
+=================+============================================================+
| AviSynth+ r2150 || Added 10/12/14 bit support to ConvertFrom* functions.     |
|                 || Added ``bits`` parameter to ConvertFrom* functions.       |
+-----------------+------------------------------------------------------------+
| AviSynth+ r2043 || ConvertTo/From* functions moved to their own plugin.      |
|                 || Added the ConvertTo/FromDoubleWidth functions.            |
+-----------------+------------------------------------------------------------+
| AviSynth+ r2022 | ConvertHbd*() functions renamed to ConvertToStacked and    |
|                 | ConvertFromStacked.                                        |
+-----------------+------------------------------------------------------------+
| AviSynth+ r2003 | Added ConvertHbdFromStacked/ConvertHbdToStacked functions. |
+-----------------+------------------------------------------------------------+

$Date: 2022/03/07 03:39:14 $

.. _use 16-bit video in a pseudo-8-bit colorspace:
    http://avisynth.nl/index.php/High_bit-depth_Support_with_Avisynth#Processing_High_Bit-depth_Video_with_AviSynth
.. _AviSyth+ high bit depth formats:
    http://avisynth.nl/index.php/Avisynthplus_color_formats
.. _Stack16:
    http://avisynth.nl/index.php/Stack16
.. _Dither tools:
    http://avisynth.nl/index.php/Dither_tools
.. _flash3kyuu_deband:
    http://avisynth.nl/index.php/Flash3kyuu_deband
.. _LSMASHSource:
    http://avisynth.nl/index.php/LSMASHSource
