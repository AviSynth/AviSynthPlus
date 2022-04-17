======
Levels
======

Adjusts brightness, contrast, and gamma. This is done using the following
*transfer* function:

.. math::

    \mathsf{output = ( (input - input\_low) / (input\_high - input\_low) )
    ^{(1 / gamma)} * (output\_high - output\_low) + output\_low}

* ``input_low`` and ``input_high`` determine what input pixel values are treated
  as pure black and pure white.
* ``output_low`` and ``output_high`` determine what output values are treated as
  pure black and pure white.
* ``gamma`` controls the degree of non-linearity in the conversion.

This is one of those filters for which it would really be nice to have a GUI.
Since we can't offer a GUI (though `AvsPmod`_ does), we at least make this filter
compatible with `VirtualDub`_'s when the clip is RGB. In that case you should be
able to take the numbers from VirtualDub's Levels dialog and pass them as
parameters to the **Levels** filter and get the same results. Unlike VirtualDub's
filter however, the input and output parameters can be larger than the maximum
value of the pixel format (for example, in 8-bit it can be greater than 255, see
table below).

When processing data in YUV mode, **Levels** only gamma-corrects the luma
information, not the chroma. Gamma correction is really an RGB concept, and is
only approximated here in YUV. If ``gamma=1.0`` (unity), the filter should have
the same effect in both RGB and YUV modes. For adjusting brightness or contrast
in YUV mode, it *may* be better (depending on the effect you are looking for) to
use :doc:`tweak` or :doc:`coloryuv`, because **Levels** changes the chroma of
the clip.

Note in AviSynth+, the parameters ``input_low``, ``input_high``, ``output_low``
and ``output_high``:

* are float instead of int.
* are not `autoscaling`_ – they are relative to the current bit depth:

 .. table::
  :widths: auto

  +--------+------+---------+----------+----------+--------+
  | Bits   | Min. |                               | Max.   |
  +========+======+=========+==========+==========+========+
  | **8**  | 0    | 16      | 128      | 235      | 255    |
  +--------+------+---------+----------+----------+--------+
  | **10** | 0    | 64      | 512      | 940      | 1023   |
  +--------+------+---------+----------+----------+--------+
  | **12** | 0    | 256     | 2048     | 3760     | 4095   |
  +--------+------+---------+----------+----------+--------+
  | **14** | 0    | 1024    | 8192     | 15040    | 16383  |
  +--------+------+---------+----------+----------+--------+
  | **16** | 0    | 4096    | 32768    | 60160    | 65535  |
  +--------+------+---------+----------+----------+--------+
  | **32** | 0    | 16/255  | 128/255  | 235/255  | 1.0    |
  +--------+------+---------+----------+----------+--------+


Syntax and Parameters
----------------------

::

    Levels (clip input, float input_low, float gamma, float input_high, float output_low, float output_high,
            bool "coring", bool "dither")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: input_low

    | Input values at ``input_low`` or lower are treated as black, and lighter
      colors are darkened proportionally.
    | Therefore, raising ``input_low`` darkens the output.

.. describe:: gamma

    | `Gamma`_ adjustment. See `examples`_.
    | Higher ``gamma`` brightens the output; lower ``gamma`` darkens the output.

.. describe:: input_high

    | Input values at ``input_high`` or higher are treated as *white*, and
      darker colors are brightened proportionally.
    | Therefore, lowering ``input_high`` brightens the output.

.. describe:: output_low

    Dark values brighten to gray as ``output_low`` becomes larger.

.. describe:: output_high

    Light values darken to gray as ``output_high`` becomes smaller.

.. describe:: coring

    For RGB this parameter is ignored and internally set to false. For Y and YUV,
    when true (default),

    #. input clip is clamped to limited range (e.g. in 8-bit, 16-235 for luma
       and 16-240 for chroma);
    #. this clamped input is scaled from limited range back to full range,
    #. the conversion takes place according to the transfer function above, and then
    #. output is scaled back to limited range.

    When false, the conversion takes place according to the transfer function,
    without any scaling.

    .. note::

       ``coring`` was created for VirtualDub compatibility, and it remains true
       by default for compatibility with older scripts.

       In the opinion of some, you should `always use coring=false`_ if you are
       working directly with luma values (whether or not your input is limited
       range).

       Limited range video can be correctly processed with ``coring=false``; for
       example::

            Levels(0, 1.6, 255, 0, 255, coring=true)

       Produces the same result as::

            Levels(16, 1.6, 235, 16, 235, coring=false)

       Except that the output is not clipped to limited range. Black and white
       levels are preserved while adjusting ``gamma``, unlike::

            Levels(0, 1.6, 255, 0, 255, coring=false)

.. describe:: dither

    When true, `ordered dithering`_ is applied to combat `banding`_.

    Default: false


Examples
--------

::

    # does nothing on a [16,235] clip, but it clamps (or rounds) a [0,255] clip to [16,235]:
    Levels(0, 1, 255, 0, 255)

::

    # the input is scaled from [16,235] to [0,255],
    # the conversion [0,255]->[16,235] takes place (accordingly to the formula),
    # and the output is scaled back from [0,255] to [16,235]:
    # (for example: the luma values in [0,16] are all converted to 30)
    Levels(0, 1, 255, 16, 235)

::

    # gamma-correct image for display in a brighter environment:
    # example: luma of 16 stays 16, 59 is converted to 79, etc.
    Levels(0, 1.3, 255, 0, 255)

::

    # invert the image (make a photo-negative):
    # example: luma of 16 is converted to 235
    Levels(0, 1, 255, 255, 0)

::

    # does nothing on a [0,255] clip; does nothing on a [16,235]:
    Levels(0, 1, 255, 0, 255, coring=false)

::

    # scales a [0,255] clip to [16,235]:
    Levels(0, 1, 255, 16, 235, coring=false)
    # note both luma and chroma components are scaled by the same
    # amount, so it's not exactly the same as ColorYUV(levels="PC->TV")

::

    # scales a [16,235] clip to [0,255]:
    Levels(16, 1, 235, 0, 255, coring=false)
    # note both luma and chroma components are scaled by the same
    # amount, so it's not exactly the same as ColorYUV(levels="TV->PC")

::

    # makes a clip 100% black
    Levels(0, 1.0, 255, 0, 0)

::

    # apply fading on gamma corrected source (same holds for resizing and smoothing)
    clip = ...
    gamma = 2.2
    clip.Levels(0, gamma, 255, 0, 255) # undo gamma (also called gamma correction)
    FadeOut(n)
    Levels(0, 1.0/gamma, 255, 0, 255) # redo gamma
    ## use bit depth >= 10 to avoid banding in dark areas


Changelog
----------

.. table::
    :widths: auto

    +-----------------+----------------------------------------------------------------+
    | Version         | Changes                                                        |
    +=================+================================================================+
    | AviSynth+ r2542 | Added 32 bit float support.                                    |
    +-----------------+----------------------------------------------------------------+
    | AviSynth+ r2487 || Added 10-16 bit support for YUV(A) and planar RGB(A) formats. |
    |                 || Added support for RGB48/64 formats.                           |
    +-----------------+----------------------------------------------------------------+
    | AviSynth 2.6.0  || Added ``dither`` parameter.                                   |
    |                 || Added support for Y8, YV16, YV24 and YV411 formats.           |
    +-----------------+----------------------------------------------------------------+
    | AviSynth 2.5.3  | Added ``coring`` parameter.                                    |
    +-----------------+----------------------------------------------------------------+

$Date: 2022/04/17 08:19:32 $

.. _AvsPmod:
    https://forum.doom9.org/showthread.php?t=175823
.. _Gamma:
    https://en.wikipedia.org/wiki/Gamma_correction
.. _VirtualDub:
    http://avisynth.nl/index.php/VirtualDub2
.. _autoscaling:
    http://avisynth.nl/index.php/Autoscale_parameter
.. _always use coring=false:
    https://web.archive.org/web/20160825211112/http://forum.doom9.org/showthread.php?p=1722885#post1722885
.. _ordered dithering:
    http://avisynth.org/mediawiki/Ordered_dithering
.. _banding:
    https://en.wikipedia.org/wiki/Colour_banding
