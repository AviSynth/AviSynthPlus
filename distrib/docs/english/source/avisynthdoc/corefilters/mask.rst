============
Mask Filters
============

Set of filters to manipulate the alpha channel:

* `AddAlphaPlane`_ adds or replaces an alpha plane.
* `RemoveAlphaPlane`_ removes an alpha plane.
* `Mask`_ also replaces an alpha plane but works different than `AddAlphaPlane`_.
* `ResetMask`_ by default sets all pixel values in the alpha plane to the
  maximum value.
* `ColorKeyMask`_ is a simple color keying filter, it sets transparency in the
  alpha channel in areas where the specified color meets the criteria.

.. _AddAlphaPlane:

AddAlphaPlane
--------------

**AddAlphaPlane** adds an alpha plane to the source clip. It can also be used to
replace the existing alpha plane of the source clip. Note that the color format
changes after using ``AddAlphaPlane()`` on an alpha-less format, for example,
RGBP16 becomes RGBAP16.

.. rubric:: Syntax and Parameters

::

    AddAlphaPlane (clip, clip mask)
    AddAlphaPlane (clip, float mask)
    AddAlphaPlane (clip, int mask)


.. describe:: clip

    Source clip; RGB24, RGB32, RGB48, RGB64, and all Planar RGB(A) and YUV(A)
    formats supported.

.. describe:: mask

    The ``mask`` parameter can be specified in 3 different ways, as a clip, an
    integer or a float value.

    The alpha source clip can either be a single channel greyscale (Y) or an
    alpha capable format (such as RGBAP8 or RGB64). If a clip is supplied,
    depending on the type, either the greyscale clip or the alpha channel is
    then copied onto the alpha channel of the source clip. The alpha source
    clip must be the same bit depth and dimensions as the source clip.

    Pixel value in integer or float. If the numeric-type mask parameter is
    supplied, it will be used as filler value of the resulting alpha plane. No
    bit depth scaling happens, parameter value is used as-is. For reference,
    the table below shows all of the maximum values in regards to bit depth.

    .. table::
        :widths: auto

        +-------+--------+--------+--------+--------+--------+
        | 8-bit | 10-bit | 12-bit | 14-bit | 16-bit | 32-bit |
        +-------+--------+--------+--------+--------+--------+
        | 255   | 1023   | 4095   | 16383  | 65535  | 1.0    |
        +-------+--------+--------+--------+--------+--------+

    A value of 0 creates a fully transparent mask (black) and the maximum pixel
    value creates a completely opaque mask (white). Range for all bit depths is
    from 0 to the max value. 32-bit Float clips are specified in the range of
    [0.0 - 1.0]. To set the alpha channel to the maximum value, it's easier to
    use `ResetMask`_.

.. _RemoveAlphaPlane:

RemoveAlphaPlane
----------------

**RemoveAlphaPlane** removes the alpha plane from the source clip. Note that the
color format changes after using ``RemoveAlphaPlane()``, for example, YUVA444P8
becomes YUV444P8.


.. rubric:: Syntax and Parameters

::

    RemoveAlphaPlane (clip)

.. describe:: clip

    Source clip; RGB32, RGB64, and all Planar RGBA and YUVA formats supported.

.. _Mask:

Mask
----

**Mask** replaces the alpha channel in the source clip with a "luma mask"
created from the specified mask clip. Internally the mask clip is converted to
greyscale (``ConvertToY(matrix="PC.601")``), this result is then used to replace
the alpha plane in the source clip. Note that **Mask** is a legacy filter, the
intended use was to convert a greyscale RGB clip that consisted of 3 identical
channels to an alpha channel. Using `AddAlphaPlane`_ is recommended.


.. rubric:: Syntax and Parameters

::

    Mask (clip, clip mask)

.. describe:: clip

    Source clip; RGB32, RGB64, and all Planar RGBA formats supported.

.. describe:: mask

    Mask clip to replace the alpha channel in the source clip. The dimensions
    and bit depth must be the same as the source clip. The alpha channel in this
    clip is ignored.

.. _ResetMask:

ResetMask
---------

**ResetMask** by default sets all of the pixels in the alpha channel to the
maximum value, effectively making it white (completely opaque).


.. rubric:: Syntax and Parameters

::

    ResetMask (clip, float "mask")

.. describe:: clip

    Source clip; RGB32, RGB64, and all Planar RGBA and YUVA formats supported.

.. describe:: mask

    Sets the pixel value of the mask channel. By default, it is set to the maximum
    value of the pixel format. For reference, the table below shows all of the
    maximum values in regards to bit depth.

    .. table::
        :widths: auto

        +-------+--------+--------+--------+--------+--------+
        | 8-bit | 10-bit | 12-bit | 14-bit | 16-bit | 32-bit |
        +-------+--------+--------+--------+--------+--------+
        | 255   | 1023   | 4095   | 16383  | 65535  | 1.0    |
        +-------+--------+--------+--------+--------+--------+

    A value of 0 creates a fully transparent mask (black) and the maximum pixel
    value creates a completely opaque mask (white). Range for all bit depths is
    from 0 to the max value. 32-bit Float clips are specified in the range of
    [0.0 - 1.0].

.. _ColorKeyMask:

ColorKeyMask
------------

Clears pixels in the alpha channel by comparing the specified color. Each pixel
with a color differing less than the tolerance is set to zero (that is black
or transparent), otherwise it is left unchanged. i.e. It is NOT set to opaque
(white). To start off with a fully opaque mask, use `ResetMask`_ beforehand,
this allows an aggregate mask to be constructed with multiple calls to
**ColorKeyMask** to build up transparent areas where each color of interest
occurs. To view or extract the mask use :doc:`ShowAlpha <showalpha>` or the
:doc:`Extract <extract>` filters. See the :ref:`examples <ColorKeyMask-example>`
section for more information.

.. rubric:: Syntax and Parameters

::

    ColorKeyMask (clip, int color, int tolB, int tolG, int tolR)

.. describe:: clip

    Source clip; RGB32, RGB64 and all Planar RGBA formats supported.

.. describe:: color

    Specify the color to compare. Color is specified as an RGB value in either
    hexadecimal or decimal notation. Hex numbers must be preceded with a $. See
    the :doc:`colors <../syntax/syntax_colors>` page for more information on
    specifying colors.


.. describe:: tolB, tolG, tolR

    Tolerance specifies the range for which close colors are considered the same.
    The range is from 0 to 255 and the values are autoscaled for bit depths
    greater than 8 bit. Note that these parameters are unnamed, however they do
    have a default. When ``tolR`` or ``tolG`` are not set, they use the value
    from ``tolB``. When ``tolB`` is not defined, it defaults to 10.


Examples
--------

.. rubric:: AddAlphaPlane

Let's create some colorful text using :doc:`ColorBars <colorbars>`,
:doc:`Subtitle <subtitle>` and `AddAlphaPlane`_ and then save it to a png:

.. code-block:: c++

    src = ColorBars(width=256, height=192, pixel_type="RGB32").Crop(0,0,0,-143)
    msk = Blankclip(src).Subtitle("AviSynth+", size=55, align=2, text_color=$FFFFFF).ExtractR()

    AddAlphaPlane(src, msk)
    ImageWriter("colorful text", type="png")

.. list-table::

   * - .. figure:: pictures/addalphaplane-colorfultext.png

          Result

Now suppose that you made a mask in an image editor and want to load it in
AviSynth to use it as an alpha for a clip. The process is similar to the
previous example::

    src = FFVideoSource(video.mkv)
    msk = FFImageSource("mask.png").ExtractR() # see note below

    AddAlphaPlane(src, msk)

If the mask was saved as a single channel greycale (Y) image or if the mask is
in the alpha channel then :doc:`ExtractR() <extract>` is not needed.

.. _ColorKeyMask-example:

.. rubric:: ColorKeyMask

Let's use `ColorKeyMask`_ to remove the background and then :doc:`overlay` the
result over a :doc:`ColorBars <colorbars>` clip:

.. code-block:: c++

    src  = ImageSource("colorkeymask-dog.png")
    base = ColorBars(width=src.Width(), height=src.Height())

    ResetMask(src)
    ColorKeyMask($F22E42, 25)
    ColorKeyMask($DC2026, 25)
    ColorKeyMask($B92708, 25)

    # MaskTools2: mt_inpand shrinks the mask and mt_deflate feathers the mask outward.
    msk = ExtractA().mt_inpand().mt_inflate().mt_inflate()

    Overlay(base, src, mask=msk)

    # To use Layer instead, comment out Overlay and use the following commands.
    # src = AddAlphaPlane(src, msk)
    # Layer(base, src)

.. list-table::

   * - .. figure:: pictures/colorkeymask-dog.png

          `Source`_

     - .. figure:: pictures/colorkeymask-dog-result.png

          Result

The result is by no means perfect, but it gives the general idea of how to use
the ColorKeyMask filter. There are more intricate ways of improving the result,
for example:

* Lightly denoising the source clip prior to ColorKeyMask often helps
  (specially with noisy sources). Scaling the source clip to a lower resolution
  can also be beneficial. Both methods help reduce variations in the target color,
  leading to lower tolerance values.
* Scaling the source clip to a higher resolution prior to ColorKeyMask can help
  retain finer details in the mask, like stray hairs or edges.
* Using color correction filters to reduce some of the color spill. This sometimes
  requires multipass masking and layering that target only the troublesome areas.
* And then there is `MaskTools2`_ which includes various filters for dealing
  with masks, as shown in the example above.

It should be noted that ColorKeyMask is a primitive color keying filter, if the
end goal is green screen removal (`chroma keying`_), it may not be the best
option. There is dedicated software that handle this type of task much better,
as discussed in the "`Chaining calls to ColorKeyMask`_" VideoHelp thread.


Changelog
---------

+-----------------+-------------------------------------------------------------+
| Version         | Changes                                                     |
+=================+=============================================================+
| AviSynth+ r2487 || Mask: support RGB64 and all Planar RGBA formats.           |
|                 || ColorKeyMask: support RGB64 and all Planar RGBA formats.   |
|                 || ResetMask: support RGB64 and all Planar RGBA/YUVA formats. |
|                 || ResetMask: new ``mask`` parameter.                         |
|                 || New filters: AddAlphaPlane and RemoveAlphaPlane.           |
+-----------------+-------------------------------------------------------------+
| AviSynth 2.5.8  | ColorKeyMask: Allow independant tolerance levels for each   |
|                 | channel.                                                    |
+-----------------+-------------------------------------------------------------+
| AviSynth 2.0.6  | Added ResetMask filter.                                     |
+-----------------+-------------------------------------------------------------+
| AviSynth 2.0.3  | Added ColorKeyMask filter.                                  |
+-----------------+-------------------------------------------------------------+

$Date: 2022/03/30 15:25:39 $

.. _chroma keying:
    https://en.wikipedia.org/wiki/Chroma_key
.. _Source:
    https://www.pexels.com/photo/banana-peel-on-the-head-of-a-french-bulldog-4587982/
.. _MaskTools2:
    http://avisynth.nl/index.php/MaskTools2
.. _Chaining calls to ColorKeyMask:
    https://forum.videohelp.com/threads/380989-Chaining-calls-to-ColorKeyMask-in-Avisynth