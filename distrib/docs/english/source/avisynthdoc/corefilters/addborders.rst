==========
AddBorders
==========

Add black or colored borders, increasing frame size. This has several common uses:

* Adjust the `aspect ratio`_ (make a 4:3 clip into 16:9 without stretching)
* :doc:`Splice <splice>` a smaller resolution clip to a larger one without resizing
* Together with :doc:`Crop <crop>`, shift a clip horizontally or vertically – see below.

See also: :doc:`letterbox`, which adds borders without changing frame size.


Syntax and Parameters
---------------------

::

    AddBorders (clip clip, int left, int top, int right, int bottom, int "color", int "color_yuv")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: left, top, right, bottom

    Border width in pixels.

    * For YUV411 sources, left and right must be `mod4`_ (divisible by 4).
    * For YUV420 sources, all four border widths must be `mod2`_ (divisible by 2).
    * For YUV422 sources, left and right must be mod2.

.. describe:: color

    | Specifies the border color; black by default.
    | Color is specified as an RGB value in either hexadecimal or decimal notation.
    | Hex numbers must be preceded with a $. See the
      :doc:`colors <../syntax/syntax_colors>` page for more information on
      specifying colors.

    * For YUV clips, colors are converted from full range to limited range
      `Rec.601`_.

    * Use ``color_yuv`` to specify full range YUV values or a color with a
      different matrix.

    Default: $000000

.. describe:: color_yuv

    Specifies the border color using YUV values. Input clip must be YUV;
    otherwise an error is raised. See the :ref:`YUV colors <yuv-colors>` for
    more information.


Examples
--------

* Add letterbox (top and bottom) borders:

  .. code-block:: c++

    # add dark blue borders, using hex color notation
    AddBorders(0, 86, 0, 86, color=$00008B)

    # same as above, using named preset color
    AddBorders(0, 86, 0, 86, color=color_darkblue)

    # full scale black border using color_yuv hex color notation
    AddBorders(0, 86, 0, 86, color_yuv=$008080)

* Be aware that many older lossy compression algorithms don't deal well with
  solid-color borders, unless the border happens to fall on a `macroblock`_
  boundary (16 pixels for MPEG).

* Use **AddBorders** in combination with **Crop** to *shift* an image without
  changing the frame size:

  .. code-block:: c++

    # Shift an image 2 pixels to the right
    Crop(0, 0, Width-2, Height)
    AddBorders(2, 0, 0, 0)

  * Note, shifting this way must be done in 1- or 2-pixel increments, depending
    on color format.
  * You can shift in sub-pixel increments with :doc:`Resize <resize>`.


Changelog
----------

+-----------------+------------------------------------------------------------------+
| Version         | Changes                                                          |
+=================+==================================================================+
| AviSynth+ 3.6.2 | Fix: AddBorders did not pass frame properties                    |
+-----------------+------------------------------------------------------------------+
| AviSynth+ 3.5.0 | New ``color_yuv`` parameter like in BlankClip                    |
+-----------------+------------------------------------------------------------------+
| AviSynth+ r2397 | AddBorders missing l/r/top/bottom vs. subsampling check for YUVA |
+-----------------+------------------------------------------------------------------+
| AviSynth 2.6.0  | Bugfix: Fixed RGB24 AddBorders with ``right=0``                  |
+-----------------+------------------------------------------------------------------+
| AviSynth 2.0.7  | New ``color`` parameter                                          |
+-----------------+------------------------------------------------------------------+

$Date: 2022/04/17 11:37:04 $

.. _aspect ratio:
    http://avisynth.nl/index.php/Aspect_ratios
.. _mod2:
    http://avisynth.nl/index.php/Modulo
.. _mod4:
    http://avisynth.nl/index.php/Modulo
.. _Rec.601:
    https://en.wikipedia.org/wiki/Rec._601
.. _macroblock:
    https://en.wikipedia.org/wiki/Macroblock
