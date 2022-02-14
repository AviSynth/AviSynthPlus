
Letterbox
=========

Fills the top and bottom *rows* of each frame, and optionally the left and right 
*columns*, with black or color. This has several common uses:

* Black out video noise from the existing black bands in an image that's already 
  letterboxed.
* Black out the video noise at the bottom of the frame in `VHS`_ tape sources.
* Black out overscan areas in `VCD`_ or `SVCD`_ sources.
* Create a quick rectangular mask for other filters – a so-called "`garbage matte`_". 

See also: :doc:`AddBorders <addborders>`, which increases frame size. 
**Letterbox** does not change frame size.

The functionality of **Letterbox** can be duplicated with a combination of 
:doc:`Crop <crop>` and :doc:`AddBorders <addborders>`, but **Letterbox** is 
faster and easier.

Generally, it's better to **Crop** video noise off than to black it out; many 
older lossy compression algorithms don't deal well with solid-color borders, 
unless the border happens to fall on a `macroblock`_ boundary (16 pixels for 
MPEG). However, in some cases, particularly for certain hardware players, it's 
better to use **Letterbox** because it lets you keep a standard frame size. 


Syntax and Parameters
----------------------

::

    Letterbox (clip, int top, int bottom, int "x1", int "x2", int "color", int "color_yuv")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: top, bottom

    Number of *top* and *bottom* rows to blank out.

    * For YUV420 sources, top and bottom must be `mod2`_ (divisible by 2).

.. describe:: x1, x2

    Number of *left* (``x1``) and *right* (``x2``) columns to blank out.

    * For YUV422 and YUV420 sources, left and right must be `mod2`_ (divisible by 2).
    * For YUV411 sources, left and right must be `mod4`_ (divisible by 4).

    Default: 0, 0

.. describe:: color

    | Fill color specified as an RGB value in either hexadecimal or decimal 
      notation.
    | Hex numbers must be preceded with a $. See the
      :doc:`colors <../syntax/syntax_colors>` page for more information on 
      specifying colors.

    * For YUV clips, colors are converted from full range (0–255) to limited 
      range (16–235) `Rec.601`_.
    * Use ``color_yuv`` to specify full range YUV values or a color with a 
      different matrix.

    Default: $000000

.. describe:: color_yuv

    | Specifies the fill color using YUV values. Input clip must be YUV.
    | See the :ref:`YUV colors <yuv-colors>` section for more information.


Changelog
----------

+-----------------+---------------------------------------------------------------+
| Version         | Changes                                                       |
+=================+===============================================================+
| AviSynth+ 3.4.1 | Added ``color_yuv`` option.                                   |
+-----------------+---------------------------------------------------------------+
| AviSynth+ r2487 | Added support for RGB48/64 and all Planar RGB(A)/YUV(A) color |
|                 | formats.                                                      |
+-----------------+---------------------------------------------------------------+
| AviSynth 2.0.7  | Added ``color`` option.                                       |
+-----------------+---------------------------------------------------------------+
| AviSynth 2.0.6  | Added optional left and right parameters (``x1`` and ``x2``). |
+-----------------+---------------------------------------------------------------+

$Date: 2022/02/08 11:37:04 $

.. _VHS:
    https://en.wikipedia.org/wiki/VHS
.. _overscan:
    https://en.wikipedia.org/wiki/Overscan#Overscan_amounts
.. _VCD:
    https://en.wikipedia.org/wiki/Video_CD
.. _SVCD:
    https://en.wikipedia.org/wiki/Super_Video_CD
.. _garbage matte:
    https://en.wikipedia.org/wiki/Matte_(filmmaking)#Garbage_and_holdout_mattes
.. _macroblock:
    https://en.wikipedia.org/wiki/Macroblock
.. _mod2:
    http://avisynth.nl/index.php/Modulo
.. _mod4:
    http://avisynth.nl/index.php/Modulo
.. _Rec.601:
    https://en.wikipedia.org/wiki/Rec._601
