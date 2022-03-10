==================
GeneralConvolution
==================
Performs a matrix convolution on any format clip.


Syntax and Parameters
---------------------

::

    GeneralConvolution (clip clip, float "bias", string "matrix", float "divisor", bool "auto",
                        bool "luma", bool "chroma", bool "alpha")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: bias

    Additive bias to adjust the total output intensity.

    Default: 0.0

.. describe:: matrix

    A 3×3, 5×5, 7×7 or 9×9 matrix with 3\ :sup:`2` (9), 5\ :sup:`2` (25),
    7\ :sup:`2` (49), 9\ :sup:`2` (81) float or integer values. Float values are
    converted to integers for 8-16 bit clips.

    Default: "0 0 0 0 1 0 0 0 0"

.. describe:: divisor

    Divides the output of the convolution before adding bias.

    Default: 1.0

.. describe:: auto

    Enables auto scaling. Auto scaling divides the output of the convolution by
    the sum of the elements of the ``matrix``. The value of ``divisor`` is applied
    in addition to this auto scaling factor. If the sum of elements is zero, auto
    scaling is disabled.

    Default: true

.. describe:: luma, chroma, alpha

    Enables processing only selected planes. For RGB clips ``luma`` and ``chroma``
    setting is ignored. Unprocessed planes are simply copied. E.g. ``alpha=false``
    can speed up RGBA/YUVA processing, usually the alpha channel is not used.

    Default: true, true, true

.. note:: The ``divisor`` is usually the sum of the elements of the ``matrix``.
    But when the sum is zero, you can leave ``divisor=1`` and use the bias setting
    to correct the pixel values. The ``bias`` could be useful if the pixel values
    are negative due to the convolution. After adding ``bias``, the pixels are
    clipped to the range 0-255 or the appropriate min-max range of the specific
    bit depth. In 32 bit float formats no clamp happens.

    Around the borders the edge pixels are simply repeated to service the matrix.


Examples
--------

* Blur::

    GeneralConvolution(0, "
       10 10 10 10 10
       10 10 10 10 10
       10 10 16 10 10
       10 10 10 10 10
       10 10 10 10 10 ", 256, False)

* Horizontal (Sobel) edge detection::

    GeneralConvolution(128, "
        1  2  1
        0  0  0
       -1 -2 -1 ", 8)

* Vertical (Sobel) Edge Detection::

    GeneralConvolution(128, "
       1  0 -1
       2  0 -2
       1  0 -1 ", 8)

* Displacement (simply move the position of the "1" for left, right, up, down)::

    GeneralConvolution(0,"
       0 1 0
       0 0 0
       0 0 0 ")

* Displacement by half pixel up (auto scaling)::

    GeneralConvolution(0,"
       0 1 0
       0 1 0
       0 0 0 ")

* Displacement by half pixel right (manual scaling)::

    GeneralConvolution(0,"
       0   0   0
       0 128 128
       0   0   0 ", 256, False)

* Sharpness filter::

    GeneralConvolution(0,"
       0   -1   0
      -1    5  -1
       0   -1   0 ", 1, True)

    # In this case, the new pixel values y(m,n) are given by
    # y(m,n) = (-1*x(m-1,n) - 1*x(m,n-1) + 5*x(m,n) - 1*x(m,n+1)
    #          - 1*x(m+1,n))/(-1-1+5-1-1)/1.0 + 0

* Slight blur filter with black level clipping and 25% brightening::

    GeneralConvolution(-16,"
       0   12   0
      12  256  12
       0   12   0 ", 0.75, True)

    # In this case, the new pixel values y(m,n) are given by
    # y(m,n) = ( 12*x(m-1,n) + 12*x(m,n-1) + 256*x(m,n) + 12*x(m,n+1)
    #          + 12*x(m+1,n) )/(12+12+256+12+12)/0.75 - 16

* Emboss filter (3D relief effect)::

    GeneralConvolution(128, "
    -1 0 0
     0 0 0
     0 0 1")

Some other examples can be found `here`_ and `also here`_.


Changelog
----------

+------------------+--------------------------------------------------------------------------+
| Version          | Changes                                                                  |
+==================+==========================================================================+
| AviSynth+ r2768  || Allow 7x7 and 9x9 matrices (was: 3x3 and 5x5).                          |
|                  || All 8-32 bit formats supported (was: RGB32 only): YUY2 is converted     |
|                  |  to/from YV16, RGB24/32/48/64 are treated as planar RGB internally.      |
|                  || Since 32 bit float input is now possible, ``matrix`` elements and       |
|                  |  ``bias`` parameter are now of float type.                               |
|                  || For 8-16 bit clips the matrix is converted to integer before use.       |
|                  || Allow chroma subsampled formats to have their luma or chroma processed. |
|                  |  E.g. set chroma=false for a YV12 input.                                 |
|                  || New parameters: ``luma, chroma, alpha``.                                |
|                  || MT friendly parameter parsing.                                          |
+------------------+--------------------------------------------------------------------------+
| AviSynth 2.5.5   | Added ``divisor``, ``auto`` parameters.                                  |
+------------------+--------------------------------------------------------------------------+
| AviSynth 2.0.0   | Initial release.                                                         |
+------------------+--------------------------------------------------------------------------+

$Date: 2022/03/10 14:18:26 $

.. _here:
    http://web.archive.org/web/20100105183639/http://www.gamedev.net/reference/programming/features/imageproc/page2.asp
.. _also here:
    https://web.archive.org/web/20120802031716/https://jeanbruenn.info/2011/03/13/avisynths-convolution-stuff-explained/
