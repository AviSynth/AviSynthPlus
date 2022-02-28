
Blur / Sharpen
==============

| **Blur** a simple 3x3 `kernel`_ blurring filter.
| **Sharpen** is 3×3 kernel sharpening filter; the inverse of **Blur**.


Syntax and Parameters
---------------------
::

    Blur (clip, float amount)
    Blur (clip, float amountH, float amountV, bool "mmx")
    Sharpen (clip, float amount)
    Sharpen (clip, float amountH, float amountV, bool "mmx")

.. describe:: clip

    Input clip; all color formats supported.

.. describe:: amount

    Blurring or sharpening strength:

    * **Blur**: the allowable range is from -1.0 to +1.58
    * **Sharpen**: the allowable range  is from -1.58 to +1.0
    * Negative **Blur** actually sharpens the image; in fact ``Sharpen(n)`` is
      just an alias for ``Blur(-n)``.

.. describe:: amountH, amountV

    You can use 2 arguments to set independent vertical and horizontal blurring
    or sharpening, for example:

    * ``Blur(0,1)`` will blur vertical only, perhaps to blend interlaced lines
      together.
    * ``Blur(1,0)`` will blur horizontal only.
    * If *amountV* is not specified, it defaults to *amountH*.

.. describe:: mmx

    **Deprecated** - this parameter is simply ignored.


Notes
-----

If you need a larger radius Gaussian blur, try chaining several Blurs together::

    Blur(1.0).Blur(1.0).Blur(1.0)

Chaining calls to **Sharpen** is not a good idea, as the image quickly deteriorates.


Developer notes
---------------

**Blur** uses the kernel [(1−1/2^amount)/2, 1/2^amount, (1−1/2^amount)/2].
The largest allowable argument for **Blur** is log2(3) (which is about 1.58),
which corresponds to a (1/3,1/3,1/3) kernel. A value of 1.0 gets you a
(1/4,1/2,1/4) kernel for example. Likewise ``Blur(1.0).Blur(1.0)`` is a
convolution of the kernel (1/4,1/2,1/4) with itself, being a
(1/4,1/2,1/4)*(1/4,1/2,1/4) = (1/16,4/16,6/16,4/16,1/16) kernel.
It can be read of `Pascal's triangle`_.


Changelog
----------

+-----------------+---------------------------------------------------------------------------+
| Version         | Changes                                                                   |
+=================+===========================================================================+
| AviSynth+ 3.7.0 | Fix: Blur right side garbage: 16 bit+AVX2+non mod32 width                 |
+-----------------+---------------------------------------------------------------------------+
| AviSynth+ r2664 | Fix: YUY2 Sharpen overflow artifacts - e.g. Sharpen(0.6)                  |
+-----------------+---------------------------------------------------------------------------+
| AviSynth+ r2636 | Fix: Blur/Sharpen crashed when YUY2.width<8, RGB32.width<4, RGB64.width<2 |
+-----------------+---------------------------------------------------------------------------+
| AviSynth+ r2632 | Enhanced: Blur, Sharpen - AVX2 for 8-16 bit planar formats and SSE2 for   |
|                 | 32 bit float formats.                                                     |
+-----------------+---------------------------------------------------------------------------+
| AviSynth+ r2487 || Blur/Sharpen: support for all planar RGB(A)/YUV(A) and RGB32/64 formats. |
|                 || Enhanced: Blur, Sharpen- SSE2/SSE4 FOR 10-16 bits planar and RGB64.      |
|                 || Fix: Blur width=16 (YV12 width=32)                                       |
+-----------------+---------------------------------------------------------------------------+
| AviSynth+ r2290 | Blur/Sharpen: added 16/32 bit support.                                    |
+-----------------+---------------------------------------------------------------------------+
| AviSynth+ r1576 || Blur/Sharpen: ``mmx`` parameter ignored.                                 |
|                 || Blur/Sharpen: filters now have C and SSE2 versions.                      |
+-----------------+---------------------------------------------------------------------------+
| AviSynth 2.6.0  | Blur/Sharpen: added support for Y8, YV411, YV16 and YV24 color formats.   |
+-----------------+---------------------------------------------------------------------------+
| AviSynth 2.5.8  | MMX routines fixed (have full 8 bit precision now); mmx=true by default.  |
+-----------------+---------------------------------------------------------------------------+
| AviSynth 2.5.7  | Added ``mmx`` option.                                                     |
+-----------------+---------------------------------------------------------------------------+

$Date: 2022/02/28 11:37:04 $

.. _kernel:
    http://en.wikipedia.org/wiki/Kernel_(image_processing)
.. _Pascal's triangle:
    https://en.wikipedia.org/wiki/Pascal's_triangle
