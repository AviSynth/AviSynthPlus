==============
Resize Filters
==============

The resize filters scale the input video frames to an arbitrary new resolution,
and can optionally crop the frame before resizing with sub-pixel precision.

The following resizers are included:

* **BicubicResize** is similar to **BilinearResize**, except that instead of a
  linear filtering function it uses the `Mitchell–Netravali`_ two-part cubic.
  The parameters ``b`` and ``c`` can be used to adjust the properties of the
  cubic; they are sometimes referred to as "blurring" and "ringing" respectively.

  If you are enlarging your video, you will get sharper results with
  **BicubicResize** than with **BilinearResize**. However, if you are shrinking
  it, you may prefer BilinearResize as it performs some `antialiasing`_.

* **BilinearResize** uses standard `bilinear filtering`_ and is a good choice
  for smoothing overly sharp sources.

* **BlackmanResize** is a modification of **LanczosResize** that has better
  control of `ringing`_ artifacts for high numbers of ``taps``.

* **GaussResize** uses a `gaussian`_ resizer, which unlike the bicubics, does
  not overshoot – but perhaps does not appear as sharp to the eye.

* **LanczosResize** is a sharper alternative to **BicubicResize**. It is NOT suited
  for low bitrate video; the various Bicubic flavours are much better for this.

* **Lanczos4Resize** is a short hand for ``LanczosResize(taps=4)``. It produces
  sharper images than **LanczosResize** with the default ``taps=3``, especially
  useful when upsizing a clip.

* **PointResize** is the simplest resizer possible. It uses a Point Sampler or
  `Nearest Neighbour`_ algorithm, which usually results in a very "blocky" image.
  So, in general this filter should only be used, if you intend to have inferior
  quality, or you need the clear pixel drawings. Useful for magnifying small
  areas for examination.

* **SincResize** uses the truncated sinc function. It is very sharp, but prone
  to `ringing`_ artifacts.

* **Spline16Resize**, **Spline36Resize** and **Spline64Resize** are three
  `Spline based`_ resizers. They are the (cubic) spline-based resizers from
  `Panorama tools`_ that fit a spline through the sample points and then derives
  the filter kernel from the resulting blending polynomials. See `this thread`_
  for the technical details.

  The rationale for Spline is to be as sharp as possible with less ringing
  artifacts than **LanczosResize** produces. **Spline16Resize** uses √16 or 4
  sample points, **Spline36Resize** uses √36 or 6 sample points, etc  ... The
  more sample points used, the more accurate the resampling. Several resizer
  comparison pages are given in the `External Links`_ section.

  * **Spline64Resize** may be the most accurate of the Resize filters. [`Dersch`_]
  * **Spline16Resize** is sharper and rings just a bit (which may be desirable
    with soft sources), and looks pleasing to the eye when enlarging or reducing
    in moderate amounts. [`Doom9`_]
  * **Spline36Resize** is somewhere in between the other two.

As with any resampling, there are trade-offs to be considered between preservation
(or augmentation) of image detail and possible artifacts (i.e., oversharpening).


Syntax and Parameters
----------------------

::

    BicubicResize (clip, int target_width, int target_height, float "b", float "c",
                   float "src_left", float "src_top", float, "src_width", float "src_height")

    BilinearResize (clip, int target_width, int target_height,
                    float "src_left", float "src_top", float "src_width", float "src_height")

    BlackmanResize (clip, int target_width, int target_height,
                    float "src_left", float "src_top", float "src_width", float "src_height", int "taps")

    GaussResize (clip, int target_width, int target_height,
                 float "src_left", float "src_top", float "src_width", float "src_height", float "p")

    LanczosResize (clip, int target_width, int target_height,
                   float "src_left", float "src_top", float "src_width", float "src_height", int "taps")

    Lanczos4Resize (clip, int target_width, int target_height,
                    float "src_left", float "src_top", float "src_width", float "src_height")

    PointResize (clip, int target_width, int target_height,
                 float "src_left", float "src_top", float "src_width", float "src_height")

    SincResize (clip, int target_width, int target_height,
                float "src_left", float "src_top", float "src_width", float "src_height", int "taps")

    Spline16Resize (clip, int target_width, int target_height,
                    float "src_left", float "src_top", float "src_width", float "src_height")

    Spline36Resize (clip, int target_width, int target_height,
                    float "src_left", float "src_top", float "src_width", float "src_height")

    Spline64Resize (clip, int target_width, int target_height,
                    float "src_left", float "src_top", float "src_width", float "src_height")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: target_width, target_height

    Width and height of the returned clip.

.. describe:: b, c

    Parameters for **BicubicResize** only.

    The default for both ``b`` and ``c`` is 1/3, which were recommended by
    Mitchell and Netravali for having the most visually pleasing results.

    Set [``b`` + 2\ ``c`` = 1] for the most numerically accurate filter. This
    gives, for ``b=0``, the maximum value of 0.5 for ``c``, which is the
    `Catmull-Rom spline`_ and a good suggestion for sharpness.

    Larger values of ``b`` and ``c`` can produce interesting op-art effects –
    for example, try ``(b=0, c= -5.0)``.

    As ``c`` exceeds 0.6, the filter starts to `"ring"`_ or overshoot. You won't
    get true sharpness – what you'll get is exaggerated edges. Negative values
    for ``b`` (although allowed) give undesirable results, so use ``b=0`` for
    values of ``c`` > 0.5.

    With ``(b=0, c=0.75)`` the filter is the same as `VirtualDub's "Precise Bicubic"`_.

    | **BicubicResize** may be the most visually pleasing of the Resize filters
      for downsizing to half-size or less. `Doom9 [2]`_
    | Try the default setting, ``(b=0, c=0.75)`` as above, or ``(b= -0.5, c=0.25)``.

    Default: 1/3, 1/3

.. describe:: src_left, src_top

    See `Cropping`_ section below.

    Cropping of the left and top edges respectively, in pixels, before resizing.

    Default: 0.0, 0.0

.. describe:: src_width, src_height

    See `Cropping`_ section below.

    As with :doc:`Crop <crop>`, these arguments have different functionality,
    depending on their value:

    * If  > zero, these set the **width** and **height** of the clip before resizing.
    * If <= zero, they set the cropping of the **right** and **bottom** edges
      respectively, before resizing.

    Note, there are certain limits:

    * clip.Width must be >= (``src_left`` + **width**)
    * clip.Width must be >  (``src_left`` + **right**)
    * clip.Height must be >= (``src_top`` + **height**)
    * clip.Height must be >  (``src_top`` + **bottom**)

    ...otherwise it would enlarge ("un-crop") the clip, or reduce width or height
    to 0, which is not allowed.

    Default: source width, source height

.. describe:: taps

    Parameters for **BlackmanResize, LanczosResize, and SincResize** only.

    Basically, taps affects sharpness. Equal to the number of filter `lobes`_
    (ignoring mirroring around the origin).

    Note: the input argument named taps should really be called "lobes". When
    discussing resizers, "taps" has a different meaning, as described below:

    “So when people talk about Lanczos2, they mean a 2-lobe Lanczos-windowed
    sinc function. There are actually 4 lobes -- 2 on each side...

    For upsampling (making the image larger), the filter is sized such that the
    entire equation falls across 4 input samples, making it a 4-tap filter. It
    doesn't matter how big the output image is going to be - it's still just 4
    taps. For downsampling (making the image smaller), the equation is sized so
    it will fall across 4 *destination* samples, which obviously are spaced at
    wider intervals than the source samples. So for downsampling by a factor of
    2 (making the image half as big), the filter covers 8 input samples, and
    thus 8 taps. For 3X downsampling, you need 12 taps, and so forth.

    The total number of taps you need for downsampling is the downsampling
    ratio times the number of lobes, times 2. And practically, one needs to
    round that up to the next even integer. For upsampling, it's always 4 taps.”
    `Don Munsil (avsforum post)`_ | `mirror`_.

    Range:

    * 1-100 for **BlackmanResize** and **LanczosResize**
    * 1-20 for **SincResize**

    Default:

    * 3 for **LanczosResize**
    * 4 for **BlackmanResize** and **SincResize**

.. describe:: p

    Parameter for **GaussResize** only.

    Sharpness. Range from about 1 to 100, with 1 being very blurry and 100 being
    very sharp.

    Default: 30.0

.. _resize-cropping:

Cropping
--------

* All resizers have an expanded syntax which **crops** the frame before resizing::

    BilinearResize(100, 150, src_left=10, src_top=10, src_width=200, src_height=300)

 ...or more succinctly::

    BilinearResize(100, 150, 10, 10, 200, 300)

* The operations are the same as if you put :doc:`Crop <crop>` before the Resize::

    Crop(10, 10, 200, 300).BilinearResize(100, 150)

* The cropping parameters are all :doc:`floating point <../syntax/syntax_script_variables>`.
  This allows any **Resize** filter to be used as a sub-pixel shifter. [`IanB`_]

* **PointResize** cannot do subpixel shifting because it uses only integer pixel
  coordinates.

* Note that :doc:`Crop <crop>` gives a hard boundary, whereas the **Resize**
  filters interpolate pixels outside the cropped region – depending on the
  resizer kernel – bilinear, bicubic etc, and not beyond the edge of the image.

* As a general rule,
    * :doc:`Crop <crop>` any hard borders or noise; **Resize** cropping may
      propagate the noise into the output.
    * Use **Resize** cropping to maintain accurate edge rendering when excising
      a part of a complete image.

* Negative cropping is allowed; this results in repeated edge pixels as shown
  below::

    FFImageSource("resize-sintel-6291.jpg")
    BilinearResize(Width, Height, -32, -32, Width, Height)

 .. list-table::

     * - .. figure::  pictures/resize-sintel-6291.jpg

            Original

       - .. figure:: pictures/resize-sintel-6291-shift.jpg

            Repeated edge pixels


Examples
--------

* Cropping::

    Crop(10, 10, 200, 300).BilinearResize(100, 150)

 which is nearly the same as::

    BilinearResize(100, 150, 10, 10, 200, 300)

* Load a video file and resize it to 240x180 (from whatever it was before)::

    AviSource("video.avi").BilinearResize(240,180)

* Load a 720x480 (`Rec. 601`_) video and resize it to 352x240 (`VCD`_),
  preserving the correct aspect ratio::

    AviSource("dv.avi").BilinearResize(352, 240, 8, 0, 704, 480)

 which is the same as::

    AviSource("dv.avi").BilinearResize(352, 240, 8, 0, -8, -0)

* Extract the upper-right quadrant of a 320x240 video and zoom it to fill the
  whole frame::

    BilinearResize(320, 240, 160, 0, 160, 120)


Notes
-----

* AviSynth has completely separate vertical and horizontal resizers. If input is
  the same as output on one axis, that resizer will be skipped. The resizer with
  the smallest downscale ratio is called first; this is done to preserve maximum
  quality, so the second resizer has the best possible picture to work with.
  :doc:`Data storing <../FilterSDK/DataStorageInAviSynth>` will have an impact on
  what `mods`_ should be used for sizes when resizing and cropping; see
  :ref:`Crop Restrictions <crop-restrictions>`.


External Links
--------------

* `AviSynth resize filter comparison`_ (hermidownloads.craqstar.de)
* `Upscaling in AviSynth – Comparison of resizers`_ (jeanbruenn.info)
* `Testing Interpolator Quality`_ (Helmut Dersch, Technical University Furtwangen)
* `Discussion of resizers for downsizing`_ (doom9.org)
* `Resampling guide`_ (guide.encode.moe)


Changelog
---------

+-----------------+---------------------------------------------------------------+
| Version         | Changes                                                       |
+=================+===============================================================+
| AviSynth+ r2768 | Resizers: don't use crop at special edge cases to avoid       |
|                 | inconsistent results across different parameters/colorspaces. |
+-----------------+---------------------------------------------------------------+
| AviSynth+ r2664 | AVX2 resizer possible access violation in extreme resizes     |
|                 | (e.g. 600->20)                                                |
+-----------------+---------------------------------------------------------------+
| AviSynth+ r2632 || Fix: Resizers for 32 bit float rare random garbage on right  |
|                 |  pixels (simd code NaN issue)                                 |
|                 || Completely rewritten 16bit and float resizers, much faster   |
|                 |  (and not only with AVX2)                                     |
|                 || 8 bit resizers: AVX2 support.                                |
+-----------------+---------------------------------------------------------------+
| AviSynth+ r2487 || Added support for RGB48/64, Planar RGB 8/16/Float formats.   |
|                 || Added support for Alpha in planar RGBA and YUVA formats.     |
+-----------------+---------------------------------------------------------------+
| AviSynth+ r2290 | Added support for 16/32 bit YUV formats (C routine only).     |
+-----------------+---------------------------------------------------------------+
| AviSynth+ r1858 | Fix: RGB resizers shift horizontally to the opposite          |
|                 | direction when ``src_left`` param is used.                    |
+-----------------+---------------------------------------------------------------+
| AviSynth 2.6.0  | Added ``SincResize``.                                         |
+-----------------+---------------------------------------------------------------+
| AviSynth 2.5.8  | Added ``BlackmanResize, Spline64Resize``.                     |
+-----------------+---------------------------------------------------------------+
| AviSynth 2.5.6  || Added ``Spline16Resize, Spline36Resize, GaussResize``.       |
|                 || Added ``taps`` parameter in LanczosResize.                   |
|                 || Added offsets in Crop part of xxxResize.                     |
+-----------------+---------------------------------------------------------------+
| AviSynth 2.5.5  | Added ``Lanczos4Resize``.                                     |
+-----------------+---------------------------------------------------------------+


$Date: 2022/03/07 15:10:22 $

.. _Mitchell–Netravali:
    http://en.wikipedia.org/wiki/Mitchell%E2%80%93Netravali_filters
.. _antialiasing:
    http://en.wikipedia.org/wiki/Spatial_anti-aliasing#Examples
.. _bilinear filtering:
    http://en.wikipedia.org/wiki/Bilinear_filtering
.. _ringing:
    http://en.wikipedia.org/wiki/Ringing_artifacts
.. _gaussian:
    http://en.wikipedia.org/wiki/Gaussian_filter
.. _Nearest Neighbour:
    http://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
.. _Spline based:
    http://en.wikipedia.org/wiki/Spline_interpolation
.. _Panorama tools:
    http://panotools.sourceforge.net/
.. _this thread:
    http://forum.doom9.org/showthread.php?t=147117
.. _Dersch:
    http://web.archive.org/web/20060827184031/http://www.path.unimelb.edu.au/~dersch/interpolator/interpolator.html
.. _Doom9:
    http://forum.doom9.org/showthread.php?p=1689519#post1689519
.. _Catmull-Rom spline:
    http://en.wikipedia.org/wiki/Cubic_Hermite_spline#Catmull.E2.80.93Rom_spline
.. _"ring":
    http://en.wikipedia.org/wiki/Ringing_artifacts
.. _VirtualDub's "Precise Bicubic":
    http://www.virtualdub.org/blog/pivot/entry.php?id=95
.. _Doom9 [2]:
    http://forum.doom9.org/showthread.php?t=172871&page=2
.. _lobes:
    http://en.wikipedia.org/wiki/Lanczos_resampling#Lanczos_kernel
.. _Don Munsil (avsforum post):
    https://www.avsforum.com/threads/lanczos-vs-bicubic-comparison.460922/page-2#post-4760581
.. _mirror:
    http://avisynth.nl/index.php/Lanczos_lobs/taps
.. _IanB:
    http://forum.doom9.org/showpost.php?p=938102&postcount=2
.. _Rec. 601:
    http://en.wikipedia.org/wiki/Rec._601
.. _VCD:
    http://en.wikipedia.org/wiki/Video_CD
.. _AviSynth resize filter comparison:
    http://web.archive.org/web/20090422150849/http://hermidownloads.craqstar.de/videoresizefiltercomparasion/
.. _mods:
    http://avisynth.nl/index.php/Modulo
.. _Upscaling in AviSynth – Comparison of resizers:
    http://web.archive.org/web/20140207171106/http://jeanbruenn.info/2011/10/30/upscaling-in-avisynth-comparison-of-resizers/
.. _Testing Interpolator Quality:
    http://web.archive.org/web/20060827184031/http://www.path.unimelb.edu.au/~dersch/interpolator/interpolator.html
.. _Discussion of resizers for downsizing:
    http://forum.doom9.org/showthread.php?t=172871
.. _Resampling guide:
    https://guide.encode.moe/encoding/resampling.html
