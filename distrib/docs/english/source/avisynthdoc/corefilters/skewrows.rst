========
SkewRows
========

Skews the rows of a clip. It can be used to correct (deskew) problem clips such
as the :doc:`error message <message>` shown below.


Syntax and Parameters
----------------------

::

    SkewRows (clip, int skew)

.. describe:: clip

    Source clip. All :doc:`interleaved <../FilterSDK/InterleavedImageFormat>`
    pixel formats are supported (including Y8).

.. describe:: skew

    Skew amount, can be positive or negative. For YUY2 sources, ``skew`` should
    be even (divisible by 2). The skewing is memory layout based, so RGB images
    are skewed from the bottom up, YUV images are skewed from the top down.

    The effect of the algorithm is to paste all of the input rows together as
    one single very long row, then slice them up based on the new width (= input
    width + ``skew``). When the skew amount is negative, the last skew pixels of
    each line are added to the beginning of the next line and you get some extra
    lines at the end. The last line is padded with grey pixels when required.

    The geometry of the output is calculated as follows:

    * ``OutWidth = InWidth + skew`` // signed skew values acceptable
    * ``OutHeight = (InHeight*InWidth + OutWidth-1) / OutWidth`` // rounded up
      to nearest integer

    In other words, the width and height of the output clip are therefore
    altered slightly.

Examples
---------

.. list-table::

    * - .. figure:: pictures/skewrows_before.png

           (original with skew problem)

    * - .. figure:: pictures/skewrows_after.png

           ``SkewRows(1)``


See also
--------

:doc:`AviSource <avisource>` and :doc:`DirectShowSource <directshowsource>`
pixel_type:

    For planar color formats, adding a '+' prefix, e.g.
    DirectShowSource(..., pixel_type="+YV12"), tells AviSynth the video rows are
    DWORD aligned in memory instead of packed. **This can fix skew or tearing of
    the decoded video when the width of the picture is not divisible by 4.**


Changelog
----------

+----------------+-----------------+
| Version        | Changes         |
+================+=================+
| AviSynth 2.6.0 | Initial release |
+----------------+-----------------+

$Date: 2022/03/17 12:22:43 $
