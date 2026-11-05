
Limiter
=======

``Limiter`` (clip, float "min_luma", float "max_luma", float "min_chroma", float
"max_chroma", string "show", bool "paramscale")

This filter is present in *v2.5*. The standard known as CCIR-601 defines the
range of pixel values considered legal for presenting on a TV. These ranges
are 16-235 for the luma component and 16-240 for the chroma component.

Pixels outside this range are known to cause problems with some TV sets, and
thus it is best to remove them before encoding if that is your intended
display device. By default this filter clips (or "clamps") pixels under 16 to
16 and over 235 (or 240) to 235 (or 240).

Prior to *v2.53* the (incorrect) default value was 236. Use Limiter(16, 235,
16, 240) for CCIR-601 compliant digital video.

The limits are given in the scale of the clip: integer formats use their own bit
depth (the fractional part is dropped), 32 bit float uses 0..1 for luma and
-0.5..0.5 for chroma. The defaults follow the bit depth.

With ``paramscale=true`` (default: false) the limits are 8 bit levels and are
converted to the bit depth of the clip, so the same script works at all bit
depths: e.g. ``max_luma=235`` is 940 at 10 bit. They can be non-integer: 234.75 is
939 at 10 bit. Integer formats round the result to their own steps.
The conversion always assumes limited range YUV, like :doc:`ConvertBits <convertbits>`
with ``fulls=false, fulld=false``; ``_ColorRange`` is not used.

In  *v2.56*, an option *show* is added. If set, it colors the pixels outside
the specified [min_luma,max_luma] or [min_chroma,max_chroma] range.

*show* can be "luma" (shows out of bounds luma in red/green), "luma_grey"
(shows out of bounds luma and makes the remaining pixels greyscale), "chroma"
(shows out of bounds chroma in yellow), "chroma_grey" (shows out of bounds
chroma and makes the remaining pixels greyscale). The coloring is done as
follows:

| YUY2 (chroma shared between two horizontal pixels p1 and p2: Y1UY2V): j,k=1,2
| YV24 (no chroma shared): j,k=1
| YV12 (chroma shared between 2x2 pixels Y11UY12V; Y21UY22V): j,k=11,12,21,22

+-----------------------------------+--------------+-----------------------+
|                                   | luma         | luma_grey             |
+===================================+==============+=======================+
| Yj < min_luma                     | red (pj)     | red (pj)              |
+-----------------------------------+--------------+-----------------------+
| Yj > max_luma                     | green (pj)   | green (pj)            |
+-----------------------------------+--------------+-----------------------+
| Yj < min_luma and Yk > max_luma   | yellow (all) | puke (pj), olive (pk) |
+-----------------------------------+--------------+-----------------------+

+-----------------------------------+--------------+-----------------------+
|                                   | chroma       | chroma_grey           |
+===================================+==============+=======================+
| U < min_chroma                    | yellow       | yellow                |
+-----------------------------------+--------------+-----------------------+
| U > max_chroma                    | yellow       | blue                  |
+-----------------------------------+--------------+-----------------------+
| V < min_chroma                    | yellow       | cyan                  |
+-----------------------------------+--------------+-----------------------+
| V > max_chroma                    | yellow       | red                   |
+-----------------------------------+--------------+-----------------------+
| U < min_chroma and V < min_chroma | yellow       | green                 |
+-----------------------------------+--------------+-----------------------+
| U > max_chroma and V < min_chroma | yellow       | teal                  |
+-----------------------------------+--------------+-----------------------+
| U < min_chroma and V > max_chroma | yellow       | orange                |
+-----------------------------------+--------------+-----------------------+
| U > max_chroma and V > max_chroma | yellow       | magenta               |
+-----------------------------------+--------------+-----------------------+

+-----------+----------------------------------------------+
| Changelog |                                              |
+===========+==============================================+
| r2728     | Added paramscale; the limits are float.      |
+-----------+----------------------------------------------+
| v2.60     | Added show for YV24.                         |
+-----------+----------------------------------------------+
| v2.56     | added show to show out of bounds luma/chroma |
+-----------+----------------------------------------------+

$Date: 2026/11/05 15:00:00 $
