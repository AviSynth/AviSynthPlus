===
Bob
===

**Bob** takes a clip and bob-deinterlaces it. This means that it enlarges each
field into its own frame by interpolating between the lines. The top fields are
nudged up a little bit compared with the bottom fields, so the picture will not
actually appear to bob up and down. However, it will appear to "shimmer" in
stationary scenes because the interpolation doesn't really reconstruct the other
field very accurately.

This filter uses :doc:`BicubicResize <resize>` to do its dirty work. See the
`Technical details`_ section for more information.


Syntax and Parameters
---------------------

::

    Bob (clip, float "b", float "c", int "height")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: b, c

    The default values correspond to the Mitchell-Netravali cubic filter. See
    :doc:`BicubicResize <resize>` for more details.

    Default: 1/3, 1/3

.. describe:: height

    Set the height of the output.

    Default: field height * 2


Technical details
-----------------

A bob filter doesn't really move the physical position of a field. It just
puts it back where it started. If you just use :doc:`SeparateFields <separatefields>`
then you have 2 half height frames: line 0 becomes line 0 of frame 0 and line 1
becomes line 0 of frame 1. Thus line 0 and 1 are now in the same place! Bob
now basically resizes each frame by a factor of two but in the first frame
uses the original lines for the even lines and in the second frame uses the
original lines for the odd lines, exactly as is supposed to be. If you just
did a resize vertically by a factor of 2 on each frame after doing a
:doc:`SeparateFields <separatefields>`, then it wouldn't work right because the physical
position of a field moves.

Schematic:
Suppose the lines 0o, 1o, 2o, 3o, ... are original lines and 0i, 1i, 2i, 3i,
... are the interpolated lines.

Start with:

.. table::
    :widths: auto

    +-------------+---------+
    | line number | frame 0 |
    +=============+=========+
    | 0           | 0o      |
    +-------------+---------+
    | 1           | 1o      |
    +-------------+---------+
    | 2           | 2o      |
    +-------------+---------+
    | 3           | 3o      |
    +-------------+---------+

Separate fields:

.. table::
    :widths: auto

    +-------------+---------+---------+
    | line number | frame 0 | frame 1 |
    +=============+=========+=========+
    | 0           | 0o      | 1o      |
    +-------------+---------+---------+
    | 1           | 2o      | 3o      |
    +-------------+---------+---------+

Double size:

.. table::
    :widths: auto

    +-------------+---------+---------+
    | line number | frame 0 | frame 1 |
    +=============+=========+=========+
    | 0           | 0o      | 1o      |
    +-------------+---------+---------+
    | 1           | 1i      | 2i      |
    +-------------+---------+---------+
    | 2           | 2o      | 3o      |
    +-------------+---------+---------+
    | 3           | 3i      | 4i      |
    +-------------+---------+---------+

But this is wrong, because the physical position of the field changed.

Bob does it right:

.. table::
    :widths: auto

    +-------------+---------+---------+
    | line number | frame 0 | frame 1 |
    +=============+=========+=========+
    | 0           | 0o      | 0i      |
    +-------------+---------+---------+
    | 1           | 1i      | 1o      |
    +-------------+---------+---------+
    | 2           | 2o      | 2i      |
    +-------------+---------+---------+
    | 3           | 3i      | 3o      |
    +-------------+---------+---------+


Examples
--------

**To strictly preserve the original fields and just fill in the missing lines.**

``Bob(0.0, 1.0)`` preserves the original fields for RGB and YUV422 and preserves
the Luma but not the Chroma for YUV420.

The filter coefficients with b=0.0 and c=1.0 give you 0 at x=1.0/2.0 and 1 at
x=0. Which with the +/-0.25 shift occurring on the original field locations,
you get a very crisp cubic filter with -1/8 5/8 5/8 -1/8 coefficients on the
x=0.5/1.5 taps for the other field.

However, since the shift on the chroma planes is only 0.125 for YUV420 the taps
don't end up on exactly the same distances. More `discussion`_.

$Date: 2022/03/10 22:35:42 $

.. _discussion: https://forum.doom9.org/showthread.php?p=826073#post826073
