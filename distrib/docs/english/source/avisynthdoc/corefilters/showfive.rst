================
ShowFiveVersions
================

**ShowFiveVersions** takes five video streams and combines them in a
staggered arrangement from left to right.


Syntax and Parameters
----------------------

::

    ShowFiveVersions (clip1, clip2, clip3, clip4, clip5)

.. describe:: clip

    Source clips; all 5 required and all must have the same color format, etc.


Examples
---------

The only use for this (that I can think of) is to help find the NTSC pulldown
pattern. You can do this using code like this:

::

    # View all five pulldown patterns at once
    DoubleWeave()
    # put a resizing filter here if necessary (see below)
    a = Pulldown(0,2).Subtitle("0,2")
    b = Pulldown(1,3).Subtitle("1,3")
    c = Pulldown(2,4).Subtitle("2,4")
    d = Pulldown(0,3).Subtitle("0,3")
    e = Pulldown(1,4).Subtitle("1,4")
    ShowFiveVersions(a,b,c,d,e)


This code displays the five pulldown patterns with some text identifying
which is which. You can then look through the movie and pick the pattern which
avoids blending frames. (In ordinary pulldown, there will actually be two
which work equally well. Look at the diagrams in the :doc:`Pulldown <pulldown>`
filter section to see why). If none of the five work properly, then you're
dealing with one of the more perverse forms of pulldown and you might want to
use :doc:`PeculiarBlend <peculiar>`.

If you're planning to capture at a high resolution and then scale down after,
you should place the :doc:`ReduceBy2 <reduceby2>` or :doc:`BilinearResize <resize>`
or whatever just after the :doc:`DoubleWeave <doubleweave>` statement in the
code above. Before :doc:`DoubleWeave <doubleweave>` it won't work correctly,
and if you postpone it any further, **ShowFiveVersions** will produce a *really*
big frame.


Changelog
---------

+----------------+----------------------------------+
| Version        | Changes                          |
+================+==================================+
| AviSynth 2.6.0 | Added Y8, YV16, and YV24 support |
+----------------+----------------------------------+
| AviSynth 2.5.6 | Added YV12 support               |
+----------------+----------------------------------+

$Date: 2022/03/17 22:08:36 $
