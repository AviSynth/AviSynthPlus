==========
Interleave
==========

*Interleaves* frames from several clips on a frame-by-frame basis, so for
example if you give three arguments, the first three frames of the output video
are the first frames of the three source clips, the next three frames are the
second frames of the source clips, and so on.


Syntax and Parameters
----------------------

::

    Interleave (clip1, clip2 [, ...])

.. describe:: clip1, clip2, ...

    Source clips; all color formats supported. The dimensions of the clips and
    their color formats must be the same. See :doc:`filters with multiple input
    clips <../filters_mult_input_clips>` for the resulting
    :doc:`clip properties <../syntax/syntax_clip_properties>`.


Examples
--------

**Interleave** is very useful for comparing two similar videos, stepping
frame-by-frame::

    Interleave(A.Subtitle("A"), B.Subtitle("B"))



$Date: 2022/03/24 22:44:06 $
