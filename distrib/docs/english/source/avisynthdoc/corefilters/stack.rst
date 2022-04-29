===============================
StackHorizontal / StackVertical
===============================

**StackHorizontal** takes two or more video clips and displays them together
in left-to-right order.

**StackVertical** is similar, working top-to-bottom.


Syntax and Parameters
----------------------

::

    StackHorizontal (clip1, clip2 [, ...])
    StackVertical (clip1, clip2 [, ...])

.. describe:: clip1, clip2, ...

    Source clips; all color formats supported.

    * The color formats must be the same for all clips.
    * **StackHorizontal**: the height must be the same for all clips.
    * **StackVertical**: the width must be the same for all clips.

    Most properties (soundtrack, frame rate, etc) are taken from the first clip,
    see :doc:`filters with multiple input clips <../filters_mult_input_clips>`
    for the resulting :doc:`clip properties <../syntax/syntax_clip_properties>`.


Examples
--------

Compare frames with and without noise reduction::

    StackVertical(last, last.SpatialSoften(2,3,6))


Show clips in variables a,b,c,d in a box like this::

    # a b
    # c d
    StackVertical(StackHorizontal(a,b),StackHorizontal(c,d))


$Date: 2022/04/29 21:28:07 $
