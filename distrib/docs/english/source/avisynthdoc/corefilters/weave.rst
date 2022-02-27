=============
Weave Filters
=============

Set of filters to weave fields, columns and rows:

* `Weave`_ combines fields together to produce interlaced frames.
* `WeaveColumns`_ takes a clip and weaves sets of columns together to produce
  composite frames.
* `WeaveRows`_ takes a clip and weaves sets of rows together to produce
  composite frames.


.. _Weave:

Weave
-----

**Weave** is the opposite of :ref:`SeparateFields`: it takes pairs of fields
from the input video clip and combines them together to produce interlaced
frames. The new clip has half the frame rate and frame count. **Weave** uses
the frame-parity information in the source clip to decide which field to put
on top. If it gets it wrong, use :ref:`ComplementParity` beforehand or
:doc:`SwapFields <swapfields>` afterwards.

All AviSynth filters keep track of field parity, so **Weave** will always join
the fields together in the proper order. If you want to change the order, you'll
have to use :ref:`ComplementParity`, :ref:`AssumeTFF <AssumeFieldFirst>` or
:ref:`AssumeBFF <AssumeFieldFirst>` beforehand or :doc:`SwapFields <swapfields>`
afterwards.

**Weave** raises an exception if the clip is already frame-based. You may want
to use :ref:`AssumeFieldBased <AssumeFrameField>` to force weave a second time.

.. rubric:: Syntax and Parameters

::

    Weave (clip)

.. describe:: clip

    Source clip; all color formats supported.


.. _WeaveColumns:

WeaveColumns
------------

**WeaveColumns** is the opposite of :ref:`SeparateColumns`: it weaves the
columns of *period* frames into a single output frame. The number of frames of
the new clip is the ceiling of the number of frames of the input clip divided
by *period*. **WeaveColumns** is a relatively slow filter due to the sparse
pixel placing required by the algorithm. In some applications it may be faster
to use :doc:`TurnLeft/Right <turn>` with `WeaveRows`_.


.. rubric:: Syntax and Parameters

::

    WeaveColumns (clip, int period)

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: period

    ``period`` must be greater than zero.


.. _WeaveRows:

WeaveRows
---------

**WeaveRows** is the opposite of :ref:`SeparateRows`: it weaves the rows of
period frames into a single output frame. The number of frames of the new
clip is the ceiling of the number of frames of the input clip divided by
*period*. **WeaveRows** is a relatively quick filter, typically costing 1
output frame blit. ``WeaveRows(2)`` is the same as ``Weave()``.


.. rubric:: Syntax and Parameters

::

    WeaveRows (clip, int period)

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: period

    ``period`` must be greater than zero.


Examples
--------

::

    # makes a black and white checkerboard
    # (without changing the spatial position of the rows and columns)

    BlankClip # black

    # frame 0 consists of the rows 0,2,4,... of the original frame 0
    # frame 1 consists of the rows 1,3,5,... of the original frame 0
    # frame 2 consists of the rows 0,2,4,... of the original frame 1
    # frame 3 consists of the rows 1,3,5,... of the original frame 1
    # etc ...
    SeparateRows(2)

    # E1 consists of even frames thus
    # rows 0,2,4,... of the original frame 0
    # rows 0,2,4,... of the original frame 1
    # etc ...
    E1 = SelectEven()

    # O1 consists of the odd frames thus
    # rows 1,3,5,... of the original frame 0
    # rows 1,3,5,... of the original frame 1
    # etc ...
    O1 = SelectOdd

    # likewise for a white clip
    BlankClip(color=$FFFFFF)
    SeparateRows(2)
    E2 = SelectEven()
    O2 = SelectOdd()

    # rows 0,2,4,... of the original black frame 0
    # rows 0,2,4,... of the original white frame 0
    # rows 0,2,4,... of the original black frame 1
    # rows 0,2,4,... of the original white frame 1
    # etc ...
    EI = Interleave(E1, E2)

    # rows 1,3,5,... of the original white frame 0
    # rows 1,3,5,... of the original black frame 0
    # rows 1,3,5,... of the original white frame 1
    # rows 1,3,5,... of the original black frame 1
    # etc ...
    OI = Interleave(O2, O1)

    # alternating black and white columns frame 0
    # alternating black and white columns frame 1
    # etc ...
    E = EI.WeaveColumns(2)

    # alternating white and black columns frame 0
    # alternating white and black columns frame 1
    # etc ...
    O = OI.WeaveColumns(2)

    # alternating black and white columns frame 0
    # alternating white and black columns frame 0
    # alternating black and white columns frame 1
    # alternating white and black columns frame 1
    # etc ...
    Interleave(E, O)

    # weaves the even and odd rows of the original clips
    # with alternating black and white columns
    WeaveRows(2)


Changelog
---------

+-----------------+---------------------------------------------------+
| Version         | Changes                                           |
+=================+===================================================+
| AviSynth+ r2487 | Weave filters: added support for 10-16bit, float, |
|                 | Planar RGB(A)/YUV(A) and RGB48/64 formats.        |
+-----------------+---------------------------------------------------+
| AviSynth 2.6.0  | Added WeaveColumns and WeaveRows.                 |
+-----------------+---------------------------------------------------+

$Date: 2022/02/27 13:38:34 $
