================
Separate Filters
================

Set of filters to separate a frame:

* `SeparateFields`_ splits each frame into its component top and bottom fields.
* `SeparateColumns`_  separates the columns of each frame into new frames.
* `SeparateRows`_  Separates the rows of each frame into new frames.

.. _SeparateFields:

SeparateFields
--------------

**SeparateFields** is the opposite of :ref:`Weave`: it takes a frame-based clip
and splits each frame into its component fields, producing a new half height
clip with twice the frame rate and twice the frame count. This is useful if you
would like to use :doc:`Trim <trim>` and similar filters with single-field
accuracy.

**SeparateFields** uses the field-dominance information in the source clip to
decide which of each pair of fields to place first in the output. If it gets
it wrong, use :ref:`ComplementParity`, :ref:`AssumeTFF <AssumeFieldFirst>` or
:ref:`AssumeBFF <AssumeFieldFirst>` before **SeparateFields**.

**SeparateFields** raises an exception if the clip is already field-based.
You may want to use :ref:`AssumeFrameBased <AssumeFrameField>`  to force
separate a second time.

.. rubric:: Syntax and Parameters

::

    SeparateFields (clip)

.. describe:: clip

    Source clip; all color formats supported.

.. _SeparateColumns:

SeparateColumns
---------------
**SeparateColumns** is the opposite of :ref:`WeaveColumns`: it separates the
columns of each frame into *interval* frames. The number of frames of the new
clip is *interval* times the number of frames of the old clip.
**SeparateColumns** is a relatively slow filter due to the sparse pixel picking
required by the algorithm. In some applications it may be faster to use
:doc:`TurnLeft/Right <turn>` with `SeparateRows`_.

.. rubric:: Syntax and Parameters

::

    SeparateColumns (clip, int interval)

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: interval

    | ``interval`` must be less than or equal width and greater than zero.
    | The width of the frame must be a multiple of ``interval``, otherwise
      an error is thrown.

.. _SeparateRows:

SeparateRows
-------------

**SeparateRows** is the opposite of :ref:`WeaveRows`: it separates the rows of
each frame into *interval* frames. The number of frames of the new clip is
*interval* times the number of frames of the old clip. **SeparateRows** like
`SeparateFields`_ is very fast as it uses zero cost subframing to perform it's
magic. ``SeparateRows(2)`` is the same as ``SeparateFields()`` except the output
is frame-based instead of field-based.

.. rubric:: Syntax and Parameters

::

    SeparateRows (clip, int interval)

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: interval

    | ``interval`` must be less than or equal height and greater than zero.
    | The height of the frame must be a multiple of ``interval``, otherwise
      an error is thrown.


Examples
--------

::

    # returns the original clip:
    AviSource("c:\file.avi")
    SeparateColumns(1)

    # returns a clip where the columns are separated:
    # frame 0 consists of the columns 0,3,6,... of the original frame 0
    # frame 1 consists of the columns 1,4,7,... of the original frame 0
    # frame 2 consists of the columns 2,5,8,... of the original frame 0
    # frame 3 consists of the columns 0,3,6,... of the original frame 1
    # etc ...
    AviSource("c:\file.avi")
    SeparateColumns(3)

    # returns a clip where the rows are separated:
    # frame 0 consists of the rows 0,2,4,... of the original frame 0
    # frame 1 consists of the rows 1,3,5,... of the original frame 0
    # frame 2 consists of the rows 0,2,4,... of the original frame 1
    # frame 3 consists of the rows 1,3,5,... of the original frame 1
    # etc ...
    AviSource("c:\file.avi")
    SeparateRows(2)


Changelog
---------

+-----------------+---------------------------------------------------+
| Version         | Changes                                           |
+=================+===================================================+
| AviSynth+ r2487 | Separate filters: added support for 10-16bit,     |
|                 | float, Planar RGB(A)/YUV(A) and RGB48/64 formats. |
+-----------------+---------------------------------------------------+
| AviSynth 2.6.0  | Added SeparateColumns and SeparateRows.           |
+-----------------+---------------------------------------------------+

$Date: 2022/02/27 14:59:41 $
