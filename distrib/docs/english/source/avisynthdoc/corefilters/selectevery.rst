
SelectEvery
===========

**SelectEvery** returns a clip with only some of the frames in every 
*step_size* selected. **SelectEvery** is a generalization of filters like 
:doc:`SelectEven <select>` and :doc:`Pulldown <pulldown>`.

Syntax and Parameters
----------------------

::
    
    SelectEvery (clip, int step_size, int offset1 [, int offset2 [, ...]])

.. describe:: clip

    Source clip.
    
.. describe:: step_size

    The number of frames in the pattern.
    
.. describe:: offset

    The offset(s) into the step to select a frame. If omitted it defaults to 0.


Examples
--------

Return even numbered frames, starting with 0::

    SelectEvery(clip, 2, 0) # identical to SelectEven(clip)
    
Return odd numbered frames, starting with 1::

    SelectEvery(clip, 2, 1) # identical to SelectOdd(clip)
    
Select frames 3, 6, 7, 13, 16, 17, ... from source clip::

    SelectEvery(clip, 10, 3, 6, 7)

Select frames 0, 9, 18, ... from source clip::
    
    SelectEvery(clip, 9, 0)
    
Fixed pattern 1 in 5 decimation, first frame in every ``step_size`` removed::

    SelectEvery(clip, 5, 1, 2, 3, 4)

Duplicate every fourth frame::

    SelectEvery(clip, 4, 0, 1, 2, 3, 3)
    
Take a 24fps progressive input clip and apply 3:2 pulldown, yielding a 30fps 
interlaced output clip::

    AssumeFrameBased()
    SeparateFields()
    SelectEvery(8, 0, 1, 2, 3, 2, 5, 4, 7, 6, 7) # standard NTSC-Film telecine pattern
    Weave()

$Date: 2022/02/06 21:28:07 $
