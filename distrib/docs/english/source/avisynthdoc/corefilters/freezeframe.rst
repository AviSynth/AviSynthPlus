
FreezeFrame
===========

**FreezeFrame** replaces all the frames between *first_frame* and *last_frame* 
with a copy of *source_frame*. The sound track is not modified. This is useful 
for covering up glitches in a video in cases where you have a similar 
glitch-free frame available.


Syntax and Parameters
^^^^^^^^^^^^^^^^^^^^^

::

    FreezeFrame (clip, int first_frame, int last_frame, int source_frame)

.. describe:: clip

    Source clip.

.. describe:: first_frame, last_frame

    First and last frame to replace (inclusive).

.. describe:: source_frame

    Frame to copy.

.. note::

    Frames are numbered starting from zero. Use :ref:`ShowFrameNumber` before
    **FreezeFrame** to make sure the right frames are selected.

Examples
--------

Replace frames 5 through 15 with frame 16::

    FreezeFrame(4, 14, 15)

Replace frames 1 through 20 with frame 10::

    FreezeFrame(0, 19, 9)

Replace all frames of a clip with the first frame::

    FreezeFrame(0, FrameCount()-1, 0)
    
    # Identical to Trim(0,-1).Loop(FrameCount(),0,0)
    # Except FreezeFrame leaves the audio untouched


$Date: 2022/02/11 22:44:06 $
