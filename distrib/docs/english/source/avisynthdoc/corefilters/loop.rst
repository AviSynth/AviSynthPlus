
Loop
====

Loops the segment from frame *start* to frame *end* a given number of times.


Syntax and Parameters
----------------------

::

    Loop (clip, int "times", int "start", int "end")

.. describe:: clip

    Source clip.

.. describe:: times

    The number of times the loop is applied.

    Setting ``times`` to -1 loops a "very large" number of times.

    Default: -1

.. describe:: start

    The frame of the clip where the loop starts.

    Default: 0

.. describe:: end

    The frame of the clip where the loop ends.

    Default: FrameCount(clip)


Examples
--------

::

    Loop()                           # play the clip (almost) endlessly.
    Loop(times=-1)                   # play the clip (almost) endlessly.
    
    Loop(times=10)                   # play the clip ten times.
    
    Loop(times=-1, start=20, end=29) # play up to frame 19;
                                     # frames 20-29 are repeated (almost) infinite times. 
    
    Loop(times=2, start=20, end=29)  # play up to frame 19; 
                                     # play frames 20-29 two times; 
                                     # continue from frame 30 to end of clip. 
    
    Loop(times=10, start=20, end=29) # play up to frame 19; 
                                     # play frames 20-29 ten times; 
                                     # continue from frame 30 to end of clip. 
    
    Loop(times=0, start=20, end=29)  # play up to frame 19;
                                     # delete frames 20-29 (play them zero times); 
                                     # continue to end of clip. 
    
    Loop(times=1, start=20, end=29)  # play all frames normally. 


$Date: 2022/02/04 22:44:06 $
