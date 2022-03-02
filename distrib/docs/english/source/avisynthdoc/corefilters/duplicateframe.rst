
DuplicateFrame
==============

**DuplicateFrame** is the opposite of :doc:`DeleteFrame <deleteframe>`. It
duplicates a set of frames given as a number of arguments. As with
:doc:`DeleteFrame <deleteframe>`, the soundtrack is not modified, so if you use
this filter to duplicate many frames you may get noticeable desynchronization.


Syntax and Parameters
----------------------

::

    DuplicateFrame (clip, int frame [, ...])

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: frame

    Frame(s) to duplicate. Note that frames are numbered starting from zero. If
    needed, use :ref:`ShowFrameNumber` before **DuplicateFrame** to make sure
    the right frames are selected.


Examples
---------

Duplicate frame number 3 twice and duplicate frame 21 and 42 once::


    DuplicateFrame(3, 3, 21, 42) # Add 4 frames


Changelog
---------

+----------------+--------------------------------------+
| Version        | Changes                              |
+================+======================================+
| AviSynth 2.5.8 | Added support for multiple arguments |
+----------------+--------------------------------------+

$Date: 2022/03/02 19:42:53 $
