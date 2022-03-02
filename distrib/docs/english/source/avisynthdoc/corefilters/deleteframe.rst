
DeleteFrame
===========

**DeleteFrame** is the opposite of :doc:`DuplicateFrame <duplicateframe>`.
It deletes a set of frames, given as a number of arguments. The soundtrack is
not modified, so if you use this filter to delete many frames you may get
noticeable desynchronization.

Syntax and Parameters
----------------------

::

    DeleteFrame (clip, int frame [, ...])

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: frame

    Frame number(s) to delete. Note that frames are numbered starting from zero.
    If needed, use :ref:`ShowFrameNumber` before **DeleteFrame** to make sure
    the right frames are selected.


Examples
--------

Delete frames 3, 9, 21 and 42:

::

    DeleteFrame(3, 9, 21, 42)

If you want to delete a range of frames (*a* to *b*, say) along with the
corresponding portion of the soundtrack, you can do it with :doc:`Trim <trim>`,
like this:

::

    Trim(0,a-1) ++ Trim(b+1,0)

Or with :doc:`Loop <loop>`, like this:

::

    Loop(0,a,b)


Changelog
----------

+----------------+--------------------------------------+
| Version        | Changes                              |
+================+======================================+
| AviSynth 2.5.8 | Added support for multiple arguments |
+----------------+--------------------------------------+

$Date: 2022/03/02 19:42:53 $
