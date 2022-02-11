
Echo
====

**Echo** forces getframe accesses to all the input clips (up to the number of 
frames of the first clip). The results from *clip2* and up are discarded. The 
result from *clip1* is returned.

This may be useful forcing alternate script paths with 
:doc:`ImageWriter <imagewriter>` and :doc:`WriteFile <write>`.


Syntax and Parameters
----------------------

::

    Echo (clip1, clip2 [, ...])

.. describe:: clip1, clip2, ...

    Source clip(s).


Examples
--------

::

    clip1 = ColorBars.Trim(0,99).ShowFrameNumber() # numbered clip (640x480)
    clip2 = clip1.SelectEvery(10, 0) # select frame 0, 10, 20, ...
    clip2 = clip2.BilinearResize(160, 120) # create thumbnails of frame 0, 10, 20, ...
    clip2 = clip2.ImageWriter(file="D:\AviSynth\Plugins\file", type="jpg")
    Echo(clip1, clip2) # returns clip1, exports clip2 (the thumbnails)


Changelog
----------

+----------------+-----------------+
| Version        | Changes         |
+================+=================+
| AviSynth 2.6.0 | Initial release |
+----------------+-----------------+

$Date: 2022/02/04 13:38:34 $
