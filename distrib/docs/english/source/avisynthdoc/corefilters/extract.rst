===============
Extract Filters
===============

Set of filters to extract a plane (channel) from the source clip:

* **Extract** family includes 7 filters, each name corresponds to the plane that
  will be extracted.
* **PlaneToY** works differently, the plane to be extracted is defined by a
  parameter.

Resulting clip is Y-only (Y8, Y10 etc. as appropriate). To extract a plane while
keeping the same color format or to cast it to another format use the
:doc:`Show filter family <showalpha>`.


Syntax and Parameters
----------------------

::

    ExtractY (clip)
    ExtractU (clip)
    ExtractV (clip)
    ExtractA (clip)
    ExtractR (clip)
    ExtractG (clip)
    ExtractB (clip)
    PlaneToY (clip, string "plane")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: plane

    The color plane to be extracted.
    If specified plane does not exist, an error is raised.

    Default: "Y"


Examples
--------

For RGB, ExtractR/G/B are identical to::

    PlaneToY("R") # also ShowRed("Y")
    PlaneToY("G") # also ShowGreen("Y")
    PlaneToY("B") # also ShowBlue("Y")

For YUV, ExtractY/U/V are identical to::

    PlaneToY("Y") # also ShowY("Y") and ConvertToY()
    PlaneToY("U") # also ShowU("Y") and UToY8()
    PlaneToY("V") # also ShowV("Y") and VToY8()

For RGBA/YUVA alpha extraction, ExtractA is identical to::

    PlaneToY("A") # also ShowAlpha("Y")


Changelog
---------

+-----------------+---------------------------------------------------------------+
| Version         | Changes                                                       |
+=================+===============================================================+
| AviSynth+ 3.7.1 | ExtractY/U/V/R/G/B/A, PlaneToY: delete _ChromaLocation        |
|                 | property. Set _ColorRange property to "full" if source is     |
|                 | Alpha plane.                                                  |
+-----------------+---------------------------------------------------------------+
| AviSynth+ r2768 || Fix: Allow ExtractY on greyscale clips.                      |
|                 || Fix: ShowY, ShowU, ShowV crash for YUV (non-YUVA) sources.   |
+-----------------+---------------------------------------------------------------+
| AviSynth+ r2331 || Fix: YUY2 PlaneToY finally works.                            |
+-----------------+---------------------------------------------------------------+
| AviSynth+ r1858 || ExtractY, PlaneToY("Y") accepts YUY2 clip.                   |
|                 || ExtractR, ExtractG, ExtractB, ExtractA, and PlaneToY("R"),   |
|                 |  PlaneToY("G"), PlaneToY("B"), PlaneToY("A") functions are    |
|                 |  accepting packed RGB input (RGB24/32/48/64). They are        |
|                 |  converted to planar RGB on-the-fly before plane extraction.  |
+-----------------+---------------------------------------------------------------+
| AviSynth+       | New functions for plane extraction ExtractX. (20161116)       |
+-----------------+---------------------------------------------------------------+

$Date: 2022/03/23 20:11:14 $
