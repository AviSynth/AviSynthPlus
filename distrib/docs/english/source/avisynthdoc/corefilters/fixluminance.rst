
FixLuminance
============

.. warning::

    This filter is considered broken/deprecated: see `discussion`_

The purpose of this filter is to progressively darken the top of the image; 
for example, to compensate for certain VCRs which make the top of each frame 
brighter than the bottom. In practice it only works for NTSC content and it 
needs its fields separated.

When applying it to an image with height > 255 pixels, you will see a repetitive 
gradient. So it doesn't work for field-separated PAL content. 


Syntax and Parameters
----------------------

::
    
    FixLuminance (clip, int intercept, int slope)

.. describe:: clip

    Source clip; only YUY2 is supported.

.. describe:: intercept

    Bottom line at which to start correction. ``intercept`` should be equal to 
    or smaller than the height of the image. 
        
.. describe:: slope

    | Slope of correction function. The top of the picture is made darker by 1 
      (on a 0-255 scale) every ``slope`` lines. 
    | Maximum darkening (at top line) = ``intercept`` / ``slope`` 


Examples
--------

* Progressively darken top lines:

 ::

    ColorBars(width=512, height=480, pixel_type="YUY2")
    Crop(0, 0, 0, 256)
    FixLuminance(100, 4)

 .. figure:: pictures/fixluminance-colorbars.png
    :align: left

    Line 99 is normal; line 0, the topmost line, is darker by 25 
    (``intercept``/``slope`` = 100/4 = 25) 

* Make a *ramp* test image:

 ::

    BlankClip(width=512, height=256, pixel_type="YUY2", color_yuv=$FF8080)
    FixLuminance(255, 1)

 .. figure:: pictures/fixluminance-ramp.png
    :align: left

$Date: 2022/02/08 22:44:06 $

.. _discussion:
    http://avisynth.nl/index.php/Talk:FixLuminance
