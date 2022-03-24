====================
Animate / ApplyRange
====================

The `Animate`_ function changes the arguments of some other function
dynamically over a range of specified frames.

`ApplyRange`_ is similar to Animate but the arguments of the specified function
remain constant throughout the range of the specified frames.

See the `Examples`_ section.

.. _Animate:

Animate
-------

**Animate** is a meta-filter which evaluates its parameter filter with
continuously varying arguments. This filter will not handle a changing
soundtrack or different output frame sizes.

The ``filtername`` argument can even be **Animate** if you want quadratic rather
than linear interpolation.

.. rubric:: Syntax and Parameters

::

    Animate (clip, int start_frame, int end_frame, string filtername, var start_args, var end_args)


.. describe:: clip

    Source clip, sent to filter ``filtername``.

.. describe:: start_frame, end_frame

    | At frame ``start_frame`` and earlier, the filter is evaluated with the
      arguments given by ``start_args``.
    | At frame ``end_frame`` and later, the filter is evaluated with the
      arguments given by ``end_args``.
    | In between, the arguments are linearly interpolated for a smooth transition.

.. describe:: filtername

    Name of any filter or function accessible to your script.

.. describe:: start_args, end_args

    | Two lists of arguments to ``filtername``. Data types must match. The two
      nested argument lists are not parenthesized.
    | Strings and video clips can't be interpolated, and therefore must be
      identical in the two argument lists.

    An important warning: If you use a clip as the first argument to **Animate**,
    that clip shouldn't be included here. Thus, for example::

        V = Version()
        Animate(V, 0, 149, "Crop", V, 0, 0, 64, 32, V, 316, 0, 64, 32)

    results in an error. The correct script would be::

        V = Version()
        Animate(V, 0, 149, "Crop", 0, 0, 64, 32, 316, 0, 64, 32)

    or alternatively, ::

        V = Version()
        Animate(0, 149, "Crop", V, 0, 0, 64, 32, V, 316, 0, 64, 32)

.. _ApplyRange:

ApplyRange
----------

**ApplyRange** is a special case of `Animate`_ where ``start_args`` = ``end_args``.
It can be used to apply a certain filter only on a certain range of frames of a
clip. Like `Animate`_, this filter will not handle a changing soundtrack or
different output frame sizes.

In cases where a large number of ranges need processing, calling **ApplyRange**
many times may cause resource issues. An alternative is found here:
:ref:`ConditionalReader: ApplyRange replacement <complicated-applyrange>`.

.. rubric:: Syntax and Parameters

::

    ApplyRange (clip, int start_frame, int end_frame, string filtername, var args)

.. describe:: clip

    Source clip, sent to filter ``filtername``.

.. describe:: start_frame, end_frame

    | Frames outside the range ``start_frame`` to ``end_frame`` are passed
      through untouched.
    | Frames inside the range ``start_frame`` to ``end_frame`` (inclusive) are
      processed by filter filtername with arguments ``args``.
      If ``start_frame``\ ==\ ``end_frame``, only one frame is processed.

.. describe:: filtername

    Name of any filter or function accessible to your script.

.. describe:: args

    | List of arguments to ``filtername``. Unlike **Animate**, ``args`` can't
      contain a clip.
    | As with **Animate**, if you use a clip as the first argument to
      **ApplyRange**, that clip shouldn't be included here.


Examples
--------

Animate Examples
^^^^^^^^^^^^^^^^

**Scrolling "Version" video** ::

    ver = Version()
    Animate(ver, 0, 149, "Crop",
    \    0, 0, 64, 32,
    \  316, 0, 64, 32)

**Fade to white** ::

    AviSource("test.avi")
    Animate(100, 200, "Levels",
    \  0, 1, 255,   0, 255,
    \  0, 1, 255, 255, 255)

**Zoom In** ::

    # Do a gradual zoom into the center of a 320x240 video, starting at
    # 1:1 magnification in frame 100 and ending with 4:1 magnification
    # in frame 200:
    clip = AviSource("test.avi")
    Animate(100, 200, "BicubicResize",
    \ clip, 320, 240,   0,  0, 320, 240,
    \ clip, 320, 240, 120, 90,  80,  60)
    # Animate(clip, 100,200,"BicubicResize",
    #\  320,240,0,0,320,240,
    #\  320,240,120,90,80,60) # also works

**Zoom Out** ::

    # Make the text "Hello, World!" zoom out from the center of a 320x240 video:
    BlankClip(width=320, height=240)
    Animate(0,48,"Subtitle",
    \  "Hello, World!", 160, 120, 0, 99999, "Arial", 0,
    \  "Hello, World!",  25, 130, 0, 99999, "Arial", 48)

**Zoom overlay 1** ::

    # Zooming clip c2 while overlaying it on c1:

    function myfunc(clip c1, clip c2, int x, int y, int w, int h)
    {
      w = w - w % 2
      h = h - h % 2
      my_c2 = BicubicResize(c2, w, h)
      Overlay(c1, my_c2, x, y)
    }

    c1 = AviSource("c1.avi") # c1 is larger than c2
    c2 = AviSource("c2.avi").BicubicResize(320,240)
    Animate(0, 1000, "myfunc",
    \  c1, c2,  10,  10,  10,  10,
    \  c1, c2, 300, 300, 360, 288)
    # or
    # Animate(c1,0,1000,"myfunc",
    #\  c2, 10, 10, 10, 10,
    #\  c2,300,300,360,288)

    # but the following doesn't work, since three clips
    # are passed to myfunc (c1, c1 and c2), while only two are allowed:
    # Animate(c1,0,1000,"myfunc",
    #\  c1,c2, 10, 10, 10, 10,
    #\  c1,c2,300,300,360,288)

**Zoom overlay 2** ::

    # A small picture enlarges on a black clip until replace the main clip:

    function res(clip clip, clip "LClip", int "width", int "height",
    \           int "centerX", int "centerY") {
        LClip = BicubicResize(LClip, width, height)
        Overlay(clip, LClip, centerX-LClip.Width/2, centerY-LClip.Height/2)
    }

    function resize(clip clip, clip "LClip",
    \               int "start_frame", int "start_width", int "start_height",
    \               int "end_frame", int "end_width", int "end_height",
    \               int "centerX", int "centerY") {
        return Animate(start_frame, end_frame, "res",
            \       clip, LClip, start_width, start_height, centerX, centerY,
            \       clip, LClip, end_width, end_height, centerX, centerY)
    }

    clip = AviSource("test.avi")
    clip = clip.ConvertToRGB()
    clip = clip.BicubicResize(640,480)
    black = BlankClip(clip)

    resize(black, clip,
    \      0, 120, 120*clip.Height/clip.Width,
    \      500, 640, 480,
    \      clip.Width/2, clip.Height/2)

See also, :ref:`Subtitle: Animated parameter demonstration <subtitle-animated-demo>`


ApplyRange Examples
^^^^^^^^^^^^^^^^^^^

::

    ver = Version()
    return ver.ApplyRange(0, 149, "Crop", 158, 0, 64, 32)
    # gives an error since cannot have different frame sizes within a clip

::

    Version()
    ApplyRange(100, 149, "Blur", 1.0) # Blur only frames 100-149

::

    AviSource("test.avi").BicubicResize(320,240)
    ApplyRange(0, 48, "Subtitle", "Hello, World!", 25, 130, 0, 99999, "Arial", 48)

    # is the same as:
    clip = AviSource("test.avi").BicubicResize(320,240)
    ApplyRange(clip, 0, 48 "Subtitle", "Hello, World!", 25, 130, 0, 99999, "Arial", 48)

    # since the frame range can be provided to Subtitle itself, this is the same as:
    AviSource("test.avi").BicubicResize(320,240)
    Subtitle("Hello, World!", 25, 130, 0, 48, "Arial", 48)


Changelog
---------

+-----------------+-----------------------------------------------+
| Version         | Changes                                       |
+=================+===============================================+
| AviSynth 2.5.3  || Added ApplyRange filter.                     |
|                 || Added support for audio, and ``start_frame`` |
|                 |  can be equal to ``end_frame``.               |
+-----------------+-----------------------------------------------+

$Date: 2022/03/24 15:10:22 $
