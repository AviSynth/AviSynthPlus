==========
ShowFrames
==========

Set of filters to overlay frame numbers, timecodes, and debug information onto a video clip:

* `ShowFrameNumber`_ displays the frame number on each frame.
* `ShowSMPTE`_ displays SMPTE timecodes.
* `ShowTime`_ displays time for the current frame.
* `ShowCRC32`_ computes and displays the CRC32 checksum of selected planes (debug filter).

See the `Examples`_ section for visuals.

.. _ShowFrameNumber:

ShowFrameNumber
---------------
Draws text on every frame indicating what frame number AviSynth sees at that
point in the script. This is sometimes useful when writing scripts. If you
apply additional filters to the clip, they will treat the text on the frame
just as they would treat an image, so the numbers may be distorted by the
time you see them. Sometimes this is what you want, as it shows frame blending etc.

The frame numbers will be drawn on the left side of the clip for *frame-based,
top field first* material; on the right side for *frame-based, bottom field first*
material and alternating on the left side and right for *field-based* material,
depending whether the field is top or bottom. Much more on the topic of
interlaced video on this page: `Interlaced Fieldbased`_.

Due to antialiased text rendering, this filter can be a little slow.

See the :ref:`ShowFrameNumber examples <ShowFrameNumber-examples>` section.

.. rubric:: Syntax and Parameters

::

    ShowFrameNumber (clip, bool "scroll", int "offset", float "x", float "y", string "font", float "size",
                     int "text_color", int "halo_color", float "font_width", float "font_angle",
                     bool "bold", bool "italic", bool "noaa", bool "gdi")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: scroll

    | If *true*, the frame number will be drawn only once on the video and
      scroll from top to bottom;
    | If *false*, it will be drawn on one side, stacked vertically as often
      as it fits.

    Default: false

.. describe:: offset

    Sets the starting frame number.

    Default: 0

.. describe:: x, y

    Text position. Their interpretation corresponds to :doc:`Subtitle's <subtitle>`
    *align=4* and the special meaning of -1 is not available. Note that ``x``
    and ``y`` must be used together or not at all; if they are present, the
    ``scroll`` option is ignored.

    Default: 0.0, 0.0

.. describe:: font

    | Font name; can be the name of any installed Windows font when ``gdi=true``.
    | When ``gdi=false``, only internal bitmap fonts are available (``"Terminus"``
      or ``"info_h"``).

    Default: "Arial" when ``gdi=true``, "Terminus" when ``gdi=false``

.. describe:: size

    Height of the text in pixels, between 1 and 1000000.

    Default 24.

.. describe:: text_color, halo_color

    | Colors for font fill and outline respectively. Default is yellow and black.
    | See :doc:`Subtitle <subtitle>` and the :doc:`colors <../syntax/syntax_colors>`
      page for more information on specifying colors.

    Default: $00FFFFFF, $00000000

.. describe:: font_width

    | Set character width in logical units, to the nearest 0.125 unit.
    | See the example section of :doc:`Subtitle <subtitle>` for an example.

    Default: 0 (use Windows' default width)

.. describe:: font_angle

    Adjust the baseline angle of text in degrees anti-clockwise to the
    nearest 0.1 degree.

    Default: 0.0 (no rotation)

.. describe:: bold

    | Using bold letters or not

    Default: true when ``gdi=true``, false when ``gdi=false``

.. describe:: italic

    | Using italic letters or not

    Default: false

.. describe:: noaa

    | Disables antialiasing when drawing the text

    Default: false

.. describe:: gdi

    | When *true*, text is rendered using the Windows GDI Antialiaser, which
      produces antialiased output but is slower.
    | When *false*, the built-in bitmap font (Terminus) is used. This is faster
      and is the only option on non-Windows platforms.
    | Ignored (forced to *false*) on non-Windows builds.

    Default: true on Windows (with GDI support), false otherwise


.. _ShowSMPTE:

ShowSMPTE
---------

Displays `SMPTE`_-style timecode labels for the current frame. Format is
HH:MM:SS:FF (for example "03:52:39:24" = 3 hours, 52 minutes, 39 seconds and
24 frames).

Frame 0 is marked "00:00:00:00", frame 1 is marked "00:00:00:01" and so on –
unless an *offset* is applied.

Due to antialiased text rendering, this filter can be a little slow.

See the :ref:`ShowSMPTE examples <ShowSMPTE-examples>` section.

.. note::
    With certain exceptions, SMPTE timecode has no concept of fractional frame
    rates (like 24.5 fps for example).

    **ShowSMPTE** source clips must have an *integer* framerate (18, 24, 25, 30,
    31,...) or a *drop-frame* rate ('29.97' being the most common). Supported
    drop-frame rates are listed in the :ref:`table below <ShowSMPTE-table>`. If
    that's not the case an error will be thrown.

    If the framerate is not integral or drop-frame (let's call it "nonstandard"
    for short), use `ShowFrameNumber`_ or `ShowTime`_ instead.

    You may encounter media sources that are almost at a standard framerate,
    but not quite – perhaps due to an error in processing at some point, or
    perhaps the source was something like a security camera or a video game
    console. In this case you should force the clip to the nearest standard
    framerate with :doc:`AssumeFPS <fps>`.

.. rubric:: Syntax and Parameters

::

    ShowSMPTE (clip, float "fps", string "offset", int "offset_f", float "x", float "y", string "font",
               float "size", int "text_color", int "halo_color", float "font_width", float "font_angle",
               bool "bold", bool "italic", bool "noaa", bool "gdi")

.. describe:: clip

    Source clip; all color formats supported. See boxed note above.

.. describe:: fps

    | Not required, unless the current fps can't be used.
    | If used, ``fps`` must be either an integer or a standard drop-frame rate
      as listed in the :ref:`table below <ShowSMPTE-table>`.

    Default: (clip.Framerate)

.. describe:: offset

    Sets the start time. Format is *HH:MM:SS:FF* (for example "03:52:39:24" -
    3 hours, 52 minutes, 39 seconds and 24 frames).

.. describe:: offset_f

    Sets the starting frame number; ignored if ``offset`` is supplied.

    Default: 0

.. describe:: x, y

    | Text position. Their interpretation corresponds to :doc:`Subtitle's <subtitle>`
      *align=2*.

    Default: 0.0, 0.0

.. describe:: font

    | Font name; can be the name of any installed Windows font when ``gdi=true``.
    | When ``gdi=false``, only internal bitmap fonts are available (``"Terminus"``
      or ``"info_h"``).

    Default: "Arial" when ``gdi=true``, "Terminus" when ``gdi=false``

.. describe:: size

    Height of the text in pixels.

    Default 24.

.. describe:: text_color, halo_color

    | Colors for font fill and outline respectively. Default is yellow and black.
    | See :doc:`Subtitle <subtitle>` and the :doc:`colors <../syntax/syntax_colors>`
      page for more information on specifying colors.

    Default: $00FFFFFF, $00000000

.. describe:: font_width

    | Set character width in logical units, to the nearest 0.125 unit.
    | See the example section of :doc:`Subtitle <subtitle>` for an example.

    Default: 0 (use Windows' default width)

.. describe:: font_angle

    Adjust the baseline angle of text in degrees anti-clockwise to the
    nearest 0.1 degree.

    Default: 0.0 (no rotation)

.. describe:: bold

    | Using bold letters or not

    Default: true when ``gdi=true``, false when ``gdi=false``

.. describe:: italic

    | Using italic letters or not

    Default: false

.. describe:: noaa

    | Disables antialiasing when drawing the text

    Default: false

.. describe:: gdi

    | When *true*, text is rendered using the Windows GDI Antialiaser, which
      produces antialiased output but is slower.
    | When *false*, the built-in bitmap font (Terminus) is used. This is faster
      and is the only option on non-Windows platforms.
    | Ignored (forced to *false*) on non-Windows builds.

    Default: true on Windows (with GDI support), false otherwise

Drop-Frame versus Non-Drop-Frame Time Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When television began, it was black-and-white only. At that time NTSC
("American" standard) television ran at 30 frames per second (60 fields per
second). When the television engineers added color, they slowed the frame rate
by the precise ratio 1000/1001, due to `technical reasons`_. NTSC televisions
now run at 30×1000/1001 or approximately 29.97002997 frames per second. This is
commonly called "29.97 fps." 29.97 is the *nominal* framerate, a convenient
shortcut term for 30×1000/1001.

This slight slowing of the framerate complicates the display of timecode.
A second of time no longer consists of a whole number of frames. If the
timecode readout simply advanced the *seconds* counter every 30 frames, the
timecode reading would be slower than clock time by about 3.6 seconds per hour.
Timecode displays cannot show "fractional" frames (their whole purpose is to
uniquely identify every frame) so they `drop`_ the display of just enough frame
numbers to make the displayed timecode correspond to real or clock time. This
is done in a prescribed and repeatable fashion: the first two frame numbers of
every minute, except for the tenth minute, are dropped, ie::

    00:00:00:00, 00:00:00:01, 00:00:00:02, ...,
    00:00:59:29, 00:01:00:02, 00:01:00:03,
    00:01:59:29, 00:02:00:02, 00:02:00:03, ...,
    00:08:59:29, 00:09:00:02, 00:09:00:03, ...,
    00:09:59:29, 00:10:00:00, 00:10:00:01, etc

.. _ShowSMPTE-table:

**ShowSMPTE** automatically assumes `drop-frame timecode`_ given certain input
framerate ranges, as listed in the table below. For example, if the input
framerate is > 29.969 and < 29.971 fps, the framerate is assumed to be
30×1000/1001 for time calculation, and drop-frame counting is used.

    .. table::
        :widths: auto

        +-----------------------------+---------------+--------------+
        | Input fps (bounds excluded) | Assumed rate  | Nominal rate |
        +=============================+===============+==============+
        | 23.975 – 23.977             | 24×1000/1001  | 23.98        |
        +-----------------------------+---------------+--------------+
        | 29.969 – 29.971             | 30×1000/1001  | 29.97        |
        +-----------------------------+---------------+--------------+
        | 47.951 – 47.953             | 48×1000/1001  | 47.95        |
        +-----------------------------+---------------+--------------+
        | 59.939 – 59.941             | 60×1000/1001  | 59.94        |
        +-----------------------------+---------------+--------------+
        | 119.879 - 119.881           | 120×1000/1001 | 119.88       |
        +-----------------------------+---------------+--------------+

You may encounter the term "NDF" - this means "*non-drop-frame*." As you would
expect, this is used for all the integer framerates. Sometimes though, video
running at *drop-frame rates* will have NDF timecode. This is most common for
short-form videos of a few minutes' duration at most: some video professionals
prefer not to skip frame numbers at all, even though the time display will be
off slightly. To get **ShowSMPTE** to show NDF timecode at drop-frame rates, see
the :ref:`examples section below <ShowSMPTE-ndf>`.

.. _ShowTime:

ShowTime
--------

Displays time for the current frame. Format is HH:MM:SS.DDD (for example
"03:52:39.800" = 3 hours, 52 minutes, 39 seconds and 800 milliseconds).

Due to antialiased text rendering, this filter can be a little slow.

See the :ref:`ShowTime examples <ShowTime-examples>` section.

.. rubric:: Syntax and Parameters

::

    ShowTime (clip, int "offset_f", float "x", float "y", string "font", float "size",
              int "text_color", int "halo_color", float "font_width", float "font_angle",
              bool "bold", bool "italic", bool "noaa", bool "gdi")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: offset_f

    Sets the starting frame number. Displayed time will be increased by
    ``offset_f/clip.FrameRate`` seconds.

    Default: 0

.. describe:: x, y

    | Text position. Their interpretation corresponds to :doc:`Subtitle's <subtitle>`
      *align=2*.

    Default: 0.0, 0.0

.. describe:: font

    | Font name; can be the name of any installed Windows font when ``gdi=true``.
    | When ``gdi=false``, only internal bitmap fonts are available (``"Terminus"``
      or ``"info_h"``).

    Default: "Arial" when ``gdi=true``, "Terminus" when ``gdi=false``

.. describe:: size

    Height of the text in pixels.

    Default 24.

.. describe:: text_color, halo_color

    | Colors for font fill and outline respectively. Default is yellow and black.
    | See :doc:`Subtitle <subtitle>` and the :doc:`colors <../syntax/syntax_colors>`
      page for more information on specifying colors.

    Default: $00FFFFFF, $00000000

.. describe:: font_width

    | Set character width in logical units, to the nearest 0.125 unit.
    | See the example section of :doc:`Subtitle <subtitle>` for an example.

    Default: 0 (use Windows' default width)

.. describe:: font_angle

    Adjust the baseline angle of text in degrees anti-clockwise to the
    nearest 0.1 degree.

    Default: 0.0 (no rotation)

.. describe:: bold

    | Using bold letters or not

    Default: true when ``gdi=true``, false when ``gdi=false``

.. describe:: italic

    | Using italic letters or not

    Default: false

.. describe:: noaa

    | Disables antialiasing when drawing the text

    Default: false

.. describe:: gdi

    | When *true*, text is rendered using the Windows GDI Antialiaser, which
      produces antialiased output but is slower.
    | When *false*, the built-in bitmap font (Terminus) is used. This is faster
      and is the only option on non-Windows platforms.
    | Ignored (forced to *false*) on non-Windows builds.

    Default: true on Windows (with GDI support), false otherwise

.. _ShowCRC32:

ShowCRC32
---------

A debug filter that computes the CRC32 checksum of selected planes of each frame and overlays
the result as an uppercase hexadecimal string.

By default (no ``channels`` or ``mode`` given) the result is a single 8-character value
(e.g. ``"AB12CD34"``).  With the ``channels`` and ``mode`` parameters the filter can cover
any combination of planes and either combine them into one CRC or show them side by side.

For **packed formats** (YUY2, RGB24/32/48/64) with default parameters the interleaved
buffer is hashed directly, matching the original pre-3.7.6 behaviour.  When ``channels``
or ``mode`` is specified the clip is internally converted to a planar equivalent first.

For **planar RGB(A)** clips planes are always processed in R, G, B, A logical order,
regardless of the physical plane storage order.

The text placement and scrolling behaviour are identical to `ShowFrameNumber`_.

.. rubric:: Syntax and Parameters

::

    ShowCRC32 (clip, bool "scroll", int "offset", float "x", float "y", string "font", float "size",
               int "text_color", int "halo_color", float "font_width", float "font_angle",
               bool "bold", bool "italic", bool "noaa", string "channels", int "mode",
               int "showmode", bool "gdi")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: scroll

    | If *true*, the checksum will be drawn only once on the video and
      scroll from top to bottom;
    | If *false*, it will be drawn on one side, stacked vertically as often
      as it fits.

    Default: false

.. describe:: offset

    Frame number offset added to the internal counter. A positive value makes the
    display start from a later logical frame; a negative value from an earlier one.

    Default: 0

.. describe:: x, y

    Text position. Their interpretation corresponds to :doc:`Subtitle's <subtitle>`
    *align=4*. ``x`` and ``y`` must be used together or not at all; when present,
    the ``scroll`` option is ignored.

    Default: 0.0, 0.0

.. describe:: font

    | Font name; can be the name of any installed Windows font when ``gdi=true``.
    | When ``gdi=false``, only internal bitmap fonts are available (``"Terminus"``
      or ``"info_h"``).

    Default: "Arial" when ``gdi=true``, "Terminus" when ``gdi=false``

.. describe:: size

    Height of the text in pixels, between 1 and 1000000.

    Default: 24

.. describe:: text_color, halo_color

    | Colors for font fill and outline respectively. Default is yellow and black.
    | See :doc:`Subtitle <subtitle>` and the :doc:`colors <../syntax/syntax_colors>`
      page for more information on specifying colors.

    Default: $00FFFF00, $00000000

.. describe:: font_width

    | Set character width in logical units, to the nearest 0.125 unit.
    | See the example section of :doc:`Subtitle <subtitle>` for an example.

    Default: 0 (use Windows' default width)

.. describe:: font_angle

    Adjust the baseline angle of text in degrees anti-clockwise to the
    nearest 0.1 degree.

    Default: 0.0 (no rotation)

.. describe:: bold

    Using bold letters or not.

    Default: true when ``gdi=true``, false when ``gdi=false``

.. describe:: italic

    Using italic letters or not.

    Default: false

.. describe:: noaa

    Disables antialiasing when drawing the text.

    Default: false

.. describe:: channels

    | Selects which planes to include in the checksum, using their initial letters.
    | For YUV(A) clips: Y, U, V, A.  For RGB(A) clips: R, G, B, A.
    | Letters are not case-sensitive and may appear in any order; unrecognised
      letters are silently ignored.
    | For planar RGB(A) planes are always processed in R, G, B, A order regardless
      of the order given here.

    Default: all planes of the current color space (same set as ``Invert``).

.. describe:: mode

    | Controls how checksums from multiple planes are combined.
    | ``0`` — compute a single CRC32 over all selected planes concatenated in order;
      the result is one 8-character hex value (e.g. ``"AB12CD34"``).
    | ``1`` — compute a separate CRC32 per selected plane; the result shows each
      plane label followed by its value (e.g. ``"Y:AB12CD34 U:12345678 V:ABCDEF01"``).

    Default: 0

.. describe:: showmode

    | Controls whether the CRC result is overlaid as text, stored as a frame property,
      or both.
    | ``0`` — display only (default; original behaviour).
    | ``1`` — display the text *and* store the CRC value(s) as an integer array frame
      property named ``ShowCRC32``.  For ``mode=0`` the array has one element; for
      ``mode=1`` it has one element per active plane, in display order.
    | ``2`` — store the frame property only; no text is drawn on the output.
    | Values are unsigned 32-bit integers stored as ``int64`` (the native array element
      type for AviSynth frame properties).

    Default: 0

.. describe:: gdi

    | When *true*, text is rendered using the Windows GDI Antialiaser, which
      produces antialiased output but is slower.
    | When *false*, the built-in bitmap font (Terminus) is used. This is faster
      and is the only option on non-Windows platforms.
    | Ignored (forced to *false*) on non-Windows builds.

    Default: true on Windows (with GDI support), false otherwise


Examples
--------

.. _ShowFrameNumber-examples:

.. rubric:: `ShowFrameNumber`_

* Default appearance if source is frame-based:

 .. list-table::

    * - .. figure:: pictures/showframenumber-sintel-4592.jpg

        ::

            LSMASHSource("sintel-2048-surround.mp4")
            ShowFrameNumber()

* Draw the frame numbers in red, scrolling from top to bottom, starting with
  "00009"::

    # this is always top field first, therefore numbers will be on the left
    Mpeg2Source("clip.d2v")
    ShowFrameNumber(scroll=true, offset=9, text_color=$ff0000)

.. _ShowSMPTE-examples:

.. rubric:: `ShowSMPTE`_

* Default appearance:

 .. list-table::

    * - .. figure:: pictures/showsmpte-sintel-4592.jpg

        ::

            LSMASHSource("sintel-2048-surround.mp4")
            ShowSMPTE()

* Change the position::

    fontheight=32

    ##bottom center
    ShowSMPTE(size=fontheight)

    ##top center
    ShowSMPTE(size=fontheight, y=(fontheight))

    ##top left
    ShowSMPTE(size=fontheight, x=(fontheight*3), y=(fontheight))

    ##top right
    ShowSMPTE(size=fontheight, x=(Width-fontheight*3), y=(fontheight))

.. _ShowSMPTE-ndf:

* Showing non-drop-frame timecode at drop-frame rates::

    ColorBars()               ## (framerate = 29.97)
    ShowSMPTE(size=24, y=24)  ## timecode (top of screen) is DF (drop-frame)
    C=Last
    AssumeFPS(30)             ## force integer framerate
    ShowSMPTE()               ## timecode (bottom of screen) is NDF
    AssumeFPS(C)              ## fps returned to original
    return Last

    ## DF (top) skips frame numbers at frames 1800, 3598, 106094...
    ## NDF (bottom) does not skip numbers but runs slower than real time
    ##  (frame 106094 = DF "00:59:00:02" == NDF "00:58:56:14")

* Using ``offset``, ``x``, ``y``, ``font``, ``size``, and ``text_color``
  arguments::

    ShowSMPTE(offset="00:00:59:29", x=360, y=576, font="georgia", size=24, text_color=$ff0000)

.. _ShowTime-examples:

.. rubric:: `ShowTime`_

* Default appearance:

 .. list-table::

    * - .. figure:: pictures/showtime-sintel-4592.jpg

        ::

            LSMASHSource("sintel-2048-surround.mp4")
            ShowTime()

.. _ShowCRC32-examples:

.. rubric:: `ShowCRC32`_

* Default behaviour on a packed RGB32 clip — the whole interleaved frame buffer is
  hashed and a single 8-character hex value is overlaid::

    ColorBars(pixel_type="RGB32").ShowCRC32()

* Equivalent explicit call — ``mode=0`` combines all RGBA planes into one CRC.
  For packed RGB32 with explicit parameters the clip is first converted to planar
  RGBA internally, so all four planes are covered::

    ColorBars(pixel_type="RGB32").ShowCRC32(mode=0, channels="RGBA")

* Per-plane mode — display a separate CRC for each of the R, G, B planes side by
  side (e.g. ``"R:AB12CD34 G:12345678 B:ABCDEF01"``)::

    ColorBars(pixel_type="RGB32").ShowCRC32(mode=1, channels="RGB")

* Store CRC values as a frame property without drawing anything on screen, then
  display the property with ``propShow``.  ``channels="RGBA"`` is specified, but
  ``RGBP8`` has no alpha plane — the ``A`` letter is silently ignored, so the
  property array will have three elements (R, G, B) rather than four.  No error
  is raised::

    ColorBars(pixel_type="RGBP8")
    ShowCRC32(mode=1, channels="RGBA", showmode=2)
    propShow(props="ShowCRC32", align=1)

Changelog
---------

+-----------------+-----------------------------------------------------------------------------+
| Version         | Changes                                                                     |
+=================+=============================================================================+
| AviSynth+ 3.7.6 || All four filters: add ``gdi`` parameter to select between Windows GDI      |
|                 |  antialiased rendering (slow, high quality) and the built-in bitmap font    |
|                 |  (fast, cross-platform). Defaults to ``true`` on Windows with GDI.          |
|                 || ``font``, ``bold`` defaults are now ``gdi``-dependent.                     |
+-----------------+-----------------------------------------------------------------------------+
| AviSynth+ 3.7.6 || ``ShowCRC32``: add ``channels``, ``mode``, and ``showmode`` parameters.    |
|                 || ``channels``: select planes individually.                                  |
|                 || ``mode``: combined (0) or per-plane (1) CRC output.                        |
|                 || ``showmode``: display only (0), display + frame property (1), property     |
|                 |  only (2). Frame property name: ``ShowCRC32`` (int64 array).                |
|                 || Planar RGB(A): planes always processed in R, G, B, A order.                |
+-----------------+-----------------------------------------------------------------------------+
| AviSynth+ 3.7.3 | Add ``bold``, ``italic`` and ``noaa``                                       |
+-----------------+-----------------------------------------------------------------------------+
| AviSynth+ 3.7.0 | Added ``ShowCRC32``.                                                        |
+-----------------+-----------------------------------------------------------------------------+
| AviSynth 2.6.0  || All functions: position (x,y) can be float (previously int) (with 0.125    |
|                 |  pixel granularity).                                                        |
|                 || ShowSMPTE: added drop-frame for other framerates (other than 30).          |
+-----------------+-----------------------------------------------------------------------------+
| AviSynth 2.5.8  || Added ShowTime function.                                                   |
|                 || Added ``font_width``, ``font_angle`` args.                                 |
+-----------------+-----------------------------------------------------------------------------+
| AviSynth 2.5.6  | Added ``offset`` and other options.                                         |
+-----------------+-----------------------------------------------------------------------------+

$Date: 2026/05/08 11:57:00 $

.. _Interlaced Fieldbased:
    http://avisynth.nl/index.php/Interlaced_fieldbased
.. _SMPTE:
    http://en.wikipedia.org/wiki/SMPTE_timecode
.. _drop-frame timecode:
    https://web.archive.org/web/20090206001750/http://teched.vt.edu/GCC/HTML/VirtualTextbook/PDFs/AdobeTutorialsPDFs/Premiere/PremiereTimecode.pdf
.. _technical reasons:
    https://web.archive.org/web/20180810220539/http://documentation.apple.com/en/finalcutpro/usermanual/index.html#chapter=D%26section=6%26tasks=true
.. _drop:
    https://en.wikipedia.org/wiki/SMPTE_timecode#Drop-frame_timecode
