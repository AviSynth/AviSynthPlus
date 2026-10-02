===============
Subtitle / Text
===============

Set of filters to overlay text on a clip:

* `Subtitle`_ adds anti-aliased text to a range of frames. It includes features
  like alpha blending, font spacing, rotation, etc. Only available on Windows.
* `Text`_ is a stripped-down version of the `Subtitle`_ filter. It only works
  with bitmap fonts and the set of features are limited. Available on Windows
  and other non-Windows OSes.

.. _Subtitle:

Subtitle
--------

The ``Subtitle`` filter adds anti-aliased text to a range of frames. All
parameters after text are optional and can be omitted or specified out of order
using the name=value syntax.

The short form (with all default parameters) is useful when you don't really
care what the subtitle looks like as long as you can see it—for example, when
you're using :doc:`StackVertical <stack>` and its ilk to display several
versions of a frame at once, and you want to label them to remember which is which.

.. rubric:: Syntax and Parameters

::

    Subtitle (clip, string text, float "x", float "y", int "first_frame",
              int "last_frame", string "font", float "size", int "text_color",
              int "halo_color", int "align", int "spc", int "lsp", float "font_width",
              float "font_angle", bool "interlaced", string "font_filename", bool "utf8",
              bool "bold", bool "italic", bool "noaa", string "placement", bool "gdi")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: text

    The text to be displayed.

.. describe:: x, y

    Text position. Can be set to -1 to automatically center the text horizontally
    or vertically, respectively. Negative values of ``x`` and ``y`` not equal to
    -1 can be used to move subtitles partially off the screen.

    Default values:

    * ``x`` = 8 if align=1,4,7 or none; -1 if align=2,5,8; or width-8 if align=3,6,9
    * ``y`` = size if align=4,5,6 or none; 0 if align=7,8,9; or height-1 if align=1,2,3

.. describe:: first_frame, last_frame

    The text will be shown starting from frame ``first_frame`` and ending with
    frame ``last_frame``.

    Default: 0, (inputclip.Framecount-1)

.. describe:: font

    Font name; can be the name of any installed Windows font.

    Default: "arial"

.. describe:: size

    Height of the text in pixels, and is rounded to the nearest 0.125 pixel.

    Default: 18

.. describe:: text_color, halo_color

    Colors for font fill and outline respectively.

    These may be any of the `preset colors`_, or specified as hexadecimal
    $AARRGGBB values, similar to HTML--except that they start with $ instead
    of # and the 4th octet specifies the alpha transparency. $\ **00**\rrggbb
    is completely opaque, $\ **FF**\rrggbb is fully transparent. You can
    disable the halo by using FF as the alpha value.

    * See :doc:`Colors <../syntax/syntax_colors>` for more information on
      specifying colors.
    * For YUV clips, the colors are converted from full range to limited range
      `Rec.601`_.

    Default: $00FFFF00, $00000000

.. describe:: align

    Set where the text is placed, based on the numeric keypad layout, as follows:

        .. image:: pictures/subtitle-align-chart.png

        .. image:: pictures/subtitle-align-demo.png

    Default: 7, or top-left. If ``x`` and/or ``y`` are given, text is positioned
    relative to the (``x,y``) location. Note there is no Y-center alignment setting.

.. describe:: spc

    Modify the inter-character spacing. If ``spc`` is less than zero, inter-character
    spacing is decreased; if greater, the spacing is increased. Default is 0:
    use Windows' default spacing.

    This is helpful for trying to match typical fonts on the PC to fonts used in
    film and television credits which are usually wider for the same height or
    to just fit or fill in a space with a fixed per-character adjustment. See
    example below.

    For more information, see the Microsoft documentation of the function
    `SetTextCharacterExtra()`_.

    Default: 0

.. describe:: lsp

    **L**\ine **S**\pacing **P**\arameter; enables *multi-line* text (where "\\n"
    enters a line break). If ``lsp`` is less than zero, inter-line spacing is
    decreased; if greater, the spacing is increased, relative to Windows'
    default spacing. By default, multi-line text disabled.

    In the unlikely event that you want to output the characters "\\n" literally
    in a multi-line text, you can do this by using "\\\\n".

    Since 3.7.4 SubTitle accept real LF (``\r``) or CR LF (``\r\n``) control
    characters for line break, without the need of specifying ``lsp`` parameter.

    String functions which return an LF separated list such as
    :doc:`ListAutoloadDirs <../syntax/syntax_plugins>` can be Subtitle'd directly.

.. describe:: font_width

    Set character width in logical units, to the nearest 0.125 unit.
    Default=0, use Windows' default width. See example below.

    Character width varies, depending on the font face and size, but "Arial" at
    ``size=16`` is about 7 units wide; if ``font_width`` is less than that, (but
    greater than zero), the text is squeezed, and if it is greater, the text is
    stretched. Negative numbers are converted to their absolute values.

    For more information, see the Microsoft documentation of the function
    `CreateFont()`_.

    Default: 0.0

.. describe:: font_angle

    Adjust the baseline angle of text in degrees anti-clockwise to the nearest
    0.1 degree. Default 0, no rotation.

    Default: 0.0

.. describe:: interlaced

    When enabled, reduces flicker from sharp fine vertical transitions on
    interlaced displays. It applies a mild vertical blur by increasing the
    anti-aliasing window to include 0.5 of the pixel weight from the lines
    above and below.

    Default: false

.. describe:: font_filename

    Allows using non-installed font, by giving the font file name. Once it is
    loaded, other **Subtitle** instances can use it.

    Default: ""

.. describe:: utf8

    Allows drawing text encoded in `UTF-8`_. Can be a bit tricky, since AviSynth
    does not support utf8 scripts. But when a unicode script containing non-ansi
    characters is saved as UTF8 without BOM, the text itself can be passed as-is
    and providing the ``utf8=true`` setting.

    Default: false

.. describe:: bold

    | Using bold letters or not
    | Default is true, since bold has better visibility.

    Default: true

.. describe:: italic

    | Using italic letters or not

    Default: false

.. describe:: noaa

    | Disables antialiasing when drawing the text.
    | Setting it true will disable antialiasing.
      Useful when someone would use ``"VCR OSD Mono"`` as-is, without beautifying the outlines,
      as it as mentioned in https://forum.doom9.org/showthread.php?t=184627

    Default: false

.. describe:: placement

    Chroma location hint. Used in subsampled YUV formats, otherwise ignored.
    Valid values are the same as in ``ChromaInPlacement`` and ``ChromaOutPlacement``
    in the :doc:`Convert <convert>` functions.

    Any placement is accepted, and silently mapped to the implemented text chroma
    kernels, the same way as the mask kernels of :doc:`Overlay <overlay>` and
    :doc:`Layer <layer>` (see the kernel table in Overlay's ``placement``):

    Both renderers (``gdi=true`` and ``gdi=false``, the latter same as ``Text``) have
    the same kernel variants: "mpeg2"-like (left-cosited H, centred V; 1-2-1 filter),
    "mpeg1"-like (centred; box average) and "top_left"-like. 4:1:1, 4:4:0 and 4:1:0
    have no "mpeg2"-like kernel. The "top_left"-like kernel is a point sample (a single
    luma sample of the block) with ``gdi=true``; with ``gdi=false`` it is a 1-2-1 filter
    centred on the top-left luma sample on 4:2:0 (3x3) and 4:4:0 (vertical), the same
    as ``"left"`` on 4:2:2/YUY2, and a point sample on 4:1:1/4:1:0.

    E.g. ``"top"`` and ``"bottom"`` use the centred kernel on 4:2:0/4:2:2/4:1:1, and on
    4:4:0 ``"left"`` (``"mpeg2"``) equals ``"center"``. YUY2 is handled like 4:2:2.

    Default (not given or ``"auto"``): the clip's ``_ChromaLocation`` frame property
    (read from frame 0; subsampled YUV incl. YUY2), otherwise the per-format default of ConvertToXXXX:
    ``"left"`` for 4:2:0/4:2:2/4:1:1 (and YUY2), ``"top"`` for 4:4:0, ``"top_left"`` for 4:1:0.

.. describe:: gdi

    | When *true*, text is rendered using the Windows GDI Antialiaser, which
      produces antialiased output and supports TrueType fonts, ``font_filename``,
      ``font_width``, ``font_angle``, ``interlaced`` and ``noaa``.
    | When *false*, the built-in bitmap font (Terminus) is used — same rendering
      as the ``Text`` filter. Faster and cross-platform compatible but limited:
      ``font_filename``, ``font_width``, ``font_angle``, ``interlaced``, ``noaa``
      and ``font`` (font name) are ignored. ``placement`` works with both; only the
      ``"top_left"``-like kernel differs (see ``placement``).

    Default: true


.. _Text:

Text
----

The **Text** filter is a stripped-down version of the `Subtitle`_ filter that
works with bitmap fonts. It includes 2 fonts or can load an external `BDF`_ font
file.

Text treats real LF (``\n``) control characters for line break, without the need 
of specifying ``lsp`` parameter.

.. rubric:: Syntax and Parameters

::

    Text (clip, string text, float "x", float "y", int "first_frame",
          int "last_frame", string "font", float "size", int "text_color",
          int "halo_color", int "align", int "spc", int "lsp", float "font_width",
          float "font_angle", bool "interlaced", string "font_filename", bool "utf8",
          bool "bold", bool "italic", bool "noaa", string "placement", bool "gdi")

.. describe:: clip

    Source clip; all color formats supported.

.. describe:: text

    The text to be displayed.

.. describe:: x, y, first_frame, last_frame

    Same as Subtitle.

.. describe:: font

    | Font name; can be "Terminus" or "Info_h" or path to a BDF font file.
    | "Info_h" is fixed to 10x20 in size.

    Default: "Terminus"

.. describe:: size

    Only applicable when ``font="Terminus"``; valid sizes are
    12, 14, 16, 18, 20, 22, 24, 28, 32.

    Default: 18

.. describe:: text_color, halo_color

    Colors for font fill and outline respectively. Works the same as Subtitle,
    except for the transparency values. There is no transparency for ``text_color``
    and for ``halo_color`` only the following options are available.

    MSB byte of ``halo_color``:

    * FF : semi-transparent box around text without halo.
    * FE : semi-transparent box around text and use ordinary color bytes of ``halo_color``.
    * 00 : use ``halo_color``.
    * 01 - FD : no halo.

    Default: $00FFFF00, $00000000

.. describe:: align, spc, lsp

    Same as Subtitle; ``spc`` not available.

.. describe:: font_width, font_angle, interlaced

    Not available.

.. describe:: font_filename

    Same as Subtitle except it only accepts BDF fonts.

    Default: ""

.. describe:: utf8

    Same as Subtitle. Much more international unicode characters (1354), use
    ``utf8=true`` under Windows.

    Default: false

.. describe:: bold

    Only applicable when ``font="Terminus"``; set to true for bold text.

    Default: false

.. describe:: italic, noaa

    These parameters have no effect in Text. They are provided only to match the parameter list 
    with SubTitle, because on non-Windows systems "Subtitle" is aliased to "Text", so
    each Subtitle parameter must exist in "Text" as well.

.. describe:: placement

    Chroma location hint. Used in subsampled YUV formats, otherwise ignored.
    Valid values are the same as in ``ChromaInPlacement`` and ``ChromaOutPlacement``
    in the :doc:`Convert <convert>` functions.

    Any placement is accepted, and silently mapped to the implemented chroma kernels
    with the kernel table of :doc:`Overlay <overlay>`'s ``placement``: "mpeg2"-like
    (``"left"``, 1-2-1 filter; 4:2:0/4:2:2/YUY2 only), "mpeg1"-like (centred, box average)
    and "top_left"-like (1-2-1 filter centred on the top-left luma sample: 3x3 on 4:2:0,
    vertical on 4:4:0; equals ``"left"`` on 4:2:2/YUY2; point sample on 4:1:1/4:1:0).
    E.g. ``"top"`` and ``"bottom"`` are centred on
    4:2:0/4:2:2/4:1:1, and on 4:4:0 ``"left"`` (``"mpeg2"``) equals ``"center"``.

    Default (not given or ``"auto"``): the clip's ``_ChromaLocation`` frame property
    (read from frame 0; subsampled YUV incl. YUY2), otherwise the per-format default of ConvertToXXXX:
    ``"left"`` for 4:2:0/4:2:2/4:1:1 (and YUY2), ``"top"`` for 4:4:0, ``"top_left"`` for 4:1:0.

.. describe:: gdi

    This parameter has no effect in ``Text``. It is provided only to match the
    parameter list with ``Subtitle``, because on non-Windows systems ``Subtitle``
    is aliased to ``Text``, so each ``Subtitle`` parameter must exist in ``Text``
    as well.

    Default: false

Examples
--------

**Center text** ::

    AviSource("clip.avi")
    Subtitle("Hello world!", align=5)

**Some text in the upper right corner of the clip with specified font, size and
color red** ::

    AviSource("clip.avi")
    Subtitle("Hello world!", font="georgia", size=24, text_color=$ff0000, align=9)

**Prints text on multiple lines without any text halo border.** ::

    BlankClip()
    Subtitle( \
      "Some text on line 1\\nMore text on line 1\n" + \
      "Some text on line 2", \
             lsp=10, halo_color=$ff000000)

It results in: ::

    Some text on line 1\nMore text on line 1
    Some text on line 2

**Prints text on multiple lines, direct control characters** ::

    BlankClip()
    Subtitle( \
      e"Some text on line 1\n" + \
      "Some text on line 2", \
             , halo_color=$ff000000)
    #note: using e-prefixed escape sequence aware string literal

**Use String() to display values of functions.** ::

    AviSource("clip.avi")
    Subtitle("Width=" + String(Width()))

**Using spc and font_width arguments** ::

    ColorBars().KillAudio()
    Subtitle("ROYGBIV", x=-1, y=100, spc=-10, font_width=6)
    Subtitle("ROYGBIV", x=-1, y=150, spc=0,  font_width=0) ## width=default
    Subtitle("ROYGBIV", x=-1, y=200, spc=10, font_width=10)
    Subtitle("ROYGBIV", x=-1, y=250, spc=20, font_width=20)

.. _Subtitle-animated-demo:

**Animated parameter demonstration** ::

    ColorbarsHD(width=640, height=360)
    AmplifyDB(-30)
    Tweak(cont=0.5, sat=0.5)
    ConvertToRGB32(matrix="PC.709")
    Trim(0, 255)
    s1 = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG."
    s3 = "THE QUICK BROWN FOX \nJUMPS OVER \nTHE LAZY DOG."
    minvalue = -32.0
    maxvalue = +128.0
    B = BlankClip(Last, length=15)
    return Animate(Last, 0, 255, "anim_aln", s3, 1.000000, s3, 9.999999)
       \ + B
       \ + Animate(Last, 0, 255, "anim_spc", s1, minvalue, s1, maxvalue)
       \ + B
       \ + Animate(Last, 0, 255, "anim_wid", s1, minvalue, s1, maxvalue)
       \ + B
       \ + Animate(Last, 0, 255, "anim_lsp", s3, minvalue, s3, maxvalue)
       \ + B
       \ + Animate(Last, 0, 255, "anim_ang", s1, -15.0000, s1, 375.0000).FadeOut(15)
    function anim_aln(clip C, string s, float f) {
        return C.Subtitle(s, align=Floor(f), lsp=0)
        \       .Subtitle("align = "+String(Floor(f)),
        \                 x=-1, y=C.Height-42, size=32, text_color=$c0c0c0)
    }
    function anim_spc(clip C, string s, float f) {
        return C.Subtitle(s, align=8, spc=0, text_color=$c0c0c0)
        \       .Subtitle(s, align=5, spc=Round(f))
        \       .Subtitle("spc = "+String(Round(f)),
        \                 align=2, size=32, text_color=$c0c0c0)
    }
    function anim_wid(clip C, string s, float f) {
        return C.Subtitle(s, align=8, font_width=0, text_color=$c0c0c0)
        \       .Subtitle(s, align=5, font_width=Round(f))
        \       .Subtitle("font_width = "+String(Round(f)),
        \                 align=2, size=32, text_color=$c0c0c0)
    }
    function anim_lsp(clip C, string s, float f) {
        return C.Subtitle(s, align=8, lsp=0, text_color=$c0c0c0)
        \       .Subtitle(s, align=5, lsp=Round(f))
        \       .Subtitle("lsp = "+String(Round(f)),
        \                 align=2, size=32, text_color=$c0c0c0)
    }
    function anim_ang(clip C, string s, float f) {
        return C.Subtitle(s, align=5, font_angle=f)
        \       .Subtitle("font_angle = "+String(f, "%03.3f"),
        \                 align=2, size=32, text_color=$c0c0c0)
    }

**UTF8 text indirectly** ::

    Title="Cherry blossom "+CHR($E6)+CHR($A1)+CHR($9C)+CHR($E3)+CHR($81)+CHR($AE)+CHR($E8)+CHR($8A)+CHR($B1)
    Subtitle(Title, utf8=true)

**Using the Text filter** ::

    Text("Terminus", size=20, align=4)
    Text("Info_h", font="Info_h", align=5)
    Text("Load bdf font", font="spleen-12x24.bdf", align=6)


Changelog
---------

+-----------------+--------------------------------------------------------------------------+
| Version         | Changes                                                                  |
+=================+==========================================================================+
| AviSynth+ 3.7.6 || Add ``placement`` parameter to ``Subtitle``: chroma placement for all   |
|                 |  subsampled YUV formats incl. YUY2; ConvertToXXXX ``ChromaInPlacement``  |
|                 |  names and default (``_ChromaLocation``, else format default). Values    |
|                 |  without an own kernel are mapped to the nearest one (same table as      |
|                 |  Overlay/Layer). YUY2 (``gdi=true``) is rendered like 4:2:2 planar: its  |
|                 |  default chroma is now left-sited (was: always centred).                 |
|                 || ``Text``: ``placement`` values without an own kernel are mapped like    |
|                 |  in ``Subtitle`` instead of falling back to ``"left"`` (e.g. ``"top"``   |
|                 |  is rendered centred on 4:2:0/4:2:2/4:1:1). ``"top_left"`` on            |
|                 |  4:2:0/4:4:0 is now a 1-2-1 filter centred on the top-left luma sample,  |
|                 |  on 4:2:2/YUY2 the same as ``"left"`` (was: the ``"left"`` kernel).      |
|                 || Add ``gdi`` parameter to ``Subtitle``. When ``false``, ``Subtitle``     |
|                 |  uses the built-in bitmap font instead of GDI rendering.                 |
|                 || Add ``gdi`` parameter to ``Text`` (accepted but ignored; for API        |
|                 |  compatibility with ``Subtitle``).                                       |
+-----------------+--------------------------------------------------------------------------+
| 3.7.4           | Feature: SubTitle to accept real LF (``\r``) or CR LF (``\r\n``) control |
|                 | characters for line break.                                               |
+-----------------+--------------------------------------------------------------------------+
| AviSynth+ 3.7.3 || Fix: "Text" filter negative x or y coordinates (e.g. 0 instead of -1)   |
|                 || Fix: "Text" filter would omit last character when x<0                   |
|                 || "Text" Fix: ``halo_color`` needs only MSB=$FF and not the exact         |
|                 |   $FF000000 constant for fade                                            |
|                 || "Text" ``halo_color`` allows to have both halo and shaded background    |
|                 || "Text" much nicer rendering of subsampled formats (#308)                |
|                 || Fix: "Subtitle" and "Text" filter to respect the explicitely (#293)     |
|                 |  given coorditanes for y=-1 or x=-1, instead of applying                 |
|                 |  vertical/horizontal center alignment.                                   |
|                 || Fix: "Text" to throw proper error message if the specified font (#293)  |
|                 |  name (e.g. Arial) is not found among internal bitmap fonts.             |
|                 || "Text": add "italic" and "noaa" (unused) parameters                     |
|                 || "SubTitle": add "bold", "italic" and "noaa" parameters                  |
|                 || "Text": add "placement" parameter                                       |
+-----------------+--------------------------------------------------------------------------+
| AviSynth+ 3.7.1 | Fix: "Text" filter would crash when y coord is odd and format has        |
|                 | vertical subsampling.                                                    |
+-----------------+--------------------------------------------------------------------------+
| AviSynth+ 3.6.0 | New "Text" filter.                                                       |
+-----------------+--------------------------------------------------------------------------+
| AviSynth+ r2632 | Fix: Subtitle for Planar RGB/RGBA: wrong text colors.                    |
+-----------------+--------------------------------------------------------------------------+
| AviSynth+ r2487 | Subtitle: new parameters "font_filename" and "utf8".                     |
+-----------------+--------------------------------------------------------------------------+
| AviSynth  2.60  | Position (x,y) can be float (previously int) (with 0.125 pixel           |
|                 | granularity).                                                            |
+-----------------+--------------------------------------------------------------------------+
| AviSynth  2.58  | Added ``font_width``, ``font_angle``, ``interlaced`` parameters and      |
|                 | alpha color blending.                                                    |
+-----------------+--------------------------------------------------------------------------+
| AviSynth  2.57  | Added multi-line text and line spacing parameter.                        |
+-----------------+--------------------------------------------------------------------------+
| AviSynth  2.07  || Added ``align`` and ``spc`` parameters.                                 |
|                 || Setting y=-1 calculates vertical center (alignment unaffected).         |
|                 || Default x and y values dependent on alignment (previously x=8, y=size). |
+-----------------+--------------------------------------------------------------------------+
| AviSynth  1.00  | Setting x=-1 uses horizontal center and center alignment                 |
|                 | (undocumented prior to v2.07)                                            |
+-----------------+--------------------------------------------------------------------------+

$Date: 2025/03/23 22:30:00 $

.. _BDF:
    https://en.wikipedia.org/wiki/Glyph_Bitmap_Distribution_Format
.. _SetTextCharacterExtra():
    https://docs.microsoft.com/en-us/windows/win32/api/wingdi/nf-wingdi-settextcharacterextra
.. _CreateFont():
    https://docs.microsoft.com/en-us/windows/win32/api/wingdi/nf-wingdi-createfonta
.. _preset colors:
    http://avisynth.nl/index.php/Preset_colors
.. _Rec.601:
    https://en.wikipedia.org/wiki/Rec._601
.. _UTF-8:
    https://en.wikipedia.org/wiki/UTF-8
