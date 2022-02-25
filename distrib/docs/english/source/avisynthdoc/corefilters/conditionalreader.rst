
ConditionalReader
=================

Import information from a text file and assign a per-frame value to a script
variable.


Syntax and Parameters
----------------------

::

    ConditionalReader (clip, string filename, string variablename, bool "show")

.. describe:: clip

    Source clip. Not touched, unless you specify ``show=true``.

.. describe:: filename

    Path to the file with the per-frame values you want to set. See
    `File format`_ below.

.. describe:: variablename

    Name of the variable you want the ``filename`` values assigned to.

.. describe:: show

    If *true*, show a text overlay with ``variablename``'s assigned value at the
    current frame.

    Default: false


File format
-----------

* The file is plain text.

  * It is not case sensitive.
  * Each line stands alone.
  * There are several different line types, as explained below.
  * Blank lines are ignored.

* The file may begin with one or more optional *comment* lines:

  Any line is ignored if the first non-whitespace character is '#' (hash), ';'
  (semicolon) or '%' (percent).

  ::

      #this line is a comment
      ;so is this line
           %and this one too.

* The first non-*comment* line should be a *type* line, which must be one of the
  following:

    ``TYPE int``

        | Tells **ConditionalReader** to expect decimal integer values.
        | (Digits ``0-9``, optionally preceded with a ``'+'`` or ``'-'`` sign)

    ``TYPE float``

        | Tells **ConditionalReader** to expect floating-point values.
        | (Decimal number with ``'.'`` decimal point, and optionally followed by
          an ``'E'`` character and decimal exponent)

    ``TYPE bool``

        | Tells **ConditionalReader** to expect boolean values.
        | (``'true'`` or ``'false'``, ``'T'`` or ``'F'``, ``'yes'`` or ``'no'``,
          ``'1'`` or ``'0'``)

    ``TYPE string``

        (Any sequence of characters, including quote, backslash etc. A line
        break ends the string.)

  The *type* line defines the *data* type to be parsed from this file and
  assigned to the variable ``variablename``. You must have one, and only one,
  *type* line per file, and it must come before any other lines, except for
  optional comments.

* A *default* line should come next. It is optional but strongly suggested:

    ``DEFAULT <value>``

        | This specifies the value to be used if there is no applicable *data*
          line for a given frame.
        | If you do not define a default, be sure to specify a value for all
          frames; otherwise your script variable could be |undefined| or
          take a random value.

* An optional *offset* line may appear:

    ``OFFSET <integer-value>``

        When specified, this will add an offset adjustment to all frame numbers
        in the *data* lines below.

* Next come the *data* line(s), which set the per-frame value(s) to be assigned
  to the script variable. There are three styles:

    *Single-frame* style:

            ``<framenumber> <value>``

            Set value for frame *framenumber* only.

    *Range* style:

            ``R <startframe> <endframe> <value>``

            | Apply value to a range of frames.
            | Note that both startframe and endframe are included in the range.

    *Interpolated* style:

            ``I <startframe> <endframe> <start-value> <stop-value>``

            | Interpolate between *start-value* and *stop-value* over a range of
              frames. This only works on *int* and *float* types.
            | Note that both *startframe* and *endframe* are included in the range.

  | Later data lines in the file overrule earlier ones. Styles may be
    mixed-and-matched; see the examples below.
  | All *data* and *default* values must be valid for the defined *type*, as
    defined above, or an error will be raised.


Examples
--------


Basic usage
~~~~~~~~~~~

File *Basic.txt*:
::

    Type float
    Default 3.45567

    R 45 300 76.5654
    2 -671.454
    72 -671.454

The file above will return float values. It will by default return 3.45567.
However, frames 45 to 300 it will return 76.5654. And frame 2 and 72 will
return -671.454.

Later data lines in the file overrule earlier ones. This is illustrated by the
'72' line: even though frame 72 is inside the range of 45-300, frame 72 will use
the value -671.454, not 76.5654. If the 'R' line had been placed after the '72'
line, the range value would have had priority.

A script to invoke this file could be:
::

    ColorBars(512,512)
    Trim(0,500)
    ScriptClip("subtitle(string(myvar))")
    ConditionalReader("file.txt", "myvar", false)

This will put the values into the variable called *"myvar"*, which is used by
:doc:`Subtitle <subtitle>`, invoked by :ref:`ScriptClip` to display the
:doc:`runtime <../syntax/syntax_runtime_environment>` value.

**Note** the **ConditionalReader** line comes *after* any use of *"myvar"* in
the script.


String values
~~~~~~~~~~~~~

File *Strings.txt:*
::

    Type string
    Default
    # (default=empty string)

    R  45  99 this is a string
    R 100 199 "quoted string"
    R 200 299 w:\xyz.txt

Try the above :doc:`Subtitle <subtitle>` script with *Strings.txt*. You will see
the strings displayed verbatim.


Adjusting Overlay
~~~~~~~~~~~~~~~~~

Script:
::

    ColorBars(512,256)
    a1 = Trim(0,600)
    a2 = MessageClip("Text clip")
    Overlay(a1,a2, y = 100, x = 110, mode="subtract", opacity=0, pc_range=true)
    ConditionalReader("opacity.txt", "ol_opacity_offset", false)
    ConditionalReader("xoffset.txt", "ol_x_offset", false)

File *xoffset.txt:*
::

    Type int
    Default -50

    I 25 50 -50 100
    R 50 250 100
    I 250 275 100 250

File *opacity.txt:*
::

    Type float
    Default 0.0

    I 25 50 0.0 1.0
    R 50 250 1.0
    I 250 275 1.0 0.0

It is easier to watch the clip above than completely describe what it does.
Basically, this example defines *keyframes* for an :doc:`overlay` x-offset and
opacity. For frames 25-50 the opacity is scaled from 0.0 to 1.0, while the text
is moving from left to right. The text is then kept steady from frame 50-250,
and thereafter it moves further to the right, while fading out.


.. _complicated-applyrange:

ApplyRange replacement
~~~~~~~~~~~~~~~~~~~~~~

Using a large number of :doc:`ApplyRange <animate>` calls in a script can lead
to resource issues. **ConditionalReader** together with
:doc:`ConditionalFilter <conditionalfilter>` can be used instead, leading to an
efficient solution:

File.txt:
::

    Type Bool
    Default False

    2 True
    R 45 60 True
    72 True
    R 200 220 True
    210 False
    315 True

By default, the script value will be False. However, for frames 2, 45-60, 72,
200-220 and 315, except for 210, it will be True. Later data lines in the file
overrule earlier ones. This is illustrated by frame '210': even though it is
inside the range of 200-220, the later value, False, will be used.

A script to make use of this file could be:
::

    Colorbars(512,512)
    Trim(0,500)
    A=Last
    FlipHorizontal() # Add a complex filter chain
    B=Last
    ConditionalFilter(A, B, "MyVar", "==", "False", false)
    ConditionalReader("File.txt", "MyVar", false)

This will put the values into the variable called *"MyVar"*, which is used by
:doc:`ConditionalFilter <conditionalfilter>` to select between the unprocessed
and flipped version of the source.


Strings values in v2.58
~~~~~~~~~~~~~~~~~~~~~~~

**ConditionalReader** cannot return strings prior to AviSynth v2.60, but one
solution is to create a list of variables with corresponding string assignments,
and |Eval| the indexed solution. For example:

::

    Import("strings.txt")
    ScriptClip("""subtitle(Eval("n"+string(mystringindex)))""")
    ConditionalReader("range_string.txt", "mystringindex")


File *strings.txt*

::

    n0=""
    n1="Intro"
    n2="Main"
    n3="Credits"

File *range_string.txt*

::

    Type int
    Default 0

    R 10 1000 1
    R 1005 3000 2
    R 3200 3800 3

Obviously *strings.txt* does not need to be a separate file, but this solution
is sometimes appropriate in some multilingual applications, e.g., multilingual
applications:

::

    language="spanish"
    Import(language + "_strings.txt")


Changelog
---------
+----------------+----------------------------------+
| Version        | Changes                          |
+================+==================================+
| AviSynth 2.6.0 | Added OFFSET, Added Type=string. |
+----------------+----------------------------------+

$Date: 2022/02/24 20:09:50 $

.. |Eval| replace:: :doc:`Eval <../syntax/syntax_internal_functions_control>`
.. |undefined| replace:: :doc:`undefined  <../syntax/syntax_internal_functions_boolean>`
