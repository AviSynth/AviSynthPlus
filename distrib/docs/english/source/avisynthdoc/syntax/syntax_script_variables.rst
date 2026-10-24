
AviSynth Syntax - Script variables
==================================

This page shows how to use *variables* to store intermediate values for
further processing in a script. It also describes the types of data that
scripts can manipulate, and how *literals* (constants) of those types are
written.

A *variable name* can be a character string of practically any length (more
than 4000 characters in Avisynth 2.56 and later) that contains (English)
letters, digits, and underscores (_), but no other characters. The name
cannot start with a digit.

You may use characters from your language system codepage (locale) in strings
and file names (ANSI 8 bit, utf8, but not a 16 bit Unicode).

A variable's placement in an *expression* is determined by the
:doc:`AviSynth Syntax <syntax_ref>`.

Variables can have a value of one of the following *types*:

-   clip

A video clip containing video and / or audio. A script must return a value of
this type.

-   string

A sequence of characters representing text. String literals are written as
text surrounded either by "quotation marks" or by """three quotes""". The
text can contain any characters except the terminating quotation mark or
triple-quote sequence. If you need to put a quotation mark inside a string, 
you need to use `Python-style`_ """three quotes""". For example:
::

    Subtitle("""AVISynth is as they say "l33t".""")

Alternatively, you can use Windows extended-ASCII curly-quotes
inside the string instead of straight quotes to get around this limitation.

Since Avisynth+ 3.6 string containing escaped characters are available.

In a string literal, an escape character is a special character that starts 
with a backslash (\) and is followed by another character or a sequence of 
characters. The escape character modifies the meaning of the following character 
or sequence, and allows you to represent characters that are not normally 
printable or that have special functions.

Converted escape sequences:

-   ``\n`` to LF-Chr(10)
-   ``\r`` to CR-Chr(13)
-   ``\t`` to TAB-Chr(9)
-   ``\0`` to NUL-Chr(0) (NUL is string terminator, use at your own risk)
-   ``\a`` to Chr(7)-audible beep
-   ``\f`` to FF-Chr(12) - Form feed
-   ``\\`` (double \) to Backslash
-   ``\"`` to " (double-quotation mark)
-   ``\'`` to ' (single-quotation mark) (since 3.7.1)
-   ``\b`` to BS-CHR(8) - backspace (since 3.7.1)
-   ``\v`` to VT-CHR(11) - vertical tab (since 3.7.1)

::

    e"Hello \n" will store actual LF (0x0A, 10) control character into the string

A string may end with an escaped backslash: ``e"C:\\Temp\\"`` gives ``C:\Temp\``
(fixed in 3.7.6; earlier versions reported a missing closing quotation mark).

-   int
-   long

Integer is signed 32-bit. Long is signed 64-bit. In general when 'integer' is 
mentioned, it can be any (32 or 64-bit) integers.
64-bit is available since Avisynth version 3.7.4 (3.8?).
An integer literal is entered as a sequence of digits, optionally with a + or - 
at the beginning.

The value can be given in *hexadecimal* by preceding them with a "$" character. 
For example ``$FF`` as well as ``$ff`` (case does not matter) are equal to 255.
Unlike decimal constants, hexadecimal constants are 32 bit by default.
Suffixed by "L" or "l" makes it 64-bit long.


-   float
-   double

Float is a single-precision, double is a 64-bit `floating-point`_ number. 
Literals are entered as a sequence of digits with a decimal point (.) somewhere 
in it and an optional + or -. For example, +1. is treated as a floating-point number. Note that
exponent-style notation is **not** supported.

-   bool

Boolean values must be either *true* or *false*. In addition they can be
written as ''yes'' or ''no'', but you should avoid using these in your
scripts (they remain for compatibility purposes only).

-   array

Array values are supported at script level in Avisynth+.
See also at :doc:`script array concept and helper functions <../script_ref/script_ref_arrays>` .

-   function

Function objects are supported in Avisynth+.
See also at :doc:`Function objects <syntax_function_objects>` .

-   val

A generic type name. It is applicable only inside a
:doc:`user defined script functions <syntax_userdefined_scriptfunctions>` argument list,
in order to be able to declare an argument variable to be of *any* type (int, float, bool, string, or clip). You must
then explicitly test for its type (using the :doc:`boolean functions <syntax_internal_functions_boolean>`) and take
appropriate actions.

There is another type which is used internally by Avisynth - the void or
'undefined' type. Its principal use is in conjunction with optional function
arguments. See the :doc:`Defined() <syntax_internal_functions_boolean>` function.

Variables can be either local (bound to the local scope of the executing
script block) or global. Global variables are bound to the global script
environment's scope and can be accessed by all :doc:`Internal functions <syntax_internal_functions>`,
:doc:`User defined script functions <syntax_userdefined_scriptfunctions>`, :doc:`runtime environment <syntax_runtime_environment>` scripts and the main
script also.

To define and / or assign a value to a global variable you must precede its
name with the keyword ``global`` at the left side of the assignment. The
keyword is not needed (actually it is not allowed) in order to read the value
of a global variable. Examples:

::

    global canvas = BlankClip(length=200, pixel_type="yv12")
    global stroke_intensity = 0.7
    ...
    global canvas = Overlay(canvas, pen, opacity=stroke_intensity, mask=brush)

To declare a variable, simply type the variable name, followed by '=' (an
equals sign), followed by its initial value. The type must not be declared;
it is inferred by the value assigned to it (and can actually be changed by
subsequent assignments). The only place where it is allowed (though not
strictly required) to declare a variable's type is in
:doc:`user defined script functions <syntax_userdefined_scriptfunctions>` argument lists. Examples:

::

    b = false      # this declares a variable named 'b' of type 'bool' and initializes it to 'false'
    x = $100       # type int (initial value is in hexadecimal)
    y = 256        # type int (initial value is in decimal)
    global f = 0.0 # type float declared globally
    ...
    function my_recolor_filter(clip c, int new_color, float amount, val
    "userdata") { ... }

Then since Avisynth+ 3.6.0 exists UseVar. UseVar is special filter, opens a clean variable environment in which only the
variables in the parameter list can be seen.

Changelog
~~~~~~~~~
+----------------+------------------------------------------------------------+
| Version        | Changes                                                    |
+================+============================================================+
| Avisynth 3.7.4 | Added 64-bit decimals                                      |
|                | Added 64-bit floating point                                |
|                | Added "L" suffixed hexadecimal notation                    |
+----------------+------------------------------------------------------------+
| Avisynth 3.6.0 | Added "Usevar"                                             |
|                | Added types: function objects and array                    |
|                | escaped string literal syntax                              |
+----------------+------------------------------------------------------------+


$Date: 2025/02/05 11:11:11 $

.. _Python-style: http://forum.doom9.org/showthread.php?s=&threadid=71597
.. _floating-point: http://en.wikipedia.org/wiki/Floating_point
