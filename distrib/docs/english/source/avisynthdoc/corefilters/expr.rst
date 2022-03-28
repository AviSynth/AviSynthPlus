
Expr
====
Applies a `mathematical function`_, defined by an *expression* string, on the
pixels of the source clip(s). A different expression may be set for each color
channel. Users of `MaskTools2`_ may be familiar with this concept.

Syntax and Parameters
---------------------

::

    Expr (clip clip[, ...], string exp[, ...],
          string "format", bool "optAvx2", bool "optSingleMode", bool "optSSE2",
          string "scale_inputs", bool "clamp_float", bool "clamp_float_UV", int "lut")

.. describe:: clip

    One or more source clips. Up to 26 input clips can be specified.

    * The first three clips are referenced by lowercase letter x, y and z; use
      'a', 'b' ... 'w' for the rest.
    * Clips may be YUV(A), RGB(A), or greyscale; 8-16 bit integer or 32 bit float.
    * Width, height and `chroma subsampling`_ should be the same; bit depths can
      be different.

.. describe:: exp

        One or more `RPN`_ expressions.

        * A different expression may be set for each color channel (or plane).
          Plane order is Y-U-V-A or R-G-B-A. (note: due to a bug, versions prior
          to r2724 used GBRA ordering).
        * When an expression string is not given, the previous one is used.
        * The empty string (``""``) is a valid expression; it causes the plane
          to be copied (see `Expressions`_ below).
        * Keyword delimiters are space, but TAB, CR and LF characters are allowed
          as whitespace as well since 3.7.1.

.. describe:: format

    Set color format of the returned clip.

    * Use `pixel format strings`_ like "YV12", "YUV420P8", "YUV444P16", "RGBP10".
    * By default, the output format is the same as the first clip.

    Default: ""

.. describe:: optAvx2

    Enables or disables `AVX2`_ code generation if available. Do nothing if AVX2
    is not supported in AviSynth. False disables AVX2.

    Default: auto

.. describe:: optSingleMode

    If true, generate assembly code using only one XMM/YMM register set
    instead of two.

    * **Expr** generates assembly code that normally uses two 128 (SSE2) or 256
      bit (AVX2) registers ("lanes"), thus processing 8 (SSE2)/16 (AVX2) pixels
      per internal cycle.
    * Experimental parameter, ``optSingleMode=true`` makes the internal compiler
      generate instructions for only one register (4/8 pixels - SSE2/AVX2). The
      parameter was introduced to test the speed of x86 code using one working
      register. Very-very complex expressions would use too many XMM/YMM registers
      which are then "swapped" to memory slots, that could be slow. Using
      ``optSingleMode=true`` may result in using less registers with no need
      for swapping them to memory slots.

    Default: false

.. describe:: optSSE2

    Enables or disables `SSE2`_ code generation when in non-AVX2 mode. Setting
    ``optSSE2=false`` and ``optAVX2=false`` forces expression processing in a
    slow interpreted way (C language). False disables SSE2.

    Default: auto

.. describe:: scale_inputs

    Autoscale any input bit depths to 8-16 bit integer or 32 bit float for
    internal expression use, the conversion method is either full range
    (stretch) or limited YUV range (like bit shift). Feature is similar to the
    one in MaskTools2 v2.2.15. The primary reason of this feature is the "easy"
    usage of formerly written expressions optimized for 8 bits.

        +----------------+--------------------------------------------------------------+
        | Option         | Description                                                  |
        +================+==============================================================+
        | ``"int"``      | Scales limited range videos, only integer formats (8-16bits) |
        |                | to 8 (or bit depth specified by 'i8'..'i16' and 'f32')       |
        +----------------+--------------------------------------------------------------+
        | ``"intf"``     | Scales full range videos, only integer formats (8-16bits)    |
        |                | to 8 (or bit depth specified by 'i8'..'i16' and 'f32')       |
        +----------------+--------------------------------------------------------------+
        | ``"float"`` or | Only scales 32 bit float format to 8 bit                     |
        | ``"floatf"``   | range (or bit depth specified by 'i8'..'i16' and 'f32')      |
        +----------------+--------------------------------------------------------------+
        | ``"floatUV"``  | Chroma pre and post shift by 0.5 for 32 bit float pixels,    |
        |                | thus having them in the range of 0..1 instead of -0.5..+0.5  |
        |                | during Expr evaluation (since v3.5)                          |
        +----------------+--------------------------------------------------------------+
        | ``"all"``      | Scales videos to 8 (or bit depth specified by 'i8'..'i16'    |
        |                | and 'f32') - conversion uses limited_range logic             |
        |                | (mul/div by two's power)                                     |
        +----------------+--------------------------------------------------------------+
        | ``"allf"``     | Scales videos to 8 (or bit depth specified by 'i8'..'i16'    |
        |                | and 'f32') - conversion uses full scale logic (stretch)      |
        +----------------+--------------------------------------------------------------+
        | ``"none"``     | No magic (default)                                           |
        +----------------+--------------------------------------------------------------+

    * E.g. ``scale_inputs="float"`` will automatically convert 32 bit float
      input to 8 bit range (but keeps the floating point precision).
    * The default 8 bit target range can be overridden by the i10 .. i16 or f32
      specifiers at the beginning of the expression string.
    * The script inside will treat the clip as a 8 bit one. This only affects
      the internal calculations, the output is properly scaled back.
    * Note: ``ymin, ymax, cmin, cmax, range_min, range_max, range_half`` and
      ``range_size`` internal variables are changed accordingly (this behaviour
      was fixed in AviSynth+ > r2900).

    Default: "none"

.. describe:: clamp_float

    If true: clamps 32 bit float to valid ranges, which is 0..1 for luma or for
    RGB color space and -0.5..0.5 for YUV chroma UV channels.

    * Until 3.4: Ignored when scale_inputs scales 32bit-float type pixels.
    * From 3.5: not ignored, even when parameter ``"scale_inputs"`` auto-scales
      32 bit float type pixels to integer.

    Default: false (as usual, 32 bit float pixels are not clamped)

.. describe:: clamp_float_UV

    This parameter affects clamping of chroma planes: chroma is clamped between
    0..1.0 instead of -0.5..0.5.

    Default: false (as usual, 32 bit float pixels are not clamped)

.. describe:: lut

    LUT (Look-up Table) mode. LUT is precalculated table. Expression values
    are calculated for all pixel value combinations in advance. Then in each
    frame the resulting pixel value is 'looked up' from the ready-made table,
    which is indexed by the actual (x) or (x,y) pixel value. Added in v3.7.1.

        +--------+-----------------------------------------------------------+
        | Option | Description                                               |
        +========+===========================================================+
        | ``0``  | Realtime calculation (default).                           |
        +--------+-----------------------------------------------------------+
        | ``1``  | 1D LUT (lutx)                                             |
        |        |                                                           |
        |        | - 1D luts are available for 8-16 bit inputs. An 8 bit 1D  |
        |        |   lut needs 256 byte memory. A 16 bit 1D lut needs 65536  |
        |        |   2-byte-words (131072 bytes).                            |
        +--------+-----------------------------------------------------------+
        | ``2``  | 2D lut (lutxy)                                            |
        |        |                                                           |
        |        | - 2D luts are available for 8-14 bit inputs. Note: a 14   |
        |        |   bit 2D lut needs (2^14)*(2^14)*2 bytes buffer in memory |
        |        |   per plane (~1GByte).                                    |
        +--------+-----------------------------------------------------------+

        .. note::

            **Caveats**

            * When lut is not available for a given bit depth then Expr will silently
              fallback to realtime (lut=0) mode.
            * In 1D or 2D lut mode some keywords and features are forbidden in the
              expression: sx, sy, sxr, syr, frameno, time, relative pixel addressing.
            * Frame property access works, but is limited to frame #0 which is read
              before LUT evaluation.
            * In lut mode the input clip's bit depths must be the same.

    Default: 0


Expressions
------------

**Expr** accepts 1 to 26 **source clips**, up to four **expression** strings
(one per color plane), an optional output format string, and some debug options.
Output video format is inherited from the first clip, when there is no format
override. All clips have to match in their width, height and `chroma subsampling`_.

Expressions are evaluated on each plane, Y, U, V (and A) or R, G, B (,A). When
an expression string is not specified, the previous expression is used for that
plane – except for plane A (alpha) which is copied by default. When an expression
is an empty string (``""``) then the relevant plane will be copied (if the output
clip bit depth is similar). When an expression is a single clip reference letter
("x") and the source/target bit depth is similar, then the relevant plane will
be copied. When an expression is constant (after constant folding), then the
relevant plane will be filled with an optimized memory fill method.

* Example: ``Expr(clip, "255", "128, "128")`` fills all three planes.
* Example: ``Expr(clip, "x", "range_half, "range_half")`` copies luma, fills U
  and V with 128/512/... (bit depth dependent).

Other optimizations: do not call GetFrame for input clips that are not referenced
or plane-copied.

Expressions are written in `RPN`_.

Expressions use 32 bit float precision internally.

For 8..16 bit formats output is rounded and clamped from the internal 32 bit
float representation to valid 8, 10, ... 16 bits range. 32 bit float output is
not clamped at all.

Expr language/RPN elements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Clips: letters *x, y, z, a..w. x* is the first clip parameter, *y* is the
  second one, etc.
* Math: ``* / + -``
* ``%`` (modulo), like fmod. Example: ``result = x - trunc(x/d)*d``. Note: the
  internal 32-bit float can hold only a 24 bit integer number (approximately).
* Math constant: ``pi``
* Functions: ``min, max, sqrt, abs, exp, log, pow ^`` (synonyms: ``pow`` and ``^``)
* Function: ``neg`` simple negates stack top
* Function: ``sgn`` simple signum function -1 if x<0; 0 when x==0; 1 if x>0
* Function: ``clip`` three operand function for clipping. Example: ``x 16 240
  clip`` means min((max(x,16),240)
* Functions: ``sin cos atan2 tan asin acos atan`` |br| On Intel x86/x64 the
  functions ``sin``, ``cos`` and ``atan2`` have SSE2/AVX2 optimization, the others
  have not (they make the whole expression to evaluate without SIMD optimization).
* Functions: ``round, floor, ceil, trunc`` operators (nearest integer - banker's
  rounding, round down, round up, round to zero). |br| On Intel builds acceleration
  requires at least SSE4.1 capable processor or else the whole expression is
  running in C mode.
* Logical: ``> < = >= <= and or xor not == & | !=`` (synonyms: ``==`` and ``=``,
  ``&`` and ``and``, ``|`` and ``or``)
* Ternary operator: ``?`` - Example: ``x 128 < x y ?``
* Duplicate stack elements: ``dup, dupn`` (dup1, dup2, ...)
* Swap stack elements: ``swap, swapn`` (swap1, swap2, ...)
* Scale by bit shift: ``scaleb`` (operand is treated as being a number in 8 bit
  range unless i8..i16 or f32 is specified).
* Scale by full scale stretch: ``scalef`` (operand is treated as being a number
  in 8 bit range unless i8..i16 or f32 is specified).


Bit-depth aware constants
^^^^^^^^^^^^^^^^^^^^^^^^^

* ``ymin, ymax`` (ymin_a .. ymin_z for individual clips) - the usual luma limits
  (16..235 or scaled equivalents).
* ``cmin, cmax`` (cmin_a .. cmin_z) - chroma limits (16..240 or scaled equivalents)
* ``range_half`` (range_half_a .. range_half_z) - half of the range, (128 or scaled
  equivalents).
* ``range_size`` (range_size_a .. range_size_z , etc..) - 256, 1024, 4096, 16384,
  65536 for integer formats, 1.0 for 32 bit float formats.
* ``range_min, range_max`` (range_min_a .. range_min_z) - chroma/luma plane aware
  constants for the actual min-max limits.
* ``yrange_min, yrange_half, yrange_max`` - Unlike the luma/chroma plane adaptive
  "range_min", "range_half", "range_max" these constants always report the luma
  (Y) values. Since v3.5.

When the constant name is ended with _x, _y, _z, _a, etc.. the constant is brought
from the specified clip (input clips can be of different formats) When by using
parameter ``"scale_inputs"`` the input is converted to e.g. 8 bits, these constants
are calculated for this internally used bit depth.

Keywords for modifying base bit depth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``i8, i10, i12, i14, i16, f32`` (used with ``scaleb`` and ``scalef``)

Spatial input variables in expr syntax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``sx, sy`` (absolute x and y coordinates, 0 to width-1 and 0 to height-1)
* ``sxr, syr`` (relative x and y coordinates, from 0 to 1.0)

Frame property input
~~~~~~~~~~~~~~~~~~~~

* Frame properties (Integer or 32 bit float type) can be accessed within the
  expressions.
* ``clipletter.frameProperty`` syntax injects actual frame property values into
  expression.
* Example: ``x._ColorRange`` will push the value of ``_ColorRange`` frame
  property on stack (of clip x).

Internal variables
~~~~~~~~~~~~~~~~~~

    **User variables**

    * Variables can freely be used during evaluation for storing and loading
      intermediate results within the expression.
    * Variable names must begin with an English letter (a to z, A to Z) or with
      _ (underscore), followed by one or more underscore, letters or digits.
      For example A, X2, _myvar, aa etc.. (until 3.7.1 only uppercase A to Z
      names were allowed).
    * Variables names are case sensitive and cannot be already reserved words.

    **Actions with variables**

    * Store: ``varname@``
        Actual stack top is assigned to a variable.
    * Store and pop from stack: ``varname^``
        Actual stack top is assigned to a variable, then is immediately removed
        from stack top. Use case: when the value assigned to the variable won't
        be used immediately.
    * Variables can be used by simply giving their names: ``varname``
        The actual content of the variable is pushed onto the stack top.
    * Example: ``"x y - A^ x y 0.5 + + B^ A B / C@ x +"``

    **Special predefined variables**

    * ``frameno`` : use current frame number in expression.
      ``0 <= frameno < clip_frame_count.`` |br| A 32 bit integer converted to
      float, so it is precise only at approximately 24 bits.
    * ``time`` : ``calculation: time = frameno/clip_frame_count``. Use relative
      time position in expression. ``0 <= time < frameno/clip_frame_count``
    * ``width, height``: clip width and clip height

Pixel addressing
~~~~~~~~~~~~~~~~

Indexed, addressable source clip pixels by relative x,y positions.

    Syntax: ``x[a,b]`` where

    * ``'x'`` : source clip letter a..z
    * ``'a'`` : horizontal shift. -width < a < width
    * ``'b'`` : vertical shift. -height < b < height

    ``'a'`` and ``'b`` should be constant. e.g.:
    ``"x[-1,-1] x[-1,0] x[-1,1] y[0,-10] + + + 4 /"``

| When a pixel would come from off-screen, the pixels are cloned from the edge.
| Optimized version of indexed pixels requires SSSE3, and no AVX2 version is
  available. Non-SSSE3 falls back to C for the whole expression.

Auto-scale inputs with "scale_inputs"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Autoscale works by converting any input bit depths to a common 8-16 integer or
32 bit float bit format for internal expression use, the conversion method is
either full range or limited YUV range. Feature is similar to the one in
MaskTools2 v2.2.15.

The primary reason of this feature is the "easy" usage of formerly written
expressions optimized for 8 bits.

Possible values for ``scale_inputs``:

* ``"int"`` : scales limited range videos, only integer formats (8-16bits) to 8
  (or bit depth specified by 'i8'..'i16','f32')
* ``"intf"`` : scales full range videos, only integer formats (8-16bits) to 8
  (or bit depth specified by 'i8'..'i16','f32')
* ``"float"`` or ``"floatf"`` : only scales 32 bit float format to 8 bit range
  (or bit depth specified by 'i8'..'i16','f32')
* ``"floatUV"`` : chroma pre and post shift by 0.5 for 32 bit float pixels, thus
  having them in the range of 0..1 instead of -0.5..+0.5 during Expr evaluation
  (since v3.5)
* ``"all"`` : scales videos to 8 (or bit depth specified by 'i8'..'i16','f32')
  - conversion uses limited_range logic (mul/div by two's power)
* ``"allf"`` : scales videos to 8 (or bit depth specified by 'i8'..'i16','f32')
  - conversion uses full scale logic (stretch)
* ``"none"`` : no magic

Usually limited range is for normal YUV videos, full scale is for RGB or
known-to-be-fullscale YUV.

| By default the internal conversion target is 8 bits, so old expressions written
  for 8 bit videos will probably work.
| This internal working bit-depth can be overwritten by the i8, i10, i12, i14,
  i16 and f32 specifiers.

| When using autoscale mode, ``scaleb`` and ``scalef`` keywords are meaningless
  for 8-16 bits, because there is nothing to scale. However, 32 bit (float)
  values will be scaled when ``"float"``, ``"floatUV"``, ``"all"``, ``"allf"``
  is specified.
| Different conversion methods cannot be set for converting before and after the
  expression. Neither can you specify different methods for distinct input clips
  (e.g. x is full, y is limited is not supported).

How it works:

* 8-32 bit inputs ar all scaled to a common bit depth value, which bit depth is
  8 by default and can be set to 10, 12, 14, 16 or 32 bits by the 'i10'..'i16',
  'f32' keywords.

| For example: ``scale_inputs="all"`` converts any inputs to 8 bit range.
  No truncation occurs however (no precision loss), because even a 16 bit data
  is converted to 8 bit in floating point precision, using division by 256.0
  (2^16/2^8).
| So the conversion is _not_ a simple shift-right-8 in the integer domain, which
  would lose precision.

* Calculates expression
* Scales the internal result back to the original video bit depth.
* Clamping (clipping to valid range) and converting to an integer output
  (if applicable) occurs here.

The predefined constants such as ``'range_max'``, etc. will behave according to
the internal working bit depth.

.. note::

    **Important note!**

        This feature was created for easy porting earlier 8-bit-video-only lut
        expressions. You have to understand how it works internally.

        Let's see a 16bit input in ``"all"`` and ``"allf"`` mode (target is the
        default 8 bits):

        * Limited range 16->8 bits conversion has a factor of 1/256.0 (Instead
          of shift right 8 in integer domain, float-division is used or else it
          would lose presision)

        * Full range 16->8 bits conversion has a factor of 255.0/65535 (chroma
          is a bit different, since it is converted by moving into signed domain
          and back such as in (x-32768)*(127/32767)+128)

        * Using bit shifts (really it's division and multiplication by 2^8=256.0):
          |br| result = calculate_value(input / 256.0) * 256.0

        * Full scale 16-8-16 bit mode (``'intf', 'allf'``): |br|
          result = calculate_value(input / 65535.0 * 255.0 ) / 255.0 * 65535.0

        * chroma: |br| result = (calculate_value((input-32768) / 32767.0 * 127.0
          + 128 ) - 128) / 127.0 * 32767.0 + 32768

        * Use ``scale_inputs = "all"`` (``"int", "float"``) for YUV
          videos with 'limited' range e.g. in 8 bits: Y=16..235, UV=16..240).

        * Use ``scale_inputs = "allf"`` (``intf, floatf``) for RGB or
          YUV videos with 'full' range e.g. in 8 bits: channels 0..255.

        * When input is 32bit float, the 0..1.0 (luma) and -0.5..0.5 (chroma)
          channel is scaled to 0..255 (8 bits), 0..1023 (i10 mode), 0..4095
          (i12 mode), 0..16383(i14 mode), 0..65535(i16 mode) then back.

Compared to MaskTools
^^^^^^^^^^^^^^^^^^^^^

Compared to `MaskTools2`_ version 2.2.15, **Expr** has functionality similar to
*mt_lut, mt_lutxy, mt_lutxyz, mt_lutxyza* and *mt_lutspa*.

MaskTools2 is very slow for 10+ bit clips, when a `LUT`_ (lookup table) cannot
be used for memory size reasons, thus the expression is evaluated/interpreted at
runtime for each pixel. MaskTools2 (from v2.2.15) however is able to pass the
expressions to this AviSynth+ **'Expr'** filter with its ``'use_expr'`` parameter,
by passing the **expression** strings, and ``clamp_float`` and ``scale_inputs``
parameters.

The `JIT compiler`_ in **Expr** (adapted from `VapourSynth`_) turns the
expression calculation into realtime assembly code which is much faster and
basically bit depth independent.

    In **Expr**:

    * Up to 26 clips are allowed (x,y,z,a,b,...w). Masktools handles only up to
      4 clips with its mt_lut, mt_lutxy, mt_lutxyz, mt_lutxyza
    * Clips with different bit depths are allowed
    * Works with 32 bit floats instead of 64 bit double internally
    * Less functions (e.g. no bit shifts)
    * Logical 'false' is 0 instead of -1
    * The ymin, ymax, etc built-in constants can have a _X suffix, where X is
      the corresponding clip designator letter. E.g. cmax_z, range_half_x
    * mt_lutspa-like functionality is available through "sx", "sy", "sxr", "syr"
      internal predefined variables
    * No y= u= v= parameters with negative values for filling plane with constant
      value, constant expressions are changed into optimized "fill" mode


Examples
--------

* Average three clips::

    c = Expr(clip1, clip2, clip3, "x y + z + 3 /")

* When input clips to have more planes than an implicitely specified output format::

    # target is Y only which needs only Y plane from YV12
    Expr(aYV12Clip, "x 255.0 /", format="Y32")

* Y-plane-only clip(s) can be used as source planes when a non-subsampled (rgb
  or 444) output format is specified::

    # In both examples, the r, g and b expression uses the Y plane
    Expr(Y, "x", "x 2.0 /", "x 3.0 /", format="RGBPS")
    Expr(Grey_r, Grey_g, Grey_b, "x", "y 2.0 /", "z 3.0 /", format="RGBPS")

* Using spatial feature::

    c = Expr(clip_for_format, "sxr syr 1 sxr - 1 syr - * * * 4096 scaleb *", "", "")

* Mandelbrot zoomer (original code and idea from this `Doom9 thread`_)::

    a="X dup * Y dup * - A + T^ X Y 2 * * B + 2 min Y^ T 2 min X^ "
    b=a+a
    c=b+b
    blankclip(width=960, height=640, length=1600, pixel_type="YUV420P8")
    Expr("sxr 3 * 2 - -1.2947627 - 1.01 frameno ^ / -1.2947627 + A@ X^ syr 2 * 1 - 0.4399695 "
    \ + "- 1.01 frameno ^ / 0.4399695 + B@ Y^ "+c+c+c+c+c+b+a+"X dup * Y dup * + 4 < 0 255 ?",
    \ "128", "128")

 For other ideas of spatial variables, see MaskTools2: `mt_lutspa`_

* Using the time variable for fades::

    Expr("x time *") # linear fade in
    Expr("x 1 time - *") # linear fade out
    Expr("x time 2 pow *") # quadratic(?) fade in
    Expr("x time pi * cos 0.5 * 0.5 + *") # sinusoidal fade out

 See Doom9 thread `"Fade with configurable time curves?"`_ for more information.

Changelog
----------

+-----------------+----------------------------------------------------------+
| Version         | Changes                                                  |
+=================+==========================================================+
| AviSynth+ 3.7.2 || Expr: ``scale_inputs`` to case insensitive and add      |
|                 |  floatUV to error message as an allowed value.           |
|                 || Fix: Expr LUT operation Access Violation on x86 + AVX2  |
|                 |  due to an unaligned internal buffer (<32 bytes).        |
+-----------------+----------------------------------------------------------+
| AviSynth+ 3.7.1 || New: ``round, floor, ceil, trunc``                      |
|                 || TAB, CR, LF are valid string delimiters inside expr     |
|                 |  string                                                  |
|                 || Enhanced: arbitrary variable names, not only A to Z     |
|                 || Access clip's frame properties:                         |
|                 |  *clipletter.frameProperty* syntax                       |
|                 || Enhanced: ``sin`` and ``cos``: SIMD acceleration        |
|                 || New: ``atan2``                                          |
|                 || New: ``neg, sgn``                                       |
|                 || Enhanced: allow ‘f32’ as internal autoscale target      |
|                 |  besides integer i8..i16                                 |
|                 || lut (Lookup table) support 1D and 2D                    |
|                 || Enhanced: special full scale conversion of chroma plane |
+-----------------+----------------------------------------------------------+
| AviSynth+ 3.5.0 || Allow ``"floatUV"`` for parameter ``"clamp_float"``     |
|                 || New parameter ``"clamp_float_UV"``                      |
+-----------------+----------------------------------------------------------+
| AviSynth+ r2724 || New three operand function: clip                        |
|                 || New parameter ``"clamp_float"``                         |
|                 || New parameter ``"scale_inputs"``                        |
+-----------------+----------------------------------------------------------+
| AviSynth+ r2574 || New: indexable source clip pixels by relative x,y       |
|                 |  positions like x[-1,1]                                  |
|                 || New functions: ``sin cos tan asin acos atan``           |
|                 || New operator: % (modulo)                                |
|                 || New: variables: uppercase letters A..Z for storing and  |
|                 |  reuse temporary results, frequently used computations.  |
|                 || New: predefined expr variables                          |
|                 |  ``'frameno', 'time', 'width', 'height'``                |
|                 || Fix: jitasm code generation at specific circumstances   |
+-----------------+----------------------------------------------------------+
| AviSynth+ r2544 |  Optimization; fix ``scalef``                            |
+-----------------+----------------------------------------------------------+
| AviSynth+ r2542 |  Initial release                                         |
+-----------------+----------------------------------------------------------+

$Date: 2022/03/28 05:58:18 $

.. _mathematical function:
    https://en.wikipedia.org/wiki/Function_(mathematics)
.. _chroma subsampling:
    https://en.wikipedia.org/wiki/Chroma_subsampling
.. _RPN:
    https://en.wikipedia.org/wiki/Reverse_Polish_notation
.. _pixel format strings:
    http://avisynth.nl/index.php/Avisynthplus_color_formats
.. _AVX2:
    https://en.wikipedia.org/wiki/Advanced_Vector_Extensions
.. _SSE2:
    https://en.wikipedia.org/wiki/SSE2
.. _Doom9 thread:
    https://forum.doom9.org/showthread.php?p=1738391#post1738391
.. _LUT:
    https://en.wikipedia.org/wiki/Lookup_table
.. _MaskTools2:
    http://avisynth.nl/index.php/MaskTools2
.. _JIT compiler:
    https://en.wikipedia.org/wiki/Just-in-time_compilation
.. _VapourSynth:
    https://www.vapoursynth.com/doc/functions/video/expr.html
.. _mt_lutspa:
    http://avisynth.nl/index.php/MaskTools2/mt_lutspa
.. _"Fade with configurable time curves?":
    https://forum.doom9.org/showthread.php?t=183934

.. |br| raw:: html

      <br>