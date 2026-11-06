
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
          string "scale_inputs", bool "clamp_float", bool "clamp_float_UV", 
          int "lut", int "optVectorC")

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

    * Use :ref:`pixel format strings <color-formats>` like "YV12", "YUV420P8", "YUV444P16", "RGBP10".
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
      bit (AVX2) registers, thus processing 8 (SSE2)/16 (AVX2) pixels
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
    slower interpreted way (C language, see also ``optVectorC``). False disables SSE2.

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

.. describe:: optVectorC

    C code is run when
    
    - non x86/x64 systems (architectures which are not supported by JIT compiler)
    - expression contains atan, asin, acos, which are not implemented in JIT.
      (and ``tan`` before 3.7.4), 
    - JIT is intentionally disabled with optSSE2=False
    
    Enables or disables a compiler friendly (more easily vectorizable) C code.
    The source is written in such a ways that compilers can easily turn it into
    efficient SIMD vector operation, the patterns - do the same operation on
    4-8-16 pixels - are easily recognizable.
    
    Even if the compiler is not very advanced (MSVC khhhhmm..) this approach is 
    faster because it has less overhead when interpreting the instruction flow. It 
    processes 16, 8, 4, and 1 pixels (32 bit floats) when handling the horizontal 
    line, taking the largest chunks it can then finishing the rest with the smaller ones.
    
    This way, a good compiler can achieve one-third the speed of the SSE2 JIT (which is considered 
    fast), so this is quite impressive. Expect 3-20x speedup compared to the old method
    (which can be tested with optVectorC=False).

    Default: True

Expressions
-----------

**Expr** accepts 1 to 26 **source clips**, up to four **expression** strings
(one per color plane), an optional output format string, and some debug options.
Output video format is inherited from the first clip, when there is no format
override. All clips have to match in their width, height and `chroma subsampling`_.

Expressions are evaluated on each plane, Y, U, V (and A) or R, G, B (,A). When
an expression string is not specified, the previous expression is used for that
plane – except for plane A (alpha) which is copied by default. When an expression
is an empty string (``""``) then the relevant plane will be copied (if the output
clip bit depth is similar, otherwise it is converted to the new bit depth as in
:doc:`ConvertBits <convertbits>`: Y, U, V as limited range, R, G, B and alpha as full 
range). When an expression is a single clip reference letter ("x") and the source/target 
bit depth is similar, then the relevant plane will be copied. When an expression is 
constant (after constant folding), then the relevant plane will be filled with an 
optimized memory fill method.

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
* ``%`` (modulo), same as C ``fmod``: ``x - trunc(x/d)*d``, the result has the sign
  of ``x``. ``d`` = 0 results in NaN. Since 3.7.6 the JIT (SSE2/AVX2) result is exact while
  \|x/d\| < 2^24; above that the result is inaccurate.
* Math constant: ``pi``
* Functions: ``min, max, sqrt, abs, exp, log, pow ^`` (synonyms: ``pow`` and ``^``)
* ``sqrt`` of a negative number or NaN is 0, on every CPU and also in a constant expression. The same goes for a
  ``0.5 pow``, which is calculated as ``sqrt``. There is no good answer for a negative number: NaN, 0 and a signed
  root (the sign of x with ``sqrt(abs(x))``) all have their reasons, this is the one chosen. With 0, a value which is
  below zero only by a float rounding error, like ``7.0 0.1 % 0.1 -`` (-0.0000001), does not turn into NaN.
* ``pow`` with a negative base follows IEEE 754 on every CPU: an integer power has a real result (``-2 ^ 3`` is -8), a
  fractional power gives NaN. Too large results are the difference between the paths: x86 JIT stops at about 2.4e38,
  C (aarch64, ``SetMaxCPU("none")``, an expression with ``acos``, ``asin``, ``atan``, constant expressions) gives
  infinity, also at ``0 ^ -1``.
* Function: ``neg`` simple negates stack top
* Function: ``sgn`` simple signum function -1 if x<0; 0 when x==0; 1 if x>0
* Function: ``clip`` three operand function for clipping. Example: ``x 16 240
  clip`` means min((max(x,16),240)
* Functions: ``sin cos atan2 tan asin acos atan`` |br| On Intel x86/x64 the
  functions ``sin``, ``cos``, ``tan``  and ``atan2`` have SSE2/AVX2 optimization, 
  the others have not (if e.g. ``acos`` is used it makes the whole expression to 
  evaluate without JitASM optimization).
  On the SSE2 path (CPUs without AVX2) the precision of ``sin``, ``cos`` and ``tan``
  decreases for \|x\| above about 50000.
  The JIT ``tan`` uses range reduction to [-pi/4, pi/4] and a polynomial approximation,
  a method described e.g. in the `SLEEF paper`_ (N. Shibata, F. Petrogalli: SLEEF: A Portable
  Vectorized Library of C Standard Mathematical Functions. IEEE Transactions on Parallel and
  Distributed Systems, Vol. 31, No. 6, June 2020).
* Functions: ``round, floor, ceil, trunc`` operators (nearest integer - banker's
  rounding, round down, round up, round to zero). |br| On Intel builds acceleration
  requires at least SSE4.1 capable processor or else the whole expression is
  running in C mode.
* Logical: ``> < = >= <= and or xor not == & | !=`` (synonyms: ``==`` and ``=``,
  ``&`` and ``and``, ``|`` and ``or``)
* Ternary operator: ``?`` - Example: ``x 128 < x y ?``
* Duplicate stack elements: ``dup, dupn`` (dup1, dup2, ...)
* Swap stack elements: ``swap, swapn`` (swap1, swap2, ...)
* Convert a limited range level to the bit depth of the clip: ``scaleb``. The operand is
  an 8 bit level unless i8..i16 or f32 is specified. See `Converting levels: scaleb, scalef`_.
* Convert a full range level to the bit depth of the clip: ``scalef``. The operand is
  an 8 bit level unless i8..i16 or f32 is specified.


Converting levels: scaleb, scalef
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``scaleb`` and ``scalef`` convert a **level**, a position on the pixel value scale, from
the base bit depth to the bit depth of the first clip. ``128 scaleb`` is neutral chroma
and ``235 scaleb`` is limited range white at any bit depth. The base is 8 bit, unless
``i8, i10, i12, i14, i16`` or ``f32`` is given. Only the last of these counts for the
whole expression. With ``f32`` the operand is a limited range float level, and on a float
clip it is passed through unchanged.

* ``scaleb`` uses limited range rules, ``scalef`` uses full range rules. These are the
  same calculations as in :doc:`ConvertBits <convertbits>`.
* ``yscaleb`` and ``yscalef`` use the luma rules on the chroma planes as well.
* The name "bit shift" of ``scaleb`` only describes the result between integer bit depths,
  where the offsets cancel out:

  .. table::
      :widths: auto

      +--------+--------------+------------------------------------------------+---------------------+
      | Plane  | Target clip  | Calculation for an 8 bit operand               | Reduces to          |
      +========+==============+================================================+=====================+
      | Luma   | 10-16 bit    | ``(x - 16) * 2^(bits-8) + (16 << (bits-8))``   | ``x << (bits-8)``   |
      +--------+--------------+------------------------------------------------+---------------------+
      | Luma   | 32 bit float | ``(x - 16) / 255 + 16/255``                    | ``x / 255``         |
      +--------+--------------+------------------------------------------------+---------------------+
      | Chroma | 10-16 bit    | ``(x - 128) * 2^(bits-8) + (128 << (bits-8))`` | ``x << (bits-8)``   |
      +--------+--------------+------------------------------------------------+---------------------+
      | Chroma | 32 bit float | ``(x - 128) / 255``                            | ``(x - 128) / 255`` |
      +--------+--------------+------------------------------------------------+---------------------+

Results:

.. table::
    :widths: auto

    +------------------------------------+-------+-------+--------+--------+---------+
    | Expression (no source pixels used) | Plane | 8 bit | 10 bit | 16 bit | Float   |
    +====================================+=======+=======+========+========+=========+
    | ``235 scaleb``                     | Y     | 235   | 940    | 60160  | 0.92157 |
    +------------------------------------+-------+-------+--------+--------+---------+
    | ``235 scaleb``                     | U, V  | 235   | 940    | 60160  | 0.41961 |
    +------------------------------------+-------+-------+--------+--------+---------+
    | ``128 scaleb``                     | U, V  | 128   | 512    | 32768  | 0.0     |
    +------------------------------------+-------+-------+--------+--------+---------+
    | ``i10 940 scaleb``                 | Y     | 235   | 940    | 60160  | 0.92157 |
    +------------------------------------+-------+-------+--------+--------+---------+
    | ``i16 60160 scaleb``               | Y     | 235   | 940    | 60160  | 0.92157 |
    +------------------------------------+-------+-------+--------+--------+---------+
    | ``f32 0.92157 scaleb``             | Y     | 235   | 940    | 60160  | 0.92157 |
    +------------------------------------+-------+-------+--------+--------+---------+
    | ``235 scalef``                     | Y     | 235   | 943    | 60395  | 0.92157 |
    +------------------------------------+-------+-------+--------+--------+---------+

.. warning::

    ``scaleb`` and ``scalef`` are for levels. Do not use them on float chroma to give a
    **difference**, that is an amount to add to or subtract from a pixel, such as an
    overshoot limit. The conversion subtracts neutral chroma, 128, from the operand, so
    ``5 scaleb`` on float U or V is -0.48235, not 5 steps of 0.01961. This is not a new
    behaviour, and scripts using it get a wrong result on float chroma.

    With ``scale_inputs="floatUV"`` the result looks almost right, but it is still about
    10% low. On luma and at integer bit depths a difference comes out right only because
    the offsets cancel there, so these are not a reason to use a level conversion for it.

For a difference in 8 bit limited range steps, scale by the size of the range. This gives
the correct value on every bit depth, with or without ``"floatUV"``:

* Chroma planes: ``x 5 224 / cmax cmin - * +``, where 224 = 240 - 16, the size of the
  8 bit limited range of chroma.
* Luma plane: ``x 5 219 / ymax ymin - * +``, where 219 = 235 - 16, the size of the
  8 bit limited range of luma.

With ``scale_inputs="all"`` the expression works on 8 bit values, there ``5`` itself is the
difference, and the operators ``scaleb`` and ``scalef`` have nothing to scale.

.. table::
    :widths: auto

    +--------------------+-------------------------------+------------------+--------------+---------------+
    | Source U (neutral) | Expression on U               | ``scale_inputs`` | Result       | Correct value |
    +====================+===============================+==================+==============+===============+
    | 128 (8 bit)        | ``x 5 scaleb +``              |                  | 133          | 133           |
    +--------------------+-------------------------------+------------------+--------------+---------------+
    | 512 (10 bit)       | ``x 5 scaleb +``              |                  | 532          | 532           |
    +--------------------+-------------------------------+------------------+--------------+---------------+
    | 0.0 (float)        | ``x 5 scaleb +``              |                  | **-0.48235** | 0.01961       |
    +--------------------+-------------------------------+------------------+--------------+---------------+
    | 0.0 (float)        | ``x 5 scaleb +``              | ``"floatUV"``    | **0.01765**  | 0.01961       |
    +--------------------+-------------------------------+------------------+--------------+---------------+
    | 0.0 (float)        | ``x 5 scaleb +``              | ``"all"``        | 0.01961      | 0.01961       |
    +--------------------+-------------------------------+------------------+--------------+---------------+
    | 128 (8 bit)        | ``x 5 224 / cmax cmin - * +`` |                  | 133          | 133           |
    +--------------------+-------------------------------+------------------+--------------+---------------+
    | 512 (10 bit)       | ``x 5 224 / cmax cmin - * +`` |                  | 532          | 532           |
    +--------------------+-------------------------------+------------------+--------------+---------------+
    | 0.0 (float)        | ``x 5 224 / cmax cmin - * +`` |                  | 0.01961      | 0.01961       |
    +--------------------+-------------------------------+------------------+--------------+---------------+
    | 0.0 (float)        | ``x 5 224 / cmax cmin - * +`` | ``"floatUV"``    | 0.01961      | 0.01961       |
    +--------------------+-------------------------------+------------------+--------------+---------------+


Bit-depth aware constants
^^^^^^^^^^^^^^^^^^^^^^^^^

* ``ymin, ymax`` (ymin_a .. ymin_z for individual clips) - the limited range luma limits
  (16..235 or scaled equivalents), whatever the range of the clip is.
* ``cmin, cmax`` (cmin_a .. cmin_z) - the limited range chroma limits (16..240 or scaled
  equivalents), whatever the range of the clip is.
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

.. note::

    Expr does not read the ``_ColorRange`` frame property of the clips by itself, and the
    constants do not depend on the range of the clip. ``ymin, ymax, cmin, cmax`` and ``scaleb``
    are always limited range values, also on a full range clip. So the script has to choose by
    name: ``ymin, ymax`` and ``scaleb`` for limited range, ``range_min, range_max`` and ``scalef``
    for full range.

.. table::
    :widths: auto

    +----------------------------------------+-----------------------------------------+---------+-----------------+
    | On the Y plane, ``_ColorRange`` 0 or 1 | Always means                            | 10 bit  | Float           |
    +========================================+=========================================+=========+=================+
    | ``ymin, ymax``                         | Limited range luma limits, 16 and 235   | 64, 940 | 0.0627, 0.9216  |
    +----------------------------------------+-----------------------------------------+---------+-----------------+
    | ``cmin, cmax``                         | Limited range chroma limits, 16 and 240 | 64, 960 | -0.4392, 0.4392 |
    +----------------------------------------+-----------------------------------------+---------+-----------------+
    | ``255 scaleb``                         | A level, by limited range rules         | 1020    | 1.0             |
    +----------------------------------------+-----------------------------------------+---------+-----------------+
    | ``255 scalef``                         | A level, by full range rules            | 1023    | 1.0             |
    +----------------------------------------+-----------------------------------------+---------+-----------------+
    | ``range_min, range_max``               | The whole range of the plane            | 0, 1023 | 0.0, 1.0        |
    +----------------------------------------+-----------------------------------------+---------+-----------------+
    | ``range_half``                         | The middle of the range                 | 512     | 0.5             |
    +----------------------------------------+-----------------------------------------+---------+-----------------+

To follow the clip, read the frame property in the expression, see `Frame property input`_.
This example gives 940 on a limited range 10 bit clip and 1023 on a full range one, on the
Y plane. A YUV clip without the property counts as limited::

    x._ColorRange 1 == ymax range_max ?

Alpha plane
~~~~~~~~~~~

Alpha is full range, but the constants are not adjusted for it: ``ymax`` is 235 or its scaled
equivalent on the alpha plane as well. In an alpha expression use ``range_min, range_max`` and
``scalef``.

A single expression string does not reach the alpha plane, it is copied, also on YA clips. On YA
the second expression string is for alpha. When ``format`` changes the bit depth, the copy
is a full range conversion to the new bit depth, as on the other planes without an expression.

.. table::
    :widths: auto

    +--------------------+-------------------------------------+---------------------+-------------+
    | Source clip, alpha | Call                                | Alpha before 3.7.7  | Alpha now   |
    +====================+=====================================+=====================+=============+
    | YA10, 800          | ``Expr("128 scalef")``              | 514, the expression | 800, copied |
    +--------------------+-------------------------------------+---------------------+-------------+
    | YUVA444P10, 800    | ``Expr("128 scalef")``              | 800, copied         | 800, copied |
    +--------------------+-------------------------------------+---------------------+-------------+
    | YA10, 800          | ``Expr("128 scalef", "range_max")`` | 1023                | 1023        |
    +--------------------+-------------------------------------+---------------------+-------------+
    | YUVA444P8, 200     | ``Expr("x", format="YUVA444P16")``  | random              | 51400       |
    +--------------------+-------------------------------------+---------------------+-------------+
    | YA8, 200           | ``Expr("x", format="YA16")``        | 200                 | 51400       |
    +--------------------+-------------------------------------+---------------------+-------------+

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
* A property which is not set or is not a number reads as 0. The exception is ``_Range`` and its
  older name ``_ColorRange``: when it is not set, it reads as the usual default of that clip's format:
  RGB full, the others limited, as ``ConvertBits`` assumes as well. ``_Range`` 1 is full, ``_ColorRange``
  0 is full.

  +----------------------------------------------------------------------------------+------------------------+--------+-------------+
  | Source clip                                                                      | Expression             | Before | Since 3.7.7 |
  +==================================================================================+========================+========+=============+
  | ``BlankClip(pixel_type="YUV444P10").propDelete("_ColorRange")``                  | ``x._ColorRange``      | 0      | 1           |
  +----------------------------------------------------------------------------------+------------------------+--------+-------------+
  | ``BlankClip(pixel_type="YUV444PS").propDelete("_ColorRange")``                   | ``x._ColorRange``      | 0      | 1           |
  +----------------------------------------------------------------------------------+------------------------+--------+-------------+
  | ``BlankClip(pixel_type="RGBP10").propDelete("_ColorRange")``                     | ``x._ColorRange``      | 0      | 0           |
  +----------------------------------------------------------------------------------+------------------------+--------+-------------+
  | ``BlankClip(pixel_type="YUV444P10").propSet("_ColorRange", 0)``                  | ``x._ColorRange``      | 0      | 0           |
  +----------------------------------------------------------------------------------+------------------------+--------+-------------+
  | ``BlankClip(pixel_type="RGBP10").propSet("_ColorRange", 1)``                     | ``x._ColorRange``      | 1      | 1           |
  +----------------------------------------------------------------------------------+------------------------+--------+-------------+

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
    * ``time`` : ``calculation: time = frameno/(clip_frame_count-1)``. Use relative
      time position in expression. ``0 <= time <= 1``; 0 on a single frame clip.
    * ``width, height``: width and height of the plane being processed. On subsampled
      chroma planes they differ from the clip size, e.g. half of it on 4:2:0.

Pixel addressing
~~~~~~~~~~~~~~~~

Indexed, addressable source clip pixels by relative x,y positions.

    Syntax: ``x[a,b]``, ``x[a,b]:mode`` or ``k x[a,b]:k`` where

    * ``'x'`` : source clip letter a..z
    * ``'a'`` : horizontal shift. -width < a < width
    * ``'b'`` : vertical shift. -height < b < height
    * ``':mode'`` : optional, how off-screen pixels are taken (since 3.7.6):

      * ``:c`` : clamp, the edge pixel is cloned (default, same as no suffix)
      * ``:m`` : mirror, the edge pixel is repeated (``d c b a | a b c d``),
        compatible with `akarin.Expr`_ ``:m``
      * ``:r`` : mirror, the edge pixel is not repeated (``e d c b | a b c d``),
        like `OpenCV`_'s default ``BORDER_REFLECT_101``
      * ``:k`` : constant, off-screen pixels get the filler value ``k``
        (``k k k k | a b c d``). ``k`` is the value before the pixel reference on the
        stack, it must be a constant: a number, or a constant expression like
        ``range_half`` or ``64 2 *``. ``k`` is in the value domain of the expression:
        a raw pixel value by default, the converted value when ``scale_inputs`` is
        used (so ``range_half`` is always mid-grey).
      * ``:w`` : wrap around, the row/column continues from its other end
        (``e f g h | a b c d``), like OpenCV's ``BORDER_WRAP``. For pictures whose
        opposite edges are neighbors, e.g. seamlessly tiling textures or 360 degree
        panoramas.

    ``'a'`` and ``'b`` should be constant. e.g.:
    ``"x[-1,-1] x[-1,0] x[-1,1] y[0,-10] + + + 4 /"``,
    ``"x[-2,0]:m x[2,0]:m + 2 /"``, ``"range_half x[0,-1]:k"``, ``"x[-10,0]:w"``

| When a pixel would come from off-screen, it is taken by the border mode: cloned from
  the edge (default ``:c``), mirrored (``:m``, ``:r``), the constant filler (``:k``) or
  wrapped around from the other end (``:w``).
| The limits of ``a`` and ``b`` come from the clip (luma) size, so on subsampled chroma
  planes they can be larger than the plane. Mirror and wrap modes reflect or wrap as many
  times as needed (periodic extension). The JIT reflects or wraps once, so a plane where
  a mirror or wrap mode offset needs more runs in C (the other planes stay JIT optimized). Example, 640x480 YV12
  (U and V planes are 320x240): ``a`` and ``b`` may be up to 639 and 479. ``x[2,1]:m``
  is JIT optimized on all planes. ``x[400,0]:m`` and ``x[0,300]:m`` run in C on U and
  V (400 > 320, 300 > 240), JIT on Y. Mirror and wrap modes run in C on a plane also when
  ``a`` > planewidth-3 (``:m``, ``:w``) or ``a`` > planewidth-4 (``:r``), e.g. ``x[318,0]:m``
  on U and V. Clamp (``:c``) and constant (``:k``) are always JIT optimized.
| Note: ``a`` and ``b`` are pixel offsets of the plane being processed. When the same
  expression is used for subsampled chroma planes (e.g. a single expression for all
  planes), the same offsets cover a larger area there: a 5x5 neighborhood (``a``, ``b``
  from -2 to 2) on Y covers 10x10 luma pixels on 4:2:0 U and V. For the same spatial
  extent give the chroma planes their own expression with offsets scaled by the
  subsampling (e.g. -1 to 1 for 4:2:0).
| Indexed pixels are JIT optimized for SSE2 and AVX2 (before 3.7.6: SSSE3 required,
  no AVX2 version: an expression with indexed pixels ran on the SSE2 JIT even on
  AVX2 CPUs).

Off-screen (border) handling modes and their names in other tools, shown at the left
edge of the row ``a b c d ...``, for reference when porting expressions or comparing
results.

.. list-table::
    :header-rows: 1
    :widths: 20 12 10 17 12 14 10

    * - Extension at the left edge
      - Edge pixel repeated
      - Expr
      - `OpenCV`_
      - `numpy.pad`_
      - `scipy.ndimage`_
      - `akarin.Expr`_
    * - ``a a a a | a b c d``
      - (clone, clamp)
      - ``:c`` (default)
      - BORDER_REPLICATE
      - edge
      - nearest
      - ``:c``
    * - ``d c b a | a b c d``
      - yes
      - ``:m``
      - BORDER_REFLECT
      - symmetric
      - reflect
      - ``:m``
    * - ``e d c b | a b c d``
      - no
      - ``:r``
      - BORDER_REFLECT_101 (OpenCV default)
      - reflect
      - mirror
      - \-
    * - ``k k k k | a b c d``
      - (constant)
      - ``k x[a,b]:k``
      - BORDER_CONSTANT
      - constant
      - constant
      - \-
    * - ``e f g h | a b c d``
      - (wrap)
      - ``:w``
      - BORDER_WRAP
      - wrap
      - wrap
      - \-

Note that the name "reflect" means different modes in numpy and scipy.

Border mode examples (``a``, ``b`` offsets are in pixels of the processed plane).
Note that for offsets of -1 and 1 ``:m`` gives the same result as ``:c`` (the mirrored
pixel next to the edge is the edge pixel itself); ``:m`` differs from ``:c`` only from
offset 2 on. So for 3x3 kernels ``:c`` and ``:m`` are identical, only ``:r`` (and ``:k``)
gives a different result at the frame border.

* ``:c`` clamp (default). 3-tap horizontal blur, the edge pixel is repeated:
  ``"x[-1,0] x x[1,0] + + 3 /"``
* ``:m`` mirror. 5-tap horizontal box blur. The edge pixel is not over-weighted at the
  frame border as with clamp (where pixel ``a`` would get 3 of the 5 weights), the
  border neighborhood keeps the texture of the picture:
  ``"x[-2,0]:m x[-1,0]:m x x[1,0]:m x[2,0]:m + + + + 5 /"``
* ``:r`` reflect101. 3x3 Laplacian edge detection (absolute value) with the border
  handling of OpenCV's default, so border pixels match ``abs(cv2.Laplacian(img, ksize=1))``:
  ``"x[-1,0]:r x[1,0]:r + x[0,-1]:r + x[0,1]:r + x 4 * - abs"``
* ``:r`` reflect101. Sobel edge magnitude ``sqrt(Gx^2 + Gy^2)`` (as in ExTools'
  ``ex_edge(mode="sobel")``; the corner pixels are stored in variables, since both
  kernels use them). With ``:r`` the frame border is handled like OpenCV's default
  (``cv2.magnitude`` of ``cv2.Sobel`` dx and dy). With ``:c`` (or no suffix) the same
  expression is VapourSynth's ``std.Sobel`` (scale=1): same kernels, magnitude and
  edge-repeat border; there the gradient across the frame edge is halved:
  ``"x[1,-1]:r C@ x[1,1]:r I@ + x[-1,-1]:r A@ - x[-1,1]:r G@ - x[1,0]:r x[-1,0]:r - 2 * + dup *
  G I + A - C - x[0,1]:r x[0,-1]:r - 2 * + dup * + sqrt"``
* ``:r`` reflect101. 3x3 binomial blur, the kernel of RemoveGrain(11) / (12)
  (``[1 2 1; 2 4 2; 1 2 1] / 16``). RemoveGrain copies the outermost rows and columns
  unprocessed; here the frame edge is filtered as well:
  ``"x[-1,-1]:r x[1,-1]:r + x[-1,1]:r + x[1,1]:r + x[0,-1]:r x[-1,0]:r + x[1,0]:r x[0,1]:r + + 2 * + x 4 * + 16 /"``
* ``:k`` constant. Shift the picture down by one pixel, the new top row is mid-grey
  at any bit depth:
  ``"range_half x[0,-1]:k"``
* ``:k`` constant. Number of set neighbors (4-neighborhood) of a 0/1 mask, off-screen
  pixels count as not set (zero padding). With clamp a pixel at the frame edge would
  count itself as its own off-screen neighbor:
  ``"0 x[-1,0]:k 0 x[1,0]:k + 0 x[0,-1]:k + 0 x[0,1]:k +"``
* ``:w`` wrap. Scroll the picture 10 pixels to the right, the pixels leaving at the
  right edge come back at the left edge:
  ``"x[-10,0]:w"``
* ``:w`` wrap. 5-tap horizontal box blur of a seamlessly tiling texture (or a 360 degree
  panorama): the left and right edges are treated as neighbors, so the result tiles
  seamlessly as well:
  ``"x[-2,0]:w x[-1,0]:w x x[1,0]:w x[2,0]:w + + + + 5 /"``

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

External Links
--------------

* `VapourSynth`_ Expr documentation: the base of this filter
* `akarin.Expr`_ (VapourSynth plugin): extended Expr, ``:m`` border mode compatible
* `MaskTools2`_ (``mt_lut`` family) and `mt_lutspa`_: similar syntax, spatial variables
* `rgba_rpn`_ RPN filters by wonkey_monkey: origin of the variable store (``A@``) and
  store-and-pop (``A^``) syntax, frame number and time variables
* `SLEEF paper`_: vectorized math functions, the method of the JIT ``tan``
* Border handling in other libraries: `OpenCV`_ border types, `numpy.pad`_ and
  `scipy.ndimage`_ modes
* `Doom9 thread`_: Mandelbrot zoomer, original code and idea
* `"Fade with configurable time curves?"`_ (doom9.org): using the ``time`` variable
* Background: `RPN`_ (Reverse Polish notation), `mathematical function`_, `JIT compiler`_,
  `LUT`_, `chroma subsampling`_, `SSE2`_, `AVX2`_


Changelog
----------

+-----------------+----------------------------------------------------------+
| Version         | Changes                                                  |
+=================+==========================================================+
| 3.7.6           || Fix: round, floor, ceil or trunc before a relative      |
|                 |  pixel load: crash on AVX2 CPUs; before asin, acos,      |
|                 |  atan: wrong results (JIT). Bug since 3.7.1.             |
|                 || Fix #505: Vector-C relative-row addressing ignored the  |
|                 |  current row, mis-scaled the row offset at 16/32 bit     |
|                 || Fix: JIT tan (SSE2/AVX2) was inaccurate, up to 8%       |
|                 |  relative error. New implementation, max ~3.5 ulp.       |
|                 |  Bug since 3.7.4.                                        |
|                 || Fix: JIT % (modulo) now gives the same result as        |
|                 |  fmodf; was 0 or slightly negative near multiples        |
|                 |  of the divisor (e.g. 73 % 7.3).                         |
|                 || Fix: constant condition ternary with a variable store   |
|                 |  (A@, A^), dup or swap among its operands: wrong result  |
|                 |  or crash (e.g. "0 7 sx A^ A ?", "0 sx dup ?").          |
|                 || Fix: relative pixel loads near the row edges with many  |
|                 |  values on the stack: wrong results or crash, mostly     |
|                 |  x86 (JIT register handling bug, also in earlier         |
|                 |  versions).                                              |
|                 || Relative pixel load (x[dx,dy]): new AVX2 JIT version,   |
|                 |  ~2x faster on AVX2 CPUs; SSE2 JIT no longer requires    |
|                 |  SSSE3, slightly quicker.                                |
|                 || New: relative pixel load border modes x[dx,dy]:c        |
|                 |  (clamp, default), :m (mirror, akarin.Expr compatible),  |
|                 |  :r (mirror without repeating the edge pixel),           |
|                 |  k x[dx,dy]:k (constant filler k from the stack),        |
|                 |  :w (wrap around).                                       |
+-----------------+----------------------------------------------------------+
| 3.7.4           || Enhancement: vectorizable C implementation helps nonJIT |
|                 || New parameter: optVectorC                               |
|                 || Implement ``tan`` for JitASM                            |
+-----------------+----------------------------------------------------------+
| AviSynth+ 3.7.2 || Expr: ``scale_inputs`` to case insensitive and add      |
|                 |  floatUV to error message as an allowed value.           |
|                 || Fix: Expr LUT operation Access Violation on x86 + AVX2  |
|                 |  due to an unaligned internal buffer (<32 bytes).        |
+-----------------+----------------------------------------------------------+
| AviSynth+ 3.7.1 || New: ``round, floor, ceil, trunc``                      |
|                 || TAB, CR, LF are valid string delimiters inside expr     |
|                 |  string                                                  |
|                 || Enhanced: arbitrary variable names, not only A to Z     |
|                 |  (up to 256 variables)                                   |
|                 || Access clip's frame properties:                         |
|                 |  *clipletter.frameProperty* syntax                       |
|                 || Enhanced: ``sin`` and ``cos``: SIMD acceleration        |
|                 || New: ``atan2`` (SIMD accelerated)                       |
|                 || New: ``neg, sgn``                                       |
|                 || Enhanced: allow ‘f32’ as internal autoscale target      |
|                 |  besides integer i8..i16                                 |
|                 || lut (Lookup table) support 1D and 2D                    |
|                 || Enhanced: special full scale conversion of chroma plane |
+-----------------+----------------------------------------------------------+
| AviSynth+ 3.5.0 || Allow ``"floatUV"`` for parameter ``"clamp_float"``     |
|                 || New parameter ``"clamp_float_UV"``                      |
|                 || Allow ``"floatUV"`` for parameter ``"scale_inputs"``    |
|                 || New: ``yrange_min, yrange_half, yrange_max`` and        |
|                 |  ``yscalef, yscaleb``: luma (non-chroma) rules even when |
|                 |  processing chroma planes                                |
+-----------------+----------------------------------------------------------+
| AviSynth+ r2724 || New three operand function: clip                        |
|                 || New parameter ``"clamp_float"``                         |
|                 || New parameter ``"scale_inputs"``                        |
+-----------------+----------------------------------------------------------+
| AviSynth+ r2574 || New: indexable source clip pixels by relative x,y       |
|                 |  positions like x[-1,1] (JIT requires SSSE3)             |
|                 || New functions: ``sin cos tan asin acos atan`` (no JIT)  |
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

$Date: 2026/11/05 12:00:00 $

.. _mathematical function:
    https://en.wikipedia.org/wiki/Function_(mathematics)
.. _chroma subsampling:
    https://en.wikipedia.org/wiki/Chroma_subsampling
.. _RPN:
    https://en.wikipedia.org/wiki/Reverse_Polish_notation
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
.. _SLEEF paper:
    https://arxiv.org/pdf/2001.09258
.. _OpenCV:
    https://github.com/opencv/opencv/blob/4.x/modules/core/include/opencv2/core/base.hpp
.. _numpy.pad:
    https://numpy.org/doc/stable/reference/generated/numpy.pad.html
.. _scipy.ndimage:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html
.. _akarin.Expr:
    https://github.com/AkarinVS/vapoursynth-plugin
.. _rgba_rpn:
    https://horman.net/avisynth/rgba_rpn.html

.. |br| raw:: html

      <br>