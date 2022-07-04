
Arrays
======

Arrays (AviSynth+)
^^^^^^^^^^^^^^^^^^

Beginning Avisynth+ 3.6 script arrays are supported. Functionality is different from `AVSLib`_ described below.

-  Arrays can have one or more dimensions
-  They even can be empty.
-  AVSValue (internal representation of any Avisynth variable) is deep copied for arrays (arrays in arrays in ...)
   (note: this is true when using them through c++ plugins. C interface is different, there is no deep-copy there)
-  untyped and unconstrained element number
-  arrays can appear as internal filter parameters (named and unnamed).
-  access with indexes or in a dictionary-like associative way
-  Array modifier functions allow multidimensional subarray indexes

  - ArrayAdd - append
  - ArrayDel - delete from position
  - ArrayIns - insert before position
  - ArraySet - replace at position

ArrayIns
^^^^^^^^

ArrayIns(array_to_mod, value_to_insert, index1 [, index2, index3...])

    Insert a value into an array or into its subarray.
    Returns a new array with value_to_insert inserted into array_to_mod (1D array) or array_to_mod[index1 (, index2, index3...)] (multi-dimensional array)
    The indexes point to the insertion point. Index 0 will insert at the beginning of the array.
    Index (ArraySize) will insert after the last element (same as ArrayAdd - append)
    Original array (as with the other functions) remains untouched.

ArrayAdd
^^^^^^^^

ArrayAdd(array_to_mod, value_to_append [, index1, index2, index3...])

    Appends value to the end of an array or its subarray
    Returns a new array with value_to_append appended to array_to_mod (1D array) or array_to_mod[index1 (, index2, index3...)] (multi-dimensional array).
    Original array (as with the other functions) remains untouched.

ArrayDel
^^^^^^^^

ArrayDel(array_to_mod, index1 (, index2, index3...])

    Returns a new array in which the requested position was deleted.
    Original array (as with the other functions) remains untouched.

ArraySet
^^^^^^^^

ArraySet(array_to_mod, replacement_value, index1 [, index2, index3...])

    Returns a new array with array_to_mod[index1 (, index2, index3...)] = replacement_value
    Original array (as with the other functions) remains untouched.

ArraySize
^^^^^^^^^

int ArraySize(array_value)

    Returns the size of the parameter.
    For getting the size of a subarray, pass the inner element.

Example:

::

      ColorbarsHD()
      # array indexes are zero based
      a = []
      a=ArrayAdd(a,[1,2]) # [[1,2]]
      a=ArrayIns(a,3,0) # [3,[1,2]]
      a=ArrayAdd(a,"s1") # [3,[1,2],"s1"]
      a=ArrayAdd(a,"s2") # [3,[1,2],"s1","s2"]
      a=ArrayDel(a,2) # [3,[1,2],"s2"]
      a=ArraySet(a,"g",1,0) # [3,["g",2],"s2"]
      a=ArrayAdd(a,"h",1) # [3,["g",2,"h"],"s2"]
      a=ArrayAdd(a,[10,11,12],1) # append to (1) -> [3,["g",2,"h",[10,11,12]],"s2"]
      a=ArrayDel(a,1,3,0) # del from (1,3,0) -> [3,["g",2,"h",[11,12]],"s2"]
      a=ArrayAdd(a,"added") # [3,["g",2,"h",[11,12]],"s2","added"]
      a=ArrayAdd(a,["yet","another","sub"]) # [3,["g",2,"h",[11,12]],"s2","added",["yet","another","sub"]]
      x=a[0] #3
      x=a[1,0] #g
      x=a[1,2] #h
      x=a[1,3,1] #12
      x=a[3] #"added"
      x=a[4,1] #"another"
      SubTitle("x = " + String(x) + " Size=" + String(a.ArraySize()))

Example:

::

      array_variable = [[1, 2, 3], [4, 5, 8], "hello"]
      dictionary = [["one", 1], ["two", 2]]
      empty = []
      subarray = array_variable[0]
      val = subarray[2]
      val2 = array_variable[1, 3]
      str = array_variable[2]
      n = ArraySize(array_variable) #3
      n2 = ArraySize(empty) #0
      val3 = dictionary["two"]

Example:

::

      a = []
      a=ArrayAdd(a,[1,2]) # [[1,2]]
      a=ArrayIns(a,3,0) # [3,[1,2]]
      a=ArrayAdd(a,"s1") # [3,[1,2],"s1"]
      a=ArrayAdd(a,"s2") # [3,[1,2],"s1","s2"]
      a=ArrayDel(a,2) # [3,[1,2],"s2"]
      
      b = ["hello", "leo"]

Example:

::

      ColorBars()
      clip=last
      a = [[1,2],[3,4]]
      aa = [1]
      b = a[1,1] + ArrayGet(a, 1,0) + aa[0]
      
      empty_array = []
      empty_array_2 = empty_array
      #n3 = empty_array_2.ArrayGet(0) # array index out out range error!
      
      black_yuv_16 = [0,32768,32768]
      grey_yuv_16 = [32768,32768,32768]
      white_yuv_16 = [65535,32768,32768]
      aSelectColors = [\
        ["black", black_yuv_16],\
        ["grey", grey_yuv_16],\
        ["white",white_yuv_16],\
        ["empty",empty_array]\
      ]
      test_array = [99, 1.0, "this is a string"] # mixed types
      test_array2 = [199, 2.0, "This is a string"]

      n = ArraySize(test_array) # 3
      n2 = ArraySize(empty_array_2) # 0
      sum = FirstNSum(grey_yuv_16,2)
      b = b
      
      clip = clip.Text(e"Array size = " + String(n) +\
       e"\n Empty array size = " + String(n2) +\
       e"\n sum = " + String(sum) +\
       e"\n b = " + String(b) +\
       e"\n white_yuv_16[1]=" + String(aSelectColors["white"][1]) + \
       e"\n [0]=" + String(ArrayGet(test_array,0)) + \
       e"\n [1]=" + String(ArrayGet(test_array,1)) + \
       e"\n [2]=" + ArrayGet(test_array,2), lsp=0, bold=true, font="info_h")
      
      return clip
      
      function FirstNSum(array x, int n)
      {
        a = 0
        for (i=0, x.ArraySize()-1) {
          a = a + x[i]
        }
        return a
      }

Arrays in user defined functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Avisynth accepts arrays in the place of "val" script function parameter type regardless of being named or unnamed.
(Note: "val" is translateed to "." in internal function signatures)

Example:

::

      BlankClip(pixel_type="yv12")
      r([1, 2, 3])
      r(n=[10,11,[12,13]])
      r("hello")
      function r(clip c, val "n")
      {
        if (IsArray(n)) {
         if (IsArray(n[2])) {
           return Subtitle(c, String(n[2,1]), align=8) #13 at the top
         } else {
           return Subtitle(c, String(n[2]), align=2) #3 at the bottom
         }
        } else {
          return Subtitle(c, String(n), align=5) #hello in the center
        }
      }

Some facts which are inherited from the compatible Avisynth functionality.

-  Array-typed parameters with "name" have the value "Undefined" when they are not passed.
-  But the value is "Defined" and its value is a zero-sized array if the parameter is unnamed, like in other Avisynth functions.

**"Array of Anything" issues**

What about parameter handling for "array of anything" parameter when array(s) would be passed directly.

Avisynth traditionally makes difference between zero-or-more and one-or-more kind of array parameters.
The special case is "array of anything"

- Avisynth signature: .* or .+
- Script function specifier val_array or val_array_nz (nz denotes to nonzero)

When parameter signature is array of anything (.+ or .*) and the
parameter is passed unnamed (even if it is a named parameter) then
there is an ambiguos situation.

Example:

    1,2,3 will be detected as [1,2,3] (compatibility: Avisynth collects arrays from comma separated function values, when such function signature is found)

    1 will be detected as [1] (compatibility)

    (nothing) will be detected as [], but marked in order to override it later directly by name

Consequences:

    Passing a direct script array [1,2,3] will be detected as [[1,2,3]], because unnamed and untyped parameters are
    put together into an array, which has the size (number of elements) of the list. This is a list of 1 element which happens to be an array.
    Avisynth cannot 'guess' whether we want to define a single array directly or this array is the only one part of the list.
    [1,2,3] or [ [1,2,3] ]

Syntax hint:

When someone would like to pass a directly specified array (e.g. [1,2,3] instead of 1,2,3) to a .+ or .* parameter
the parameter must be passed by name! Or better: instead of "array of anything" use the val (function signature ".") type.
It will acceopt any type, including arrays. Then you can check inside your function with IsArray() and ArraySize() if it is really an array.

Because of the existing AviSynth syntax rule: arguments given as unnamed in the place of an array-of-anything parameter
are considered to be list elements from which Avisynth creates an array

::

      function foo(val_array "n")
        Call                          n
        foo()                   O.K.  Undefined
        foo(1)                  O.K.  [1] (compatible Avisynth way)
        foo(1,2,3)              O.K.  [1,2,3] (compatible Avisynth way)
        foo([1,2,3])            !     [[1,2,3]] (compatible Avisynth way)
        foo([1,2,3],[4,5])      !     [[1,2,3],[4,5]] (compatible Avisynth way)
        foo(n=[1,2,3])          O.K.  [1,2,3]
        foo(n=[[1,2,3],[4,5]])  O.K.  [[1,2,3],[4,5]]
        foo(n=[])               O.K.  []
        foo(n="hello")          Syntax error, "hello" is not an array

        // unnamed signature
      function foo(val_array n)
        Call                          n
        foo()                   O.K.  [] (defined and array size is zero) Avisynth compatible behaviour

Script functions supports avisynth function array 
- signature '+' (one or more) with _nz type suffix. E.g. int_array_nz
- signature '*' (zero or more) without _nz type suffix. E.g. float_array

E.g.: val_array -> .* val_array_nz -> .+, int_array -> i* int_array_nz -> i+
Each basic type has its array and array_nz variant.
Such as bool_array_nz, float_array_nz, string_array_nz, clip_array_nz, func_array_nz.

Note 1: There is an error message when a script array is passed to a non-array named function argument
(e.g. foo(sigma=[1.1,1.1]) to [foo]f parameter signature

Note 2: Type-free unnamed arrays ".+" or ".*" cannot be followed by additional parameters

Note 3: A backward compatible way (AVS 2.6 and non-script-array AviSynth+ versions) of using named
or unnamed arrays is to specify a single type as "." and the plugin would check the argument type by IsArray

User defined functions get array parameter types:

- "array" or "val_array": array of any type.

    When unnamed, then this kind of parameter must be the very last one.
    Unnamed free-typed parametes cannot be followed by any other parameter.
    Translates to ".*" in a plugin parameter definition rule.

-  "bool_array" "int_array", "float_array", "string_array", "clip_array", "func_array"

    Translates to "b*", "i*", "f*", "s*", "c*", "f*" in a plugin parameter definition rule.

-  "bool_array_nz" "int_array_nz", "float_array_nz", "string_array_nz", "clip_array_nz", "func_array_nz"

    Translates to "b+", "i+", "f+", "s+", "c+", "n+" in a plugin parameter definition rule.

Example:

::

      a = [1.0, 2.0, 4.2]
      b = [3, 4, 5]
      multi = [a,b]

      sum = Summa(multi[0], multi[1], 2)
      SubTitle(Format({sum}))

      Function Summa(array "x", array "y", int "N")
      {
        sum = 0.0
        FOR(i=0,N-1) {
          sum = sum + x[i] * y[i]
        }
        return sum
      }

      or

      Function Summa(float_array x, float_array y, int "N")
      {
        sum = 0.0
        FOR(i=0,N-1) {
          sum = sum + x[i] * y[i]
        }
        return sum
      }

Arrays (pre AviSynth+: AVSLib)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before Avisynth+ 3.6 arrays were not supported natively by the
scripting language.

However, a library named [`AVSLib`_] exists that provides a functional
interface for creating and manipulating arrays. Coupled with Avisynth's OOP
style for calling functions, one can treat arrays as objects with methods,
which is a familiar and easy to understand and code scripting concept.

Therefore, two preparatory steps are needed before being able to create and
manipulate process arrays into your script:

-   [`Download`_] and install the most current version of AVSLib into
    your system.
-   Import the needed AVSLib files in your script as follows (see the
    instructions inside the library's documentation to fill-in the gaps):
-   AVSLib 1.1.x versions: Enter ``LoadPackage("avslib", "array")`` to
    load the array implementation files, or ``LoadLibrary("avslib",
    CONFIG_AVSLIB_FULL)`` to load entire AVSLib.
-   AVSLib 1.0.x versions: Enter an appropriate :doc:`Import <../corefilters/import>` ({path to AVSLib
    header}) statement as the first line of your script.

Now you are ready to create your first array! In order to provide an almost
real case example let's assume the following (which are commonplace in many
situations) about the script you want to create:

-   The script selects a distinct range of frames from each video clip.
-   Some of the input clips may have different size, fps, audio and/or
    colorspace; thus they need to be converted.
-   Some of the filtering parameters are distinct for each clip.

Having done that, let's proceed to the actual code:

First, we create the array; ..1.., ..2.., etc. are actual filename strings.
Clip loading is made by :doc:`AviSource <../corefilters/avisource>` in the example but
:doc:`DirectShowSource <../corefilters/directshowsource>` may also be specified.

::

    inp = ArrayCreate( \
        AviSource(..1..), \
        AviSource(..2..), \
        ... \
        AviSource(..n..) )

Then we convert to same fps, audio, colorspace and size by using
:doc:`AssumeFPS <../corefilters/fps>`, :doc:`ConvertAudioTo16bit <../corefilters/convertaudio>`,
:doc:`ConvertToYV12 <../corefilters/convert>` and :doc:`BilinearResize <../corefilters/resize>`
respectively (or any resizer that you find fit). We use OOP + chaining to
make compact expressions.

Note that since Avisynth does not provide a way for in-place variable
modification we must reassign to an array variable after each array operation
(usually the same).

::

    inp = inp.ArrayOpFunc("AssumeFPS", "24").ArrayOpFunc("ConvertAudioTo16bit" \
        ).ArrayOpFunc("ConvertToYV12").ArrayOpFunc("BilinearResize", "640,480")

To perform trimming we will use arrays of other types also. Below *ts* stands
for first frame to trim, *te* for last; each number corresponds to a clip in
*inp* variable.

::

    ts = ArrayCreate(12, 24, ..., 33) # n numbers in total
    te = ArrayCreate(8540, 7834, ..., 5712) # n numbers in total

We also need a counter to make things easier; we will use ArrayRange to
create an array of 0,1,2,...

::

    cnt = ArrayRange(0, inp.ArrayLen()-1)

In addition we must define a user function that will accept *inp*, *ts*, *te*
and *cnt* and do the trimming.

Since ArrayOpArrayFunc only accepts two arrays for per-element processing, it
is easier to pass 'inp' and *cnt* as array elements and *ts*, *te* as entire
arrays.

::

    Function MyTrim(clip c, int count, string fs, string fe) {
        return c.Trim(fs.ArrayGet(count), fe.ArrayGet(count))
    }

Now we are ready to do the trim (line below).

::

    inp = ArrayOpArrayFunc(inp, cnt, "MyTrim", StrQuote(ts)+","+StrQuote(te))

We will finish the processing with a final tweak on brightness with different
settings on each clip and on hue with same settings for all clips.

::

    bright = ArrayCreate(2.0, 1.5, ..., 3.1) # n numbers in total

    Function MyTweak(clip c, float br) {
        return c.Tweak(bright=br, hue=12.3)
    }

    inp = ArrayOpArrayFunc(inp, bright, "MyTweak")

And now we are ready to combine the results and return them as script's
output. We will use `Dissolve`_ for a smoother transition.

::

    return inp.ArraySum(sum_func="Dissolve", sum_args="5")

This is it; the n input clips have been converted to a common video and audio
format, trimmed and tweaked with individual settings and returned as a single
video stream with only 11 lines of code (excluding comments).

Other types of array processing are also possible (slicing ie operation on a
subset of elements, joining, multiplexing, etc.) but these are topics to be
discussed in other pages. Those that are interested can browse the `AVSLib`_
documentation. One can also take a closer look at the `examples section`_
of the AVSLib documentation.

--------

Back to :doc:`scripting reference <script_ref>`.

$Date: 2008/04/20 19:07:33 $

.. _AVSLib: http://avslib.sourceforge.net/
.. _Download: http://sourceforge.net/projects/avslib/
.. _Dissolve: http://avisynth.org/mediawiki/Dissolve
.. _examples section: http://avslib.sourceforge.net/examples/index.html
