
AviSynth Syntax - Frame properties
==================================


The concept was ported from VapourSynth.

Frame properties are per-frame data. Source or other filters can store useful 
data into frame properties such as information on colorimetry.

Imagine them as an array of zero or more elements of PropertyName=Value pairs.

Reading or writing them is possible either in runtime environment or at the normal script level. 

Avisynth provides the necessary framework for handling (set, read, clear, pass) such properties,
both from users scripts and from the Filter SDK.

Before v3.6.2test7 none of its internal functions (such as color space converters) were able
to set or work upon frame property values. However the set of frame properties which are 
actively used by Avisynth core can extend in later versions. E.g. in 3.7.x color space converters or 
even "Text" filter read and/or set some colormetry or color matrix related properties.

And there are external filters (like avsresize) and source plugins that use frame properties.

Developer info (see also in Filter SDK):

On filter level the set of frame properties should be copied from a previous (or specific) input frame.
When a filter is using ``env->MakeWritable`` in its ``GetFrame`` method this happens automatically. 

A simple ``env->NewVideoFrame`` however means a dead-end for frame property passing mechanism. Filter 
must either use ``env->NewVideoFrameP`` which has an additional parameter: a source frame (origin) of 
the frame properties. Or frame properties have to be copied later, programatically by the ``copyFrameProps``
``ScriptEnvironment`` function. 

For compatibility reasons ``NewVideoFrameP`` must be called adaptively instead of the traditional 
``NewVideoFrame``, since it does not exist in older Avisynth versions. Only usable after detecting 
Avisynth interface version 8 (NewVideoFrameP was introduced in v8).

What happens if this copy is not done properly? Nothing deadly, but then the filter is a dead-end for 
the flow of frame properties since NewVideoFrame constructs new frame without them. Old filters which are
not written with this in mind will behave like this.

Frame property getter functions are evaluated at every frame. They can be used inside the scripts passed 
to runtime filters (ScriptClip).

Frame property setters can be called both at script level and in runtime. When a property is set on script 
level it will be constant along the lifetime of the whole clip, unless it is changed in runtime, inside ScriptClip. 

Definition
----------

Frame properties are stored as key-value pairs.
Each video frame can contain such a list, which can even be different from frame to frame.

* Key is an alphanumeric identifier.

* Value can be of single value or array of type

  * 64 bit integer (32 bit integer when converted to AVSValue inside Avisynth+, AVSValue limitation)
  * 64 bit double (32 bit float when converted to AVSValue inside Avisynth+, AVSValue limitation)
  * string (or byte data) with given length. Strings are null terminated.
    The content whether it is string or binary data can be given as a hint; from Avisynth scripting 
    only strings allowed.

  * frame reference (PVideoFrame)
  * clip reference (PClip)

Manipulation modes
------------------

Property setting has 3 modes: 0-replace 1-append 2-touch.
(for developers: see ``AVSPropAppendMode`` in avisynth.h)

* 0 - single value for that key is replaced

* 1 - property is appended (make arrays by calling with mode=1 consecutively)

* 2 - touch (deprecated, just creates the key)

Reserved frame property names
-----------------------------

There are quasi-standard frame property names which are used widespread.

Some of them are used in Avisynth core:

- _ChromaLocation
- _Range (since 3.7.7)
- _ColorRange
- _Matrix

but the others are used only by external plugins.

Since 3.7.7 ``_Range`` (the ITU-T H.273 conform name) is the primary name of the range property, and ``_ColorRange``
is used in parallel, as an alias of the same property. ``_ColorRange`` stays a valid name, no deprecation period is
set for it.

Frame property concept came from VapourSynth (see "Reserved Frame Properties" 
at http://www.vapoursynth.com/doc/apireference.html). But it is convenient that 
Avisynth plugins use the same property names and their and values. 

Keys starting with _ have strictly defined meanings specified below. It is acceptable 
to not set any of these keys if they are unknown. It is also a fatal error to set them 
to a value not specified below.

_ChromaLocation
~~~~~~~~~~~~~~~

    int _ChromaLocation

    Chroma sample position in YUV formats.

        0=left, 1=center, 2=topleft, 3=top, 4=bottomleft, 5=bottom.

        Synonyms:

        * left, mpeg2
        * center, jpeg, mpeg1

_Range
~~~~~~

    int _Range

    Full or limited range (PC/TV range). Primarily used with YUV formats, but can mark Studio RGB as well.
    Follows ITU-T H.273 numbering, and it is the same as in VapourSynth. The primary name since 3.7.7.

        1=full range, 0=limited range (RGB: narrow range).

_ColorRange
~~~~~~~~~~~

    int _ColorRange

    The older name of ``_Range``, reserved as well, with 0 and 1 swapped:

        0=full range, 1=limited range (RGB: narrow range).

    Up to 3.7.6 this was the only range property AviSynth knew. Since 3.7.7 it is used in parallel with ``_Range``,
    for some time; it stays valid, no deprecation period is set for it.

    The two names are one property, the frame holds it as ``_Range``:

    * Read, write or delete by either name reaches the same property, the last write counts. Integer 0 and 1 are
      swapped when used as ``_ColorRange``, any other integer is the same under both names.
    * A value which is not an integer (a string, for example) is not a valid range. It is stored as it is, and it
      can be read by both names with no swap.
    * An array is not a valid range either. The script functions and ``propGetInt`` swap its elements, the API's
      ``propGetIntArray`` returns it as ``_Range`` holds it.
    * ``propNumKeys`` and ``propGetKeyByIndex`` list ``_Range`` only. ``propGetAll`` and ``propShow`` give both.
    * ``props`` of ``propCopy``, ``propDelete`` and ``propShow`` accept both names. Wildcards and regular expressions
      are matched against ``_Range`` only.

    +-------------------------------+--------------------------+-------------------------------+
    | Written                       | ``propGetInt("_Range")`` | ``propGetInt("_ColorRange")`` |
    +===============================+==========================+===============================+
    | ``propSet("_ColorRange", 0)`` | 1                        | 0                             |
    +-------------------------------+--------------------------+-------------------------------+
    | ``propSet("_Range", 0)``      | 0                        | 1                             |
    +-------------------------------+--------------------------+-------------------------------+
    | ``propSet("_ColorRange", 2)`` | 2                        | 2                             |
    +-------------------------------+--------------------------+-------------------------------+

    For plugins which run on both AviSynth+ and VapourSynth, or on an older AviSynth+: read ``_Range``, when it is not
    set read ``_ColorRange`` and swap 0 and 1, when none of them is set use the default: RGB full, the others limited.
    Up to AviSynth+ 3.7.6 ``_Range`` is an ordinary property which nothing reads, there ``_ColorRange`` has to be written.
    A plugin can check whether the AviSynth it runs on understands ``_Range``: interface version 14.1 (3.7.7) or above,
    i.e. ``AEP_INTERFACE_VERSION`` above 14, or 14 with ``AEP_INTERFACE_BUGFIX`` at least 1 (C: ``AVS_AEP_xxx``, the
    loader's ``has_at_least_v14_1``).

_Primaries
~~~~~~~~~~

    int _Primaries

    Color primaries as specified in ITU-T H.265 Table E.3. 

_Matrix
~~~~~~~

    int _Matrix

    Matrix coefficients as specified in ITU-T H.265 Table E.5. 

_Transfer
~~~~~~~~~

    int _Transfer

    Transfer characteristics as specified in ITU-T H.265 Table E.4. 

_FieldBased
~~~~~~~~~~~

    int _FieldBased

    If the frame is composed of two independent fields (interlaced).

        0=frame based (progressive), 1=bottom field first, 2=top field first 

_AbsoluteTime
~~~~~~~~~~~~~

    float _AbsoluteTime

    The frame’s absolute timestamp in seconds if reported by the source filter. Should only be set 
    by the source filter and not be modified. Use durations for all operations that depend on frame length. 

_DurationNum
~~~~~~~~~~~~

    int _DurationNum

_DurationDen
~~~~~~~~~~~~

    int _DurationDen

    The frame’s duration in seconds as a rational number.
    Filters that modify the framerate should also change these values.

    This fraction (_DurationNum/_DurationDen) should always be normalized. 

_Combed
~~~~~~~

    int _Combed (boolean)

    Whether or not the frame needs postprocessing, usually hinted from field matching filters. 

_Field
~~~~~~

    int _Field

    This property signals which field was used to generate this frame.
    VapourSynth is using it in core.std.SeparateFields, in Avisynth core it is not used at all.

        0=from bottom field, 1=from top field. 

_PictType
~~~~~~~~~

    string _PictType

    A single character describing the frame type. It uses the common IPB letters but other
    letters may also be used for formats with additional frame types.

_SARNum
~~~~~~~

    int _SARNum

_SARDen
~~~~~~~

    int _SARDen

    Pixel (sample) aspect ratio as a rational number. 

_SceneChangeNext
~~~~~~~~~~~~~~~~

    int _SceneChangeNext (boolean)

    If 1, this frame is the last frame of the current scene. The next frame starts a new scene. 

_SceneChangePrev
~~~~~~~~~~~~~~~~

    int _SceneChangePrev (boolean)

    If 1, this frame starts a new scene. 

Property set
------------

Input value of setter functions are to be come either

- from the return value of "function objects"
- direct value 

Property setter function names begin with ``propSet``

Common parameters:

::

    clip c,
    string key_name,
    direct value of supported types (integer, float, string, array, clip) or a "function object"
    int "mode": 0=replace (default), 1=append, 2=touch. There is no append mode for inserting a full array into the property. 

propSet
~~~~~~~
::

    propSet(clip, string key_name, function func_obj [, integer "mode"])

generic property setter, automatic type recognition by the return value of the function object 

::

    propSet(clip, string key_name, integer value [, integer "mode"])
    propSet(clip, string key_name, float value [, integer "mode"])
    propSet(clip, string key_name, string value [, integer "mode"])
    propSet(clip, string key_name, array value)
    propSet(clip, string key_name, clip value [, integer "mode"])

The above functions are setting a property from a directly passed values 

note: array must contain only the similarly typed values, e.g. cannot mix strings with integers.

propSetInt
~~~~~~~~~~
::

    propSetInt(clip, string key_name, function func_obj [, integer "mode"])

propSetFloat
~~~~~~~~~~~~
::

    propSetFloat(clip, string key_name, function func_obj [, integer "mode"])

propSetString
~~~~~~~~~~~~~
::

    propSetString(clip, string key_name, function func_obj [, integer "mode"])

propSetArray
~~~~~~~~~~~~

::

    propSetArray(clip, string key_name, function func_obj [, integer "mode"])

propSetClip
~~~~~~~~~~~

::

    propSetClip(clip, string key_name, function func_obj [, integer "mode"])

these setters accept only the specific type 

Property get
------------

Get properties by name or as a whole.

Since AviSynth 3.7.1: allow propGetXXX property getter functions called as normal functions, outside runtime.

By default frame property values are read from frame#0 which index can be overridden by the offset parameter.

When called from inside runtime functions will return frame properties of the actual frame (+offset).


Common parameters:

::

    clip c, 
    string key_name, 
    integer "index", (default 0): for zero based indexing array access 
    integer "offset" (default 0), similar to the other runtime functions: frame offset (e.g. -1: previous, 2: next next) 

propGetAny
~~~~~~~~~~
::

    propGetAny(clip, string key_name[, integer "index", integer "offset"])

returns the automatically detected type 

propGetInt
~~~~~~~~~~
::

    propGetInt(clip, string key_name[, integer "index", integer "offset"])

returns only if value is integer, throws an error otherwise (note: unlike Avisynth integer frame properties internally use 64 bit integers) 

propGetFloat
~~~~~~~~~~~~
::

    propGetFloat(clip, string key_name[, integer "index", integer "offset"])

If the content is not of floating point type, it throws an error. Until version 3.7.3, 
the internal representation of floating point numbers was 32-bit float. Starting from 
Avisynth 3.7.4, float frame properties are read as real 64-bit doubles.

propGetString
~~~~~~~~~~~~~
::

    propGetString(clip, string key_name[, integer "index", integer "offset"])

returns only if value is string, throws an error otherwise 

propGetAsArray
~~~~~~~~~~~~~~
::

    propGetAsArray(clip, string key_name[, integer "offset"])

A frame property can hold multiple values of the same type: it can be an array.

``propGetAsArray`` returns an array. For a single property the array size will be 1.
For a property which is not set, it returns an empty (zero sized) array.
Frame type properties (set by plugins) cannot be returned, an error is thrown, like in ``propGetAny``.

propGetClip
~~~~~~~~~~~
::

    propGetClip(clip, string key_name[, integer "index", integer "offset"])

returns only if value is clip, throws an error otherwise

propGetAll
~~~~~~~~~~
::

    propGetAll(clip [, integer "offset"])

Returns all frame properties in an array of [key-value] pairs. Array size will be ``numProps``

Each key-value pair is contained in a two dimensional subarray.
If the property value itself is an array again (integer, float, string or clip) then "value" will be an array as well.
The value of a frame type property (set by plugins) is undefined.

This array contains all properties of the specific frame.

They are accessible with the associative feature of AviSynth array access. (See syntax there)

**Example:**

::

    ScriptClip("""last.propSet("cica","hello"+String(current_frame)).\
      propSetInt("test_i1",function[](clip c) { return current_frame*3 }).\
      propSet("test_i2", current_frame * 2) """)
    ScriptClip("""p = propGetAll() \
    SubTitle("size:" + String(p.ArraySize()) + " " + \
                      String(p["test_i1"]) + " " + \
                      String(p["cica"]) + " " + \
                      String(p["test_i2"]))""")
    ScriptClip("""p = propGetAll() \
    SubTitle("size:" + String(p.ArraySize()) + " " + \
       String(p[0,1]) + " " + \
       String(p[1,1]) + " " + \
       String(p[2,1]), x=0, y=20)""")

**Example (read-write basic)**

::

    ColorBars()
    
    # just practicing with function objects
    ScriptClip(function[](clip c) { c.Subtitle(String(current_frame)) })
    
    # write frame properties with function object
    ScriptClip("""propSetInt("frameprop_from_str",func(YPlaneMax))""")
    # write frame properties with traditional script string
    ScriptClip(function[](clip c) { propSetInt("frameluma_sc_func",func(AverageLuma)) })
    
    # read frame properties (function object, string)
    ScriptClip(function[](clip c) { SubTitle(string(propGetInt("frameprop_from_str")), y=20) })
    ScriptClip("""SubTitle(string(propGetInt("frameluma_sc_func")), y=40)""")
    
    return last

**Example (almost everything)**

::

    ColorBars(width=640, height=480, pixel_type="yv12", staticframes=true)
    
    ScriptClip(function[](clip c) { propSetString("s",function[](clip c) { return "Hello " + string(current_frame) }) })
    ScriptClip(function[](clip c) { propSetString("s",function[](clip c) { return "Hello array element #2 " }, mode=1) })
    ScriptClip(function[](clip c) { propSetString("s",function[](clip c) { return "Hello array element #3 "}, mode=1 ) })
    
    ScriptClip(function[](clip c) { propSetString("s2",function[](clip c) { return "Another property "} ) })
    
    ScriptClip(function[](clip c) { propSetInt("s_int",function[](clip c) { return current_frame*1 }) })
    ScriptClip(function[](clip c) { propSetInt("s_int",function[](clip c) { return current_frame*2 }, mode=1) })
    ScriptClip(function[](clip c) { propSetInt("s_int",function[](clip c) { return current_frame*4 }, mode=1 ) })
    
    ScriptClip(function[](clip c) { propSetFloat("s_float",function[](clip c) { return current_frame*1*3.14 }) })
    ScriptClip(function[](clip c) { propSetFloat("s_float",function[](clip c) { return current_frame*2*3.14 }, mode=1) })
    ScriptClip(function[](clip c) { propSetFloat("s_float",function[](clip c) { return current_frame*3*3.14 }, mode=1 ) })
    
    ScriptClip(function[](clip c) { propSetArray("s_float_arr",function[](clip c) { return [1.1, 2.2] } ) })
    ScriptClip(function[](clip c) { propSetArray("s_int_arr",function[](clip c) { return [-1,-2,-5] } ) })
    ScriptClip(function[](clip c) { propSetArray("s_string",function[](clip c) { return ["ArrayElementS_1", "ArrayElementS_2"] } ) })
    #ScriptClip("""propDelete("s")""")
    ScriptClip(function[](clip c) {
      y = 0
      SubTitle("Prop Key count =" + String(propNumKeys), y=y)
      y = y + 15
      numKeys = propNumKeys() - 1
      for ( i = 0 , numKeys) {
        propName = propGetKeyByIndex(index = i)
        propType = propGetType(propName)
        SubTitle("#"+String(i) + " property: '" + propName + "', Type = " + String(propType) , y=y)
        y = y + 15
    
        for(j=0, propNumElements(propName) - 1) {
          SubTitle("element #" + String(j) + ", size = " + String(propType == 3 ? propGetDataSize(propName, index=j) : 0) + ", Value = " + String(propGetAny(propName, index=j)), y = y)
          #SubTitle("element #" + String(j) + " size = " + String(propType == 3 ? propGetDataSize(propName, index=j) : 0) + ", Value = " + String(propGetAny(propName, index=j)), y = y)
          y = y + 15
        }
      }
      return last
    })
    
    ScriptClip(function[](clip c) {
      a = propGetAsArray("s")
      y = 100
      x = 400
      SubTitle(string(a.ArraySize()), x=x, y=y)
      for(i=0, a.ArraySize()-1) {
        SubTitle("["+String(i)+"]="+ String(a[i]),x=x,y=y)
        y = y + 15
      }
      return last
    })
    
    # get int array one pass
    ScriptClip(function[](clip c) {
      a = propGetAsArray("s_int")
      y = 440
      x = 400
      SubTitle("Array size=" + string(a.ArraySize()), x=x, y=y)
      y = y + 15
      for(i=0, a.ArraySize()-1) {
        SubTitle("["+String(i)+"]="+ String(a[i]),x=x,y=y)
        y = y + 15
      }
      return last
    })
    
    # get float array one pass
    ScriptClip(function[](clip c) {
      a = propGetAsArray("s_float")
      y = 440
      x = 200
      SubTitle("Array size=" + string(a.ArraySize()), x=x, y=y)
      y = y + 15
      for(i=0, a.ArraySize()-1) {
        SubTitle("["+String(i)+"]="+ String(a[i]),x=x,y=y)
        y = y + 15
      }
      return last
    })
    
    # get string array
    ScriptClip(function[](clip c) {
      a = propGetAsArray("s_string")
      y = 440
      x = 000
      SubTitle("Array size=" + string(a.ArraySize()), x=x, y=y)
      y = y + 15
      for(i=0, a.ArraySize()-1) {
        SubTitle("["+String(i)+"]="+ String(a[i]),x=x,y=y)
        y = y + 15
      }
      return last
    })

AviSynth 3.7.1: allow propGetXXX property getter functions called as normal functions, outside runtime.
By default frame property values are read from frame#0 which index can be overridden by the offset parameter.

**Example:**

::

    Colorbars()
    PropSet(last, "hello", 1) # Set to 1 for all frames
    # Override to 2 with runtime function except for frameNo=1
    ScriptClip("""if(current_frame!=1) {propSet("hello",2)}""")
    n0 = propGetInt("hello") # same as propGetInt("hello",offset=0)
    # or get the frame property from the Nth frame
    n1 = propGetInt("hello",offset=1)
    n2 = propGetInt("hello",offset=2)
    # n0 and n2 is 2 (overridden in runtime)
    # n1 will be 1 (keeps global setting)
    SubTitle("n0/n1/n2=" + "{n0}/{n1}/{n2}".Format)

Deleting properties
-------------------

Deletes specific properties or all property entries 

propDelete
~~~~~~~~~~
::

    propDelete(clip, string)

Deletes a property by name. If property does not exist, do nothing.

* clip (required) specifies clip.
* string (required) or string array: key_name (case sensitive) specifies the 
  
  - name of the parameter to delete 
  - a regex expression, starts with "^", ends with "$"
  - a wildcard name specifier (contains ``*``)

*Example:*

::

    ScriptClip("""propDelete("my_spec_prop")""")
    propDelete("_*") # deletes all properties starting with "_"

propClearAll
~~~~~~~~~~~~
::

    propClearAll(clip)

Clears all properties for a given video frame

* clip (required) specifies clip. 

Other property functions
------------------------

propShow
~~~~~~~~

    This debug filter lists frame properties on screen. 
    Listing appears as a name = value list and can be filtered
    
    See here: :doc:`propShow <../corefilters/propShow>`.

propGetDataSize
~~~~~~~~~~~~~~~
::

    propGetDataSize(clip, string key_name [, integer "index", integer "offset"])

returns the size of the string or underlying data array 

propNumElements
~~~~~~~~~~~~~~~
::

    propNumElements(clip, string key_name [, integer "offset"])

returns the array size of a given property. 1=single value 

propNumKeys
~~~~~~~~~~~
::

    propNumKeys(clip, [, integer "offset"])

returns number of entries (keys) for a frame 

propGetKeyByIndex
~~~~~~~~~~~~~~~~~
::

    propGetKeyByIndex(clip, integer "index" [, integer "offset")

returns the key name for the Nth property (zero based, 0<=index<propNumKeys) 

propGetType
~~~~~~~~~~~
::

    propGetType(clip, string key_name [, integer "offset"])

returns the type of the given key

* unset: 0
* integer: 1
* float: 2
* string: 3
* clip: 4
* frame: 5 

propCopy
~~~~~~~~
::

    propCopy(clip, clip [,bool "merge"], [,string[] "props"], [,bool "exclude"]

Copies the frame properties of the second clip to the first.

Parameter ``merge`` (default false): 

* when false: exact copy (original target properties will be lost)
* when true: keeps original properties, appends relevant properties from source but 
  overwrite if a parameter with the same name already exists. 

Parameter ``props`` 
  a string or string array:

  - name of the parameter to copy
  - a regex expression, starts with "^", ends with "$"
  - a wildcard name specifier (contains ``*``)
  
Parameter ``exclude``

  - when true specify that the property list is a negative list.

*Example:*

::

    PropCopy(a) # make all properties the same with of clip a
    PropCopy(a, merge=true) # Adds new properties from clip a, overwrites if already exists
    
    # Add properties from clip "a" which end with "1" or "2" 
    b.PropCopy(a, merge=true, props=["*2","*1"])
    
    # Add properties from clip "a" where not started with "_"
    b.PropCopy(a, merge=true, props="_*", exclude = True)
    
    # same with regex
    b.PropCopy(a, merge=true, props="^[^_].*$")
    
    # erase all, then copy all, except listed ones
    propCopy(org,props=["_Matrix", "_ColorRange"], exclude = true)

Changelog
---------
+----------------+--------------------------------------------------+
| Version        | Changes                                          |
+================+==================================================+
| 3.7.7          | | ``_Range``, ``_ColorRange`` is its older name  |
+----------------+--------------------------------------------------+
| 3.7.4          | | wildcard and regex support in props parameter  |
|                |   for propDelete, propShow, propCopy             |
|                | | add "props" parameter to propShow as well      |
+----------------+--------------------------------------------------+
| 3.7.2          | | exclude parameter for negative list            |
+----------------+--------------------------------------------------+
| AviSynth 3.7.1 | | allow propGetXXX called outside                |
|                |   runtime functions                              |
|                | | add propCopy                                   |
+----------------+--------------------------------------------------+
| AviSynth 3.7.0 | add propGetType                                  |
+----------------+--------------------------------------------------+

Back to :doc:`Internal functions <syntax_internal_functions>`.

$Date: 2026/11/09 11:40:00 $
