
AviSynth Syntax - Boolean functions
===================================

Boolean functions return true or false, if the condition that they test holds
or not, respectively.

IsBool
------
::

    IsBool(var)

Tests if *var* is of the bool type. *var* can be any expression allowed by
the :doc:`AviSynth Syntax <syntax>`.

*Examples:*
::

    b = false
    IsBool(b) = true
    IsBool(1 < 2 && 0 == 1) = true
    IsBool(123) = false

IsClip
------
::

    IsClip(var)

Tests if *var* is of the clip type. *var* can be any expression allowed by
the :doc:`AviSynth Syntax <syntax>`.

*Examples:*
::

    c = AviSource(...)
    IsClip(c) = true
    IsClip("c") = false

IsFloat
-------
::

    IsFloat(var)

Tests if *var* is of the float type. *var* can be any expression allowed by
the :doc:`AviSynth Syntax <syntax>`.

*Examples:*
::

    f = Sqrt(2)
    IsFloat(f) = true
    IsFloat(true) = false
    IsFloat("42.") = false
    IsFloat(2) = true   # ints are considered to be floats by this function
    IsReallyFloat(2) = false # see below

As a workaround for the issue noted above, you may use the following user function: 
::

    ## return true for floats only
    function IsReallyFloat(val v)
    {
        return (IsInt(v)==false) && IsFloat(v)
    }

IsInt
-----
::

    IsInt(var)

Tests if *var* is of the int type. *var* can be any expression allowed by the
:doc:`AviSynth Syntax <syntax>`.

*Examples:*
::

    IsInt(2) = true
    IsInt(2.1) = false
    IsInt(true) = false

IsString
--------
::

    IsString(var)

Tests if *var* is of the string type. *var* can be any expression allowed by
the :doc:`AviSynth Syntax <syntax>`.

*Examples:*
::

    IsString("test") = true
    IsString(2.3) = false
    IsString(String(2.3)) = true

IsFunction
----------
::

    IsFunction(var)

Tests if *var* is of the function type. *var* can be any expression allowed by
the :doc:`AviSynth Syntax <syntax>`.

*Examples:*
::

    function MyFunc(clip c) {
      return c.Invert()
    }
    
    IsFunction("MyFunc") = true

Defined
-------
::

    Defined(var)

Tests if *var* is defined. Can be used inside :doc:`Script functions <syntax_userdefined_scriptfunctions>` to test if
an optional argument has been given an explicit value. More formally, the
function returns false if its argument (normally a function argument or
variable) has the void ('undefined') type, otherwise it returns true.

*Examples:*
::

    b_arg_supplied = Defined(arg)
    myvar = b_arg_supplied ? ... : ...


Exist
-----
::

    Exist(string filename)

Tests if the file specified by *filename* exists.

*Examples:*
::

    filename = ...
    clp = Exist(filename) ? AviSource(filename) : Assert(false,
    \ "file: " + filename + " does not exist")


FunctionExists
--------------
::

    FunctionExists(string name)

Tests if the function or filter or :doc:`clip property <syntax_clip_properties>` name is defined in the script. 

*Examples:*
::

    ## using a filter only if it exists (AVS+ only)
    ColorBars  
    return FunctionExists("MyFilter") 
    \ ? Apply("MyFilter", Last, "TEST") 
    \ : Last 


InternalFunctionExists
----------------------
::

    InternalFunctionExists(string name)

Tests if the function, filter or property name is defined natively within AviSynth+.

Unlike `FunctionExists`, returns false for external plugins and user-defined functions. 


VarExist
---------
::

    VarExist(string name)

Tests if the "name" variable exists or not.

Note: if variable exists, it returns true regardless of the "defined" state of the variable.


Changelog
---------
+----------------+----------------------------------+
| Version        | Changes                          |
+================+==================================+
| Avisynth+      | | Added "IsFunction"             |
|                | | Added "FunctionExists"         |
|                | | Added "InternalFunctionExists" |
|                | | Added "VarExists"              |
+----------------+----------------------------------+

Back to :doc:`Internal functions <syntax_internal_functions>`.

$Date: 2024/01/05 10:54:00 $
