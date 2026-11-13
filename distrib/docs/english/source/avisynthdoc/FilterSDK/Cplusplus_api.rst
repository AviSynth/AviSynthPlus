C++ API
=======

The header, avisynth.h, declares all the classes, structures and
miscellaneous constants of the C++ API that you might need when writing
a plugin. All external plugins should #include it:
::

    #include "avisynth.h"

Note, sometimes there is a reference to a version number of the plugin
api (for example v3, v6, v8, v10). This refers to the value of
:doc:`AVISYNTH_INTERFACE_VERSION <AviSynthInterfaceVersion>`. The
classes and miscellaneous constants are described below.


.. toctree::
    :maxdepth: 4

.. contents:: Table of contents


.. _cplusplus_createscriptenvironment:

CreateScriptEnvironment
-----------------------

::

    IScriptEnvironment* __stdcall CreateScriptEnvironment(int version = AVISYNTH_INTERFACE_VERSION);


AviSynth exports this. It enables you to use AviSynth as a library,
without writing an AviSynth script or without going through AVIFile.
[todo add link]


Classes
-------


.. _cplusplus_avisyntherror:

AvisynthError
~~~~~~~~~~~~~

::

    AvisynthError(const char* _msg)


Wrap your code in try/catch statements to enable exception handling.
AvisynthError will tell you what's wrong.
::

    try
    {
        Val = Env->Invoke("Import", Args, 0);
        Clip = Val.AsClip();
        VidInfo = Clip->GetVideoInfo();
        Frame = Clip->GetFrame( 1, Env);
    }
     catch (AvisynthError err)
    {
        printf("%s\n", err.msg);
        return 1;
    }


.. _cplusplus_videoframebuffer:

VideoFrameBuffer
~~~~~~~~~~~~~~~~

VideoFrameBuffer (VFB) holds information about a memory block which is
used for video data. For efficiency, instances of this class are not
deleted when the refcount reaches zero; instead they are stored in a
linked list to be reused. In Avisynth+ this is called frame registry.
The instances are deleted when the corresponding AVS file is closed.
Or more accurately, a VideoFrameBuffer once new'd generally is not 
released until the IScriptEnvironment is deleted, except if SetMemoryMax
is exceeded by too much then not in use VideoFrameBuffer's are forcible
deleted until SetMemoryMax is satisfied.


.. _cplusplus_videoframe:

VideoFrame
~~~~~~~~~~

VideoFrame holds a "window" into a VideoFrameBuffer, and since v10 it can
store the exact underlying video format as well. Operator new is
overloaded to recycle class instances. Its members can be called by:
::

    PVideoFrame src = child->GetFrame(n, env);
    src->GetReadPtr(..)


VideoFrame has the following members: GetPitch, GetRowSize, GetHeight,
GetReadPtr, GetWritePtr, IsWritable, IsPropertyWritable (v9),
GetPixelType (v10) and AmendPixelType (v10).

The getter functions (except GetPixelType) will give you a property (pitch,
rowsize, etc ...) of a plane (of the frame it points to). The
interleaved formats (BGR(A) or YUY2) consist of one plane, and the
planar formats consists of one (Y), three (YUV, planar RGB) or four 
(YUVA, planar RGBA) planes. The default plane is just the first plane
which is plane Y for the planar YUV formats (and G for planar RGB).

GetPixelType was introduced in v10, the exact video format of the frame
is now stored in VideoFrame::pixel_type. Before, there was no reliable way
of knowing it on a frame from propGetFrame.

The pixel_type is automatically maintained behind the scenes.

pixel_type is set on calling NewVideoFrame, and is kept with MakeWritable.
Calling SubFrame will automatically convert the format to a single plane
greyscale Y8-Y32 from planar origins. Calling SubframePlanar will strip
alpha from the format specifier.

AmendPixelType can be used in special cases, to override the pixel_type.
E.g. when a filter just overrides the format of VideoInfo in its constructor
but would return the frame unaltered, which would inherit a wrong pixel_type.


.. _cplusplus_getpitch:

GetPitch
^^^^^^^^

::

    int GetPitch(int plane=0) const;


The "pitch" (also called stride) of a frame buffer is the offset (in
bytes) from the beginning of one scan line to the beginning of the
next. The source and destination buffers won't necessarily have the
same pitch. The pitch can vary among frames in a clip, and it can
differ from the width of the clip. [todo add link]

| The scan line will be padded to a multiple of 8 or 16 (classic Avisynth) 
  or even 64 bytes (Avisynth+) due to speed reasons, so the pitch will 
  always be a multiple of that (e.g. mod64). Image processing is expensive, 
  so SIMD instructions are used to speed tasks up:

| SSE uses 128 bit = 16 byte registers, so 16 byte-pixels (4 floats) can be processed
  the same time.

| AVX uses 256 bit = 32 byte registers, so 32 byte-pixels (8 floats) can be
  processed the same time.

| AVX512 uses 512 bit = 64 byte registers, so 64 byte-pixels (16 floats) can be
  processed the same time.

NOTE that the pitch can change anytime, so in most use cases you must
request the pitch dynamically.


Usage:

GetPitch must be used on every plane (interleaved like YUY2 means 1
plane...) of every PVideoFrame that you want to read or write to. It is
the only way to get the size of the Video Buffer (e.g. get the size of
PVideoFrame):
::

    int buffer_size = src->GetPitch() * src->GetHeight(); //YUY2, interleaved


This will give you the pitch of the U-plane (it will be zero if the
plane doesn't exist):
::

    PVideoFrame src = child->GetFrame(n, env);
    const int src_pitchUV = src->GetPitch(PLANAR_U);


.. _cplusplus_getrowsize:

GetRowSize
^^^^^^^^^^

::

    int GetRowSize(int plane=0) const;


GetRowSize gives the length of each row in bytes (thus not in pixels).
It's usually equal to the pitch or slightly less, but it may be
significantly less if the frame in question has been through Crop. This
will give you the rowsize of a frame for the interleaved formats, or
the rowsize of the Y-plane for the planar formats (being the default
plane).
::

    const int src_width = src->GetRowSize();


.. _cplusplus_getheight:

GetHeight
^^^^^^^^^

::

    int GetHeight(int plane=0) const;


GetHeight gives the height of the plane in pixels.


.. _cplusplus_getreadptr:

GetReadPtr
^^^^^^^^^^

::

    const BYTE* GetReadPtr(int plane=0) const;


GetReadPtr gives you a read pointer to a plane. This will give a read
pointer to the default plane:
::

    PVideoFrame src = child->GetFrame(n, env);
    const unsigned char* srcp = src->GetReadPtr()


.. _cplusplus_getwriteptr:

GetWritePtr
^^^^^^^^^^^

::

    BYTE* GetWritePtr(int plane=0) const;


GetWritePtr gives you a write pointer to a plane.

Any buffer you get from NewVideoFrame is guaranteed to be writable (as
long as you only assign it to one PVideoFrame). Our filter's dst came
from NewVideoFrame, so we can safely call dst->GetWritePtr(). However,
frames you get from other clips via GetFrame may not be writable, in
which case GetWritePtr() will return a null pointer.
::

    PVideoFrame dst = env->NewVideoFrame(vi);
    unsigned char* dstp = dst->GetWritePtr();


If you want to write a frame which is not new (the source frame for
example), you will have to call MakeWritable first:
::

    PVideoFrame src = child->GetFrame(n, env);
    env->MakeWritable(&src);
    unsigned char* srcp = src->GetWritePtr(PLANAR_Y);


See IsWritable for more details.


.. _cplusplus_iswritable:

IsWritable
^^^^^^^^^^

::

    bool IsWritable() const;


All frame buffers are readable, but not all are writable. This method
can be used to find out if a buffer is writable or not, and there's a
MakeWritable callback (described below) to ensure that it is.

The rule about writability is this: A buffer is writable if and only if
there is exactly one PVideoFrame pointing to it. In other words, you
can only write to a buffer if no one else might be reading it. This
rule guarantees that as long as you hold on to a PVideoFrame and don't
write to it yourself, that frame will remain unchanged. The only
drawback is that you can't have two PVideoFrames pointing to a writable
buffer.

MakeWritable makes the properties writable as well.
::

    PVideoFrame src = child->GetFrame(n, env);
    if (src->IsWritetable()) {...}


.. _cplusplus_ispropertywritable:

IsPropertyWritable V9
^^^^^^^^^^^^^^^^^^^^^

::

    bool IsPropertyWritable() const;


All frame properties connected to frame buffers are readable, but not all are writable.
This method can be used to find out if a property set is writable or not.

The rule about writability is this: A buffer is writable if and only if
there is exactly one PVideoFrame pointing to it. In other words, you
can only write to a buffer if no one else might be reading it. This
rule guarantees that as long as you hold on to a PVideoFrame and don't
write to it yourself, that frame will remain unchanged.

See also :ref:`getFramePropsRW <cplusplus_getframepropsrw>`.

::

    PVideoFrame src = child->GetFrame(n, env);
    if (!src->IsPropertyWritable())
      env->MakePropertyWritable(&src);
    }
    AVSMap *props = env->getFramePropsRW(dst);


.. _cplusplus_getpixeltype:

GetPixelType V10
^^^^^^^^^^^^^^^^

::

    int GetPixelType() const;


Since v10 a Videoframe can store the exact underlying video format in 
VideoFrame::pixel_type. Before, there was no reliable way of knowing it on a 
frame from :ref:`propGetFrame <cplusplus_propgetframe>`.

The pixel_type is automatically maintained behind the scenes.

pixel_type is set on calling :ref:`NewVideoFrame <cplusplus_newvideoframe>` or 
:ref:`NewVideoFrameP <cplusplus_newvideoframep>`, and is kept with
:ref:`MakeWritable <cplusplus_makewritable>`.

Calling :ref:`SubFrame <cplusplus_subframe>` will automatically convert 
the format to a single plane greyscale Y8-Y32 from planar origins.

Calling :ref:`SubFramePlanar <cplusplus_subframeplanar>` will strip alpha
from the format specifier.


.. _cplusplus_amendpixeltype:

AmendPixelType V10
^^^^^^^^^^^^^^^^^^

::

    void AmendPixelType(int new_pixel_type);


AmendPixelType can be used in special cases, to override the pixel_type.
E.g. when a filter just overrides the format (VideoInfo::pixel_type) in its constructor
but otherwise would return the frame unaltered, this results in an inconsistent
format between the actual VideoFrame and VideoInfo.
(Filters which are now using AmendPixelType are ConvertFromDoubleWidth, ConvertToDoubleWidth,
ConvertBits and CombinePlanes)


Changes the color format metadata on this frame. Using it on a frame that isn't
writable leads to an inconsistent state, because other filters depend on it.
So, use :ref:`MakeWritable <cplusplus_makewritable>` before.

::

    PVideoFrame src = child->GetFrame(n, env);
    if (format_change_only)
    {
      // for 10-16 bit: simple format override in constructor
      env->MakeWritable(&src);
      src->AmendPixelType(vi.pixel_type);
      return src;
    }


.. _cplusplus_alignplanar:

AlignPlanar
~~~~~~~~~~~

::

    AlignPlanar(PClip _clip);


AlignPlanar does nothing, if the pitch of a frame is at least mod16 (16
bytes, being the default frame alignment for luma and chroma).
Otherwise it realigns the image, by blitting it to a larger buffer.

Filters can enforce a lower pitch, but they must always apply the
AlignPlanar filter after itself, if they intend to return a frame with
a lower pitch. VFW delivers a 4 byte alignment for example, so the
AlignPlanar filters needs to be applied on all frames when using
AviSource.



.. _cplusplus_iscriptenvironment:

IScriptEnvironment
~~~~~~~~~~~~~~~~~~

AviSynth exports an IScriptEnvironment interface. It enables you to use
AviSynth as a library, without writing an AVS script or without going
through AVIFile. Its members can be called by:
::

    IScriptEnvironment* env
    env->Invoke(..)


IScriptEnvironment has the following members: ThrowError, GetCPUFlags,
SaveString, Sprintf, VSprintf, Invoke, BitBlt, AtExit, AddFunction,
MakeWritable, FunctionExists, GetVar, GetVarDef, SetVar, SetGlobalVar,
PushContext, PopContext, NewVideoFrame, CheckVersion, Subframe,
SubframePlanar, SetMemoryMax, SetWorkingDir, DeleteScriptEnvironment
and ApplyMessage and many others They are described in the following 
subsections.


.. _cplusplus_throwerror:

ThrowError
^^^^^^^^^^

::

    __declspec(noreturn) virtual void __stdcall ThrowError(const char* fmt, ...) = 0;


ThrowError throws an exception (of type AvisynthError). Usually, your
error message will end up being displayed on the user's screen in lieu
of the video clip they were expecting:
::

    if (!vi.IsRGB()) {
        env->ThrowError("RGBAdjust requires RGB input");
    }


.. _cplusplus_getcpuflags:
.. _cplusplus_getcpuflagsex:

GetCPUFlags
^^^^^^^^^^^
GetCPUFlagsEx (v12)
^^^^^^^^^^^^^^^^^^^

::

    virtual int GetCPUFlags();
    virtual int64_t GetCPUFlagsEx();


GetCPUFlagsEx (and the old GetCPUFlags) returns the instruction set of your CPU. 
Interface V12 introduced the Ex version which returns 64 bit flags.
Reason: the many Intel AVX512 subfeatures did not fit in the 32 bits int returned by the original function.

** Intel architecture flags: **

Though individual AVX512 features can be tested, the recommended way is to
test for group of features. Avisynth supports the following AVX512 group feature flags:
CPUF_AVX512_BASE, CPUF_AVX512_FAST, and later probably: CPUF_AVX10 with version>=2.

The GetCPUFlags() - the old 32 bit version - is still usable up to AVX512_FAST group feature set.
Other future group features and individual flags will only be available through GetCPUFlagsEx.

For features up to AVX2 and some AVX512 exensions, both functions return the same flags.

To find out if you're running for example on a CPU that supports AVX2, test:
::

    env->GetCPUFlags() & CPUF_AVX2

To test against the default Avisynth+ "fast" flags, test:
::

    if ((env->GetCPUFlags() & CPUF_AVX512_FAST) == CPUF_AVX512_FAST) {
        // all "fast" AVX512 features supported (in AviSynth: *_avx512.cpp)
        // function dispatch here
    } else if ((env->GetCPUFlags() & CPUF_AVX512_BASE) == CPUF_AVX512_BASE)) {
        // "base" AVX512 features supported (in AviSynth: *_avx512b.cpp)
        // function dispatch here
    } else if (env->GetCPUFlags() & CPUF_AVX2) {
        // AVX2 path
    } else if (env->GetCPUFlags() & CPUF_SSE2) {
        // SSE2 path
    } else {
        // simple C++ path
    }

In Avisynth AVX512 function dispatchers check for CPUF_AVX512_BASE and CPUF_AVX512_FAST for calling function from 
``*_avx512b.cpp`` or ``*_avx512.cpp`` files, repectively.

Compiler flags for ``*_avx512b.cpp`` and ``*_avx512.cpp`` files are automatically set for MSVC as ``/arch:AVX512``.

For gcc or LLVM (clang-cl) builds the relevant flags in CMakeLists.txt are set as

* "Base" (``*_avx512b.cpp``) : ``" -mfma -mbmi2 -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl "``
* "Fast" (``*_avx512.cpp``) : ``" -mfma -mbmi2 -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mavx512vbmi -mavx512vbmi2 -mavx512bitalg -mavx512vpopcntdq "``

Similar to AVX2, FMA must be added explicitly for GCC/Clang. We also enable BMI2, as it is standard since AVX2 and useful for 
AVX512 mask operations like _bzhi_u64/u32. The core subsets (F, CD, BW, DQ, VL) comprising the CPUF_AVX512_BASE 
flag are present on all AVX-512-capable architectures.

We note again: early Xeon (e.g., Skylake-X/Cascadelake) may exhibit severe thermal throttling even on a single thread. 
In AviSynth+, AVX512_BASE must be manually enabled in-script unless ``CPUF_AVX512_FAST`` is also detected.
Functions in ``*_avx512b.cpp`` should be dispatched on ``CPUF_AVX512_BASE``.
Functions in ``*_avx512.cpp`` should be dispatched on ``CPUF_AVX512_FAST``.

AVX512 is considered to be "Fast" when either 

- pre AVX10, and minimum Ice Lake architecture is found (VNNI, VBMI, VBMI2, BITALG, VPOPCNTDQ and three crypto flags).
- or AVX10 (any version) is found.

Ice Lake and AVX10 common flags are: VNNI, VBMI, VBMI2, BITALG, VPOPCNTDQ; this is what you can expect from CPUF_AVX512_FAST feature flag.

** Aarch64 architecture flags: **

See :ref:`Aarch64 (ARM64) SIMD tiers<aarch64_simd_tiers>` for details about ARMv8/v9 CPU feature flags.



There's a complete list of flags in ``avs/cpuid.h`` (C++) or in ``avisynth_c.h`` (C).

See also :ref:`CPU Feature Flags<cplusplus_cpufeatureflags>`.


.. _cplusplus_savestring:

SaveString
^^^^^^^^^^

::

    virtual char* SaveString(const char* s, int length = -1);


This function copies its argument to a safe "permanent" location and
returns a pointer to the new location. Each ScriptEnvironment instance
has a buffer set aside for storing strings, which is expanded as
needed. The strings are not deleted until the ScriptEnvironment
instance goes away (when the script file is closed, usually). This is
usually all the permanence that is needed, since all related filter
instances will already be gone by then.

Though the returned pointer is not const-qualified, don't overwrite the buffer,
due to the string cacheing introduced after Avisynth 3.7.3. 

Until Avisynth 3.7.3 you could safely do this, the documentation said:
"you're welcome to write to it, as long as you don't stray beyond the 
bounds of the string."

This (formerly valid) example usage is no longer safe (converting a string to upper case)
and was replaced in Avisynth core:
::

    AVSValue UCase(AVSValue args, void*, IScriptEnvironment* env) {
        return _strupr(env->SaveString(args[0].AsString()));
    }


.. _cplusplus_sprintf_vsprintf:

Sprintf and VSprintf
^^^^^^^^^^^^^^^^^^^^

::

    virtual char* Sprintf(const char* fmt, ...);
    virtual char* VSprintf(const char* fmt, char* val);


These store strings away in the same way as SaveString, but they treat
their arguments like printf and vprintf. Currently there's a size limit
of 4096 characters on strings created this way. (The implementation
uses _vsnprintf, so you don't need to worry about buffer overrun.)


.. _cplusplus_invoke:

Invoke
^^^^^^

::

    virtual AVSValue Invoke(const char* name, const AVSValue args, const char** arg_names=0);


You can use this to call a script function. There are many script
functions which can be useful from other filters; for example, the Bob
filter uses SeparateFields, and several source filters use
UnalignedSplice. Some functions, like Weave, are implemented entirely
in terms of other functions. If you're calling a function taking
exactly one argument, you can simply pass it in the args parameter;
Invoke will convert it into an array for you. In order to call a
function taking multiple arguments, you will need to create the array
yourself; it can be done like this:
::

    AVSValue up_args[3] = {child, 384, 288};
    PClip resized = env->Invoke("LanczosResize", AVSValue(up_args,3)).AsClip();


In this case LanczosResize would need to have a parameter-type string
like "cii".

The arg_names parameter can be used to specify named arguments. Named
arguments can also be given positionally, if you prefer.

Invoke throws IScriptEnvironment::NotFound if it can't find a matching
function prototype. You should be prepared to catch this unless you
know that the function exists and will accept the given arguments.


.. _cplusplus_bitblt:

BitBlt
^^^^^^

::

    virtual void BitBlt(unsigned char* dstp, int dst_pitch, const unsigned char* srcp, int src_pitch, int row_size, int height);


This brilliantly-named function does a line-by-line copy from the
source to the destination. It's useful for quite a number of things;
the built-in filters DoubleWeave, FlipVertical, AddBorders,
PeculiarBlend, StackVertical, StackHorizontal, and ShowFiveVersions all
use it to do their dirty work.

In AddBorders it's to copy the Y-plane from the source to the
destination frame (for planar formats):
::

    const int initial_black = top*dst_pitch + vi.BytesFromPixels(left);
    if (vi.IsPlanar()) {
        BitBlt(dstp+initial_black, dst_pitch, srcp, src_pitch, src_row_size, src_height);
        ...
    }


left is the number of pixels which is added to the left, top the number
which is added to the top. So the first source pixel, srcp[0], is
copied to its new location dstp[x], and so on. The remaining bytes are
zeroed and can be refilled later on.


.. _cplusplus_atexit:

AtExit
^^^^^^

::

    virtual void AtExit(ShutdownFunc function, void* user_data);


When IScriptEnvironment is deleted on script close the AtExit functions
get run. When you register the function you can optionally provide some
user data. When the function is finally called this data will be
provided as the argument to the procedure.

The example below (thanks to tsp) loads a library and automatically
unloads it (by using AtExit) after the script is closed. It can be
useful when your plugin depends on a library and you want to load the
library in your script (the plugin fft3dfilter.dll depends on the
library fftw3.dll for example):
::

    void __cdecl UnloadDll(void* hinst, IScriptEnvironment* env) {
        if (hinst)
        FreeLibrary(static_cast<HMODULE>(hinst));
    }

    AVSValue __cdecl LoadDll(AVSValue args, void* user_data, IScriptEnvironment* env){
        HMODULE hinst = 0;
        hinst = LoadLibrary(args[0].AsString()); // loads a library
        env->AtExit(UnloadDll, hinst); // calls UnloadDll to unload the library upon script exit
        return hinst!=NULL;
    }


.. _cplusplus_addfunction:

AddFunction
^^^^^^^^^^^

::

    virtual void __stdcall AddFunction(const char* name, const char* params, ApplyFunc apply, void* user_data) = 0;


The main purpose of the AvisynthPluginInit2 (or AvisynthPluginInit3)
function is to call env->AddFunction.
::

    env->AddFunction("Sepia", "c[color]i[mode]s", Create_Sepia, 0);


AddFunction is called to let Avisynth know of the existence of our
filter. It just registers a function with Avisynth's internal function
table. This function takes four arguments: the name of the new script
function; the parameter-type string; the C++ function implementing the
script function; and the user_data cookie.

.. _cplusplus_plugin_export:

Exporting the plugin entry point
""""""""""""""""""""""""""""""""

``AvisynthPluginInit3`` must be visible in the dynamic symbol table of the plugin.
From 3.7.7 ``avisynth.h`` defines the macro ``AVS_PLUGIN_EXPORT`` for it:
::

    extern "C" AVS_PLUGIN_EXPORT const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)

- Windows: it is ``__declspec(dllexport)``.
- Linux, macOS and other POSIX systems: it is ``__attribute__((visibility("default")))``.
  Use it if the plugin is built with ``-fvisibility=hidden`` (CMake: ``CXX_VISIBILITY_PRESET hidden``), as a plugin
  built that way would otherwise hide the entry point and AviSynth could not load it.

The older spelling ``extern "C" __declspec(dllexport)`` (as in the samples below) still works on every platform.
On POSIX ``__declspec(...)`` expands to nothing, so it has no effect on the visibility there:
the plugin has to be built with the default (visible) symbol visibility.
For the C interface, ``AVSC_EXPORT`` of ``avisynth_c.h`` does the same job
(``avisynth_c_plugin_init`` and ``avisynth_c_plugin_init2``).

Since 3.7.7 the AviSynth+ core itself on Linux and macOS exports only the C API (``avs_*``),
``CreateScriptEnvironment`` and ``CreateScriptEnvironment2``. A plugin gets everything else through ``AVS_Linkage``
and the ``IScriptEnvironment`` interface. Symbols such as ``BitBlt`` or ``AVS_linkage`` of the core were exported
by earlier versions only because no symbol visibility was set; they are not part of the interface.

The added function is of type AVSValue and can therefore return any
AVSValue. Here are a few options how to return from the "added"
function:
::

    AVSValue __cdecl returnSomething(AVSValue args, void* user_data, IScriptEnvironment* env){

    char *strlit = "AnyOldName";
    int len = strlen(strlit);
    char *s = new char[len+1];

    if (s==NULL)
        env->ThrowError("Cannot allocate string mem");

    strcpy(s, strlit); // duplicate
    char *e = s+len; // point at null

    // make safe copy of string (memory is freed on Avisynth closure)
    AVSValue ret = env->SaveString(s,e-s); // e-s is text len only (excl null) {SaveString uses memcpy)

    // alternative, Avisynth uses strlen to ascertain length
    // AVSValue ret = env->SaveString(s);

    delete []s; // delete our temp s buffer
    return ret; // return saved string as AVSValue

    // alternative to MOST of above code char* converted to AVSValue.
    // return strlit;

    // alternative to ALL of above code char* converted to AVSValue.
    // return "AnyOldName";

    // String literals are read only and at constant address and so need not be saved.
    }

see also :ref:`c_avs_add_function` and :ref:`c_avs_add_function_r` in C API


.. _cplusplus_makewritable:

MakeWritable
^^^^^^^^^^^^

::

    virtual bool __stdcall MakeWritable(PVideoFrame* pvf) = 0;


MakeWritable only copies the active part of the frame to a completely
new frame with a default pitch. You need this to recieve a valid write
pointer to an existing frame.
::

    PVideoFrame src = child->GetFrame(n, env);
    env->MakeWritable(&src);


.. _cplusplus_functionexists:

FunctionExists
^^^^^^^^^^^^^^

::

    virtual bool __stdcall FunctionExists(const char* name) = 0;


FunctionExists returns true if the specified filter exists, otherwise
returns false:
::

    if (env->FunctionExists("Import")) {
        env->ThrowError("Yes, the IMPORT function exist.");
    } else {
        env->ThrowError("No, the IMPORT function don't exist.");
    }


.. _cplusplus_getvar:

GetVar
^^^^^^

::

    virtual AVSValue __stdcall GetVar(const char* name) = 0;


GetVar can be used to access AviSynth variables. It will throw an error
if the variable doesn't exist.

Internal and external (plugin) functions are, for example, exported as
AviSynth variables:

* $InternalFunctions$ Should contain a string consisting of function
  names of all internal functions.
* $InternalFunctions!Functionname!Param$ Should contain all
  parameters for each internal function.
* $PluginFunctions$ Should contain a string of all plugins in your
  autoloading plugin folder.
* $Plugin!Functionname!Param$ Should contain all parameters.

Use env->GetVar() to access them. This example returns a string
consisting of all parameters of ConvertToYV12:
::

    const char* plugin_dir;
    plugin_dir = env->GetVar("$Plugin!ConverttoYV12!Param$").AsString();


This example returns the plugin folder which is used to autoload your
plugins (and returns an error if it's not set):
::

    try {
        const char* plugin_dir;
        plugin_dir = env->GetVar("$PluginDir$").AsString();
        env->ThrowError(plugin_dir);
    } catch(...) {
        env->ThrowError("Plugin directory not set.");
    }


If you are making a conditional filter you can use it to get the
current framenumber:
::

    // Get current frame number
    AVSValue cf = env->GetVar("current_frame");
    if (!cf.IsInt())
        env->ThrowError("MinMaxAudio: This filter can only be used within ConditionalFilter");
    int n = cf.AsInt();
    PVideoFrame src = child->GetFrame(n, env);


.. _cplusplus_getvardef:

GetVarDef, v6
^^^^^^^^^^^^^

::

    virtual AVSValue __stdcall GetVarDef(const char* name, const AVSValue& def=AVSValue()) = 0;


GetVarDef can be used to access AviSynth variables. It will return
'def' if the variable doesn't exist (instead of throwing an error):
::

    int error;
    AVSValue error = env->GetVarDef("VarUnknown", AVSValue(-1)); // returns -1 when 'VarUnknown' doesn't exist
    if (error==-1)
        env->ThrowError("Plugin: The variable 'VarUnknown' doesn't exist!");


.. _cplusplus_setvar:

SetVar
^^^^^^

::

    virtual bool __stdcall SetVar(const char* name, const AVSValue& val) = 0;


It will return true if the variable was created and filled with the
given value. It will return false in case the variable was already
there and we just updated its value.

SetVar can be used to set/create AviSynth variables. The created
variables are only visible in the local scope, e.g. script functions
have a new scope.

This example sets the autoloading plugin folder to ``"C:\\"``
::

    if (env->SetVar("$PluginDir$", AVSValue("C:\\"))) {
        //variable was created
    } else {
        //variable was already existing and updated
    }


This example sets variables in GetFrame which can be accessed later on
in a script within the conditional environment:
::

    // saves the blue value of a pixel
    int BlueValue;
    BlueValue = srcp[x];
    env->SetVar("BlueValue", AVSValue(BlueValue));


.. _cplusplus_setglobalvar:

SetGlobalVar
^^^^^^^^^^^^

::

    virtual bool __stdcall SetGlobalVar(const char* name, const AVSValue& val) = 0;


Usage:

SetGlobalVar can be used to create or set AviSynth variables that are
visible within global scope. It is possible that a single filter may
want to use SetVar in order to exchange signals to possible other
instances of itself.

There are at least 4 different components that make use of
Set(Global)Var functions:

* the core itself
* the user within the avs script
* filters/plugins
* a custom application that invoked the environment

All of above may have their own requirements for the SetVar function.
Some may want to be visible globally, others may not.


.. _cplusplus_pushcontext:

PushContext
^^^^^^^^^^^

::

    virtual void __stdcall PushContext(int level=0) = 0;

Usage:

PushContext and PopContext are used to implement local script function
name space contexts. A plugin could use these to create a new local name 
space before Invoke'ing or Eval'ing script expressions in order to avoid 
clashes with similarly named variables in the calling script.

PushContext() and PopContext() should always be used in matched pairs, and 
the code region of interest be try/catch protected with a PopContext() in 
the catch case.

Example from ``ScriptFunction::Execute``:
::

    env->PushContext();
    for (int i=0; i<args.ArraySize(); ++i)
      env->SetVar(self->param_names[i], // Force float args that are actually int to be float
        (self->param_floats[i] && args[i].IsInt()) ? float(args[i].AsInt()) : args[i]);
    AVSValue result;
    try {
      result = self->body->Evaluate(env);
    }
    catch (...) {
      env->PopContext();
      throw;
    }
    env->PopContext();

| // TODO - see (also similar functions)
| http://forum.doom9.org/showthread.php?p=1595750#post1595750


.. _cplusplus_popcontext:

PopContext
^^^^^^^^^^

::

    virtual void __stdcall PopContext() = 0;

Usage:

See PopContext


.. _cplusplus_newvideoframe:

NewVideoFrame
^^^^^^^^^^^^^

::

    virtual PVideoFrame __stdcall NewVideoFrame(const VideoInfo& vi, int align=FRAME_ALIGN) = 0;
    // default align is 16

See also :ref:`NewVideoFrameP <cplusplus_newvideoframep>`.

The NewVideoFrame callback allocates space for a video frame of the
supplied size. (In this case it will hold our filter's output.) The
frame buffer is uninitialized raw memory (except that in the debug
build it gets filled with the repeating byte pattern 0A 11 0C A7 ED,
which is easy to recognize because it looks like "ALLOCATED"). "vi" is
a protected member of GenericVideoFilter. It is a structure of type
VideoInfo, which contains information about the clip (like frame size,
frame rate, pixel format, audio sample rate, etc.). NewVideoFrame uses
the information in this structure to return a frame buffer of the
appropriate size.

The following example creates a new VideoInfo structure and creates a
new video frame from it:
::

    VideoInfo vi;
    PVideoFrame frame;
    memset(&vi, 0, sizeof(VideoInfo));
    vi.width = 640;
    vi.height = 480;
    vi.fps_numerator = 30000;
    vi.fps_denominator = 1001;
    vi.num_frames = 107892; // 1 hour
    vi.pixel_type = VideoInfo::CS_BGR32;
    vi.sample_type = SAMPLE_FLOAT;
    vi.nchannels = 2;
    vi.audio_samples_per_second = 48000;
    vi.num_audio_samples = vi.AudioSamplesFromFrames(vi.num_frames);
    frame = env->NewVideoFrame(vi);



.. _cplusplus_checkversion:

CheckVersion
^^^^^^^^^^^^

::

    virtual void __stdcall CheckVersion(int version = AVISYNTH_INTERFACE_VERSION) = 0;


CheckVersion checks the interface version (avisynth.h). It throws an
error if 'version' is bigger than the used interface version. The
following interface versions are in use:

AVISYNTH_INTERFACE_VERSION = 1 (v1.0-v2.0.8), 2 (v2.5.0-v2.5.5), 3
(v2.5.6-v2.5.8), 5 (v2.6.0a1-v2.6.0a5), 6 (v2.6.0), 8 (Avisynth+) from a specific build [version 4 doesn't exist].

This example will throw an error if v2.5x or an older AviSynth version
is being used:
::

    env->CheckVersion(5)


This can be used in a plugin, for example, if it needs at least a
certain interface version for it to work.

Interface V9 (8.1) introduced new methods for establishing actual interface version both on C++ and C interfaces.
See :ref:`GetEnvProperty <cplusplus_getenvproperty>`

Interface V9.1 introduced important fixes for C interface methods: avs_new_video_frame_p(_a), avs_prop_get_data

Interface V12 introduced global locks, GetCPUFlagsEx, query L2 cache size

::

    Example of usage with CPP interface (through avisynth.h).

    IScriptEnvironment *env = ...
    avisynth_if_ver = 6; // guessed minimum
    avisynth_bugfix_ver = 0;
    try { env->CheckVersion(8); avisynth_if_ver = 8; }
    catch (const AvisynthError&) {}
    try { 
      env->CheckVersion(9); // if this works, we are at least V9, can use GetEnvProperty with AEP_INTERFACE_VERSION
      avisynth_if_ver = env->GetEnvProperty(AEP_INTERFACE_VERSION); // only since V9!
      avisynth_bugfix_ver = env->GetEnvProperty(AEP_INTERFACE_BUGFIX);      
    } 
    catch (const AvisynthError&) {}
    
    has_at_least_v8 = avisynth_if_ver >= 8; // frame properties, NewVideoFrameP, other V8 environment functions
    has_at_least_v8_1 = avisynth_if_ver > 8 || (avisynth_if_ver == 8 && avisynth_bugfix_ver >= 1);
    // 8.1: C interface frameprop access fixed, IsPropertyWritable/MakePropertyWritable support, extended GetEnvProperty queries
    has_at_least_v9 = avisynth_if_ver >= 9; // future
    has_at_least_v9_1 = avisynth_if_ver > 9 || (avisynth_if_ver == 9 && avisynth_bugfix_ver >= 1);
    // 9.1: C interface fixes: avs_new_video_frame_p(_a), avs_prop_get_data
    has_at_least_v12 = avisynth_if_ver >= 12; // global locks, GetCPUFlagsEx, query L2 cache size
    

.. _cplusplus_subframe:

Subframe
^^^^^^^^

::

    virtual PVideoFrame __stdcall Subframe(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height) = 0;


Subframe (for interleaved formats) extracts a part of a video frame.
For planar formats use SubframePlanar. For examples see SubframePlanar.


.. _cplusplus_subframeplanar:

SubframePlanar
^^^^^^^^^^^^^^

::

    virtual PVideoFrame __stdcall SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV) = 0;


SubframePlanar (for planar formats) extracts a part of a video frame.
The example below returns the first field of a frame:
::

    vi.height >>= 1; // sets new height in the constructor
    PVideoFrame frame = child->GetFrame(n, env);
    if (vi.IsPlanar()) { // SubframePlanar works on planar formats only
        const int frame_pitch = frame->GetPitch(PLANAR_Y);
        const int frame_width = frame->GetRowSize(PLANAR_Y);
        const int frame_height = frame->GetHeight(PLANAR_Y);
        const int frame_pitchUV = frame->GetPitch(PLANAR_U);
        return env->SubframePlanar(frame, 0, 2*frame_pitch, frame_width, frame_height>>1, 0, 0, 2*frame_pitchUV);
    }


Note that it copies the first row of pixels and moves on to the third
row (by moving the offset by '2*frame_pitch'). After frame_height/2 it
stops reading.

The following example keeps the left 100 pixels of a clip (it leaves
the height unaltered) and throws away the rest:
::

    vi.width = 100; // sets new width in the constructor
    PVideoFrame frame = child->GetFrame(n, env);
    if (vi.IsPlanar()) { // SubframePlanar works on planar formats only
        const int frame_pitch = frame->GetPitch(PLANAR_Y);
        const int frame_height = frame->GetHeight(PLANAR_Y);
        const int frame_pitchUV = frame->GetPitch(PLANAR_U);
        return env->SubframePlanar(frame, 0, frame_pitch, 100, frame_height, 0, 0, frame_pitchUV);
    }


Note that it copies 100 pixels and moves on to the next row (by moving
the offset by 'frame_pitch').

You need to check somewhere that the source frames is more than 100
pixels wide, otherwise throw an error.


.. _cplusplus_setmemorymax:

SetMemoryMax
^^^^^^^^^^^^

::

    virtual int __stdcall SetMemoryMax(int mem) = 0;


There is a builtin cache automatically inserted in between all filters.
You can use SetmemoryMax to increase the size.

SetMemoryMax only sets the size of the frame buffer cache. It is
independent of any other memory allocation. Memory usage due to the
frame cache should ramp up pretty quickly to the limited value and stay
there. Setting a lower SetMemoryMax value will make more memory
available for other purposes and provide less cache buffer frames. It
is pointless having more buffers available than are needed by the
scripts temporal requirements. If each and every frame generated at
each and every stage of a script is only ever used once then the cache
is entirely useless. By definition a cache is only useful if a
generated element is needed a second or subsequent time.


.. _cplusplus_setworkingdir:

SetWorkingDir
^^^^^^^^^^^^^

::

    virtual int __stdcall SetWorkingDir(const char * newdir) = 0;


Sets the default directory for AviSynth.


.. _cplusplus_deletescriptenvironment:

DeleteScriptEnvironment, v5
^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    virtual void __stdcall DeleteScriptEnvironment() = 0;


Provides a method to delete the ScriptEnvironment which is created with
CreateScriptEnvironment. End of the world, end of everything. AtExit procedures are 
called during the destroy as a side effect.


.. _cplusplus_applymessage:

ApplyMessage, v5
^^^^^^^^^^^^^^^^
ApplyMessageEx, v12
^^^^^^^^^^^^^^^^^^^

::

    virtual void _stdcall ApplyMessage(PVideoFrame* frame, const VideoInfo& vi, const char* message, int size, int textcolor, int halocolor, int bgcolor) = 0;
    virtual void _stdcall ApplyMessageEx(PVideoFrame* frame, const VideoInfo& vi, const char* message, int size, int textcolor, int halocolor, int bgcolor, bool utf8) = 0;


ApplyMessage writes text on a frame. For example:
::

    char BUF[256];
    PVideoFrame src = child->GetFrame(n, env);
    env->MakeWritable(&src);
    sprintf(BUF, "Filter: Frame %d is processed.", n);
    env->ApplyMessage(&src, vi, BUF, vi.width/4, 0xf0f080, 0, 0);


With ApplyMessageEx you can use UTF-8 encoded strings on Windows ANSI code pages. On Posix, or on Windows set to "utf8 (beta)" everything is UTF-8 already.

C interface (v14): ``avs_apply_message``. Its message is always UTF-8 encoded, as there is no need
for the C interface to carry over the legacy code page handling.


.. _cplusplus_getavslinkage:

GetAVSLinkage, v5
^^^^^^^^^^^^^^^^^

::

    virtual const AVS_Linkage* const __stdcall GetAVSLinkage() = 0;

Returns the :doc:`AVSLinkage <AVSLinkage>`.

todo: how and when to use that ...


.. _cplusplus_subframeplanara:

SubframePlanarA, v8
^^^^^^^^^^^^^^^^^^^

::

    virtual PVideoFrame __stdcall SubframePlanarA(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA) = 0;

Alpha plane aware version of SubframePlanar.


.. _cplusplus_copyframeprops:

copyFrameProps, v8
^^^^^^^^^^^^^^^^^^

::

    virtual void __stdcall copyFrameProps(const PVideoFrame& src, PVideoFrame& dst) = 0;

copy frame properties between video frames.


.. _cplusplus_getframepropsro:

getFramePropsRO, v8
^^^^^^^^^^^^^^^^^^^

::

    virtual const AVSMap* __stdcall getFramePropsRO(const PVideoFrame& frame) = 0;

get pointer for reading frame properties


.. _cplusplus_getframePropsrw:

getFramePropsRW, v8
^^^^^^^^^^^^^^^^^^^

::

    virtual AVSMap* __stdcall getFramePropsRW(PVideoFrame& frame) = 0;

get pointer for reading/writing frame properties.

Important note: a frame property set is safely writable if

- frame is just obtained with NewVideoFrame
- frame is obtained with SubFrame, SubFramePlanar or SubFramePlanarA
- env->MakeWritable is used
- or env->MakePropertyWritable is used

MakePropertyWritable (v9) vs MakeWritable: MakePropertyWritable does not make
a full copy of video buffer content, just re-references the frame (internally
is working like SubFramePlanarA)

.. _cplusplus_propnumkeys:

propNumKeys, v8
^^^^^^^^^^^^^^^

::

    virtual int __stdcall propNumKeys(const AVSMap* map) = 0;

 get number of frame properties for a frame.


.. _cplusplus_propgetkey:

propGetKey, v8
^^^^^^^^^^^^^^

::

    virtual const char* __stdcall propGetKey(const AVSMap* map, int index) = 0;

 get name of key by index.


.. _cplusplus_propnumelements:

propNumElements, v8
^^^^^^^^^^^^^^^^^^^

::

    virtual int __stdcall propNumElements(const AVSMap* map, const char* key) = 0;

get array size of a property


.. _cplusplus_propGetType:

propGetType, v8
^^^^^^^^^^^^^^^

::

    virtual char __stdcall propGetType(const AVSMap* map, const char* key) = 0;

get property data type.

::

    // enums for frame property types
    enum AVSPropTypes {
      PROPTYPE_UNSET = 'u', // ptUnset
      PROPTYPE_INT = 'i', // peType
      PROPTYPE_FLOAT = 'f', // ptFloat
      PROPTYPE_DATA = 's', // ptData
      PROPTYPE_CLIP = 'c', // ptClip
      PROPTYPE_FRAME = 'v' // ptFrame
      //  ptFunction = 'm'
    };


.. _cplusplus_propgetint:

propGetInt, v8
^^^^^^^^^^^^^^

::

    virtual int64_t __stdcall propGetInt(const AVSMap* map, const char* key, int index, int* error) = 0;

get property value as integer (int64).
You can pass nullptr to error, but if given, the following error codes are set (0 = O.K.)

With ``error`` = nullptr a failed read (missing key, wrong type, index out of range, map with error set)
throws ``AvisynthError``, so pass nullptr only for properties known to exist.
(VapourSynth API4 dies with a fatal error in this case.) The same applies to all ``propGetXXX``
functions taking an ``error`` parameter, except the ``Saturated`` ones, which never throw.
Before 3.7.6 the throw happened inside a ``noexcept`` function and terminated the process;
``propGetClip`` and ``propGetFrame`` (3.7.4 and 3.7.5) neither threw nor set ``*error``.
Though AVSValue in Avisynth does not support int64_t (as of December 2021), you can freely
use int64_t frame property values in plugins. Internally there is no special 32 bit integer
version, only 64 bit integer exists.

Possible error codes defined in Avisynth.h:

::

  enum AVSGetPropErrors {
    GETPROPERROR_SUCCESS = 0,
    GETPROPERROR_UNSET = 1, // peUnset
    GETPROPERROR_TYPE = 2, // peType
    GETPROPERROR_ERROR = 3, // map has error state set
    GETPROPERROR_INDEX = 4 // peIndex
  };

.. _cplusplus_propgetintsaturated:

propGetIntSaturated, v11
^^^^^^^^^^^^^^^^^^^^^^^^

::

    virtual int __stdcall propGetIntSaturated(const AVSMap* map, const char* key, int index, int* error) = 0;

get property value as integer (and _not_ int64).

The same as propGetInt, but clamps the underlying int64 values to valid integer ranges.

Backported from VapourSynth API 4.


.. _cplusplus_propgetfloat:

propGetFloat, v8
^^^^^^^^^^^^^^^^

::

    virtual double __stdcall propGetFloat(const AVSMap* map, const char* key, int index, int* error) = 0;

Get property value as float (double).
No special 32 bit float is handled for frame properties, only 64 bit double,
but from v11 the value can be requested in float, see propGetFloatSaturated.


.. _cplusplus_propgetfloatsaturated:

propGetFloatSaturated, v11
^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    virtual float __stdcall propGetFloatSaturated(const AVSMap* map, const char* key, int index, int* error) = 0;

Get property value as float (and _not_ double).

The same as propGetFloat, but clamps the underlying double value to valid 32 bit float range.

Backported from VapourSynth API 4.


.. _cplusplus_propgetdata:

propGetData, v8
^^^^^^^^^^^^^^^

::

    virtual const char* __stdcall propGetData(const AVSMap* map, const char* key, int index, int* error) = 0;

get property value as string buffer.

Note: C interface counterpart avs_prop_get_data behaviour was fixed and made similar to C++ propGetData in interface version 9.1


.. _cplusplus_propgetdatasize:

propGetDataSize, v8
^^^^^^^^^^^^^^^^^^^

::

    virtual int __stdcall propGetDataSize(const AVSMap* map, const char* key, int index, int* error) = 0;

get string/data buffer size.
String length is without the terminating 0.

.. _cplusplus_propgetdatatypehint:

propGetDataTypeHint, v11
^^^^^^^^^^^^^^^^^^^^^^^^

::

    virtual int __stdcall propGetDataTypeHint(const AVSMap* map, const char* key, int index, int* error) = 0;

get string/data type, given by a hint earlier.

The result is one of AVSPropDataTypeHint enums: unknown, binary or string data.

Backported from VapourSynth API4: VSDataTypeHint

.. _cplusplus_propgetclip:

propGetClip, v8
^^^^^^^^^^^^^^^

::

    virtual PClip __stdcall propGetClip(const AVSMap* map, const char* key, int index, int* error) = 0;

get property value as Clip.


.. _cplusplus_propgetframe:

propGetFrame, v8
^^^^^^^^^^^^^^^^

::

    virtual const PVideoFrame __stdcall propGetFrame(const AVSMap* map, const char* key, int index, int* error) = 0;

get property value as Frame.


.. _cplusplus_propdeletekey:

propDeleteKey, v8
^^^^^^^^^^^^^^^^^

::

    virtual int __stdcall propDeleteKey(AVSMap* map, const char* key) = 0;

removes a frame property by name (key).


.. _cplusplus_propsetint:

propSetInt, v8
^^^^^^^^^^^^^^

::

    virtual int __stdcall propSetInt(AVSMap* map, const char* key, int64_t i, int append) = 0;

sets integer (int64) frame property.
In setter function the append parameter rules that the key is replaced if exists, added otherwise (PROPAPPENDMODE_REPLACE).
For populating an array use PROPAPPENDMODE_APPEND. For just creating the key use PROPAPPENDMODE_TOUCH.

::

  enum AVSPropAppendMode {
    PROPAPPENDMODE_REPLACE = 0, // paReplace
    PROPAPPENDMODE_APPEND = 1, // paAppend
    PROPAPPENDMODE_TOUCH = 2 // paTouch
  };


.. _cplusplus_propsetfloat:

propSetFloat, v8
^^^^^^^^^^^^^^^^

::

    virtual int __stdcall propSetFloat(AVSMap* map, const char* key, double d, int append) = 0;

sets float (double) frame property.


.. _cplusplus_propsetdata:

propSetData, v8
^^^^^^^^^^^^^^^

::

    virtual int __stdcall propSetData(AVSMap* map, const char* key, const char* d, int length, int append) = 0;

sets string (byte buffer) frame property.

Use propSetDataH instead, available from v11 (3.7.4). 


.. _cplusplus_propsetdatah:

propSetDataH, v11
^^^^^^^^^^^^^^^^^

::

    virtual int __stdcall propSetDataH(AVSMap* map, const char* key, const char* d, 
                                       int length, int type, int append) = 0;

sets string (byte buffer) frame property along with a type hint: binary or string. 
Hint is then used as display (propShow) or serialization (to json) helper.

For type hints see AvsPropDataTypeHint enum:
::

  enum AVSPropDataTypeHint {
    PROPDATATYPEHINT_UNKNOWN = -1, // dtUnknown = -1,
    PROPDATATYPEHINT_BINARY = 0, // dtBinary = 0,
    PROPDATATYPEHINT_UTF8 = 1 // dtUtf8 = 1
  };

Backported from VapourSynth API4: ``mapSetData`` (their API however defines the hintless version as ``mapSetData3``)

.. _cplusplus_propsetclip:

propSetClip, v8
^^^^^^^^^^^^^^^

::

    virtual int __stdcall propSetClip(AVSMap* map, const char* key, PClip& clip, int append) = 0;

sets PClip type frame property.


.. _cplusplus_propsetframe:

propSetFrame, v8
^^^^^^^^^^^^^^^^

::

    virtual int __stdcall propSetFrame(AVSMap* map, const char* key, const PVideoFrame& frame, int append) = 0;

sets PVideoFrame type frame property..


.. _cplusplus_propgetintarray:

propGetIntArray, v8
^^^^^^^^^^^^^^^^^^^

::

    virtual const int64_t* __stdcall propGetIntArray(const AVSMap* map, const char* key, int* error) = 0;

array version of propGetInt.


.. _cplusplus_propgetfloatarray:

propGetFloatArray, v8
^^^^^^^^^^^^^^^^^^^^^

::

    virtual const double* __stdcall propGetFloatArray(const AVSMap* map, const char* key, int* error) = 0;

array version of propGetFloat.


.. _cplusplus_propsetintarray:

propSetIntArray, v8
^^^^^^^^^^^^^^^^^^^

::

    virtual int __stdcall propSetIntArray(AVSMap* map, const char* key, const int64_t* i, int size) = 0;

array version of propSetInt.


.. _cplusplus_propsetfloatarray:

propSetFloatArray, v8
^^^^^^^^^^^^^^^^^^^^^

::

    virtual int __stdcall propSetFloatArray(AVSMap* map, const char* key, const double* d, int size) = 0;

array version of propSetFloat.


.. _cplusplus_createmap:

createMap, v8
^^^^^^^^^^^^^

::

    virtual AVSMap* __stdcall createMap() = 0;

Creates a standalone, empty property map, not attached to a frame. Use it with the ``propXxx``
functions, e.g. as a temporary key-value store; free it with ``freeMap``.
C interface (v14): ``avs_create_map``.

A map is the same container as the frame properties, modelled after VapourSynth's ``VSMap``:
each key holds an array of values of one type (``int64_t``, ``double``, data (string or binary),
``PClip`` or ``PVideoFrame``); a single value is a one-element array. Unlike ``AVSValue`` arrays,
these arrays are flat: no nesting, no mixed types, no map inside a map, and no function type.


.. _cplusplus_freemap:

freeMap, v8
^^^^^^^^^^^

::

    virtual void __stdcall freeMap(AVSMap* map) = 0;

Frees a map created by ``createMap``, with its content (clip and frame references stored in it
are released). Do not call it on a frame's map (``getFramePropsRO``, ``getFramePropsRW``), that one
is owned by the frame. C interface (v14): ``avs_free_map``.


.. _cplusplus_clearmap:

clearMap, v8
^^^^^^^^^^^^

::

    virtual void __stdcall clearMap(AVSMap* map) = 0;

clears all properties for a frame.


.. _cplusplus_newvideoframep:

NewVideoFrameP, v8
^^^^^^^^^^^^^^^^^^

::

    virtual PVideoFrame __stdcall NewVideoFrameP(const VideoInfo& vi, const PVideoFrame* prop_src, int align = FRAME_ALIGN) = 0;

NewVideoFrame with frame property source.

Note: C interface counterpart avs_new_video_frame_p(_a) crash was fixed in interface version 9.1


.. _cplusplus_getenvproperty:

GetEnvProperty, v8
^^^^^^^^^^^^^^^^^^

::

    virtual size_t  __stdcall GetEnvProperty(AvsEnvProperty prop) = 0;

Query to ask for various system (not frame!) properties.

::

    // IScriptEnvironment GetEnvProperty
    enum AvsEnvProperty
    {
      AEP_PHYSICAL_CPUS = 1,
      AEP_LOGICAL_CPUS = 2,
      AEP_THREADPOOL_THREADS = 3,
      AEP_FILTERCHAIN_THREADS = 4,
      AEP_THREAD_ID = 5,
      AEP_VERSION = 6,
      AEP_HOST_SYSTEM_ENDIANNESS = 7, // V9
      AEP_INTERFACE_VERSION = 8, // V9
      AEP_INTERFACE_BUGFIX = 9,  // V9
      AEP_CACHESIZE_L2 = 10, // v12

      // Neo additionals
      AEP_NUM_DEVICES = 901,
      AEP_FRAME_ALIGN = 902,
      AEP_PLANE_ALIGN = 903,

      AEP_SUPPRESS_THREAD = 921,
      AEP_GETFRAME_RECURSIVE = 922,
    };


AEP_HOST_SYSTEM_ENDIANNESS (c++) AVS_AEP_HOST_SYSTEM_ENDIANNESS (c)
...................................................................

Populated by 'little', 'big', or 'middle' based on what GCC and/or Clang report at compile time.

AEP_INTERFACE_VERSION (c++) AVS_AEP_INTERFACE_VERSION (c)
.........................................................

For requesting actual interface (main) version. An long awaited function. 
So far the actual interface version could be queried only indirectly, with trial and error, by starting from e.g. 10 then
going back one by one until CheckVersion() did not report an exception/error code. 

Even for V8 interface this was a bit tricky, the only way to detect was the infamous

::

      has_at_least_v8 = true;
      try { env->CheckVersion(8); } catch (const AvisynthError&) { has_at_least_v8 = false; }

method.

Now (starting from interface version 8.1) a direct version query is supported as well.
Of course this (one or two direct call only) is the future.
Programs or plugins which would like to identify older systems still must rely partially on the CheckVersion method.

CPP interface (through avisynth.h).

::

    IScriptEnvironment *env = ...
    avisynth_if_ver = 6; // guessed minimum
    avisynth_bugfix_ver = 0;
    try { env->CheckVersion(8); avisynth_if_ver = 8; }
    catch (const AvisynthError&) {}
    try { 
      env->CheckVersion(9); // if this works, we are at least V9, can use GetEnvProperty with AEP_INTERFACE_VERSION
      avisynth_if_ver = env->GetEnvProperty(AEP_INTERFACE_VERSION); // only since V9!
      avisynth_bugfix_ver = env->GetEnvProperty(AEP_INTERFACE_BUGFIX);      
    } 
    catch (const AvisynthError&) {}
    
    has_at_least_v8 = avisynth_if_ver >= 8; // frame properties, NewVideoFrameP, other V8 environment functions
    has_at_least_v8_1 = avisynth_if_ver > 8 || (avisynth_if_ver == 8 && avisynth_bugfix_ver >= 1);
    // 8.1: C interface frameprop access fixed, IsPropertyWritable/MakePropertyWritable support, extended GetEnvProperty queries
    has_at_least_v9 = avisynth_if_ver >= 9;
    has_at_least_v12 = avisynth_if_ver >= 12; // global locks, GetCPUFlagsEx, query L2 cache size

C interface (through avisynth_c.h)

::

    AVS_ScriptEnvironment *env = ...
    int avisynth_if_ver = 6; // guessed minimum
    int avisynth_bugfix_ver = 0;
    int retval = avs_check_version(env, 8);
    if (retval == 0) {
      avisynth_if_ver = 8;
      retval = avs_check_version(env, 9);
      if (retval == 0) {
        // V9 at least, we have AVS_AEP_INTERFACE_VERSION supported
        size_t retval_getenv = avs_get_env_property(env, AVS_AEP_INTERFACE_VERSION);
      if(env->error == 0) {
          avisynth_if_ver = retval_getenv;
          retval_getenv = avs_get_env_property(env, AVS_AEP_INTERFACE_BUGFIX);
        if(env->error == 0)
            avisynth_bugfix_ver = retval_getenv;
      }
    }
    }
    has_at_least_v8 = avisynth_if_ver >= 8; // frame properties, NewVideoFrameP, other V8 environment functions
    has_at_least_v8_1 = avisynth_if_ver > 8 || (avisynth_if_ver == 8 && avisynth_bugfix_ver >= 1);
    // 8.1: C interface frameprop access fixed, IsPropertyWritable/MakePropertyWritable support, extended GetEnvProperty queries
    has_at_least_v9 = avisynth_if_ver >= 9;
    has_at_least_v12 = avisynth_if_ver >= 12; // global locks, avs_get_cpu_flags_ex, query L2 cache size


AEP_INTERFACE_BUGFIX (c++) AVS_AEP_INTERFACE_BUGFIX (c)
.......................................................

Denotes situations where there isn't a breaking change to the API,
but we need to identify when a particular change, fix or addition
to various API-adjacent bits might have occurred.  Could also be
used when any new functions get added.

Since the number is modelled as 'changes since API bump' and
intended to be used in conjunction with checking the main
AVISYNTH_INTERFACE_VERSION, whenever the main INTERFACE_VERSION
gets raised, the value of INTERFACE_BUGFIX should be reset to zero.

The BUGFIX version is added here with already incremented once,
both because the addition of AVISYNTH_INTERFACE_BUGFIX_VERSION
itself would require it, but also because it's intended to signify
the fix to the C interface allowing frame properties to be read
back (which was the situation that spurred this define to exist
in the first place).


AEP_CACHESIZE_L2 (c++) AVS_AEP_CACHESIZE_L2 (c)
...............................................

Since V12. Returns the size of the L2 CPU cache in bytes.

::

    size_t l2_cache_size = env->GetEnvProperty(AEP_CACHESIZE_L2);
    // or in C interface
    size_t l2_cache_size = avs_get_env_property(env, AVS_AEP_CACHESIZE_L2);
    if (env->error == 0) {
      // l2_cache_size is valid
    }


.. _cplusplus_allocate:

Allocate, v8
^^^^^^^^^^^^

::

    virtual void* __stdcall Allocate(size_t nBytes, size_t alignment, AvsAllocType type) = 0;

buffer pool allocate.

Primary goal of ``AVS_NORMAL_ALLOC`` was to have the Avisynth-reserved memory 
counter up-to-date and to take into account this memory area as well. 
Thus when reaching the MemoryMax limit the core would free up memory from 
this resource as well (along with frame registry and cache entries).
Works like a normal _aligned_alloc.

But ``AVS_POOLED_ALLOC`` has (or had) a lot more important benefit. Pooled allocations 
are meant for filters that don't need the buffers between frames and can free 
them between calls to ``GetFrame()``. Then multiple filters can use the same buffers 
instead of each filter unnecessarily clinging onto them even while a different filter 
is executing. This resulted in tons of memory savings, which was really useful, as most 
of the complex scripts on HD material used to be memory-bound in that era (around 2015).
The main reason ``env->Allocate`` and ``env->Free`` were created at all was to support these 
pooled allocations and memory re-use between filters. Adding ``AVS_NORMAL_ALLOC`` was just a bonus.

The memory savings might not be that important today (as of 2025) since PCs now have a lot 
more RAM, but the savings are still there.

::

    // IScriptEnvironment Allocate
    enum AvsAllocType
    {
      AVS_NORMAL_ALLOC = 1,
      AVS_POOLED_ALLOC = 2
    };


.. _cplusplus_free:

Free, v8
^^^^^^^^

::

    virtual void __stdcall Free(void* ptr) = 0;

buffer pool free. Pair of ``Allocate``.


.. _cplusplus_getvartry:

GetVarTry, v8
^^^^^^^^^^^^^

::

    virtual bool  __stdcall GetVarTry(const char* name, AVSValue* val) const = 0;

get variable with success indicator.
Returns true and the requested variable. If the method fails, returns false and does not touch 'val'.


.. _cplusplus_getvarbool:

GetVarBool, v8
^^^^^^^^^^^^^^

::

    virtual bool __stdcall GetVarBool(const char* name, bool def) const = 0;

Get bool value with default.
Return the value of the requested variable. If the variable was not found or had the wrong type, return the supplied default value.


.. _cplusplus_getvarint:

GetVarInt, v8
^^^^^^^^^^^^^

::

    virtual int  __stdcall GetVarInt(const char* name, int def) const = 0;

Get int value with default.
Return the value of the requested variable. If the variable was not found or had the wrong type, return the supplied default value.


.. _cplusplus_getvardouble:

GetVarDouble, v8
^^^^^^^^^^^^^^^^

::

    virtual double  __stdcall GetVarDouble(const char* name, double def) const = 0;

Get floating point value with default. As of 2021 there is no double support for variables, 32 bit float only.
Return the value of the requested variable. If the variable was not found or had the wrong type, return the supplied default value.


.. _cplusplus_getvarstring:

GetVarString, v8
^^^^^^^^^^^^^^^^

::

    virtual const char* __stdcall GetVarString(const char* name, const char* def) const = 0;

Get string with default.
Return the value of the requested variable. If the variable was not found or had the wrong type, return the supplied default value.


.. _cplusplus_getvarlong:

GetVarLong, v8
^^^^^^^^^^^^^^

::

    virtual int64_t __stdcall GetVarLong(const char* name, int64_t def) const = 0;

Get int64 with default. As of 2021 there is no int64 support for variables. On Windows it is 32 bit int.
Return the value of the requested variable. If the variable was not found or had the wrong type, return the supplied default value.


.. _cplusplus_makepropertywritable:

MakePropertyWritable v9
^^^^^^^^^^^^^^^^^^^^^^^

::

    virtual bool __stdcall MakePropertyWritable(PVideoFrame* pvf) = 0;

like MakeWritable but for frame properties only.
See also 
- :ref:`IsPropertyWritable <cplusplus_ispropertywritable>` like IsWritable but for frame properties only.
- :ref:`getFramePropsRW <cplusplus_getframepropsrw>`.


.. _cplusplus_acquiregloballock:
.. _cplusplus_releasegloballock:

AcquireGlobalLock, v12
^^^^^^^^^^^^^^^^^^^^^^
ReleaseGlobalLock, v12
^^^^^^^^^^^^^^^^^^^^^^

::

    // C++ interface (IScriptEnvironment virtual methods)
    virtual bool __stdcall AcquireGlobalLock(const char* name) = 0;
    virtual void __stdcall ReleaseGlobalLock(const char* name) = 0;

    // C interface (global avs_ functions)
    AVS_EXPORT int __stdcall avs_acquire_global_lock(AVS_ScriptEnvironment* env, const char* name);
    AVS_EXPORT void __stdcall avs_release_global_lock(AVS_ScriptEnvironment* env, const char* name);

These functions provide a global, named mutex mechanism to synchronize access to shared 
resources across different plugins within the same Avisynth process. This is essential for 
libraries like fftw3 that have non-thread-safe global state (e.g., their planner functions).

When AcquireGlobalLock (or avs_acquire_global_lock) is called, the calling thread will block 
until the named lock is available. The lock is then exclusively held by that thread until 
ReleaseGlobalLock (or avs_release_global_lock) is called.

The name parameter allows for different independent global locks.
For FFTW, the recommended name is "fftw".

For safe use, set and check ``has_at_least_v12``, see interface version check methods above.

.. warning::

    **A local mutex is no replacement for the global lock.**

    - Pre-V12 fallback: a ``static std::mutex`` exists once per plugin DLL. It keeps the threads of
      that plugin apart, but two different FFTW plugins (e.g. FFT3DFilter and DFTTest) can still
      run the FFTW planner at the same time. On AviSynth V12 and above use the global lock only.
    - Do not fall back to the local mutex when ``AcquireGlobalLock`` fails on V12 and above. It fails
      only on invalid input (e.g. a NULL lock name): throw an error (``ThrowError``) instead. A silent
      fallback looks protected, but the cross-plugin race remains, showing up as rare crashes inside FFTW.
    - Exception: in a destructor do not throw (``ThrowError`` there ends in ``std::terminate``),
      see the destructor example below.

**Example#1 RAII for C++ interface plugins**

C++ plugins, which operate with the IScriptEnvironment* interface, should use a 
Resource Acquisition Is Initialization (RAII) wrapper. This GlobalLockGuard class example 
ensures the lock is automatically released when the guarding object goes out of scope,
even if exceptions occur.

The constructor of this GlobalLockGuard expects an IScriptEnvironment* directly, 
as C++ plugins will have access to this pointer.
::

    // FFTW is not thread-safe, need to guard around its functions (except fftw_execute).
    // http://www.fftw.org/fftw3_doc/Thread-safety.html#Thread-safety
    // Pre V12, not 100%, does not guard locks from multiple plugins using FFTW at the same time.
    static std::mutex fftw_legacy_mutex; // defined as static

    // Since Avisynth IF v12 use global lock which handles fftw locks for different plugins which use the same FFTW library.
    class GlobalLockGuard
    {
    public:
      // env_ptr should be the IScriptEnvironment* received by the plugin's function.
      // lock_name is the name of the lock (e.g., "fftw").
      // use_v12_global_lock: true on AviSynth V12 and above (has_at_least_v12), uses the global lock.
      // false: older AviSynth, falls back to the plugin-local legacy mutex.
      GlobalLockGuard(IScriptEnvironment* env_ptr, const char* lock_name, bool use_v12_global_lock)
        : m_env_ptr(env_ptr), m_lockName(lock_name), m_acquired(false), m_is_legacy_lock(false)
      {
        if (!m_env_ptr || !m_lockName) {
          // Invalid parameters, cannot acquire lock.
          return;
        }

        // Attempt to acquire V12 global lock if requested and available.
        // This assumes IScriptEnvironment provides a way to check its version.
        if (use_v12_global_lock)
        {
          // AcquireGlobalLock blocks until the named lock is available.
          // It returns false only on invalid input (e.g. lock name is nullptr). This is an operational
          // failure, not a missing capability, so do not fall back silently to the legacy mutex:
          // that would lose the cross-plugin exclusion the global lock is meant to provide.
          // An underlying mutex failure is reported by a std::system_error exception instead.
          if (!m_env_ptr->AcquireGlobalLock(m_lockName))
            m_env_ptr->ThrowError("GlobalLockGuard: failed to acquire global lock '%s'", m_lockName);
          m_acquired = true;
          m_is_legacy_lock = false; // Successfully acquired V12 lock
          return; // Lock acquired, exit constructor
        }

        // If we reach here, V12 lock was not requested (interface is older than v12), use legacy mutex.
        if (strcmp(m_lockName, "fftw") == 0) {
          fftw_legacy_mutex.lock(); // Acquire the legacy mutex
          m_acquired = true;
          m_is_legacy_lock = true; // Acquired legacy lock
        }
        // else { // Handle unrecognized lock_name for legacy fallback if needed }
      }

      // Destructor releases the lock.
      ~GlobalLockGuard()
      {
        if (m_acquired) // Only attempt to release if successfully acquired
        {
          if (m_is_legacy_lock) {
            fftw_legacy_mutex.unlock(); // Release legacy mutex
          }
          else {
            // Release V12 global lock
            if (m_env_ptr) { // Safety check
              m_env_ptr->ReleaseGlobalLock(m_lockName);
            }
          }
        }
      }

      bool is_acquired() const { return m_acquired; }

      // Disallow copying and assignment to prevent common errors with mutexes.
      GlobalLockGuard(const GlobalLockGuard&) = delete;
      GlobalLockGuard& operator=(const GlobalLockGuard&) = delete;

    private:
      IScriptEnvironment* m_env_ptr; // Store the C++ interface pointer directly
      const char* m_lockName;
      bool m_acquired;
      bool m_is_legacy_lock; // true if legacy mutex was used, false if V12 global lock was used
    };

**Example for lock, C++ plugin using RAII**

::

    // In a C++ plugin's source file (e.g., fft3dfilter, dfttest.cpp)
    #include "AvsLockGuard.h" // Assuming GlobalLockGuard is in this header

    // ... plugin setup ...

    // Use a scope to define the critical section for FFTW planning.
    {
        // Acquire the global "fftw" lock. It's automatically released when this scope exits.
        GlobalLockGuard fftw_lock(env, "fftw", has_at_least_v12);

        // --- CRITICAL SECTION START ---
        // Code here is protected by the global "fftw" lock.
        // Only one thread across all dynamically linked FFTW plugins
        // within this process can execute this section concurrently.
        fftwf_plan my_plan = fftwf_plan_dft_r2c_3d(...); // Perform FFTW planning
        // --- CRITICAL SECTION END ---

    } // `fftw_lock` goes out of scope here, automatically releasing the lock.


**Example for lock in a C++ filter destructor**

Destroying an FFTW plan needs the lock as well, but a destructor has no ``env`` parameter.
Save the ``env`` received by the constructor, as FFT3DFilter does.

Why the saved pointer is valid in the destructor: the ``env`` a filter constructor receives (when
the filter is created by the script, the usual case) lives until the end of the ScriptEnvironment's
destruction. AviSynth releases the clips (that runs the
filter destructors) before it deletes this environment, also when the clips are only released at
the environment's deletion. The global lock is usable there as well. The host must release its
clips before deleting the environment; using a clip after that is a host bug anyway.

A destructor must not throw: ``ThrowError``, and so the ``GlobalLockGuard`` above, ends in
``std::terminate`` there. Lock and unlock directly instead:

::

    class MyFFTFilter : public GenericVideoFilter
    {
      IScriptEnvironment* env_saved; // for the destructor (global lock)
      bool has_at_least_v12;
      fftwf_plan plan;
      ...
    public:
      MyFFTFilter(PClip _child, ..., IScriptEnvironment* env)
        : GenericVideoFilter(_child), env_saved(env)
      {
        has_at_least_v12 = true;
        try { env->CheckVersion(12); }
        catch (const AvisynthError&) { has_at_least_v12 = false; }
        {
          GlobalLockGuard fftw_lock(env, "fftw", has_at_least_v12); // constructor: may throw
          plan = fftwf_plan_dft_r2c_2d(...);
        }
      }

      ~MyFFTFilter()
      {
        // No env parameter: env_saved. No exceptions: no GlobalLockGuard, no ThrowError.
        if (has_at_least_v12) {
          // fails only on a NULL name (cannot happen with a literal): then destroy unprotected
          const bool locked = env_saved->AcquireGlobalLock("fftw");
          fftwf_destroy_plan(plan);
          if (locked)
            env_saved->ReleaseGlobalLock("fftw");
        }
        else {
          std::lock_guard<std::mutex> lock(fftw_legacy_mutex); // pre-V12 fallback
          fftwf_destroy_plan(plan);
        }
      }
    };

**Example#2 RAII for C++ plugins using C-Compatible interface**

C-compatible plugins, which receive an ``AVS_ScriptEnvironment*``, should also use an 
RAII wrapper for safe lock management. This ``GlobalLockGuardC`` will internally call 
the ``avs_`` C functions for acquiring and releasing the lock.

::

    // Note: no pre-V12 fallback is shown here

    // It operates on the AVS_ScriptEnvironment* handle and calls the C-interface functions.
    class GlobalLockGuardC
    {
    public:
        // env_handle should be the AVS_ScriptEnvironment* received by the plugin's function.
        GlobalLockGuardC(AVS_ScriptEnvironment* env_handle, const char* lock_name)
            : m_env_handle(env_handle), m_lockName(lock_name), m_acquired(false)
        {
            if (m_env_handle && m_lockName)
            {
                m_acquired = (avs_acquire_global_lock(m_env_handle, m_lockName) == 1);
            }
        }

        // Destructor releases the lock using the C-interface functions.
        ~GlobalLockGuardC()
        {
            if (m_acquired && m_env_handle && m_lockName)
            {
                avs_release_global_lock(m_env_handle, m_lockName);
            }
        }

        bool is_acquired() const { return m_acquired; }

        // Disallow copying and assignment to prevent common errors with mutexes.
        GlobalLockGuardC(const GlobalLockGuardC&) = delete;
        GlobalLockGuardC& operator=(const GlobalLockGuardC&) = delete;

    private:
        AVS_ScriptEnvironment* m_env_handle; // Store the C-compatible handle
        const char* m_lockName;
        bool m_acquired;
    };



**Example for lock, C interface, with above RAII**
::

    // In a C-compatible plugin's source file (e.g., myfilter.cpp that uses C++ features)
    // You'd typically also include the C-compatible RAII wrapper

    // Plugin function signature for C-compatible plugins (receives AVS_ScriptEnvironment*)
    static AVS_Value AVSC_CC Create_xxxx(AVS_ScriptEnvironment* env, AVS_Value args, void* param)
    {
        // ... plugin setup ...

        // Use a scope to define the critical section for FFTW planning.
        {
            // Acquire the global "fftw" lock using the C-compatible RAII wrapper.
            GlobalLockGuardC fftw_lock(env, "fftw");

            // avs_acquire_global_lock blocks until the lock is available.
            // It fails only on invalid input => we return an error AVS_Value
            if (!fftw_lock.is_acquired()) {
                AVS_Value err;
                avs_set_to_error(&err, "MyFilter: failed to acquire global lock 'fftw'");
                return err;
            }

            // --- CRITICAL SECTION START ---
            // Code here is protected by the global "fftw" lock.
            fftwf_plan my_plan = fftwf_plan_dft_r2c_3d(...); // Perform FFTW planning
            // --- CRITICAL SECTION END ---

        } // `fftw_lock` goes out of scope here, automatically releasing the lock.

        // ... rest of the plugin logic ...
    }

**Example for lock in free_filter (C interface filters)**

A filter created through the C interface (``avs_new_c_filter``) has no ``IClip`` class of its own,
also when the plugin is written in C++: the ``free_filter`` callback serves as its destructor.
Use ``fi->env`` there: it lives as long as the filter. Do not store the ``env`` received by the
create function (``AVS_ApplyFunc``) or by ``avisynth_c_plugin_init`` for later use: those are
temporary, valid only during that call.

A C++ plugin can use RAII in the callback, ``GlobalLockGuardC fftw_lock(fi->env, "fftw");`` (it
does not throw). In pure C the lock is acquired and released manually:

::

    static void AVSC_CC free_my_filter(AVS_FilterInfo* fi)
    {
        MyFilterData* data = (MyFilterData*)fi->user_data;
        // fi->env, not an env saved in the create function
        int locked = avs_acquire_global_lock(fi->env, "fftw");
        fftwf_destroy_plan(data->plan);
        if (locked)
            avs_release_global_lock(fi->env, "fftw");
        free(data);
    }


**Example for use from pure C-compatible**

Pure C plugins do not have access to C++ RAII. They must manually call ``avs_acquire_global_lock``
and ``avs_release_global_lock``, ensuring that every acquisition has a corresponding release.

::

    fftwf_plan my_plan = NULL;
    int lock_acquired = 0; // Flag to track if lock was acquired

    // Acquire the lock
    lock_acquired = avs_acquire_global_lock(env, "fftw");
    if (!lock_acquired) {
        // Handle error: Return an error AVS_Value.
        AVS_Value err;
        avs_set_to_error(&err, "MyCFilter: Failed to acquire global FFTW planner lock!");
        return err;
    }

    // --- CRITICAL SECTION START ---
    // Code here is protected by the global "fftw" lock.
    my_plan = fftwf_plan_dft_r2c_3d(...); // Perform FFTW planning
    // --- CRITICAL SECTION END ---

    // Release the lock manually
    avs_release_global_lock(env, "fftw");



See also https://github.com/AviSynth/AviSynthPlus/issues/444
and https://www.fftw.org/doc/Thread-safety.html 


.. _cplusplus_pvideoframe:

PVideoFrame
~~~~~~~~~~~

PVideoFrame is a smart pointer to VideoFrame.

In this example it gives a pointer to frame 'n' from child:
::

    PVideoFrame src = child->GetFrame(n, env);


"child" is a protected member of GenericVideoFilter, of type PClip. It
contains the clip that was passed to the constructor. For our filter to
produce frame n we need the corresponding frame of the input. If you
need a different frame from the input, all you have to do is pass a
different frame number to child->GetFrame.

In this example it gives a pointer to a new created VideoFrame from vi
(which is a VideoInfo structure):
::

    PVideoFrame dst = env->NewVideoFrame(vi);

Interface V8 introduced frame properties. One can create a new frame with
specifying a source frame from which the frame properties are copied.

In this example it gives a pointer to a new created VideoFrame from vi,
with the actual child clip as frame property source:
::

    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrameP(vi, src);


"vi" is another protected member of GenericVideoFilter (the only other
member, actually). It is a structure of type VideoInfo, which contains
information about the clip (like frame size, frame rate, pixel format,
audio sample rate, etc.). NewVideoFrame uses the information in this
struct to return a frame buffer of the appropriate size.


.. _cplusplus_iclip:

IClip
~~~~~

An Avisynth filter is simply a C++ class implementing the IClip
interface. IClip has four pure virtual methods: GetVideoInfo, GetFrame,
GetParity, and GetAudio. The class GenericVideoFilter is a simple
do-nothing filter defined in avisynth.h. It derives from IClip and
implements all four methods. Most filters can inherit from
GenericVideoFilter rather than directly from IClip; this saves you from
having to implement methods that you don't care about, like GetAudio.

IClip has the following members: GetVersion, GetFrame, GetParity,
GetAudio, SetCacheHints and GetVideoInfo. They are described in the
following subsections.


.. _cplusplus_getversion:

GetVersion
^^^^^^^^^^

::

    virtual int __stdcall GetVersion() { return AVISYNTH_INTERFACE_VERSION; }


GetVersion returns the interface version of the Clip instance, but
it would show, against which avisynth.h version was the filter built.

AVISYNTH_INTERFACE_VERSION = 1 (v1.0-v2.0.8), 2 (v2.5.0-v2.5.5), 3
(v2.5.6-v2.5.8), 5 (v2.6.0a1-v2.6.0a5), 6 (v2.6.0), 8 (Avisynth+ from a specific build on) [version 4 doesn't exist].

This only returns the interface version of how the plugin was built against 
an actual avisynth header.

Since in Avisynth the closest (pseudo) clip is Avisynth's internal MT or cache-guard clip,
it usually returns the real Avisynth version, but for non-cached, non-MT  filters it may 
show the filter's avisynth header version.

In interface V9 (working version 8.1) use GetEnvProperty(AEP_INTERFACE_VERSION) and GetEnvProperty(AEP_INTERFACE_BUGFIX) for this task.


.. _cplusplus_getframe:

GetFrame
^^^^^^^^

::

    virtual PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) = 0;


GetFrame returns a video frame. In this example, the even frames (0, 2,
4, ...) of 'child' are returned:
::

    PVideoFrame src = child->GetFrame(2*n, env);


You should do all the GetFrame() calls BEFORE you get any pointers and
start manipulating any data.


.. _cplusplus_getparity:

GetParity
^^^^^^^^^

::

    virtual bool __stdcall GetParity(int n) = 0;


GetParity returns the field parity if the clip is field-based,
otherwise it returns the parity of first field of a frame. In other
words, it distinguishes between top field first (TFF) and bottom field
first (BFF). When it returns true, it means that this frame should be
considered TFF, otherwise it should be considered BFF.


.. _cplusplus_getaudio:

GetAudio
^^^^^^^^

::

    virtual void __stdcall GetAudio(void* buf, __int64 start, __int64 count, IScriptEnvironment* env) = 0;


Audio processing is handled through the GetAudio method. You must fill
in the buffer with count samples beginning at the sample start. A
sample may vary from one to four bytes, depending on whether the audio
is 8, 16, 24 or 32-bit (float is also 32-bit). The flag vi.SampleType()
will tell you this.

If you cannot do your audio processing in-place, you must allocate your
own buffer for the source audio using new or malloc.

In this example, the audio of frame 'n' is returned (in the buffer
'samples'):
::

    VideoInfo vi = child->GetVideoInfo();
    PVideoFrame src = child->GetFrame(n, env);
    const __int64 start = vi.AudioSamplesFromFrames(n);
    const __int64 count = vi.AudioSamplesFromFrames(1);
    SFLOAT* samples = new SFLOAT[count*vi.AudioChannels()];
    child->GetAudio(samples, max(0,start), count, env);


.. _cplusplus_setcachehints:

SetCacheHints
^^^^^^^^^^^^^

::

    int __stdcall SetCacheHints(int cachehints, int frame_range) = 0 ;
    // We do not pass cache requests upwards, only to the next filter.


See ``CachePolicyHint`` enum in ``avisynth.h`` for possible cachehints values,
or AVS_CACHE_* defines in avisynth_c.h for C interface.

Some use cases from the filter side:


CACHE_GET_MTMODE
................

Filter specifies its required MT (multithread) mode.

A filter when requested with CACHE_GET_MTMODE can return its 
multithreaded model to the core. MT_NICE_FILTER (fully reentrant), 
MT_MULTI_INSTANCE and MT_SERIALIZED (not MT friendly) can be returned.

::

    class ConvertToRGB : public GenericVideoFilter {
      ...
    int __stdcall SetCacheHints(int cachehints, int frame_range) override {
      AVS_UNUSED(frame_range);
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }

::

    enum MtMode
    {
      MT_INVALID = 0,
      MT_NICE_FILTER = 1,
      MT_MULTI_INSTANCE = 2,
      MT_SERIALIZED = 3,
      MT_SPECIAL_MT = 4, // do not use, test only
      MT_MODE_COUNT = 5
    }; 


CACHE_INFORM_NUM_THREADS
........................

Filter is informed about about the number or threads which was set by Prefetch.

Since v12.

Avisynth core uses the ``CachePolicyHint::CACHE_INFORM_NUM_THREADS`` enum to inform the filter 
about the number of threads by ``SetCacheHints`` when ``Prefetch`` is applied to that script section.

The parameter specifies the thread count.

For ``MT_MULTI_INSTANCE`` filters it is called for each instance.
For ``MT_NICE_FILTER`` or ``MT_SERIALIZED`` only once.
Not called if no ``Prefetch`` is used for that script section.

Possible use cases: 

- filter may disable its internal MT if received thread count > 1.
- for ``num_threads`` > 1, in Intel intrinsics optimization, the code can 
  use ``_mm_stream_si128`` which does not pollute caches, when writing output frames. 
  For ``num_threads`` = 1 your code version can use functions optimized for single thread 
  and can use ``_mm_store_si128``.

::

    class ConvertToRGB : public GenericVideoFilter {
      private:
        int num_threads; // initialize it, since CACHE_INFORM_NUM_THREADS is not called if no Prefetch is used
      ...
      int __stdcall SetCacheHints(int cachehints, int frame_range) override {
        if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
        if (cachehints == CACHE_INFORM_NUM_THREADS) {
          num_threads = frame_range;
        }
        return 0;
      }


CACHE_DONT_CACHE_ME
...................

Filter specifies that it needs no caching.

Filter returns 1 to the CACHE_DONT_CACHE_ME query.

::

    int __stdcall NonCachedGenericVideoFilter::SetCacheHints(int cachehints, int frame_range)
    {
      switch(cachehints)
      {
        case CACHE_DONT_CACHE_ME:
          return 1;
        case CACHE_GET_MTMODE:
          return MT_NICE_FILTER;

        case CACHE_GET_DEV_TYPE:
          return (child->GetVersion() >= 5) ? child->SetCacheHints(CACHE_GET_DEV_TYPE, 0) : 0;

        default:
          return GenericVideoFilter::SetCacheHints(cachehints, frame_range);
      }
    }


CACHE_USER_FORWARD_BEGIN .. CACHE_USER_FORWARD_END
..................................................

User defined hints for communication between plugins.

Since v13.1 (interface version 13, ``AVISYNTHPLUS_INTERFACE_BUGFIX_VERSION`` 1).

Hints in the range ``CACHE_USER_FORWARD_BEGIN`` (0x40000000) to ``CACHE_USER_FORWARD_END``
(0x7FFFFFFF, inclusive) are passed upstream through the invisible wrappers the core puts
around a filter: cache, MT guard, ``Prefetch`` and ``OnCPU``/``OnCUDA``. The result of the
filter is returned to the caller. Hints outside this range are not forwarded by the
core. C interface: ``AVS_CACHE_USER_FORWARD_BEGIN``, ``AVS_CACHE_USER_FORWARD_END``,
``avs_set_cache_hints``.

The meaning of a key and of the ``frame_range`` parameter is up to the cooperating
plugins. Choose keys that are unlikely to collide with other plugins.

Rules:

- The hint still goes "only to the next filter". Other filters in between (e.g.
  ``Trim``, or any ``GenericVideoFilter`` that does not handle the key) return 0 and cut it.
  A cooperating filter may pass a key on to its own child.
- ``MT_SERIALIZED`` filters receive the hint under the same lock as ``GetFrame``.
- ``MT_MULTI_INSTANCE`` filters: the hint is passed to every existing instance; the
  result is the one of the first instance. Instances created later, when ``Prefetch``
  is applied, do not receive hints sent earlier (e.g. from a constructor).
  Use the range for queries, or for state that every instance can rebuild by itself.
  Hint delivery can be partial: if an instance throws, the instances before it have
  already received the hint, and the caller gets the exception instead of a result.
- Parameter and result are 32-bit ``int``. For a 64-bit value use two keys (low and
  high part) as a workaround.
- Return 0 for unknown keys, as for every other hint.
- Detection: interface version 13 and ``env->GetEnvProperty(AEP_INTERFACE_BUGFIX)`` >= 1
  (C: ``avs_get_env_property(env, AVS_AEP_INTERFACE_BUGFIX)``), or any interface version above 13.

::

    // shared header of the cooperating plugins; the key must be inside the range
    const int MYPLUGIN_HINT_QUERY = CACHE_USER_FORWARD_BEGIN + 0x1234;

    // upstream filter
    int __stdcall SetCacheHints(int cachehints, int frame_range) override {
      if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
      if (cachehints == MYPLUGIN_HINT_QUERY) return my_answer(frame_range);
      return 0;
    }

    // downstream filter, e.g. in its constructor
    int answer = (child->GetVersion() >= 5) ? child->SetCacheHints(MYPLUGIN_HINT_QUERY, param) : 0;



Old, up to Avisynth 2.6
.......................

In Avisynth+ frame cacheing was completely rewritten compared to Avisynth 5 or 6.
Specifying cache ranges are no longer relevant. The following part describes
the old behaviour for historical reasons only.

SetCacheHints could be used in filters that request multiple frames
from any single PClip source per input GetFrame call. frame_range is
maximal 21.

The possible values of cachehints are:
::

    CACHE_NOTHING=0    // Filter requested no caching.
    CACHE_RANGE=1      // An explicit cache of "frame_range" frames around the current frame.
    CACHE_ALL=2        // This is default operation, a simple LRU cache.
    CACHE_AUDIO=3      // Audio caching.
    CACHE_AUDIO_NONE=4 // Filter requested no audio caching.
    CACHE_AUDIO_AUTO=5 // Audio caching (difference with CACHE_AUDIO?).


When caching video frames (cachehints=0, 1, 2), frame_range is the
radius around the current frame. When caching audio samples
(cachehints=3, 4, 5), the value 0 creates a default buffer of 64kb and
positive values allocate frame_range bytes for the cache.

E.g. If you have a single PClip source, i.e. child and you get asked
for frame 100 and you in turn then ask for frames 98, 99, 100, 101 and
102 then you need to call CACHE_RANGE with frame_range set to 3:
::

    child->SetCacheHints(CACHE_RANGE, 3);

Frames outside the specified radius are candidate for normal LRU
caching.


.. _cplusplus_getvideoinfo:

GetVideoInfo
^^^^^^^^^^^^

::

    virtual const VideoInfo& __stdcall GetVideoInfo() = 0;


GetVideoInfo returns a :doc:`VideoInfo <VideoInfo>` structure.


.. _cplusplus_pfunction:

PFunction v8
~~~~~~~~~~~~

PFunction is a smart pointer to an IFunction, and IFunction is a generic abstract
class.. It maintains a reference count on the IFunction object and
automagically deletes it when the last PFunction referencing it goes away.
For obvious reasons, you should always use PFunction rather than IFunction* to
refer to function.

Like a genuine pointer, a PFunction is only four/eight bytes long, so you can
pass it around by value. Also like a pointer, a PFunction can be assigned a
null value (0), which is often useful as a sentinel. Unlike a pointer,
PFunction is initialized to 0 by default.

A function is a new object type in Avisynth+ since V8, originally introduced in Neo fork.


.. _cplusplus_pclip:

PClip
~~~~~

PClip is a smart pointer to an IClip, and IClip is a generic abstract
class.. It maintains a reference count on the IClip object and
automagically deletes it when the last PClip referencing it goes away.
For obvious reasons, you should always use PClip rather than IClip* to
refer to clips.

Like a genuine pointer, a PClip is only four bytes long, so you can
pass it around by value. Also like a pointer, a PClip can be assigned a
null value (0), which is often useful as a sentinel. Unlike a pointer,

PClip is initialized to 0 by default.

You'll need to make sure your class doesn't contain any circular PClip
references, or any PClips sitting in dynamically allocated memory that
you forget to delete. Other than that, you don't have to worry about
the reference-counting machinery.

AviSynth filters have a standardized output channel via IClip, but
(unlike VirtualDub filters) no standardized input channel. Each filter
is responsible for obtaining its own source material -- usually (as in
this case) from another clip, but sometimes from several different
clips, or from a file.

The clip functionality must be provided by some concrete subclass of
IClip which implements the functions GetFrame(), etc. So you cannot
create a PClip without having an appropriate IClip subclass. For most
filters, the GenericVideoFilter class provides the basis for this, but
'source' filters (which is basically what you have) do not have a
parent clip and so GenericVideoFilter is not appropriate.


.. _cplusplus_avsvalue:

AVSValue
~~~~~~~~

AVSValue is a variant type which can hold any one of the following
types: a boolean value (true/false); an integer; a floating-point
number; a string; a video clip (PClip); an array of AVSValues; 
a function (Avisynth+) or nothing ("undefined").

It holds an array of AVSValues in the following way:
::

    AVSValue(const AVSValue* a, int size) { type = 'a'; array = a; array_size = size; }


For example:
::

    AVSValue up_args[3] = {child, 384, 288};
    PClip resized = env->Invoke("LanczosResize", AVSValue(up_args,3)).AsClip();


Note that
::

    AVSValue(up_args,3)


returns the following:
::

    {'a'; {child, 384, 288}; 3}


Also Invoke returns an AVSValue (see its declaration) which in that
case is a PClip.


.. _cplusplus_avsvaluegettype:

GetType V10
^^^^^^^^^^^

::

    AvsValueType GetType() const;


AVSValue::GetType returns the underlying type of the variant.

Returns an ``AvsValueType`` enum directly, one can use it instead of calling
all IsXXX functions to establish the exact type.

Note that although 'l'ong and 'd'ouble are defined, 64 bit data is not
(and in 32 bit Avisynth will never be) supported.

From v11 interface version (3.7.4) 64 bit 'l'ong and 'd'ouble data 
types are supported even on 32-bit systems,
despite the rather pessimistic "never be supported" note.

(Unlike frame properties, which support them by design).

::

    enum AvsValueType {
      VALUE_TYPE_UNDEFINED = 'v',
      VALUE_TYPE_BOOL = 'b',
      VALUE_TYPE_INT = 'i',
      VALUE_TYPE_LONG = 'l',
      VALUE_TYPE_FLOAT = 'f',
      VALUE_TYPE_DOUBLE = 'd',
      VALUE_TYPE_STRING = 's',
      VALUE_TYPE_CLIP = 'c',
      VALUE_TYPE_FUNCTION = 'n',
      VALUE_TYPE_ARRAY = 'a'
    };


.. _cplusplus_avsvalueasfunction:

AsFunction V14
^^^^^^^^^^^^^^

::

    PFunction AsFunction() const;


Returns the function object of a function value (``'n'`` type, ``IsFunction()``), e.g. a script
function passed to a plugin parameter of type ``n``. Call it with ``env->Invoke3`` or ``env->Invoke3Try``.
Check ``IsFunction()`` first. On a runtime older than V14 it returns an empty ``PFunction``.

A ``PFunction`` is reference counted: it keeps the function alive after the ``AVSValue`` is gone.

A complete plugin example:

::

    #include "avisynth.h"

    const AVS_Linkage* AVS_linkage = nullptr;

    // script: ApplyInt(function(x) { return x + 1 }, 5)  # 6
    AVSValue __cdecl ApplyInt(AVSValue args, void* user_data, IScriptEnvironment* env)
    {
      if (!args[0].IsFunction())
        env->ThrowError("ApplyInt: not a function");
      return env->Invoke3(AVSValue(), args[0].AsFunction(), AVSValue(args[1]), nullptr);
    }

    extern "C" __declspec(dllexport)
    const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
    {
      AVS_linkage = vectors;
      env->AddFunction("ApplyInt", "ni", ApplyInt, nullptr); // "ni": a function and an int parameter
      return "ApplyInt sample plugin";
    }

``Invoke3`` has no default for the implicit ``last``: pass an empty ``AVSValue()`` if none.
``AVSValue(args[1])`` is the one-element argument list.


.. _cplusplus_structures:

Structures
----------

The following structure is available: VideoInfo structure. It holds
global information about a clip (i.e. information that does not depend
on the framenumber). The GetVideoInfo method in IClip returns this
structure. A description (for AVISYNTH_INTERFACE_VERSION=6, 8 and above) of it can
be found :doc:`here <VideoInfo>`.


.. _cplusplus_constants:

Constants
---------

The following constants are defined in avisynth.h:
::

    // Audio Sample information
    typedef float SFLOAT;


::

    enum AvsSampleType {
      SAMPLE_INT8  = 1 << 0,
      SAMPLE_INT16 = 1 << 1,
      SAMPLE_INT24 = 1 << 2,  // Int24 is a very stupid thing to code, but it's supported by some hardware.
      SAMPLE_INT32 = 1 << 3,
      SAMPLE_FLOAT = 1 << 4
    };


::

    enum AvsPlane {
      DEFAULT_PLANE = 0,
      PLANAR_Y = 1 << 0,
      PLANAR_U = 1 << 1,
      PLANAR_V = 1 << 2,
      PLANAR_ALIGNED = 1 << 3,
      PLANAR_Y_ALIGNED = PLANAR_Y | PLANAR_ALIGNED,
      PLANAR_U_ALIGNED = PLANAR_U | PLANAR_ALIGNED,
      PLANAR_V_ALIGNED = PLANAR_V | PLANAR_ALIGNED,
      PLANAR_A = 1 << 4,
      PLANAR_R = 1 << 5,
      PLANAR_G = 1 << 6,
      PLANAR_B = 1 << 7,
      PLANAR_A_ALIGNED = PLANAR_A | PLANAR_ALIGNED,
      PLANAR_R_ALIGNED = PLANAR_R | PLANAR_ALIGNED,
      PLANAR_G_ALIGNED = PLANAR_G | PLANAR_ALIGNED,
      PLANAR_B_ALIGNED = PLANAR_B | PLANAR_ALIGNED,
    };


::

    enum CachePolicyHint {
      CACHE_25_NOTHING_26_UNUSED_0 = 0,
      // Values 0 to 5 are reserved for old 2.5 plugins
      // do not use them in new plugins

      // New 2.6 explicitly defined cache hints.
      CACHE_NOTHING=10, // Do not cache video.
      CACHE_WINDOW=11, // Hard protect upto X frames within a range of X from the current frame N.
      CACHE_GENERIC=12, // LRU cache upto X frames.
      CACHE_FORCE_GENERIC=13, // LRU cache upto X frames, override any previous CACHE_WINDOW.

      CACHE_GET_POLICY=30, // Get the current policy.
      CACHE_GET_WINDOW=31, // Get the current window h_span.
      CACHE_GET_RANGE=32, // Get the current generic frame range.

      CACHE_AUDIO=50, // Explicitly cache audio, X byte cache.
      CACHE_AUDIO_NOTHING=51, // Explicitly do not cache audio.
      CACHE_AUDIO_NONE=52, // Audio cache off (auto mode), X byte intial cache.
      CACHE_AUDIO_AUTO=53, // Audio cache on (auto mode), X byte intial cache.

      CACHE_GET_AUDIO_POLICY=70, // Get the current audio policy.
      CACHE_GET_AUDIO_SIZE=71, // Get the current audio cache size.

      CACHE_PREFETCH_FRAME=100, // Queue request to prefetch frame N.
      CACHE_PREFETCH_GO=101, // Action video prefetches.

      CACHE_PREFETCH_AUDIO_BEGIN=120, // Begin queue request transaction to prefetch audio (take critical section).
      CACHE_PREFETCH_AUDIO_STARTLO=121, // Set low 32 bits of start.
      CACHE_PREFETCH_AUDIO_STARTHI=122, // Set high 32 bits of start.
      CACHE_PREFETCH_AUDIO_COUNT=123, // Set low 32 bits of length.
      CACHE_PREFETCH_AUDIO_COMMIT=124, // Enqueue request transaction to prefetch audio (release critical section).
      CACHE_PREFETCH_AUDIO_GO=125, // Action audio prefetches.

      CACHE_GETCHILD_CACHE_MODE=200, // Cache ask Child for desired video cache mode.
      CACHE_GETCHILD_CACHE_SIZE=201, // Cache ask Child for desired video cache size.
      CACHE_GETCHILD_AUDIO_MODE=202, // Cache ask Child for desired audio cache mode.
      CACHE_GETCHILD_AUDIO_SIZE=203, // Cache ask Child for desired audio cache size.

      CACHE_GETCHILD_COST=220, // Cache ask Child for estimated processing cost.
        CACHE_COST_ZERO=221, // Child response of zero cost (ptr arithmetic only).
        CACHE_COST_UNIT=222, // Child response of unit cost (less than or equal 1 full frame blit).
        CACHE_COST_LOW=223, // Child response of light cost. (Fast)
        CACHE_COST_MED=224, // Child response of medium cost. (Real time)
        CACHE_COST_HI=225, // Child response of heavy cost. (Slow)

      CACHE_GETCHILD_THREAD_MODE=240, // Cache ask Child for thread safetyness.
        CACHE_THREAD_UNSAFE=241, // Only 1 thread allowed for all instances. 2.5 filters default!
        CACHE_THREAD_CLASS=242, // Only 1 thread allowed for each instance. 2.6 filters default!
        CACHE_THREAD_SAFE=243, //  Allow all threads in any instance.
        CACHE_THREAD_OWN=244, // Safe but limit to 1 thread, internally threaded.

      CACHE_GETCHILD_ACCESS_COST=260, // Cache ask Child for preferred access pattern.
        CACHE_ACCESS_RAND=261, // Filter is access order agnostic.
        CACHE_ACCESS_SEQ0=262, // Filter prefers sequential access (low cost)
        CACHE_ACCESS_SEQ1=263, // Filter needs sequential access (high cost)

      CACHE_AVSPLUS_CONSTANTS = 500,    // Smaller values are reserved for classic Avisynth

      CACHE_DONT_CACHE_ME,              // Filters that don't need caching (eg. trim, cache etc.) should return 1 to this request
      CACHE_SET_MIN_CAPACITY,
      CACHE_SET_MAX_CAPACITY,
      CACHE_GET_MIN_CAPACITY,
      CACHE_GET_MAX_CAPACITY,
      CACHE_GET_SIZE,
      CACHE_GET_REQUESTED_CAP,
      CACHE_GET_CAPACITY,
      CACHE_GET_MTMODE,

      CACHE_IS_CACHE_REQ,
      CACHE_IS_CACHE_ANS,
      CACHE_IS_MTGUARD_REQ,
      CACHE_IS_MTGUARD_ANS,

      CACHE_AVSPLUS_CUDA_CONSTANTS = 600,

      CACHE_GET_DEV_TYPE,           // Device types a filter can return
      CACHE_GET_CHILD_DEV_TYPE,    // Device types a fitler can receive

      CACHE_USER_CONSTANTS = 1000,      // Smaller values are reserved for the core

      // v13.1
      CACHE_USER_FORWARD_BEGIN = 0x40000000, // user defined hints passed upstream by the core wrappers
      CACHE_USER_FORWARD_END = 0x7FFFFFFF    // inclusive

    };


.. _cplusplus_cpufeatureflags:

CPU Feature Flags
-----------------

In ``avs/cpuid.h`` (C++) or in ``avisynth_c.h`` (C).
The ``LL`` suffix indicates these are 64-bit values; at least the last ones exceed 32 bits.

From Interface version V12 you can use :ref:`GetCPUFlagsEx<cplusplus_getcpuflagsex>` which returns a 64-bit integer
with all CPU feature flags, unlike GetCPUFlags() which returns only a 32-bit integer, thus missing some AVX512 flags.

See also :ref:`SetMaxCPU <setmaxcpu>`.

Intel x86 and AMD64 CPU feature flags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. table:: CPU Feature Flags - Intel (used in Avisynth+ for optimized code paths)
   :align: center

   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | C++ name                     | Flag value         | Remark                                                                                                                    |
   +==============================+====================+===========================================================================================================================+
   | ``CPUF_SSE2``                | ``0x20``           | PIV, K8                                                                                                                   |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_SSE3``                | ``0x100``          | PIV+, K8 Venice                                                                                                           |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_SSSE3``               | ``0x200``          | Core 2                                                                                                                    |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_SSE4``                | ``0x400``          |                                                                                                                           |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_SSE4_1``              | ``0x400``          | Penryn, Wolfdale, Yorkfield                                                                                               |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX``                 | ``0x800``          | Sandy Bridge, Bulldozer                                                                                                   |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_SSE4_2``              | ``0x1000``         | Nehalem                                                                                                                   |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX2``                | ``0x2000``         | Haswell                                                                                                                   |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_FMA3``                | ``0x4000``         |                                                                                                                           |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_F16C``                | ``0x8000``         |                                                                                                                           |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_MOVBE``               | ``0x10000``        | Big Endian move                                                                                                           |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_POPCNT``              | ``0x20000``        |                                                                                                                           |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AES``                 | ``0x40000``        |                                                                                                                           |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_FMA4``                | ``0x80000``        |                                                                                                                           |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512F``             | ``0x00100000``     | AVX-512 Foundation.                                                                                                       |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512DQ``            | ``0x00200000``     | AVX-512 DQ (Double/Quad granular) Instructions                                                                            |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512PF``            | ``0x00400000``     | AVX-512 Prefetch                                                                                                          |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512ER``            | ``0x00800000``     | AVX-512 Exponential and Reciprocal                                                                                        |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512CD``            | ``0x01000000``     | AVX-512 Conflict Detection                                                                                                |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512BW``            | ``0x02000000``     | AVX-512 BW (Byte/Word granular) Instructions                                                                              |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512VL``            | ``0x04000000``     | AVX-512 VL (128/256 Vector Length) Extensions                                                                             |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512IFMA``          | ``0x08000000``     | AVX-512 IFMA integer 52 bit                                                                                               |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512VBMI``          | ``0x10000000``     | AVX-512 VBMI, byte/word shuffling, sign/zero extension, and general pixel manipulation                                    |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512_BASE``         | ``0x20000000``     | AVX-512 Base group feature set. When F, CD, BW, DQ, VL flags exist. Avisynth sets it only if CPUF_AVX512_FAST exists      |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512_FAST``         | ``0x40000000``     | Base + VNNI, VBMI, VBMI2, BITALG, VPOPCNTDQ. Spec detection logic excludes older/throttling models that also have these   |
   |                              |                    | features                                                                                                                  |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX10`` (?)           | ``0x80000000LL``   | RFU: AVX10 group feature set: version query not from flags                                                                |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512VNNI``          | ``0x00100000000LL``| VNNI, accumulated dot product on 8/16 bit integers                                                                        |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512VBMI2``         | ``0x00200000000LL``| VBMI2: Byte/word load, store, & concatenation with shift for unaligned memory and packed data re-arrangement.             |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512BITALG``        | ``0x00400000000LL``| BITALG, Bit Manipulation Instructions                                                                                     |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512VPOPCNTDQ``     | ``0x00800000000LL``| VPOPCNTDQ, Vector Population Count Double/Quadword                                                                        |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512FP16``          | ``0x01000000000LL``| FP16, Half-precision floating-point operations                                                                            |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512BF16``          | ``0x02000000000LL``| Bfloat16 floating-point operations                                                                                        |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+

AVX-512 features are grouped into two categories:

- CPUF_AVX512_BASE: Base (F, CD, BW, DQ, VL)
- CPUF_AVX512_FAST: The "usable" AVX-512, starting with Intel's 10th gen Ice Lake, incl. AMD Zen4/5. The features Base + (VNNI, VBMI, VBMI2, BITALG, VPOPCNTDQ) are guaranteed.

Important note for user of old AVX512 CPUs, where only the Base flags exist (not an Ice Lake or better level CPU): 
the "Base" group flag ``CPUF_AVX512_BASE`` is not enabled automatically, even if the individual base flags exist!
The reason is to prevent the AVX512 Base-only optimizations in Avisynth for possibly old AVX512 systems.
However when users know what they are doing (know they do have non-throttling old CPU), they can enable it with ``SetMaxCPU("avx512base+")``

"Fast" means a usable AVX-512 implementation without severe throttling penalties on client CPUs. It basically guarantees a feature set 
similar to Intel Ice Lake/Rocket Lake and AMD Zen4/Zen5 and excludes older AVX-512 implementations (Skylake-X, Cascade Lake, Ice Lake-SP) 
which have severe throttling issues on client CPUs.

The categorization "Fast" mainly follows the ffmpeg project's "ICL" (Ice Lake arch.) approach.

Besides the group feature flags Avisynth provides detailed individual flags as well, some of them accessible only via GetCPUFlagsEx() due to
the shortage of 32 bits in GetCPUFlags().

Regarding the CMake-based build process: compiler flags for ``*_avx512.cpp`` files are automatically set to match with "CPUF_AVX512_FAST" for gcc 
or LLVM (clang-cl) builds in CMakeLists.txt. These are equivalent to using the following flags:
as ``"-mfma -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mavx512vbmi -mavx512vbmi2 -mavx512bitalg -mavx512vpopcntdq "`` .

Due to the large number of AVX-512 sub-features, the following group and composite flags are defined (mentioned already in the full list above):

.. table:: AVX-512 Group Feature Flags
   :align: center

   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512_BASE``         | ``0x20000000``     | AVX-512 Base group feature set. When F, CD, BW, DQ, VL flags. Avisynth sets it only if CPUF_AVX512_FAST exists            |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+
   | ``CPUF_AVX512_FAST``         | ``0x40000000``     | AVX-512 Advance group FAST (Ice Lake/Rocket Lake/Zen4/Zen5) feature set: Base + VNNI, VBMI, VBMI2, BITALG, VPOPCNTDQ.     |
   +------------------------------+--------------------+---------------------------------------------------------------------------------------------------------------------------+

.. note::

    **Usability of AVX-512 and Throttling**

    While AVX-512 features were first introduced with Skylake-SP (Intel Xeon), the earliest client CPUs
    to implement a *truly usable* 512-bit wide vector unit were **Ice Lake (10th Generation Mobile) and 
    Rocket Lake (11th Generation Desktop)**. These later microarchitectures significantly reduced the severe 
    clock-speed throttling penalty and the voltage/frequency impact (AVX-512 down-binning) that plagued 
    earlier implementations.

    For this reason, high-performance projects like **FFmpeg** often use feature checks similar to ``"avx512fast"`` 
    to classify a processor as having "good" or "usable" AVX-512 support. Like ICL CPU feature flag in ffmpeg.
    Compiling code solely with checking only ``"avx512base"`` (which often only implies first-generation AVX-512) 
    can lead to poor performance due to aggressive clock throttling on older hardware.

    Developers should generally use ``"avx512fast"`` group in their optimizer function dispatchers, as a minimum for 
    realistic performance testing of 512-bit code paths. Avisynth (e.g. in resamplers) has both base and fast optimizations
    implemented. However for accessing the optimizations for "Base" AVX512 on old CPUs, ``SetMaxCPU("avx512base+")`` script 
    command must be used to re-enable AVX512 on these systems.


.. note::

    **The transition to AVX10.1 and standardization with AVX10.2**

    Intel Advanced Vector Extensions 10 (Intel AVX10) is the successor to AVX-512, aiming to address the very 
    performance and consistency issues noted above. The new ISA is designed to create a **converged vector ISA** supported 
    consistently across all future Intel cores, including **Performance-cores (P-cores) and Efficient-cores (E-cores)** 
    For video filter developers, this means simplifying the instruction set selection process.

    * **AVX10.1: The Transitional Version (Server-focused):** This initial version was announced in 2023 and first 
        supported by server processors (e.g., 6th Generation Intel Xeon "Granite Rapids"). It is primarily a 
        transitional layer, and **does not provide the key performance guarantee** sought by desktop/client developers.
        As of late 2025, no consumer CPUs currently support AVX10.1 in general availability. It also does not natively 
        include the new FP8/BF16 data types or the full feature set of AVX10.2.

    * **AVX10.2: The Standardized Converged ISA (Future Client/Server):** The significant standardization that 
        removes frequency throttling uncertainty, is found in **AVX10.2**. This version mandates **512-bit vector length (VLEN)** support 
        for all cores (P-core and E-core) on a processor that reports AVX10.2 capability.

        This standardization ensures:

        1.  **Reliable Performance (Predictable Latency):** It addresses the AVX-512 hybrid core issue by forcing a consistent 512-bit 
            execution environment across all supported cores. This consistency is vital for real-time video processing and 
            decoding/encoding pipelines.
        2.  **Standardization and Modern Features:** It establishes the new, clean vector programming model. 
            AVX10.2 is designed to be binary-compatible with existing AVX-512 code, though recompilation is recommended to leverage new 
            features and the simplified CPUID check. It also introduces new instructions and data types, such as those for FP8 and BFloat16 conversions,
            beneficial for AI-driven filters and specialized media processing.
        3.  **Targeting:** Developers should target AVX10.2 as the minimum version for future consistent, 
            high-performance 512-bit code paths.

    * **Current Outlook:** The first consumer desktop CPUs (e.g., "Nova Lake") supporting the standardized 
        **AVX10.2** are generally expected in the **late 2026** timeframe. For the current Avisynth user base, 
        existing robust AVX-512 checks (like ``"avx512fast"``) remain the relevant method for detecting modern 
        vector capability.


.. table:: CPU Feature Flags (obsolete ones)
   :align: center

   +--------------------------+--------------------+------------------------------+
   | C++ name                 |  Flag value        | Remark                       |
   +==========================+====================+==============================+
   | ``CPUF_FORCE``           |  ``0x01``          | N/A                          |
   +--------------------------+--------------------+------------------------------+
   | ``CPUF_FPU``             |  ``0x02``          | 386/486DX                    |
   +--------------------------+--------------------+------------------------------+
   | ``CPUF_MMX``             |  ``0x04``          | P55C, K6, PII                |
   +--------------------------+--------------------+------------------------------+
   | ``CPUF_INTEGER_SSE``     |  ``0x08``          | PIII, Athlon                 |
   +--------------------------+--------------------+------------------------------+
   | ``CPUF_SSE``             |  ``0x10``          | PIII, Athlon XP/MP           |
   +--------------------------+--------------------+------------------------------+
   | ``CPUF_3DNOW``           |  ``0x40``          | K6-2                         |
   +--------------------------+--------------------+------------------------------+
   | ``CPUF_3DNOW_EXT``       |  ``0x80``          | Athlon                       |
   +--------------------------+--------------------+------------------------------+
   | ``CPUF_X86_64``          |  ``0xA0``          | Hammer                       |
   +--------------------------+--------------------+------------------------------+

Note: "C" interface names are the same but with ``AVS_CPUF_`` prefix instead of ``CPUF_``.
They are defined in ``avisynth_c.h``.

.. _aarch64_simd_tiers:

Avisynth Aarch64 SIMD feature tiers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First: a useful link about AArch64 architecture and its SIMD capabilities:
`Developer documentation at ARM <https://developer.arm.com/documentation/>`__

Avisynth defines multiple categories for AArch64 (mentioned also as ARM64, ARMv8/v9) SIMD support, allowing for feature-based dispatch similar
to the x86 SSE2/AVX2.

The implementation relies on runtime detection of hardware capabilities (using Linux's ``getauxval`` and ``hwcap``, or macOS's ``sysctl``).
Unlike x86, AArch64 (ARM64) is using a feature-based approach rather than relying solely on CPU model strings. Features are not
necessarily tied to specific CPU models and levels, feature become optional at a specific ARMv8 revision, then it becomes mandatory in a later revision.
By detecting a feature flag, we can only be sure, that the CPU level is at least the one where the feature was introduced as optional.

Avisynth header ``avs/config.h`` defines ``ARM64`` when it detects ``aarch64`` platform through the actual compiler's predefined macros.
When ``ARM64`` is defined, we can assume we are compiling for ARMv8-A (AArch64) architecture, regardless of actual OS or compiler.
As of late 2025 Avisynth supports aarch64 with Linux, Windows, BSD and macOS. gcc, llvm, clang-cl and MSVC compilers are supported.

Avisynth currently defines only a few ARM SIMD tiers, focusing on the most relevant features for video processing workloads, instead of
trying to cover every possible ARM extension (50+ different ones).

The ARMv8-A flag values shared with X86 flags, their usage can be guarded by platform macros.

.. table:: CPU Feature Flags - aarch64 (ARM64)
   :align: center

   +------------------------------+---------------+-----------------------------------------------------------+
   | C++ name                     | Flag value    | Remark                                                    |
   +==============================+===============+===========================================================+
   | ``CPUF_ARM_NEON``            | ``0x01``      | NEON flag, minimum for aarch64                            |
   +------------------------------+---------------+-----------------------------------------------------------+
   | ``CPUF_ARM_DOTPROD``         | ``0x02``      |                                                           |
   +------------------------------+---------------+-----------------------------------------------------------+
   | ``CPUF_ARM_SVE2``            | ``0x04``      |                                                           |
   +------------------------------+---------------+-----------------------------------------------------------+
   | ``CPUF_ARM_I8MM``            | ``0x08``      |                                                           |
   +------------------------------+---------------+-----------------------------------------------------------+
   | ``CPUF_ARM_SSE2_1``          | ``0x10``      |                                                           |
   +------------------------------+---------------+-----------------------------------------------------------+

These are in ``avs/cpuid.h`` (C++) or in ``avisynth_c.h`` (C).

Usage:
::

    #include <avisynth.h>
    #include <avs/cpuid.h>
    #ifdef NEON_INTRINSICS
    if (env->GetCPUFlags() & CPUF_ARM_DOTPROD) {
        // Use DOTPROD optimized code path
    } else {
        // Fallback to NEON baseline
    } else 
    #endif
    {
        // Plain c++ non-ARM64 code path
    }


AArch64 Versioning Scheme (v8.x-A, v9.x-A)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AArch64 is a living standard. Key features relevant to Avisynth are introduced in minor revisions:

* **ARMv8.0-A:** The baseline for AArch64. **NEON is mandatory.**
* **ARMv8.2-A:** Introduces optional ``DOTPROD`` (Dot Product, ``FEAT_DotProd``) and ``I8MM`` (8-bit Matrix Multiply).
* **ARMv8.4-A:** Makes ``DOTPROD`` mandatory.
* **ARMv8.5-A:** Introduces optional ``SVE2`` (Scalable Vector Extension 2).
* **ARMv8.6-A:** Makes ``I8MM`` (8-bit Matrix Multiply) mandatory.
* **ARMv9.0-A:** Compulsory ``SVE2`` support for ARMv9-A compliant CPUs.
* **ARMv9.1-A:** Introduces further ``SVE2`` extensions (``FEAT_SVE2p1``).



Avisynth ARM SIMD Tiers
^^^^^^^^^^^^^^^^^^^^^^^

- CPUF_ARM_NEON (Baseline SIMD)

    This is the foundational level required for any optimized Avisynth kernel on ARM.

    * **Minimum Requirement:** ARMv8.0-A (AArch64).
    * **Register Width:** Fixed **128-bit** (V-registers V0-V31).
    * **x86 Analogy:** Comparable to **SSE2**. It establishes 128-bit wide floating-point and 
      integer vector processing.
    * **Usable Instruction Sets:** All basic NEON operations (addition, subtraction, multiplication, 
      shuffle, load/store).
    * Compilers with aarch64 support automatically set Armv8.0-a baseline which includes 'fp' and 'neon' (SIMD).

- CPUF_ARM_DOTPROD (Advanced 128-bit)

    This tier represents the first major performance jump for modern video encoding and processing. It is 
    defined by the availability of specialized multiply-accumulate instructions.

    * **Minimum Requirement:** ARMv8.2-A (where the **DOTPROD** feature became optional; mandatory from ARMv8.4-A).
    * **Register Width:** Still fixed **128-bit** (V-registers).
    * **x86 Analogy:** Comparable to **SSSE3** or **SSE4.1** maybe a little **AVX2**, since v8.1-A and v8.2-A introduced other new instructions as well.
        * Unlike AVX2 which doubled the register width (128-bit to 256-bit), this ARM tier provides the *instructions* without changing the physical register width.
    * **Usable Instruction Sets:** The **DOTPROD** instructions (e.g., ``SDOT`` and ``UDOT``) for integer dot product operations.
    * **Video Processing Relevance:** Absolutely critical for accelerating modern video codecs (HEVC/VVC/AV1) by providing highly efficient paths for:
        * Motion Compensation
        * Intra-Prediction
        * Integer Matrix Operations (analogous to x86's VNNI).
    * Compiler flags: ``-march=armv8.2-a+dotprod`` (gcc/clang/clang-cl) or ``/arch:armv8.2`` (MSVC).
      The armv8.2-a base is required by GCC 11/12: their ``arm_neon.h`` defines the dot product intrinsics
      (e.g. ``vdotq_u32``) for ``arch=armv8.2-a+dotprod``, ``-march=armv8.1-a+dotprod`` fails there
      (Clang accepts it). Dispatch on the ``CPUF_ARM_DOTPROD`` feature flag, not on an architecture version.

See also:
`The-Armv8-2-architecture-extension <https://developer.arm.com/documentation/109697/2025_12/Feature-descriptions/The-Armv8-2-architecture-extension?lang=en>`__
and
`The-Armv8-4-architecture-extension <https://developer.arm.com/documentation/109697/2025_12/Feature-descriptions/The-Armv8-4-architecture-extension?lang=en>`__

- CPUF_ARM_I8MM (8 bit matrix multiplication)

    Another useful instruction set, optional since v8.2-a, mandatory in v8.6-a.

See also:
`The-Armv8-2-architecture-extension <https://developer.arm.com/documentation/109697/2025_12/Feature-descriptions/The-Armv8-2-architecture-extension?lang=en#md447-the-armv82-architecture-extension__feat_FEAT_I8MM>`__
and
`SIMD-FP-Instructions/USDOT <https://developer.arm.com/documentation/ddi0602/2025-12/SIMD-FP-Instructions/USDOT--vector---Dot-product-with-unsigned-and-signed-integers--vector--?lang=en>`__

- CPUF_ARM_SVE2 (Scalable Vector Extension 2)

    A next leap to scalable vector architecture, we skipped simple SVE in Avisynth.

    * **Minimum Requirement:** ARMv8.5-A (SVE2 feature flag).
    * **Register Width:** **Scalable Vector Length (SVL)**. This register length is defined by the hardware 
      (e.g., 256-bit or 512-bit) and is uniform across a single processor.
    * **x86 Analogy:** Comparable to **AVX-512** or **AVX10.2**. Both represent a significant shift to 
      very wide vector processing.
    * **Usable Instruction Sets:** All SVE2 instructions, which extend SVE with NEON-style operations, 
      providing masked, predicated, and full vector operations.
    * **Video Processing Relevance:** Designed for maximum performance in HPC and deep learning workloads, 
      this tier offers the largest data throughput per clock cycle for highly parallel operations like 
      convolutions.

- CPUF_ARM_SVE2_1 (Scalable Vector Extension 2.1 FEAT_SVE2p1)

    Even more useful instructions added in ARMv9.1-A


.. _AArch64_SIMD_Compilation:

AArch64 SIMD Compilation and Source Code Rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AviSynth utilizes file naming conventions within its source tree to automatically apply the correct, feature-specific compiler flags for AArch64 (ARM64) architectures, 
ensuring optimal SIMD performance while maintaining wide compatibility.

1. Source File Location
.......................

This is how Avisynth source code is organized for AArch64-specific SIMD implementations:

All AArch64-specific SIMD C++ files must be located within the **aarch64** subdirectory, parallel to any existing ``intel`` optimization folders.
CMakeLists.txt is configured to include this directory when building for ARM64 targets and the ``ENABLE_NEON_SIMD`` option is enabled for CMake.
In source code we should check ``NEON_INTRINSICS`` define for NEON specific code.
Like this:
::

    #include "turn.h"
    #ifdef INTEL_INTRINSICS
    #include "intel/turn_sse.h"
    #endif
    #ifdef NEON_INTRINSICS
    #include "aarch64/turn_neon.h"
    #endif


2. File Patterns and Automatic Compiler Options Rule
....................................................

The build system (via CMake) uses the file name suffix to select the appropriate compiler flags for the target feature. This ensures that only the intended instructions are generated for each source file.

+-------------------+--------------------------------------------------+------------------------------------+------------------------------------+
| File Pattern      | Feature / Minimum Arch                           | Compiler Flag (GCC/Clang)          | Compiler Flag (MSVC)               |
+===================+==================================================+====================================+====================================+
| ``*_neon.cpp``    | Baseline NEON (ASIMD)                            | Implicit (or ``-march=armv8-a``)   | ``/arch:armv8.0``                  |
+-------------------+--------------------------------------------------+------------------------------------+------------------------------------+
| ``*_dp.cpp``      | Dot Product (``FEAT_DotProd``) / ARMv8.2-A       | ``-march=armv8.2-a+dotprod``       | ``/arch:armv8.2``                  |
+-------------------+--------------------------------------------------+------------------------------------+------------------------------------+
| ``*_i8mm.cpp``    | Int8 Matrix Multiply (``FEAT_I8MM``) / ARMv8.2-A | ``-march=armv8.2-a+i8mm``          | ``/arch:armv8.2``                  |
+-------------------+--------------------------------------------------+------------------------------------+------------------------------------+
| ``*_sve2.cpp``    | Scalable Vector Extension 2                      | ``-march=armv8.5-a+sve2``          | ``/arch:armv8.5``                  |
|                   | (``FEAT_SVE2``) / ARMv8.5-A                      |                                    |                                    |
+-------------------+--------------------------------------------------+------------------------------------+------------------------------------+
| ``*_sve2_1.cpp``  | Scalable Vector Extension 2.1                    | ``-march=armv9.1-a+sve2p1``        | ``/arch:armv9.1``                  |
|                   | (``FEAT_SVE2p1``) / ARMv9.1-A                    |                                    |                                    |
+-------------------+--------------------------------------------------+------------------------------------+------------------------------------+

.. note::
    * The SVE2.1 modifier ``+sve2p1`` needs GCC 15 or a newer Clang (Clang 15 does not know it,
      Clang 22 of Visual Studio 2026 does);
      a ``*_sve2_1.cpp`` file would need a compiler version check in CMake.
    * MSVC supports the ``/arch:armvX.X`` syntax for convenience, individual features are not supported as separate flags.
    * This distinction is necessary when targeting features for Clang-cl and gcc, the allowed C++ intrinsics are stricter
      than in MSVC where if a more modern instruction is used, MSVC will compile it without error. For gcc/llvm/Clang-cl, so-called function attributes 
      (like ``__attribute__((target("...")))`` can enable additional features in a lower-architecture targeted context.

3. Compiler Knowledge of Minimum Version
........................................

The flags above explicitly define the minimum architectural version (e.g., ``armv8.2-a``, ``armv8.5-a``) rather than just the baseline ``armv8-a`` plus the feature.

* The compilers do not raise the base architecture by themselves: ``+dotprod`` only adds the feature to the given base.
  The intrinsics headers may still require a base: GCC 11/12 define the dot product intrinsics only for
  ``arch=armv8.2-a+dotprod`` (``-march=armv8.1-a+dotprod`` fails with "target specific option mismatch"),
  Clang checks only the feature (``__ARM_FEATURE_DOTPROD``). So state the version where the feature
  became available (ARMv8.2-A for DotProd and I8MM, ARMv8.5-A for SVE2).
* A file compiled for a higher base may use any instruction of that base, not only the feature's.
  Such code runs only after the runtime feature check, and the check is for the feature, so keep these files
  small and isolated; it is worth checking the generated code (for example the ``*_dp.cpp`` objects contain only
  ARMv8.0 instructions plus ``udot``). Do not build them with link time optimization, which could inline them
  into baseline code.

4. Source Code and Dispatch
...........................

Specialized files contain the highly optimized code using the respective **ACLE Intrinsics** (Arm C Language Extensions).

* **Runtime Dispatch:** To correctly use these files, the main filter logic must include runtime CPU feature detection to check the capabilities 
  of the host CPU, Avisynth has its own CPUFlags feature set, which relies on the OS reported hardware capabilities. (e.g., via the Linux ``getauxval()`` 
  or Windows ``IsProcessorFeaturePresent()`` functions).
* The dispatcher function will then call the appropriate implementation available: ``SVE2 or I8MM -> DOTPROD -> NEON (ASIMD)``.

5. Useful tutorial videos
.........................
LVC21-309 SVE & SVE2 in LLVM 
https://www.youtube.com/watch?v=v6NmKOkQ2LE

MSVC ARM64 optimization in Visual Studio:
https://www.youtube.com/watch?v=O5XAdeMTRWk

Real-World Use Case: FFmpeg Categorization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Projects like FFmpeg use similar feature checks to manage their optimized assembly code paths.
FFmpeg primarily checks for individual ``hwcap`` bits, which map directly to some of our tiers:

1.  **Baseline:** Determined by the presence of **NEON** (mandatory).
2.  **Advanced 128-bit:** Primarily determined by the **DOTPROD** flag.
3.  **I8MM:** Checked via dedicated **I8MM** flags.

The **DOTPROD** flag is often the most important threshold for enabling the fastest H.264/HEVC/VVC assembly functions.

Raspberry Pi Feature Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For common single-board computers, the support is as follows:

* **Raspberry Pi 4 (Broadcom BCM2711):**
    * **CPU:** Quad-core Arm Cortex-A72.
    * **Architecture:** **ARMv8.0-A**.
    * **Extensions:** NEON only, no **DOTPROD** (``asimddp`` is not listed in ``/proc/cpuinfo``).
    * **Category:** Belongs to **CPUF_ARM_NEON** only.

* **Raspberry Pi 5 (Broadcom BCM2712):**
    * **CPU:** Quad-core Arm Cortex-A76.
    * **Architecture:** **ARMv8.2-A** (with some features of later revisions).
    * **SVE/SVE2 Support:** **The Cortex-A76 core does not support SVE/SVE2.** 
      It supports **DOTPROD** and other extensions that are part of the newer ARMv8.x standards.
    * **Category:** Belongs to **CPUF_ARM_DOTPROD**, but **NOT CPUF_ARM_SVE2**.

____

Back to :doc:`FilterSDK`

$Date: 2026/10/19 10:00:00 $
