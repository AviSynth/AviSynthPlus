C++ API
=======

The header, avisynth.h, declares all the classes, structures and
miscellaneous constants of the C++ API that you might need when writing
a plugin. All external plugins should #include it:
::

    #include "avisynth.h"

Note, sometimes there is a reference to a version number of the plugin
api (for example v3, v6, v8). This refers to the value of
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

VideoFrame holds a "window" into a VideoFrameBuffer. Operator new is
overloaded to recycle class instances. Its members can be called by:
::

    PVideoFrame src = child->GetFrame(n, env);
    src->GetReadPtr(..)


VideoFrame has the following members: GetPitch, GetRowSize, GetHeight,
GetReadPtr, GetWritePtr, IsWritable and IsPropertyWritable

All those filters (except IsWritable and IsPropertyWritable) will give you a property (pitch,
rowsize, etc ...) of a plane (of the frame it points to). The
interleaved formats (BGR(A) or YUY2) consist of one plane, and the
planar formats consists of one (Y) or three (YUV) planes. The default
plane is just the first plane (which is plane Y for the planar
formats).


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


.. _cplusplus_fillborder:

FillBorder
~~~~~~~~~~

::

    FillBorder(PClip _clip);


This function fills up the right side of the picture on planar images
with duplicates of the rightmost pixel if the planes are not aligned.
That is, if src->GetRowSize(PLANAR_Y) !=
src->GetRowSize(PLANAR_Y_ALIGNED).


.. _cplusplus_convertaudio:

ConvertAudio
~~~~~~~~~~~~

::

    ConvertAudio(PClip _clip, int prefered_format);


ConvertAudio converts the sample type of the audio to one of the
following sample types: SAMPLE_INT8 (8 bits), SAMPLE_INT16 (16 bits),
SAMPLE_INT24 (24 bits), SAMPLE_INT32 (32 bits) or SAMPLE_FLOAT (float).

The following example converts the sample type of the clip child to
SAMPLE_INT16 (16 bit) if the input isn't 16 bit.
::

    ConvertAudio(child, SAMPLE_INT16);


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
and ApplyMessage. They are described in the following subsections.


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

GetCPUFlags
^^^^^^^^^^^

::

    virtual long GetCPUFlags();


GetCPUFlags returns the instruction set of your CPU. To find out if
you're running for example on a CPU that supports MMX, test:
::

    env->GetCPUFlags() & CPUF_MMX


There's a complete list of flags in avisynth.h.


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
instances will already be gone by then. The returned pointer is not
const-qualified, and you're welcome to write to it, as long as you
don't stray beyond the bounds of the string.

Example usage (converting a string to upper case):
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


| // TODO - see (also similar functions)
| http://forum.doom9.org/showthread.php?p=1595750#post1595750


.. _cplusplus_popcontext:

PopContext
^^^^^^^^^^

::

    virtual void __stdcall PopContext() = 0;


?


.. _cplusplus_popcontextglobal:

PopContextGlobal
^^^^^^^^^^^^^^^^

::

    virtual void __stdcall PopContextGlobal() = 0;


?


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

::

    Example of usage with CPP interface (through avisynth.h).

    IScriptEnvironment *env = ...
    int avisynth_if_ver = 6;
    int avisynth_bugfix_ver = 0;
    try { 
      avisynth_if_ver = env->GetEnvProperty(AEP_INTERFACE_VERSION); 
      avisynth_bugfix_ver = env->GetEnvProperty(AEP_INTERFACE_BUGFIX);      
    } 
    catch (const AvisynthError&) { 
      try { env->CheckVersion(8); avisynth_if_ver = 8; } catch (const AvisynthError&) { }
    }
    has_at_least_v8 = avisynth_if_ver >= 8; // frame properties, NewVideoFrameP, other V8 environment functions
    has_at_least_v8_1 = avisynth_if_ver > 8 || (avisynth_if_ver == 8 && avisynth_bugfix_ver >= 1);
    // 8.1: C interface frameprop access fixed, IsPropertyWritable/MakePropertyWritable support, extended GetEnvProperty queries
    has_at_least_v9 = avisynth_if_ver >= 9; // future


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
CreateScriptEnvironment.


.. _cplusplus_applymessage:

ApplyMessage, v5
^^^^^^^^^^^^^^^^

::

    virtual void _stdcall ApplyMessage(PVideoFrame* frame, const VideoInfo& vi, const char* message, int size, int textcolor, int halocolor, int bgcolor) = 0;


ApplyMessage writes text on a frame. For example:
::

    char BUF[256];
    PVideoFrame src = child->GetFrame(n, env);
    env->MakeWritable(&src);
    sprintf(BUF, "Filter: Frame %d is processed.", n);
    env->ApplyMessage(&src, vi, BUF, vi.width/4, 0xf0f080, 0, 0);


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
Though AVSValue in Avisynth does not support int64_t (as of December 2021), you can freely
use int64_t frame property values in plugins. Internally there is no special 32 bit integer
version, only 64 bit integer exists.

Possible error codes defined in Avisynth.h:

::

  enum AVSGetPropErrors {
    GETPROPERROR_UNSET = 1, // peUnset
    GETPROPERROR_TYPE = 2, // peType
    GETPROPERROR_INDEX = 4 // peIndex
  };



.. _cplusplus_propgetfloat:

propGetFloat, v8
^^^^^^^^^^^^^^^^

::

    virtual double __stdcall propGetFloat(const AVSMap* map, const char* key, int index, int* error) = 0;

Get property value as float (double).
No special 32 bit float is handled for frame properties, only 64 bit double.


.. _cplusplus_propgetdata:

propGetData, v8
^^^^^^^^^^^^^^^

::

    virtual const char* __stdcall propGetData(const AVSMap* map, const char* key, int index, int* error) = 0;

get property value as string buffer.


.. _cplusplus_propgetdatasize:

propGetDataSize, v8
^^^^^^^^^^^^^^^^^^^

::

    virtual int __stdcall propGetDataSize(const AVSMap* map, const char* key, int index, int* error) = 0;

get string/data buffer size.
String length is without the terminating 0.


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

internal use only, creating frame property buffer.


.. _cplusplus_freemap:

freeMap, v8
^^^^^^^^^^^

::

    virtual void __stdcall freeMap(AVSMap* map) = 0;

internal use only, frees up frame property buffer.


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

    virtual PVideoFrame __stdcall NewVideoFrameP(const VideoInfo& vi, PVideoFrame* propSrc, int align = FRAME_ALIGN) = 0;

NewVideoFrame with frame property source.


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

      // Neo additionals
      AEP_NUM_DEVICES = 901,
      AEP_FRAME_ALIGN = 902,
      AEP_PLANE_ALIGN = 903,

      AEP_SUPPRESS_THREAD = 921,
      AEP_GETFRAME_RECURSIVE = 922,
    };

AEP_HOST_SYSTEM_ENDIANNESS (c++) AVS_AEP_HOST_SYSTEM_ENDIANNESS (c)

Populated by 'little', 'big', or 'middle' based on what GCC and/or Clang report at compile time.

AEP_INTERFACE_VERSION (c++) AVS_AEP_INTERFACE_VERSION (c)

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
    int avisynth_if_ver = 6;
    int avisynth_bugfix_ver = 0;
    try { 
      avisynth_if_ver = env->GetEnvProperty(AEP_INTERFACE_VERSION); 
      avisynth_bugfix_ver = env->GetEnvProperty(AEP_INTERFACE_BUGFIX);      
    } 
    catch (const AvisynthError&) { 
      try { env->CheckVersion(8); avisynth_if_ver = 8; } catch (const AvisynthError&) { }
    }
    has_at_least_v8 = avisynth_if_ver >= 8; // frame properties, NewVideoFrameP, other V8 environment functions
    has_at_least_v8_1 = avisynth_if_ver > 8 || (avisynth_if_ver == 8 && avisynth_bugfix_ver >= 1);
    // 8.1: C interface frameprop access fixed, IsPropertyWritable/MakePropertyWritable support, extended GetEnvProperty queries
    has_at_least_v9 = avisynth_if_ver >= 9; // future

C interface (through avisynth_c.h)

::

    AVS_ScriptEnvironment *env = ...
    int avisynth_if_ver = 6; // guessed minimum
    int avisynth_bugfix_ver = 0;
    int retval = avs_check_version(env, 8);
    if (retval == 0) {
      avisynth_if_ver = 8;
      // V8 at least, we have avs_get_env_property but AVS_AEP_INTERFACE_VERSION query may not be supported
      int retval = avs_get_env_property(env, AVS_AEP_INTERFACE_VERSION);
      if(env->error == 0) {
        avisynth_if_ver = retval;
        retval = avs_get_env_property(env, AVS_AEP_INTERFACE_BUGFIX);
        if(env->error == 0)
          avisynth_bugfix_ver = retval;
      }
    }
    has_at_least_v8 = avisynth_if_ver >= 8; // frame properties, NewVideoFrameP, other V8 environment functions
    has_at_least_v8_1 = avisynth_if_ver > 8 || (avisynth_if_ver == 8 && avisynth_bugfix_ver >= 1);
    // 8.1: C interface frameprop access fixed, IsPropertyWritable/MakePropertyWritable support, extended GetEnvProperty queries
    has_at_least_v9 = avisynth_if_ver >= 9; // future


AEP_INTERFACE_BUGFIX (c++) AVS_AEP_INTERFACE_BUGFIX (c)

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


.. _cplusplus_allocate:

Allocate, v8
^^^^^^^^^^^^

::

    virtual void* __stdcall Allocate(size_t nBytes, size_t alignment, AvsAllocType type) = 0;

buffer pool allocate.

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

buffer pool free.


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


GetVersion returns the interface version of the loaded avisynth.dll:

AVISYNTH_INTERFACE_VERSION = 1 (v1.0-v2.0.8), 2 (v2.5.0-v2.5.5), 3
(v2.5.6-v2.5.8), 5 (v2.6.0a1-v2.6.0a5), 6 (v2.6.0), 8 (Avisynth+ from a specific build on) [version 4 doesn't exist].
2021 note: this returns the interface version of how the plugin was built against actual avisynth header.
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

    void __stdcall SetCacheHints(int cachehints, int frame_range) = 0 ;
    // We do not pass cache requests upwards, only to the next filter.

    // int __stdcall SetCacheHints(int cachehints, int frame_range) = 0 ;
    // We do not pass cache requests upwards, only to the next filter.


Avisynth+: frame cacheing was completely rewritten compared to Avisynth 5 or 6.
Specifying cache ranges are not relevant.

Avisynth classic up to v6:

SetCacheHints should be used in filters that request multiple frames
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

Avisynth+: A filter when requested with CACHE_GET_MTMODE can return its 
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

| // TODO - describe input and output for v5 //
| http://forum.doom9.org/showthread.php?p=1595750#post1595750


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


.. _cplusplus_structures:

Structures
----------

The following structure is available: VideoInfo structure. It holds
global information about a clip (i.e. information that does not depend
on the framenumber). The GetVideoInfo method in IClip returns this
structure. A description (for AVISYNTH_INTERFACE_VERSION=6, 8 and 9) of it can
be found :doc:`here <VideoInfo>`.


.. _cplusplus_constants:

Constants
---------

The following constants are defined in avisynth.h:
::

    // Audio Sample information
    typedef float SFLOAT;


::

    enum { // sample types
        SAMPLE_INT8 = 1<<0,
        SAMPLE_INT16 = 1<<1,
        SAMPLE_INT24 = 1<<2,
        SAMPLE_INT32 = 1<<3,
        SAMPLE_FLOAT = 1<<4
    };


::

    enum {
       PLANAR_Y=1<<0,
       PLANAR_U=1<<1,
       PLANAR_V=1<<2,
       PLANAR_ALIGNED=1<<3,
       PLANAR_Y_ALIGNED=PLANAR_Y|PLANAR_ALIGNED,
       PLANAR_U_ALIGNED=PLANAR_U|PLANAR_ALIGNED,
       PLANAR_V_ALIGNED=PLANAR_V|PLANAR_ALIGNED,
       PLANAR_A=1<<4,
       PLANAR_R=1<<5,
       PLANAR_G=1<<6,
       PLANAR_B=1<<7,
       PLANAR_A_ALIGNED=PLANAR_A|PLANAR_ALIGNED,
       PLANAR_R_ALIGNED=PLANAR_R|PLANAR_ALIGNED,
       PLANAR_G_ALIGNED=PLANAR_G|PLANAR_ALIGNED,
       PLANAR_B_ALIGNED=PLANAR_B|PLANAR_ALIGNED,
      };


::

    enum CachePolicyHint {
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

      CACHE_USER_CONSTANTS = 1000       // Smaller values are reserved for the core

    };  


::

    // For GetCPUFlags.  These are backwards-compatible with those in VirtualDub.
    // ending with SSE4_2
    // For emulation see https://software.intel.com/en-us/articles/intel-software-development-emulator
    enum {
                        /* oldest CPU to support extension */
      CPUF_FORCE        =  0x01,   //  N/A
      CPUF_FPU          =  0x02,   //  386/486DX
      CPUF_MMX          =  0x04,   //  P55C, K6, PII
      CPUF_INTEGER_SSE  =  0x08,   //  PIII, Athlon
      CPUF_SSE          =  0x10,   //  PIII, Athlon XP/MP
      CPUF_SSE2         =  0x20,   //  PIV, K8
      CPUF_3DNOW        =  0x40,   //  K6-2
      CPUF_3DNOW_EXT    =  0x80,   //  Athlon
      CPUF_X86_64       =  0xA0,   //  Hammer (note: equiv. to 3DNow + SSE2, which
                                   //          only Hammer will have anyway)
      CPUF_SSE3         = 0x100,   //  PIV+, K8 Venice
      CPUF_SSSE3        = 0x200,   //  Core 2
      CPUF_SSE4         = 0x400,
      CPUF_SSE4_1       = 0x400,   //  Penryn, Wolfdale, Yorkfield
      CPUF_AVX          = 0x800,   //  Sandy Bridge, Bulldozer
      CPUF_SSE4_2       = 0x1000,  //  Nehalem
      // AVS+
      CPUF_AVX2         = 0x2000,   //  Haswell
      CPUF_FMA3         = 0x4000,
      CPUF_F16C         = 0x8000,
      CPUF_MOVBE        = 0x10000,  // Big Endian move
      CPUF_POPCNT       = 0x20000,
      CPUF_AES          = 0x40000,
      CPUF_FMA4         = 0x80000,

      CPUF_AVX512F      = 0x100000,  // AVX-512 Foundation.
      CPUF_AVX512DQ     = 0x200000,  // AVX-512 DQ (Double/Quad granular) Instructions
      CPUF_AVX512PF     = 0x400000,  // AVX-512 Prefetch
      CPUF_AVX512ER     = 0x800000,  // AVX-512 Exponential and Reciprocal
      CPUF_AVX512CD     = 0x1000000, // AVX-512 Conflict Detection
      CPUF_AVX512BW     = 0x2000000, // AVX-512 BW (Byte/Word granular) Instructions
      CPUF_AVX512VL     = 0x4000000, // AVX-512 VL (128/256 Vector Length) Extensions
      CPUF_AVX512IFMA   = 0x8000000, // AVX-512 IFMA integer 52 bit
      CPUF_AVX512VBMI   = 0x10000000,// AVX-512 VBMI
    };
 

____

Back to :doc:`FilterSDK`

$Date: 2015/01/13 00:24:50 $
