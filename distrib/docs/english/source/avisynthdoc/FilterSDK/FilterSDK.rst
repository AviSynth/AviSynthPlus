
AviSynth FilterSDK
==================

Note: May not fully up-to-date

AviSynth external Filter SDK is a package for developers to create your own
filters (plugins and console applications) for AviSynth.

The package consists of:

-   these documentation text files (in HTML or Wiki format);
-   the header file 'avisynth.h' (recent version) with all declarations
    to include in plugin source code;
-   several plugin and console application source codes;
-   some extra files in 'Extra' folder.

.. toctree::
    :maxdepth: 3

.. contents:: Table of contents


Necessary software
------------------

You must have some :doc:`necessary software. <SDKNecessaries>`

Writing a plugin
----------------

We will start by writing some simple plugins to show you the basics.

Processing video
~~~~~~~~~~~~~~~~

* :doc:`InvertNeg` produces a photo-negative of the input clip.
* :doc:`SimpleSample` has some very simple examples covering development
  of a filter that draws a variable sized square, in the middle of
  the screen. It does so for all color formats.

One thing not covered in SimpleSample, is how to :doc:`change frame size <ChangeFrameSize>` in a
filter.

Processing audio
~~~~~~~~~~~~~~~~

* xxx

Also have a look at :doc:`Getting started with Audio <GettingStartedWithAudio>`.

Runtime functions
~~~~~~~~~~~~~~~~~

See :doc:`Non-clip sample <Non-ClipSample>` how to create runtime AviSynth functions.

Source filters
~~~~~~~~~~~~~~

* :doc:`GradientMask <GradientMask>` creates a simple Gradient. The example explains a
  few things how source filters work.

Speeding up your plugin using assembler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
todo: having SIMD intrinsic widespread this topic is a bit outdated:
You can also browse various topic on :doc:`Assembler Optimizing <AssemblerOptimizing>`.

Making dual plugins
~~~~~~~~~~~~~~~~~~~

This old topic once was about making dual 2.5 and 2.6 plugins.
Now in 2020 it is still important to give support earlier (now it's the 2.6) interface
which is used by earlier Avisynth+ and classic Avisynth 2.6.

One of the features of AviSynth+ is that 2.6 plugins (plugins compiled with plugin
api v6) can still be used in AviSynth+.
Avisynth+ evolved from classic Avisynth 2.6 so this is no wonder.
There are differences though since AviSynth+ introduced high bit depth
video handling and there came new support functions. AviSynth+'s avisynth.h is
responsible for the compatibility, functions which did not exist in classic Avisynth
(such as BitsPerComponent() for VideoInfo) return a compatible value (8 in this
case since Avisynth 2.6 handled only 8 bit videos)

Beginning with the v8 (v8.1 for C interface due to early bugs) interface - frame property 
support - dual interfaces became important again. Plugins have to be able to detect the 
availability of v8 interface and behave accordingly.

(todo: They are described :doc:`here <DualPlugins>`).

Writing console applications that access AviSynth
-------------------------------------------------

When writing console applications (commandline programs) it is possible
to access AviSynth in two ways.

The first one is to use the VfW api (using the `AVIFile library`_)
like is done in avs2avi (see `[1]`_ or `[2]`_), avs2yuv (`using
the C api`_, or `using the C++ api`_), or `avs2wav`_ for example. See
also `here`_ to get you going.

The second one is to call AviSynth directly like is done in
`avs2pipe`_ for example (it uses the C api). It's a tool to output y4m
video or wav audio to stdout. The way to do this is importing
avisynth.dll via loadlibrary() and getprocaddress(). Then creating a
scriptenvironment and importing a script using invoke(). At that point
you will have a clip from which you can call getframe() to access the
video and getaudio() to access the audio.

For C api clients see the examples (ffmpeg, x265, x264, avs2yuv) here:
:doc:`Introducing to C API <C_api>`.

Below are some examples that have a script as input and raw video or
audio as output:

* :doc:`avs2yuv` reads a script and outputs raw video.
* :doc:`avs2pcm` reads a script and outputs raw audio.

Compiling plugins and console applications that access AviSynth
---------------------------------------------------------------

How to compile plugins and console applications that access AviSynth is
described :doc:`here <CompilingAvisynthPlugins>`.

Debugging plugins and console applications that access AviSynth
---------------------------------------------------------------

How to debug plugins and console applications that access AviSynth is
described :doc:`here <DebuggingAvisynthPlugins>`.

AviSynth and its plugin api's
-----------------------------

AviSynth exists as an instance of the ScriptEnvironment class, that
implements the IScriptEnvironment interface. The IScriptEnvironment
interface is defined in avisynth.h (and avisynth_c.h) and it is the
only way for plugins and external applications to communicate with
AviSynth. A pointer to ScriptEnvironment object is passed along to all
plugins, so that they can use AviSynth facilities. Plugins are
forbidden from storing this pointer. AviSynth creates one
ScriptEnvironment object when Windows attempts to use AviSynth to open
a script via AVIFile API.

When ScriptEnvironment is created, it checks for CPU extensions (to
provide this information to filters that are able to use CPU
extensions), sets up memory limits for itself and performs pre-scanning
of all plugins.

AviSynth has the capability to load third-party libraries that include
their own video and audio filters. It comes with two language
interfaces (or plugin api's):

* C++ API (through avisynth.h) - The classes and miscellaneous
  constants are described in :doc:`here <Cplusplus_api>`.
* C API (through avisynth_c.h) - The classes and miscellaneous
  constants are described in :doc:`here <C_api>`.

The built-in filters use the C++ API. This Filter SDK (or Source
Development Kit) describes how to create plugins using both interfaces.

Although not included in AviSynth itself, several people wrote other
language interfaces in Delphi, Purebasic, NET and Java. They can be
found `here <http://forum.doom9.org/showthread.php?p=566904#post566904>`__.
(comment from 2025: update needed)

...

There are several different Colorspaces in AviSynth. See more information
about :doc:`Color Spaces <ColorSpaces>` and :doc:`Working With Images <WorkingWithImages>`.

What's new in the 2.6 api
-------------------------

- C++ API (AVISYNTH_INTERFACE_VERSION = 6):

    - Plugin api v3 and older contained :doc:`baked code <AVSLinkage>` meaning code
      that is "baked" into all and every plugin instead being called
      from avisynth.dll. Starting from 2.6 the version 2.5 plugins
      are supported directly (with current baked code; meaning that
      plugins compiled for 2.5 can be loaded when using 2.6) and all
      the baked code from 2.6+ plugins is removed and the plugins
      are still source compatible. Note that the baked code is moved
      to interface.cpp, where also the structure :doc:`AVS_Linkage <AVSLinkage>` is
      defined.
    - The :ref:`IScriptEnvironment <cplusplus_iscriptenvironment>` interface has several new members:

        - :ref:`ApplyMessage <cplusplus_applymessage>` writes text on a frame.
        - :ref:`DeleteScriptEnvironment <cplusplus_deletescriptenvironment>` provides a method to delete
          the ScriptEnvironment which is created with
          CreateScriptEnvironment.
        - :ref:`GetAVSLinkage <cplusplus_getavslinkage>` returns the AVSLinkage.
        - :ref:`GetVarDef <cplusplus_getvardef>` can be used to access AviSynth variables.
          It will not throw an error if the variable doesn't exist.

    - Avisynth+ header still uses integer for things that are memory sizes (e.g. pitch)
      This only affects x64 versions.
      (Although Avisynth 2.6 defined them as size_t, AviSynth+ developers have left it as integer.
      AviSynth 2.6 had never reached a widespread x64 version so AviSynth+ variant "won")
    - New colorformats are added: Y8, YV411, YV16 and YV24.
      AviSynth+ extended these formats with 10, 12, 14, 16 bit integer and 32 bit floating point formats.
      Aside from 411, all other 4:2:0, 4:2:2 and 4:4:4 formats are available at higher bit depths.
      AviSynth+ introduced planar RGB formats for all the above mentioned bit depths.
      RGB48 and RGB64 is available as the 16 bit variants of RGB24 and RGB32 packed RGB formats.
      Alpha plane option was added for YUV (YUVA) and planar RGB formats (Planar RGBA).
    - :doc:`VideoInfo` has several new constants and functions (the
      ones relating to the new colorformats, the chroma placement
      constants, GetPlaneHeightSubsampling,
      GetPlaneWidthSubsampling).
      And in AviSynth+:
      BitsPerComponent, NumComponents, ComponentSize
      IsRGB48, IsRGB64, Is420, Is422, Is444, IsY,
      IsYUVA, IsPlanarRGB, IsPlanarRGBA
    - Some new cache and cpu constants for GetCPUFlags (the v5/v6 ones).
      In AviSynth+: CPU constants up to AVX512F, AVX512BW and on.
    - SetCacheHints changed from void to int.
    - AviSynth+: SetCacheHints constants helping automatic MT mode registration for plugins

- C API (AVISYNTH_INTERFACE_VERSION = 6):
    - The following functions are added to the interface:
      avs_is_yv24, avs_is_yv16, avs_is_yv12, avs_is_yv411,
      avs_is_y8, avs_is_color_space,
      avs_get_plane_width_subsampling,
      avs_get_plane_height_subsampling, avs_bits_per_pixel,
      avs_bytes_from_pixels, avs_row_size, avs_bmp_size,
      avs_get_row_size_p, avs_get_height_p and
      avs_delete_script_environment.
    - And in AviSynth+:
      avs_is_rgb48, avs_is_rgb64,
      avs_is_444, avs_is_422, avs_is_420, avs_is_y,
      avs_is_yuva, avs_is_planar_rgb, avs_is_planar_rgba
      avs_num_components, avs_component_size and avs_bits_per_component
      (and others which are not mentioned because they are deprecated)

What's new in the api V8
------------------------

- C++ API (AVISYNTH_INTERFACE_VERSION = 8):
    - The :ref:`IScriptEnvironment <cplusplus_iscriptenvironment>` interface has several new members:

        - :ref:`SubframePlanarA <cplusplus_subframeplanara>` alpha aware version of SubframePlanar.

        - :ref:`copyFrameProps <cplusplus_copyframeprops>` copy frame properties between video frames.
        - :ref:`getFramePropsRO <cplusplus_getframepropsro>` get pointer for reading frame properties
        - :ref:`getFramePropsRW <cplusplus_getframePropsrw>` get pointer for reading/writing frame properties.

        - :ref:`propNumKeys <cplusplus_propnumkeys>` get number of frame properties for a frame.

        - :ref:`propGetKey <cplusplus_propgetkey>` get name of key by index.
        - :ref:`propNumElements <cplusplus_propnumelements>` get array size of a property.
        - :ref:`propGetType <cplusplus_propGetType>` get property data type.

        - :ref:`propGetInt <cplusplus_propgetint>` get property value as integer (int64).
        - :ref:`propGetFloat <cplusplus_propgetfloat>` get property value as float (double).
        - :ref:`propGetData <cplusplus_propgetdata>` get property value as string buffer.
        - :ref:`propGetDataSize <cplusplus_propgetdatasize>` get string/data buffer size.
        - :ref:`propGetClip <cplusplus_propgetclip>` get property value as Clip.
        - :ref:`propGetFrame <cplusplus_propgetframe>` get property value as Frame.

        - :ref:`propDeleteKey <cplusplus_propdeletekey>` removes a frame property by name (key).

        - :ref:`propSetInt <cplusplus_propsetint>` sets integer (int64) frame property.
        - :ref:`propSetFloat <cplusplus_propsetfloat>` sets float (double) frame property.
        - :ref:`propSetData <cplusplus_propsetdata>` sets string (byte buffer) frame property.
        - :ref:`propSetClip <cplusplus_propsetclip>` sets PClip type frame property.
        - :ref:`propSetFrame <cplusplus_propsetframe>` sets PVideoFrame type frame property..

        - :ref:`propGetIntArray <cplusplus_propgetintarray>` array version of propGetInt.
        - :ref:`propGetFloatArray <cplusplus_propgetfloatarray>` array version of propGetFloat.
        - :ref:`propSetIntArray <cplusplus_propsetintarray>` array version of propSetInt.
        - :ref:`propSetFloatArray <cplusplus_propsetfloatarray>` array version of propSetFloat.

        - :ref:`createMap <cplusplus_createmap>` internal use only, creating frame property buffer.
        - :ref:`freeMap <cplusplus_freemap>` internal use only, frees up frame property buffer.
        - :ref:`clearMap <cplusplus_clearmap>` clears all properties for a frame.

        - :ref:`NewVideoFrameP <cplusplus_newvideoframep>` NewVideoFrame with frame property source.

        - :ref:`GetEnvProperty <cplusplus_getenvproperty>` Query to ask for various system (not frame!) properties.

        - :ref:`Allocate <cplusplus_allocate>` buffer pool allocate.
        - :ref:`Free <cplusplus_free>` buffer pool free.

        - :ref:`GetVarTry <cplusplus_getvartry>` get variable with success indicator.
        - :ref:`GetVarBool <cplusplus_getvarbool>` get bool value with default.
        - :ref:`GetVarInt <cplusplus_getvarint>` get int value with default.
        - :ref:`GetVarDouble <cplusplus_getvardouble>` get floating point value with default.
        - :ref:`GetVarString <cplusplus_getvarstring>` get string with default.
        - :ref:`GetVarLong <cplusplus_getvarlong>` get int64 with default.

        - enumeration constants for frame property, system property access
        - various other constants (MT modes, cache modes)

- C API (AVISYNTH_INTERFACE_VERSION = 8):
        - mostly the same functions as provided in C++ interface.
          Naming convention is kept. E.g. propSetFloat in C++ is prop_set_float in C
        - Important note: frame property access in V8 is broken. Safely available since V8.1.
          or check simply for V9 like ffmpeg does.

- C API (AVISYNTH_INTERFACE_VERSION = 8, AVISYNTH_INTERFACE_BUGFIX = 1):
        - working frame property access (see v9 comments for fixes)

What's new in the API V9
------------------------

- C and C++ API (AVISYNTH_INTERFACE_VERSION = 9):
        - :ref:`MakePropertyWritable <cplusplus_makepropertywritable>` like MakeWritable but for frame properties only.
        - :ref:`IsPropertyWritable <cplusplus_ispropertywritable>` like IsWritable but for frame properties only.
        - C interface equivalents: avs_make_property_writable and avs_is_property_writable

- C API (AVISYNTH_INTERFACE_VERSION = 9, AVISYNTH_INTERFACE_BUGFIX = 1):
        - Fix: C interface crash when using avs_new_video_frame_p(_a)
        - Fix: C interface avs_prop_get_data behave like C++ counterpart.

- C API (AVISYNTH_INTERFACE_VERSION = 9, AVISYNTH_INTERFACE_BUGFIX = 2):
        - Fix: C API undefined behavior when upstream throw runtime error

What's new in the API V10
-------------------------

- C and C++ API (AVISYNTH_INTERFACE_VERSION = 10):
        - Technical fix: made ``VideoFrameBuffer`` destructor public like in other classes of the public API to prevent compiler errors downstream when calling non-const member functions
        - New: :ref:`VideoFrame <cplusplus_videoframe>` (c++) and AVS_VideoFrame (c) now have its own pixel_type field. Before, there was no reliable way of knowing it on a frame from :ref:`propGetFrame <cplusplus_propgetframe>`.
        - New: :ref:`VideoFrame::GetPixelType <cplusplus_getpixeltype>` (avs_video_frame_get_pixel_type) returns the video format of a VideoFrame, ideally kept in sync with VideoInfo::pixel_type.
        - New: :ref:`VideoFrame::AmendPixelType <cplusplus_amendpixeltype>` (avs_video_frame_amend_pixel_type) changes the pixel_type field of a VideoFrame (special cases)

          C interface equivalents: avs_video_frame_get_pixel_type and avs_video_frame_amend_pixel_type
        - New: :ref:`AVSValue::GetType <cplusplus_avsvaluegettype>` returns the underlying type directly
        - Added ``AvsValueType`` enum for the above case to avisynth.h
        - Added DEFAULT_PLANE and AVS_DEFAULT_PLANE to plane enum (avisynth.h, avisynth_c.h)
        - Gave all enums of public C++ API a name (avisynth.h): AvsVersion, AvsSampleType, AvsPlane, AvsColorFormat, AvsImageTypeFlags, AvsChromaPlacement
        - prop_src argument of :ref:`NewVideoFrameP <cplusplus_newvideoframep>` (c++) and of 'avs_new_video_frame_p' (c) is now const
        - New: :ref:`VideoInfo::GetChannelMask <videoinfo_getchannelmask>` (c++) and 'avs_get_channel_mask' (c) Audio channel layout support. returns the channel mask stored in VideoInfo struct
        - New: :ref:`VideoInfo::SetChannelMask <videoinfo_setchannelmask>` (c++) and 'avs_set_channel_mask' Audio channel layout support. sets the validity and the layout mask value into VideoInfo
        - New: :ref:`VideoInfo::IsChannelMaskKnown <videoinfo_ischannelmaskknown>` (c++) and 'avs_is_channel_mask_known' (c) Audio channel layout support. Returns if clip has valid channel mask in its VideoInfo

.. _api_v11_whats_new:

What's new in the API V11
-------------------------

- C and C++ API (AVISYNTH_INTERFACE_VERSION = 11):
        - General: support 64 bit data types in ``AVSValue``/``AVS_Value``: ``double`` and ``long`` (``int64_t``), also for 32 bit Avisynth!
        - C++ interface
        
          - changed: ``AVSValue::IsFloat`` true for any 32/64 bit floating point or integer types
          - changed: ``AVSValue::IsInt`` true for any 32/64 bit integer types
          - new: ``AVSValue::IsFloatfStrict`` : returns true only if AVSValue is stricly 32 bit float
          - new: ``AVSValue::IsLongStrict`` : returns true only if AVSValue is stricly 64 bit integer
          - new: ``AVSValue::AsLong`` : returns int64_t
          - new: ``AVSValue::AsLong(int64_t def)``
          - No change: since AsFloat return type was double --> no change, it retrieves double values as well
          - new AVSValue constructors for 64 bit types
          - new: ``IScriptInterface::propGetIntSaturated`` and ``IScriptInterface::propGetFloatSaturated``
          - new: ``IScriptInterface::propGetDataTypeHint`` and ``IScriptInterface::propSetDataH``
          - New enum ``AVSPropDataTypeHint``: ``DATATYPEHINT_UNKNOWN``, ``DATATYPEHINT_BINARY`` and ``DATATYPEHINT_UTF8``.
          
        - C interface

          - New getter API calls: ``avs_get_as_long``, ``avs_get_as_int``, ``avs_get_as_float``
            and the rest: ``avs_get_as_bool``, ``avs_get_as_string``, ``avs_get_as_error``, ``avs_get_as_array``
          
          - New API call for array size query: ``avs_get_array_size``
          - New API call for array (or array-like value) content: ``avs_get_array_elt``
          - Modified INLINE typecheck and getter helpers for 64-bit data type awareness:
            
            * ``avs_is_int``, ``avs_is_float``
            * ``avs_as_int``, ``avs_as_float``
          
          - New INLINE type check helpers:
            
            * ``avs_is_long_strict``, ``avs_is_floatf_strict``

          - New INLINE getter helpers for 64-bit data (prefer using API calls):

            * ``avs_as_long``, ``avs_as_float``
          - New setter API calls: 
            
            * ``avs_set_to_double``, ``avs_set_to_long``
            * ``avs_set_to_array`` (deep arrays, deep copy, standard in AviSynth+)
              (Note: avs_release_value and avs_copy_value are required for destruct or copy arrays)
          - changed: ``avs_copy_value`` and ``avs_release_value`` now deep-copy and deep-release arrays,
            matching C++ ``AVSValue`` behavior. Previously the C interface only shallow-copied (single level)
            and did not free array elements (internally, arrays were flagged as non-deep-copy via a now unused
            code path). See :ref:`avs_copy_value <c_avs_copy_value>`.
          - API version of existing INLINE value setters (``new_value_xxx``) for the rest value types, to make the world round:
            
            * ``avs_set_to_error``, ``avs_set_to_bool``, ``avs_set_to_int``, ``avs_set_to_float``, ``avs_set_to_string``
            
          - new API function for assign a 'v'oid undefined value to AVS_Value
          
            * ``avs_set_to_void``
            
          - API version of existing INLINE type checks
          
            * ``avs_val_defined``, ``avs_val_is_error``, ``avs_val_is_bool``, ``avs_val_is_int``, ``avs_val_is_string``,
              ``avs_val_is_float``, ``avs_val_is_floatf_strict``, ``avs_val_is_long_strict``, ``avs_val_is_array``
           
          - New optional plugin entry point: ``avisynth_c_plugin_init2``
            
            * A C plugin signals to AviSynth that it is V11 interface (64-bit data) ready by implementing ``avisynth_c_plugin_init2`` as well.
            * ``avisynth_c_plugin_init2`` has the same signature as ``avisynth_c_plugin_init`` and can
              simply call forward to the old ``avisynth_c_plugin_init`` entry point. Both entry points can be implemented; 
              AviSynth+ will first check ``avisynth_c_plugin_init2``, then ``avisynth_c_plugin_init``.
              Don't forget to add a new 
              ::
              
                avisynth_c_plugin_init2@4 = _avisynth_c_plugin_init2@4
                
              line to your existing .def file on Win32.
              
          - New ``avs_prop_get_int_saturated`` and ``avs_prop_get_float_saturated``
          - New ``avs_prop_get_data_type_hint`` and ``avs_prop_set_data_h``
          - New constants AVS_PROPDATATYPEHINT_UNKNOWN, AVS_PROPDATATYPEHINT_BINARY and AVS_PROPDATATYPEHINT_UTF8
          
          - New alternative avs_add_function_r and APPLYFUNCR functions for cases where the callback cannot return 
            AVS_Value struct (functions/filters written Python / ctypes)

          - Deprecated inline helper functions. 
            
            * ``avs_get_pitch`` => ``avs_get_pitch_p(p, AVS_DEFAULT_PLANE)``
            * ``avs_get_row_size`` => ``avs_get_row_size_p(p, AVS_DEFAULT_PLANE``)
            * ``avs_get_height`` => ``avs_get_height_p(p, AVS_DEFAULT_PLANE)``
            * ``avs_get_read_ptr`` => ``avs_get_read_ptr_p(p, AVS_DEFAULT_PLANE)``
            * ``avs_get_write_ptr`` => ``avs_get_write_ptr_p(p, AVS_DEFAULT_PLANE)``
            * ``avs_release_frame`` => ``avs_release_video_frame``
            * ``avs_copy_frame`` => ``avs_copy_video_frame``
            * Use ``#define AVSC_ALLOW_DEPRECATED`` before including ``avisynth_c.h`` if they still need for you, 
              but better fix your code: use the recommended replacements.
              
          - Add missing ``AVS_MT_xxxx mode`` constants to ``avisynth_c.h`` (similar to c++ ``avisynth.h`` header ``enum MtMode``)

.. _api_v12_whats_new:

What's new in the API V12
-------------------------

- C and C++ API (AVISYNTH_INTERFACE_VERSION = 12):
        - Global Lock support: 

          * ``env->AcquireGlobalLock``, ``env->ReleaseGlobalLock`` (C++),
          * ``avs_acquire_global_lock``, ``avs_release_global_lock`` (C)
          
          See :ref:`global lock support<cplusplus_acquiregloballock>`

        - new ``CachePolicyHint::CACHE_INFORM_NUM_THREADS`` enum to inform the filter about the 
          number of threads by ``SetCacheHints`` (C interface: ``AVS_CACHE_INFORM_NUM_THREADS`` 
          and ``avs_set_cache_hints``).
          See :ref:`SetCacheHints<cplusplus_setcachehints>` .

        - wider GetCPUFlagsEx() 

          * ``env->GetCPUFlagsEx`` (C++),
          * ``avs_get_cpu_flags_ex`` (C)

          See :ref:`Getting CPU Flags<cplusplus_getcpuflags>`

        - L2 query with new ``AvsEnvProperty``:
        
          * ``AEP_CACHESIZE_L2`` (C++), 
          * ``AVS_AEP_CACHESIZE_L2`` (C) 
          
          to query L2 cache size in bytes by ``IScriptEnvironment->GetEnvProperty()``.
          x86/x64 architecture only for now. See :ref:`AvsEnvProperty<cplusplus_getenvproperty>` .

          See :ref:`GetEnvProperty<cplusplus_getenvproperty>` .

        - ARM64 (aarch64) CPU flags

- C++ API

          * ``env->ApplyMessageEx`` (C++),
          
          see :ref:`ApplyMessageEx<cplusplus_applymessage>`

.. _api_v13_whats_new:

What's new in the API V13
-------------------------

- C and C++ API (AVISYNTH_INTERFACE_VERSION = 13):
        - New API functions for additional pixel formats:

          * ``avs_is_440``, ``avs_is_410``, ``avs_is_411``, ``avs_is_ya`` (C)
          * ``Is440``, ``Is410``, ``Is411``, ``IsYA`` (C++)


Some history
------------

:doc:`Ben's AviSynth Docs <BensAviSynthDocs>` is the documentation written for AviSynth 1.0
by Ben Rudiak-Gould, in its original form.

See more about the modifications for AviSynth 2.5 in the :doc:`AviSynth Two-Five SDK <AviSynthTwoFiveSDK>`.

Please read AviSynth :doc:`SDK History <SDKHistory>`. ---

Other sources ???
-----------------

Once you've got the basics down on AVS development (Ben's text is quite
good), the `[SDK]`_ for VirtualDub is also a good read. Good news is you
won't have to worry about writing `[function pointers]`_ and `[raw Win32]`_;
meanwhile, Avery knows his stuff when it comes to video & CPU optimization
techniques, so you better pay attention.

Some video related ebooks (PDF) can be downloaded freely from `[Snell & Wilcox]`_. edit -
`this`_???

License terms
-------------

Note: Avisynth Filter SDK parts are under specific :doc:`SDK license <SDKLicense>` terms.

$Date: 2026/08/25 22:40:00 $

Latest online Avisynth+ version is at https://avisynthplus.readthedocs.io/en/latest/avisynthdoc/FilterSDK/FilterSDK.html
This one is maintained properly.

Latest online mediaWiki version is at http://avisynth.nl/index.php/Filter_SDK

.. _[SDK]: http://virtualdub.org/filtersdk
.. _[function pointers]: http://function-pointer.org/
.. _[raw Win32]: http://www.charlespetzold.com/pw5/index.html
.. _[Snell & Wilcox]: http://www.snellwilcox.com/reference.html
.. _[AviSynth Development forum]:
    http://forum.doom9.org/forumdisplay.php?s=&f=69
.. _AVIFile library: http://msdn.microsoft.com/en-us/library/windows/desktop/dd756808(v=vs.85).aspx
.. _[1]: http://forum.doom9.org/showthread.php?t=71493
.. _[2]: http://www.avisynth.info/?%E3%82%A2%E3%83%BC%E3%82%AB%E3%82%A4%E3%83%96#t952b3b1
.. _using the C api: http://komisar.gin.by/tools/avs2yuv/avs2yuv-0.24bm2.zip
.. _using the C++ api: http://kemuri9.net/dev/avs/avs2yuv/avs2yuv.zip
.. _avs2wav: http://forum.doom9.org/showthread.php?p=1502613#post1502613
.. _here: http://forum.doom9.org/showthread.php?p=589781#post589781
.. _avs2pipe: http://forum.doom9.org/showthread.php?t=160383
.. _this: http://www.snellgroup.com/support/documentation/engineering-guides
