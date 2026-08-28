
C API
=====

.. toctree::
    :maxdepth: 3

.. contents:: Table of contents

Preface
~~~~~~~

The ABI closely mirrors the native C++ ABI. It should provide all the functionality 
of C++. The main differences from the C++ ABI are:

- The way a new filter is created since virtual functions can't be used.
- Error handling since exceptions can't be used.
- Memory management since smart pointers can't be used.

The header, ``avisynth_c.h``, and some of its helpers in the ``avs/`` folder, declare 
all the classes, structures, and miscellaneous constants that you might need when 
writing a plugin or a client. All external plugins should ``#include`` it (which 
includes some more helper headers from the ``avs/`` directory)::

    #include "avisynth_c.h"

or if proper paths are set to the installed package and SDK include files::

    #include <avisynth_c.h>

Using C interface is an option, you can use it from C++ programs.

The Avisynth library can be used in two ways:

Dynamic loading
----------------

- Dynamic loading ``AviSynth.dll``/``libavisynth.so``/``.dylib``.

  Advantage: independence of the actual AviSynth version.

  AviSynth versions with different API levels can be supported, because we can detect 
  on loading the necessary API methods whether they exist or not. Earlier AviSynth 
  versions may contain fewer API functions. By detecting the loaded AviSynth/interface 
  version, it's the caller's responsibility to call only those API functions which 
  have valid function pointers and are documented to work. For example, you should 
  only use frame property-related functions when ``lib.avs_get_version(clip) >= 9``.

  - Use ``#define AVSC_NO_DECLSPEC`` for function pointer definitions only. 
    (``avisynth_c.h`` provides prototypes and a helper function for loading the 
    library in Windows.)
  - Load the library dynamically and get the necessary API functions as needed.

Static linking
--------------

- Static linking ``avisynth.lib``/``libavisynth``.

  - Leave ``AVSC_NO_DECLSPEC`` undefined.
  - Provide ``avisynth.lib`` to the linker.

  Drawback: Your plugin/software won't work with older AviSynth instances. If your 
  plugin/client is using newer API functions, your plugin or client will fail to 
  start due to dependency issues. (On Windows: platform returned code 127.)

Note: The library name for Windows is ``avisynth.lib``. On Unix-like systems, it is 
called ``libavisynth.so``, and on macOS, it is called ``libavisynth.dylib``.



Useful source links
~~~~~~~~~~~~~~~~~~~

Note from 2025: until this part is updated properly, check these excellent
examples for using C API as a client or in a plugin:

- ffmpeg (client)

  https://github.com/FFmpeg/FFmpeg/blob/release/7.1/libavformat/avisynth.c

- x265mod (client)

  https://github.com/Patman86/x265-Mod-by-Patman/blob/master/source/input/avs.cpp

- x264 (client)

  https://code.videolan.org/videolan/x264/-/blob/master/input/avs.c?ref_type=heads
  
- avs2yuv (client)

  https://github.com/DJATOM/avs2yuv
  
- Many-many plugins from Asd-g:

  https://github.com/Asd-g/
  
- AvsInPaint plugin

  https://github.com/pinterf/AvsInpaint
  
- AssRender plugin

  https://github.com/pinterf/assrender


Quick list
~~~~~~~~~~

+-------------------------------------+-----------------------+-------+
| Function                            | Area                  | ver   |
+-------------------------------------+-----------------------+-------+
| avs_add_function                    | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_at_exit                         | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_bit_blt                         | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_check_version                   | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_clip_get_error                  | AVS_Clip              | 3     |
+-------------------------------------+-----------------------+-------+
| avs_copy_clip                       | AVS_Clip              | 3     |
+-------------------------------------+-----------------------+-------+
| avs_copy_value                      | AVS_Value             | 3     |
+-------------------------------------+-----------------------+-------+
| avs_copy_video_frame                | AVS_VideoFrame        | 3     |
+-------------------------------------+-----------------------+-------+
| avs_create_script_environment       | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_function_exists                 | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_get_audio                       | AVS_Clip              | 3     |
+-------------------------------------+-----------------------+-------+
| avs_get_cpu_flags                   | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_get_frame                       | AVS_Clip              | 3     |
+-------------------------------------+-----------------------+-------+
| avs_get_parity                      | AVS_Clip              | 3     |
+-------------------------------------+-----------------------+-------+
| avs_get_var                         | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_get_version                     | AVS_Clip              | 3     |
+-------------------------------------+-----------------------+-------+
| avs_get_video_info                  | AVS_Clip              | 3     |
+-------------------------------------+-----------------------+-------+
| avs_invoke                          | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_make_writable                   | AVS_VideoFrame        | 3     |
+-------------------------------------+-----------------------+-------+
| avs_new_c_filter                    | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_new_video_frame_a               | AVS_VideoFrame        | 3     |
+-------------------------------------+-----------------------+-------+
| avs_release_clip                    | AVS_Clip              | 3     |
+-------------------------------------+-----------------------+-------+
| avs_release_value                   | AVS_Value             | 3     |
+-------------------------------------+-----------------------+-------+
| avs_release_video_frame             | AVS_VideoFrame        | 3     |
+-------------------------------------+-----------------------+-------+
| avs_save_string                     | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_set_cache_hints                 | AVS_Clip              | 3     |
+-------------------------------------+-----------------------+-------+
| avs_set_global_var                  | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_set_memory_max                  | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_set_to_clip                     | AVS_Value             | 3     |
+-------------------------------------+-----------------------+-------+
| avs_set_var                         | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_set_working_dir                 | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_subframe                        | AVS_VideoFrame        | 3     |
+-------------------------------------+-----------------------+-------+
| avs_take_clip                       | AVS_Value             | 3     |
+-------------------------------------+-----------------------+-------+
| avs_vsprintf                        | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_sprintf                         | AVS_ScriptEnvironment | 3     |
+-------------------------------------+-----------------------+-------+
| avs_delete_script_environment       | AVS_ScriptEnvironment | 6     |
+-------------------------------------+-----------------------+-------+
| avs_subframe_planar                 | AVS_VideoFrame        | 6     |
+-------------------------------------+-----------------------+-------+
| avs_get_error                       | AVS_ScriptEnvironment | 6     |
+-------------------------------------+-----------------------+-------+
| avs_is_yv24                         | AVS_VideoInfo         | 6     |
+-------------------------------------+-----------------------+-------+
| avs_is_yv16                         | AVS_VideoInfo         | 6     |
+-------------------------------------+-----------------------+-------+
| avs_is_yv12                         | AVS_VideoInfo         | 6     |
+-------------------------------------+-----------------------+-------+
| avs_is_yv411                        | AVS_VideoInfo         | 6     |
+-------------------------------------+-----------------------+-------+
| avs_is_y8                           | AVS_VideoInfo         | 6     |
+-------------------------------------+-----------------------+-------+
| avs_is_color_space                  | AVS_VideoInfo         | 6     |
+-------------------------------------+-----------------------+-------+
| avs_get_plane_width_subsampling     | AVS_VideoInfo         | 6     |
+-------------------------------------+-----------------------+-------+
| avs_get_plane_height_subsampling    | AVS_VideoInfo         | 6     |
+-------------------------------------+-----------------------+-------+
| avs_bits_per_pixel                  | AVS_VideoInfo         | 6     |
+-------------------------------------+-----------------------+-------+
| avs_bytes_from_pixels               | AVS_VideoInfo         | 6     |
+-------------------------------------+-----------------------+-------+
| avs_row_size                        | AVS_VideoFrame        | 6     |
+-------------------------------------+-----------------------+-------+
| avs_bmp_size                        | AVS_VideoInfo         | 6     |
+-------------------------------------+-----------------------+-------+
| avs_get_pitch_p                     | AVS_VideoFrame        | 6     |
+-------------------------------------+-----------------------+-------+
| avs_get_row_size_p                  | AVS_VideoFrame        | 6     |
+-------------------------------------+-----------------------+-------+
| avs_get_height_p                    | AVS_VideoFrame        | 6     |
+-------------------------------------+-----------------------+-------+
| avs_get_read_ptr_p                  | AVS_VideoFrame        | 6     |
+-------------------------------------+-----------------------+-------+
| avs_is_writable                     | AVS_VideoFrame        | 6     |
+-------------------------------------+-----------------------+-------+
| avs_get_write_ptr_p                 | AVS_VideoFrame        | 6     |
+-------------------------------------+-----------------------+-------+
| avs_is_yuv444p16                    | AVS_VideoInfo         | 6+  X |
+-------------------------------------+-----------------------+-------+
| avs_is_yuv422p16                    | AVS_VideoInfo         | 6+  X |
+-------------------------------------+-----------------------+-------+
| avs_is_yuv420p16                    | AVS_VideoInfo         | 6+  X |
+-------------------------------------+-----------------------+-------+
| avs_is_y16                          | AVS_VideoInfo         | 6+  X |
+-------------------------------------+-----------------------+-------+
| avs_is_yuv444ps                     | AVS_VideoInfo         | 6+  X |
+-------------------------------------+-----------------------+-------+
| avs_is_yuv422ps                     | AVS_VideoInfo         | 6+  X |
+-------------------------------------+-----------------------+-------+
| avs_is_yuv420ps                     | AVS_VideoInfo         | 6+  X |
+-------------------------------------+-----------------------+-------+
| avs_is_y32                          | AVS_VideoInfo         | 6+  X |
+-------------------------------------+-----------------------+-------+
| avs_num_components                  | AVS_VideoInfo         | 6+    |
+-------------------------------------+-----------------------+-------+
| avs_component_size                  | AVS_VideoInfo         | 6+    |
+-------------------------------------+-----------------------+-------+
| avs_bits_per_component              | AVS_VideoInfo         | 6+    |
+-------------------------------------+-----------------------+-------+
| avs_is_444                          | AVS_VideoInfo         | 6+    |
+-------------------------------------+-----------------------+-------+
| avs_is_422                          | AVS_VideoInfo         | 6+    |
+-------------------------------------+-----------------------+-------+
| avs_is_420                          | AVS_VideoInfo         | 6+    |
+-------------------------------------+-----------------------+-------+
| avs_is_y                            | AVS_VideoInfo         | 6+    |
+-------------------------------------+-----------------------+-------+
| avs_is_yuva                         | AVS_VideoInfo         | 6+    |
+-------------------------------------+-----------------------+-------+
| avs_is_planar_rgb                   | AVS_VideoInfo         | 6+    |
+-------------------------------------+-----------------------+-------+
| avs_is_planar_rgba                  | AVS_VideoInfo         | 6+    |
+-------------------------------------+-----------------------+-------+
| avs_is_rgb48                        | AVS_VideoInfo         | 6+    |
+-------------------------------------+-----------------------+-------+
| avs_is_rgb64                        | AVS_VideoInfo         | 6+    |
+-------------------------------------+-----------------------+-------+
| avs_subframe_planar_a               | AVS_VideoFrame        | 8     |
+-------------------------------------+-----------------------+-------+
| avs_copy_frame_props                | AVS_VideoFrame        | 8     |
+-------------------------------------+-----------------------+-------+
| avs_get_frame_props_ro              | AVS_VideoFrame        | 8     |
+-------------------------------------+-----------------------+-------+
| avs_get_frame_props_rw              | AVS_VideoFrame        | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_num_keys                   | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_get_key                    | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_num_elements               | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_get_type                   | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_get_int                    | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_get_float                  | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_get_data                   | AVS_Map               | 9.1 ! |
+-------------------------------------+-----------------------+-------+
| avs_prop_get_data_size              | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_get_clip                   | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_get_frame                  | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_delete_key                 | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_set_int                    | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_set_float                  | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_set_data                   | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_set_clip                   | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_set_frame                  | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_get_int_array              | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_get_float_array            | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_set_int_array              | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_prop_set_float_array            | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_clear_map                       | AVS_Map               | 8     |
+-------------------------------------+-----------------------+-------+
| avs_new_video_frame_p               | AVS_ScriptEnvironment | 8     |
+-------------------------------------+-----------------------+-------+
| avs_new_video_frame_p_a             | AVS_ScriptEnvironment | 8     |
+-------------------------------------+-----------------------+-------+
| avs_get_env_property                | AVS_ScriptEnvironment | 8     |
+-------------------------------------+-----------------------+-------+
| avs_pool_allocate                   | AVS_ScriptEnvironment | 8     |
+-------------------------------------+-----------------------+-------+
| avs_pool_free                       | AVS_ScriptEnvironment | 8     |
+-------------------------------------+-----------------------+-------+
| avs_get_var_try                     | AVS_ScriptEnvironment | 8     |
+-------------------------------------+-----------------------+-------+
| avs_get_var_bool                    | AVS_ScriptEnvironment | 8     |
+-------------------------------------+-----------------------+-------+
| avs_get_var_int                     | AVS_ScriptEnvironment | 8     |
+-------------------------------------+-----------------------+-------+
| avs_get_var_double                  | AVS_ScriptEnvironment | 8     |
+-------------------------------------+-----------------------+-------+
| avs_get_var_string                  | AVS_ScriptEnvironment | 8     |
+-------------------------------------+-----------------------+-------+
| avs_get_var_long                    | AVS_ScriptEnvironment | 8     |
+-------------------------------------+-----------------------+-------+
| avs_is_property_writable            | AVS_VideoFrame        | 9     |
+-------------------------------------+-----------------------+-------+
| avs_make_property_writable          | AVS_VideoFrame        | 9     |
+-------------------------------------+-----------------------+-------+
| avs_video_frame_get_pixel_type      | AVS_VideoFrame        | 10    |
+-------------------------------------+-----------------------+-------+
| avs_video_frame_amend_pixel_type    | AVS_VideoFrame        | 10    |
+-------------------------------------+-----------------------+-------+
| avs_is_channel_mask_known           | AVS_VideoInfo         | 10.1  |
+-------------------------------------+-----------------------+-------+
| avs_set_channel_mask                | AVS_VideoInfo         | 10.1  |
+-------------------------------------+-----------------------+-------+
| avs_get_channel_mask                | AVS_VideoInfo         | 10.1  |
+-------------------------------------+-----------------------+-------+
| avs_set_to_void                     | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_set_to_error                    | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_set_to_bool                     | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_set_to_int                      | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_set_to_string                   | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_set_to_float                    | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_set_to_double                   | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_set_to_long                     | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_set_to_array                    | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_get_as_bool                     | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_get_as_int                      | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_get_as_long                     | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_get_as_string                   | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_get_as_float                    | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_get_as_error                    | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_get_as_array                    | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_get_array_elt                   | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_get_array_size                  | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_val_defined                     | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_val_is_error                    | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_val_is_bool                     | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_val_is_int                      | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_val_is_string                   | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_val_is_float                    | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_val_is_floatf_strict            | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_val_is_long_strict              | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_val_is_array                    | AVS_Value             | 11    |
+-------------------------------------+-----------------------+-------+
| avs_prop_get_int_saturated          | AVS_Map               | 11    |
+-------------------------------------+-----------------------+-------+
| avs_prop_get_float_saturated        | AVS_Map               | 11    |
+-------------------------------------+-----------------------+-------+
| avs_prop_get_data_type_hint         | AVS_Map               | 11    |
+-------------------------------------+-----------------------+-------+
| avs_prop_set_data_h                 | AVS_Map               | 11    |
+-------------------------------------+-----------------------+-------+
| avs_add_function_r                  | AVS_ScriptEnvironment | 11    |
+-------------------------------------+-----------------------+-------+
| avs_get_cpu_flags_ex                | AVS_ScriptEnvironment | 12    |
+-------------------------------------+-----------------------+-------+
| avs_is_440                          | AVS_VideoInfo         | 13    |
+-------------------------------------+-----------------------+-------+
| avs_is_410                          | AVS_VideoInfo         | 13    |
+-------------------------------------+-----------------------+-------+
| avs_is_411                          | AVS_VideoInfo         | 13    |
+-------------------------------------+-----------------------+-------+
| avs_is_ya                           | AVS_VideoInfo         | 13    |
+-------------------------------------+-----------------------+-------+


Reference
~~~~~~~~~

.. _c_avs_add_function:

avs_add_function
----------------


::

    int avs_add_function(AVS_ScriptEnvironment *, 
                         const char * name, const char * params,
                         AVS_ApplyFunc apply, void * user_data);

    int avs_add_function_r(AVS_ScriptEnvironment *, 
                         const char * name, const char * params,
                         AVS_ApplyFuncR apply, void * user_data);


Both forms define the function name, parameter signature, and the callback function itself. The 
difference lies in the type of the callback function (apply).

``avs_add_function`` and ``avs_add_function_r`` are used to inform AviSynth of the existence of
our filter. These functions register a function with AviSynth's internal function table.

The base ``avs_add_function`` takes four arguments: the name of the new script function, the 
parameter-type string, the C function (callback) implementing the script function, and the 
user_data cookie.

The added function returns a type ``AVS_Value`` and can therefore return any AVS_Value type, Clip, string, 
integer, double, etc.. In this version the function returns AVS_Value directly.

The second form, ``avs_add_function_r``, is an alternative approach where the function result is 
provided by filling the ``AVS_Value`` result into a passed pointer.

This is particularly useful when interfacing with Python. In Python 3.13 (as of 2025), our callback
written in Python cannot return structs (like ``AVS_Value``) directly to the C caller via a function 
return value. However, it can accept and fill an ``AVS_Value`` C struct passed as a pointer.

The first form avs_add_function, the callback (``apply``) returns result as return value (``AVS_Value``).
This is the callback type used by ``avs_add_function``:
::

    typedef AVS_Value (AVSC_CC * AVS_ApplyFunc)(AVS_ScriptEnvironment *, 
                                                AVS_Value args, void * user_data);

Int the alternative form, the callback (``apply``) returns result in byref parameter (``AVS_Value *``)
This is the callback type used by ``avs_add_function_r``:
::

    typedef void(AVSC_CC* AVS_ApplyFuncR)(AVS_ScriptEnvironment*, 
                                          AVS_Value* ret, AVS_Value args, void* user_data);

Its main purpose in ``avisynth_c_plugin_init`` or ``avisynth_c_plugin_init2`` to add and define plugin filters.
But a client or a plugin can define its own functions or filters as well.

Example:
::

    static AVS_Value AVSC_CC Create_JincResize(AVS_ScriptEnvironment* env, AVS_Value args, void* param) {
    ...
    }

    avs_add_function(env, "JincResize", "cii[src_left]f[src_top]f[src_width]f[src_height]f[quant_x]i[quant_y]i"
                                        "[tap]i[blur]f[cplace]s[threads]i[opt]i", Create_JincResize, 0);

Example (alternative version): a simple (non-Clip oriented) example which would add X to the input.
::

    static void AVSC_CC Create_IncreaseBy(AVS_ScriptEnvironment* env, AVS_Value *retval, AVS_Value args, void* param) {
    ...
    }

    avs_add_function_r(env, "IncreaseBy", "[delta]i", Create_IncreaseBy, 0);


The added function is of type AVSValue and can therefore return any AVSValue. 

For more info and examples see also :ref:`cplusplus_addfunction` in C++ API.

Related also :ref:`c_callbacks`


.. _c_avs_at_exit:

avs_at_exit
-----------

Sets the callback, which will be called when the ``AVS_ScriptEnvironment`` is destroyed by Avisynth core.

Related also :ref:`c_at_exit`
See also: :ref:`cplusplus_atexit`

.. _c_avs_bit_blt:

avs_bit_blt
-----------

See also: :ref:`cplusplus_bitblt`

.. _c_avs_check_version:

avs_check_version
-----------------

See also: :ref:`cplusplus_checkversion`

.. _c_avs_clip_get_error:

avs_clip_get_error
------------------

Use this after an avs_get_frame, there is no other way to check if GetFrame was successful.
No c++ API equivalent.

.. _c_avs_copy_clip:

avs_copy_clip
-------------

Clones an ``AVS_Clip`` variable, this is important because clip objects are reference counted. When the reference
count reaches zero, the clip is destroyed from memory.

This clip must be released manually by ``avs_release_clip``.

In c++ a simple PClip clip = clip2 assignment does this task, PClip is a smart pointers in C++.

See also: :ref:`cplusplus_pclip`

.. _c_avs_copy_value:

avs_copy_value
--------------

Clones an ``AVS_Value`` variable, this is important because AVS_Value objects can be reference counted (clip, function object - latter 
is not available in the C interface), or resource is allocated for them in Avisynth core (for 64 bit long, double and (dynamic) arrays, 
maybe strings in the future). 

At clip content, its reference count is increased by one.

This ``avs_copy_value`` is the _proper_ copy of an ``AVS_Value``. See also ``avs_array_elt`` which obtains an ``AVS_Value`` element from
an array; but it just gets a duplicate (field-by-field single AVS_Value copy, no reference counts are affected) from its ``AVS_Value`` 
content. 

The variable obtained with ``avs_copy_value`` must be released with ``avs_release_value``.

The function performs deep-copy if the source is an array, the result is a dynamic array (managed by Avisynth) in this case.

As mentioned above, avs_copy_value results must be released, unlike ``array_elt``'d values, which must not be freed; 
as a best practice, keep the array_elt'd value as-is and always free the original C array elements.

If array is an Avisynth+ ``dynamic array`` (all arrays are smart dynamic arrays in script and C++ AVSValue), then 
call ``avs_release_value`` for the array alone would deep-free the whole content. 
But when the array is a C allocated one, free their elements with ``avs_release_value`` one by one 
(except if the value is another C array). You can do anyhow, but remember, release only once. Calling avs_release_value
on a C array would result in crash.

In c++ a simple ``AVSValue var1 = var2`` assignment does this task, ``AVSValue`` is a smart pointer in C++.

See also: :ref:`cplusplus_avsvalue`

.. _c_avs_copy_video_frame:

avs_copy_video_frame
--------------------

Clones an ``AVS_VideoFrame`` variable, but not only copies a single pointer but increases the frame's reference counter.
This is important because video frame objects are reference counted. On release the reference count is decreased by one.
When the reference count reaches zero, the video frame is destroyed from memory.

The obtained video frame must be released manually by ``avs_release_video_frame``.

In c++ a simple ``PVideoFrame frame_dst = frame_src`` assignment does this task, ``PVideoFrame`` is a smart pointers in C++.

See also: :ref:`cplusplus_videoframe`

.. _c_avs_create_script_environment:

avs_create_script_environment
-----------------------------

See also: :ref:`cplusplus_createscriptenvironment`

.. _c_avs_function_exists:

avs_function_exists
-------------------

See also: :ref:`cplusplus_functionexists`

.. _c_avs_get_audio:

avs_get_audio
-------------

See also: :ref:`cplusplus_getaudio`

.. _c_avs_get_cpu_flags:

avs_get_cpu_flags
-----------------

See also: :ref:`cplusplus_getcpuflags`

.. _c_avs_get_cpu_flags_ex:

avs_get_cpu_flags_ex
--------------------

See also: :ref:`cplusplus_getcpuflagsex`

.. _c_avs_get_frame:

avs_get_frame
-------------

The obtained video frame must be released with avs_release_video_frame.

See also: :ref:`cplusplus_getframe`

.. _c_avs_get_parity:

avs_get_parity
--------------

See also: :ref:`cplusplus_getparity`

.. _c_avs_get_var:

avs_get_var
-----------

See also: :ref:`cplusplus_getvar`

.. _c_avs_get_version:

avs_get_version
---------------

For checking the actual Avisynth interface version, use the ``avs_check_version`` and 
the ``avs_get_env_property`` functions with AVS_AEP_xxx query values.

See also: :ref:`cplusplus_getversion`

.. _c_avs_get_video_info:

avs_get_video_info
------------------

See also: :ref:`GetVideoInfo<cplusplus_getvideoinfo>`

.. _c_avs_invoke:

avs_invoke
----------

See also: :ref:`cplusplus_invoke`

.. _c_avs_make_writable:

avs_make_writable
-----------------

See also: :ref:`cplusplus_makewritable`

.. _c_avs_new_c_filter:

avs_new_c_filter
----------------

::

    // This is the callback type used by avs_add_function
    typedef AVS_Value (AVSC_CC * AVS_ApplyFunc)
                            (AVS_ScriptEnvironment *, AVS_Value args, void * user_data);

    // v11 alternative of avs_add_function with return value by reference
    // This is the callback type used by avs_add_function_r
    typedef void(AVSC_CC* AVS_ApplyFuncR)
    (AVS_ScriptEnvironment*, AVS_Value* ret, AVS_Value args, void* user_data);

    typedef struct AVS_FilterInfo AVS_FilterInfo;
    struct AVS_FilterInfo
    {
      // these members should not be modified outside of the AVS_ApplyFunc or AVS_ApplyFuncR callback
      AVS_Clip * child;
      AVS_VideoInfo vi;
      AVS_ScriptEnvironment * env;
      AVS_VideoFrame * (AVSC_CC * get_frame)(AVS_FilterInfo *, int n);
      int (AVSC_CC * get_parity)(AVS_FilterInfo *, int n);
      int (AVSC_CC * get_audio)(AVS_FilterInfo *, void * buf,
                                      int64_t start, int64_t count);
      int (AVSC_CC * set_cache_hints)(AVS_FilterInfo *, int cachehints,
                                            int frame_range);
      void (AVSC_CC * free_filter)(AVS_FilterInfo *);

      // Should be set when ever there is an error to report.
      // It is cleared before any of the above methods are called
      const char * error;
      // this is to store whatever and may be modified at will
      void * user_data;
    };

    // Create a new filter
    // 'fi' is set to point to the AVS_FilterInfo so that you can
    //   modify it once it is initialized.
    // 'store_child' should generally be set to true.  If it is not
    //   set then ALL methods (the function pointers) must be defined
    // If it is set then you do not need to worry about freeing the child
    //    clip.
    AVSC_API(AVS_Clip *, avs_new_c_filter)(AVS_ScriptEnvironment * e,
                                           AVS_FilterInfo * * fi,
                                           AVS_Value child, int store_child);

In a filter defintion lifetime ``avs_add_function`` defines a filter with its name, parameter signature, and a 
callback (APPLYFUNC/APPLYFUNCR) function, which is called by Avisynth when it instantiates the filter: when founds 
its name in the script and parameters are matching and their values are known. When a script contains multiple
occurances in the filter, each with different parameter list, this APPLYFUNC is called for each occurances.

``avs_new_c_filter`` must be called from inside this filter creating ``APPLYFUNC`` or ``APPLYFUNCR`` type function. 
Dynamic resource allocation for a struct for holding the filter actual parameters can also be done, typically 
address of this struct is passed as a ``(void *)`` cookie in user_data in filter defition, in e.g. the ``get_frame`` 
callback, and accesible at the end of the filter liferime: in free_filter callback.

Through the ``AVS_FilterInfo`` struct the callbacks ``get_frame``, ``get_audio``, ``set_cache_hints``, ``get_parity`` 
can be set. Yes, ``avs_new_c_filter`` passes a pointer to pointer of ``AVS_FilterInfo``.

``free_filter`` is a callback, practically serves as the desctructor, like the desctructor of C++ IClip descendant).
Resources allocated at filter creation can be released in there.

Errors during APPLYFUNC/APPYFUNCR must be signed with a filled ``const char* error`` field in ``AVS_FilterInfo``

C++ equivalent: no. It's much different. C++ works with ``IClip`` descendants, like ``GenericVideoFilter`` or
``NonCachedGenericVideoFilter`` classes.

.. _c_avs_new_video_frame_a:

avs_new_video_frame_a
---------------------

.. _c_avs_release_clip:

avs_release_clip
----------------

Decreases the ``AVS_Clip`` reference count by one. When it reaches zero, the underlying PClip object is destroyed.
When there was a C function (filter) associated with it, its ``free_filter`` callback is called.

In C++ PClip is a smart pointer: reference counting and release is automatically done.

See also: :ref:`c_avs_new_c_filter`
See also: :ref:`c_avs_take_clip`
See also: :ref:`c_avs_copy_clip`
See also: :ref:`cplusplus_pclip`

.. _c_avs_release_value:

avs_release_value
-----------------

If the AVS_Value holds a clip or video frame, their reference is decreased by one. When a clip or video_frame
reference count reaches zero, it will be also freed up.

If the AVS_Value holds an array, Avisynth will try to deep-free the elements then deallocates
the occupied memory. DO NOT call avs_release_value to a preallocated array from your C or C++ code.

Resource freeing can happen for 64 bit long and double types on 32-bit systems.
And maybe on strings in the future.

So as a best practice, call avs_release_value for all types, except a C array variable itself, as necessary.
Take care of value obtained by ``avs_array_elt``, since this does not do real resource duplication. Either the
obtained value, or the AVS_Value in the original C array must be freed.

In C++ AVSValue is a smart pointer, its desctructor is called automatically.

See also: :ref:`cplusplus_avsvalue`

.. _c_avs_release_video_frame:

avs_release_video_frame
-----------------------

Videoframe objects are created by ``get_frame`` (its return value), ``avs_copy_video_frame``, ``avs_new_video_frame_a``, 
``avs_new_video_frame_p``, ``avs_new_video_frame_p_a``, ``avs_subframe_planar`` and ``avs_subframe_planar_a``.

The function decreases the ``AVS_VideoFrame`` reference count by one. When it reaches zero, the underlying PVideoFrame object is destroyed.

Except when it serves as the filter return value, AVS_VideoFrame must be released.

In C++ PVideoFrame is a smart pointer: reference counting and release is automatically done.

See also: :ref:`cplusplus_videoframe`

.. _c_avs_save_string:

avs_save_string
---------------

Avisynth must make copy of a volatile string, in order able to pass them further on in the filter chain.
If a function would just fill a local string buffer with text, the content disappears after the function exit.
With avs_save_string Avisynth with take a static copy of the string, preserving it for good (until the ScriptEnvironment lives).

See also: :ref:`cplusplus_savestring`

.. _c_avs_set_cache_hints:

avs_set_cache_hints
-------------------

See also: :ref:`cplusplus_setcachehints`

.. _c_avs_set_global_var:

avs_set_global_var
------------------

See also: :ref:`cplusplus_setglobalvar`

.. _c_avs_set_memory_max:

avs_set_memory_max
------------------

See also: :ref:`cplusplus_setmemorymax`

.. _c_avs_set_to_clip:

avs_set_to_clip
---------------

Stuffs an AVS_Clip into an AVS_Value. Clip reference count is increased by one. After this, the original
AVS_Clip can be released with avs_release_clip.

In C++ a simple direct assignment is done: ``PClip clip = ... ; AVSValue retval = clip;``

See also: :ref:`c_avs_take_clip`
See also: :ref:`cplusplus_pclip`

.. _c_avs_set_var:

avs_set_var
-----------

.. _c_avs_set_working_dir:

avs_set_working_dir
-------------------

.. _c_avs_subframe:

avs_subframe
------------

See also: :ref:`cplusplus_subframe`

.. _c_avs_take_clip:

avs_take_clip
-------------

Reverse of ``avs_set_to_clip``. Extracts an ``AVS_Clip`` from an ``AVS_Value``. Clip reference count is increased by
one. After this, the original ``AVS_Value`` can be released with ``avs_release_value``.

In C++ a simple direct assignment is done: ``AVSValue val = args[0]; PClip retval = val``

See also: :ref:`c_avs_set_to_clip`
See also: :ref:`cplusplus_pclip`

.. _c_avs_vsprintf:

avs_vsprintf
------------

.. _c_avs_sprintf:

avs_sprintf
-----------

.. _c_avs_delete_script_environment:

avs_delete_script_environment
-----------------------------

End of the world, end of everything. AtExit procedures are called during the destroy as a side effect.

See also: :ref:`cplusplus_deletescriptenvironment`

.. _c_avs_subframe_planar:

avs_subframe_planar
-------------------

See also: :ref:`cplusplus_subframeplanar`

.. _c_avs_get_error:

avs_get_error
-------------

.. _c_avs_is_yv24:

avs_is_yv24
-----------

.. _c_avs_is_yv16:

avs_is_yv16
-----------

.. _c_avs_is_yv12:

avs_is_yv12
-----------

.. _c_avs_is_yv411:

avs_is_yv411
------------

.. _c_avs_is_y8:

avs_is_y8
---------

.. _c_avs_is_color_space:

avs_is_color_space
------------------

.. _c_avs_get_plane_width_subsampling:

avs_get_plane_width_subsampling
-------------------------------

.. _c_avs_get_plane_height_subsampling:

avs_get_plane_height_subsampling
--------------------------------

.. _c_avs_bits_per_pixel:

avs_bits_per_pixel
------------------

.. _c_avs_bytes_from_pixels:

avs_bytes_from_pixels
---------------------

.. _c_avs_row_size:

avs_row_size
------------

.. _c_avs_bmp_size:

avs_bmp_size
------------

.. _c_avs_get_pitch_p:

avs_get_pitch_p
---------------

.. _c_avs_get_row_size_p:

avs_get_row_size_p
------------------

.. _c_avs_get_height_p:

avs_get_height_p
----------------

.. _c_avs_get_read_ptr_p:

avs_get_read_ptr_p
------------------

.. _c_avs_is_writable:

avs_is_writable
---------------

.. _c_avs_get_write_ptr_p:

avs_get_write_ptr_p
-------------------

.. _c_avs_is_yuv444p16:

avs_is_yuv444p16
----------------

.. _c_avs_is_yuv422p16:

avs_is_yuv422p16
----------------

.. _c_avs_is_yuv420p16:

avs_is_yuv420p16
----------------

.. _c_avs_is_y16:

avs_is_y16
----------

.. _c_avs_is_yuv444ps:

avs_is_yuv444ps
---------------

.. _c_avs_is_yuv422ps:

avs_is_yuv422ps
---------------

.. _c_avs_is_yuv420ps:

avs_is_yuv420ps
---------------

.. _c_avs_is_y32:

avs_is_y32
----------

.. _c_avs_num_components:

avs_num_components
------------------

See also: :doc:`VideoInfo struct <VideoInfo>`

.. _c_avs_component_size:

avs_component_size
------------------

See also: :doc:`VideoInfo struct <VideoInfo>`

.. _c_avs_bits_per_component:

avs_bits_per_component
----------------------

See also: :doc:`VideoInfo struct <VideoInfo>`

.. _c_avs_is_444:

avs_is_444
----------

See also: :doc:`VideoInfo struct <VideoInfo>`

.. _c_avs_is_422:

avs_is_422
----------

.. _c_avs_is_420:

avs_is_420
----------

.. _c_avs_is_y:

avs_is_y
--------

.. _c_avs_is_yuva:

avs_is_yuva
-----------

.. _c_avs_is_planar_rgb:

avs_is_planar_rgb
-----------------

.. _c_avs_is_planar_rgba:

avs_is_planar_rgba
------------------

.. _c_avs_is_rgb48:

avs_is_rgb48
------------

.. _c_avs_is_rgb64:

avs_is_rgb64
------------

.. _c_avs_is_440:

avs_is_440
----------

.. _c_avs_is_410:

avs_is_410
----------

.. _c_avs_is_411:

avs_is_411
----------

.. _c_avs_is_ya:

avs_is_ya
---------

.. _c_avs_subframe_planar_a:

avs_subframe_planar_a
---------------------

See also: :ref:`cplusplus_subframeplanara`

.. _c_avs_copy_frame_props:

avs_copy_frame_props
--------------------

.. _c_avs_get_frame_props_ro:

avs_get_frame_props_ro
----------------------

.. _c_avs_get_frame_props_rw:

avs_get_frame_props_rw
----------------------

.. _c_avs_prop_num_keys:

avs_prop_num_keys
-----------------

.. _c_avs_prop_get_key:

avs_prop_get_key
----------------

.. _c_avs_prop_num_elements:

avs_prop_num_elements
---------------------

.. _c_avs_prop_get_type:

avs_prop_get_type
-----------------

.. _c_avs_prop_get_int:

avs_prop_get_int
----------------

.. _c_avs_prop_get_float:

avs_prop_get_float
------------------

.. _c_avs_prop_get_data:

avs_prop_get_data
-----------------

.. _c_avs_prop_get_data_size:

avs_prop_get_data_size
----------------------

.. _c_avs_prop_get_clip:

avs_prop_get_clip
-----------------

.. _c_avs_prop_get_frame:

avs_prop_get_frame
------------------

.. _c_avs_prop_delete_key:

avs_prop_delete_key
-------------------

.. _c_avs_prop_set_int:

avs_prop_set_int
----------------

.. _c_avs_prop_set_float:

avs_prop_set_float
------------------

.. _c_avs_prop_set_data:

avs_prop_set_data
-----------------

.. _c_avs_prop_set_clip:

avs_prop_set_clip
-----------------

.. _c_avs_prop_set_frame:

avs_prop_set_frame
------------------

.. _c_avs_prop_get_int_array:

avs_prop_get_int_array
----------------------

.. _c_avs_prop_get_float_array:

avs_prop_get_float_array
------------------------

.. _c_avs_prop_set_int_array:

avs_prop_set_int_array
----------------------

.. _c_avs_prop_set_float_array:

avs_prop_set_float_array
------------------------

.. _c_avs_clear_map:

avs_clear_map
-------------

.. _c_avs_new_video_frame_p:

avs_new_video_frame_p
---------------------

.. _c_avs_new_video_frame_p_a:

avs_new_video_frame_p_a
-----------------------

.. _c_avs_get_env_property:

avs_get_env_property
--------------------

.. _c_avs_pool_allocate:

avs_pool_allocate
-----------------

.. _c_avs_pool_free:

avs_pool_free
-------------

.. _c_avs_get_var_try:

avs_get_var_try
---------------

.. _c_avs_get_var_bool:

avs_get_var_bool
----------------

.. _c_avs_get_var_int:

avs_get_var_int
---------------

.. _c_avs_get_var_double:

avs_get_var_double
------------------

.. _c_avs_get_var_string:

avs_get_var_string
------------------

.. _c_avs_get_var_long:

avs_get_var_long
----------------

.. _c_avs_is_property_writable:

avs_is_property_writable
------------------------

.. _c_avs_make_property_writable:

avs_make_property_writable
--------------------------

.. _c_avs_video_frame_get_pixel_type:

avs_video_frame_get_pixel_type
------------------------------

.. _c_avs_video_frame_amend_pixel_type:

avs_video_frame_amend_pixel_type
--------------------------------

.. _c_avs_is_channel_mask_known:

avs_is_channel_mask_known
-------------------------

.. _c_avs_set_channel_mask:

avs_set_channel_mask
--------------------

.. _c_avs_get_channel_mask:

avs_get_channel_mask
--------------------

.. _c_avs_set_to_void:

avs_set_to_void
---------------

.. _c_avs_set_to_error:

avs_set_to_error
----------------

.. _c_avs_set_to_bool:

avs_set_to_bool
---------------

.. _c_avs_set_to_int:

avs_set_to_int
--------------

.. _c_avs_set_to_string:

avs_set_to_string
-----------------

.. _c_avs_set_to_float:

avs_set_to_float
----------------

.. _c_avs_set_to_double:

avs_set_to_double
-----------------

.. _c_avs_set_to_long:

avs_set_to_long
---------------

.. _c_avs_set_to_array:

avs_set_to_array
--------------------

.. _c_avs_get_as_bool:

avs_get_as_bool
---------------

.. _c_avs_get_as_int:

avs_get_as_int
--------------

.. _c_avs_get_as_long:

avs_get_as_long
---------------

.. _c_avs_get_as_string:

avs_get_as_string
-----------------

.. _c_avs_get_as_float:

avs_get_as_float
----------------

.. _c_avs_get_as_error:

avs_get_as_error
----------------

.. _c_avs_get_as_array:

avs_get_as_array
----------------

.. _c_avs_get_array_elt:

avs_get_array_elt
-----------------

.. _c_avs_get_array_size:

avs_get_array_size
------------------

.. _c_avs_val_defined:

avs_val_defined
---------------

.. _c_avs_val_is_clip:

avs_val_is_clip
---------------

.. _c_avs_val_is_bool:

avs_val_is_bool
---------------

.. _c_avs_val_is_int:

avs_val_is_int
--------------

.. _c_avs_val_is_long_strict:

avs_val_is_long_strict
----------------------

.. _c_avs_val_is_float:

avs_val_is_float
----------------

.. _c_avs_val_is_floatf_strict:

avs_val_is_floatf_strict
------------------------

.. _c_avs_val_is_string:

avs_val_is_string
-----------------

.. _c_avs_val_is_array:

avs_val_is_array
----------------

.. _c_avs_val_is_error:

avs_val_is_error
----------------

.. _c_avs_prop_get_int_saturated:

avs_prop_get_int_saturated
--------------------------

.. _c_avs_prop_get_float_saturated:

avs_prop_get_float_saturated
----------------------------

.. _c_avs_prop_get_data_type_hint:

avs_prop_get_data_type_hint
---------------------------

.. _c_avs_prop_set_data_h:

avs_prop_set_data_h
-------------------

.. _c_avs_add_function_r:

avs_add_function_r
------------------

See :ref:`c_avs_add_function`


.. _c_avs_scriptenvironment:

AVS_ScriptEnvironment
~~~~~~~~~~~~~~~~~~~~~

In C++ terminology: ``IScriptEnvironment``.

See at :ref:`cplusplus_createscriptenvironment`.


AVS_Value
~~~~~~~~~

In C++ terminology: ``AVSValue``.

See at :ref:`cplusplus_avsvalue`.

AVS_VideoInfo
~~~~~~~~~~~~~

In C++ terminology: ``VideoInfo``.
::

    // AVS_VideoInfo is laid out identically to VideoInfo
    typedef struct AVS_VideoInfo {
      int width, height;    // width=0 means no video
      unsigned fps_numerator, fps_denominator;
      int num_frames;

      int pixel_type;

      int audio_samples_per_second;   // 0 means no audio
      int sample_type;
      int64_t num_audio_samples;
      int nchannels;

      // Image type properties
      // BFF, TFF, FIELDBASED. Also used for storing Channel Mask
      // Manipulate it through the channelmask interface calls 
      int image_type;
    } AVS_VideoInfo;


See at :doc:`VideoInfo <VideoInfo>`.


AVS_VideoFrame
~~~~~~~~~~~~~~

In C++ terminology: ``PVideoFrame``.

Internal AVS structure which holds the frame buffer and the plane pointers, and the frame property data pointer.

See at :ref:`cplusplus_videoframe`.


AVS_Clip
~~~~~~~~~~~~~~

In C++ terminology: ``PClip``.

See at :ref:`cplusplus_pclip`.


.. _c_callbacks:

Callbacks
~~~~~~~~~

Functions, filters
------------------

::

    typedef AVS_Value (AVSC_CC * AVS_ApplyFunc) (AVS_ScriptEnvironment *, AVS_Value args, void * user_data)
    int avs_add_function(AVS_ScriptEnvironment *, const char * name, const char * params, AVS_ApplyFunc apply, void * user_data) 

    typedef void (AVSC_CC * AVS_ApplyFuncR) (AVS_ScriptEnvironment *, AVS_Value *retval, AVS_Value args, void * user_data)
    int avs_add_function_r(AVS_ScriptEnvironment *, const char * name, const char * params, AVS_ApplyFuncR apply, void * user_data) 

See also at :ref:`c_avs_add_function`.

In order to create a new filter ``avs_add_function`` or ``avs_add_function_r`` must be used to register 
a call back function of type ``AVS_ApplyFunc`` or ``AVS_ApplyFuncR``
::

    typedef struct AVS_FilterInfo AVS_FilterInfo;
    struct AVS_FilterInfo
    {
      // these members should not be modified outside of the AVS_ApplyFunc or AVS_ApplyFuncR callback
      AVS_Clip * child;
      AVS_VideoInfo vi;
      AVS_ScriptEnvironment * env;
      AVS_VideoFrame * (AVSC_CC * get_frame)(AVS_FilterInfo *, int n);
      int (AVSC_CC * get_parity)(AVS_FilterInfo *, int n);
      int (AVSC_CC * get_audio)(AVS_FilterInfo *, void * buf,
                                      int64_t start, int64_t count);
      int (AVSC_CC * set_cache_hints)(AVS_FilterInfo *, int cachehints,
                                            int frame_range);
      void (AVSC_CC * free_filter)(AVS_FilterInfo *);

      // Should be set when ever there is an error to report.
      // It is cleared before any of the above methods are called
      const char * error;
      // this is to store whatever and may be modified at will
      void * user_data;
    };

This is the structure that contains the essence of a filter. The ``AVS_ApplyFunc`` or ``AVS_ApplyFuncR`` 
callback must manipulate it appropriately.

::

    AVS_Clip * avs_new_c_filter(AVS_ScriptEnvironment * e, AVS_FilterInfo * * fi,
                                AVS_Value child, int store_child)

``avs_new_c_filter`` creates a new filter. It should be called inside the ``AVS_ApplyFunc`` or ``AVS_ApplyFuncR`` callback. 
``fi`` is set to point to the ``AVS_FilterInfo`` so that you can modify it once it is initialized. ``store_child`` should 
generally be set to ``true``. If it is not set, then ALL methods (the function pointers) must be defined. 
If it is set, then you do not need to worry about freeing the child clip.

In your ``FilterInfo`` the ``free_filter`` callback is a function which is called when the clip (AVS_Clip) is destroyed, that is its reference count
reaches zero. In C++ terminology, this is the destructor of your filter.


.. _c_at_exit:

at_exit
-------

::

    typedef void (AVSC_CC *AVS_ShutdownFunc)(void* user_data, AVS_ScriptEnvironment * env)
    void avs_at_exit(AVS_ScriptEnvironment *, AVS_ShutdownFunc function, void * user_data)

This is a callback, which is called when the AVS_ScriptEnvironment is destroyed by Avisynth core.

Client can be set with ``avs_at_exit`` : :ref:`c_avs_at_exit`


plugin initializers
-------------------
::

    const char * AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment* env)
    const char * AVSC_CC avisynth_c_plugin_init2(AVS_ScriptEnvironment* env)

``avisynth_c_plugin_init`` and/or ``avisynth_c_plugin_init2`` is the entry point for the plugin and must be defined. 
The latter is supported from V11 interface, Avisynth+ 3.7.4 (3.7.3.r4198 test).

When Avisynth 'pings' a DLL, it will search for an init function.

Avisynth version 3.7.4 tries ``avisynth_c_plugin_init2`` first. If found Avisynth+ will know that the plugin is v11 
(64-bit data) capable. When a plugin only has ``avisynth_c_plugin_init``, then 64-bit int and double parameter values 
will be truncated to int and float (32-bit data types) when Avisynth calls the plugin/function. This is a compatibility
measure because internally AviSynth can have 64 bit long and double data types.

When you want to be sure that your DLL will surely be found, don't forget to update the .def file as follows:

::

    LIBRARY AvsInpaint
    EXPORTS
      avisynth_c_plugin_init@4 = _avisynth_c_plugin_init@4
      avisynth_c_plugin_init2@4 = _avisynth_c_plugin_init2@4

**Example**

::

    AVSC_EXPORT const char * AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment * Env)
    {
      avs_add_function(Env, "InpaintLogo", "c[Mask]c[Radius]f[Sharpness]f[PreBlur]f[PostBlur]f
                      [ChromaWeight]f[PreBlurSize]f[PostBlurSize]f[ChromaTensor]b[PixelAspect]f
                      [Steps]i", Inpaint_Create, 0);
      avs_add_function(Env, "DeblendLogo", "cc[Alpha]c", Deblend_Create, 0);
      avs_add_function(Env, "AnalyzeLogo", "c[Mask]c[ComputeAlpha]b[DeviationWeight]f
                      [SubsamplingWeight]f", Analyze_Create, 0);
      avs_add_function(Env, "DistanceFunction", "c[Scale]f[PixelAspect]f", DistanceFunction_Create, 0);
      return "Logo Inpainting Filter";
    }

    // can co-exist with avisynth_c_plugin_init
    AVSC_EXPORT const char* AVSC_CC avisynth_c_plugin_init2(AVS_ScriptEnvironment* Env)
    {
      return avisynth_c_plugin_init(Env);
    }

Historical content
~~~~~~~~~~~~~~~~~~

* to be checked and updated *

source: http://forum.doom9.org/showthread.php?p=1464911#post1464911

compiling plugins: http://forum.doom9.org/showthread.php?p=1092380#post1092380

example: http://forum.doom9.org/showthread.php?p=1001260#post1001260

| For now the api is described here:
| http://www.kevina.org/avisynth_c/api.html.

| An example is given here:
| http://www.kevina.org/avisynth_c/example.html.

In v2.60 (AVISYNTH_INTERFACE_VERSION = 6) the following functions are
added to the C interface:
::

    avs_is_yv24
    avs_is_yv16
    avs_is_yv12
    avs_is_yv411
    avs_is_y8
    avs_is_color_space
    avs_get_plane_width_subsampling
    avs_get_plane_height_subsampling
    avs_bits_per_pixel
    avs_bytes_from_pixels
    avs_row_size
    avs_bmp_size
    avs_get_row_size_p
    avs_get_height_p

____

Back to :doc:`FilterSDK`

$Date: 2026/08/25 22:52:00 $
