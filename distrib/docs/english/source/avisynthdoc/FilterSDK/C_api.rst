
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

.. _c_api_dynamic_loading:

Dynamic loading
----------------

- Dynamic loading ``AviSynth.dll``/``libavisynth.so``/``.dylib``. Recommended.

  Advantage: independence of the actual AviSynth version.

  With static linking (import library) every API function the plugin or host uses must exist
  in the loaded AviSynth, or the plugin (or the program) does not load at all, even if that
  function is never called. E.g. a plugin calling ``avs_acquire_global_lock`` (V12) with a
  fallback for older versions does not load on an AviSynth older than V12; dynamically
  loaded, the missing function is just NULL.

  AviSynth versions with different API levels can be supported, because we can detect 
  on loading the necessary API methods whether they exist or not. Earlier AviSynth 
  versions may contain fewer API functions. By detecting the loaded AviSynth/interface 
  version, it's the caller's responsibility to call only those API functions which 
  have valid function pointers and are documented to work. For example, you should 
  only use frame property-related functions when the interface version is at least 9.

  - Use ``#define AVSC_NO_DECLSPEC`` for function pointer definitions only: each API
    function ``name`` gets a function pointer type ``name_func`` instead of a declaration.
    The ``AVSC_INLINE`` helpers which call into the API (e.g. ``avs_new_video_frame``) are not
    defined then: call the API function itself.
  - Load the library dynamically and get the necessary API functions as needed.

Ready loader: avisynth_c_loader.h
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the loader for use from **C**. From **C++** use the other variant, ``avisynth_c_loader.hpp`` (the
class ``avsc::Library``, described below).

``avisynth_c_loader.h`` is a loader for Windows, Linux, macOS, BSD and Haiku, made of header files only:
there is no source file to build or library to link. Besides ``avisynth_c.h`` it uses the common header
``avisynth_c_loader_common.h`` (below), which has to be copied with it.
It replaces the Windows-only ``avs_load_library`` which was part of ``avisynth_c.h`` before 3.7.6.

It is a sample: copy it into your project, as it is or trimmed. ``avisynth_c.h`` and its ``avs/``
folder must be on the include path (from V14, 3.7.6 on). Where to find it:

- source tree: ``distrib/FilterSDK/c_loader``
- Windows packages and the SDK of the Windows installer: ``FilterSDK/c_loader``
- CMake install: ``share/doc/avisynth/examples/c_loader``

In 3.7.6 ``avisynth_c_loader.h`` was installed with the AviSynth headers; since 3.7.7 it is not.

Since 3.7.7 there are two variants sharing the loading, binding and compatibility handling;
their common part (function list, fallbacks, version query, platform calls) is ``avisynth_c_loader_common.h``,
copy it as well. ``example_host.c`` and ``example_host.cpp`` in the same folder are minimal hosts,
``example_plugin.cpp`` is a minimal C plugin written in C++: it shows the loader in a plugin, the two
entry points (both pass their value-support flag to a shared initializer) and the
64 bit values. The parameter letters are the same for both entry points (``i``, ``f``; there are no ``l``,
``d`` letters): through ``avisynth_c_plugin_init2`` an ``i`` parameter may hold a long and an ``f`` parameter
a double, and ``avs_invoke`` results keep them too; through ``avisynth_c_plugin_init`` AviSynth converts them
to 32 bit int and float.

- ``avisynth_c_loader.hpp`` (C++17): the class ``avsc::Library``, for C++ hosts and plugins using the
  C API, the usual case (the C API keeps them independent of the compiler AviSynth was built with).
  Construct it empty, then load: ``avsc::Library lib; if (!lib.load()) { ... }`` (``load()``: the default
  name from the library search path, ``load(path)``: a given file). No constructor loads, so a failed load
  cannot go unnoticed. Optionally ``lib.error()`` tells why a load failed (see ``avs_load_library_ex``).
  It releases any owned library reference in its destructor, is movable, and throws no exceptions.
  The C API functions and the version fields are its members, e.g.
  ``lib.avs_invoke(...)``, ``lib.has_at_least_v14``, ``lib.at_least(14, 1)``; the functions below are
  ``load``, ``attach``, ``bind_plugin`` (see ``avs_bind_plugin_library`` below),
  ``create_script_environment``, ``detect_plugin_env``, ``env_failed``.
  Plugins use ``static avsc::Library lib;`` and ``lib.bind_plugin()``: it releases the temporary
  library reference and retains only function pointers and version/capability fields. The C++ runtime
  destroys the static object at plugin unload; no explicit cleanup hook or ``FreeLibrary``/``dlclose``
  call is needed.
  ``loaded()`` and ``operator bool`` report a usable binding, including a non-owning plugin binding;
  ``handle()`` returns only an owned handle, and is NULL for a plugin binding.
  The inline helpers of ``avisynth_c.h`` which are not defined with ``AVSC_NO_DECLSPEC`` are members, with the
  same names and arguments: ``lib.avs_new_video_frame(env, vi)``, ``avs_new_frame``, ``avs_new_value_clip``,
  ``avs_is_same_colorspace``, the default plane ``avs_get_pitch``, ``avs_get_row_size``, ``avs_get_height``,
  ``avs_get_read_ptr``, ``avs_get_write_ptr``, and ``avs_release_frame``, ``avs_copy_frame``. Also
  ``lib.get_cpu_flags_ex(env)``: the 64 bit CPU flags on every AviSynth (before V12 the 32 bit flags, zero
  extended), so no check of the ``avs_get_cpu_flags_ex`` pointer is needed. The C loader has no such helpers:
  call the API function through ``lib->``.
- ``avisynth_c_loader.h`` (C11, also compiles as C++): the struct ``AVS_Library`` and the functions
  below. In C they are ``static inline``, in C++ ``inline`` (one copy for the program).

- ``AVS_Library* avs_load_library(const char* library_path)``

  Loads AviSynth and fills the ``AVS_Library`` struct with every C API function.
  Functions missing from the loaded library (an older AviSynth) get a fallback where possible
  (since 3.7.7, see ``avs_library_set_fallbacks`` below), the others are NULL. A non-NULL entry
  therefore does not tell the version: use the version fields (``has_at_least_xxx``).
  Returns NULL only if the library cannot be loaded, or if it is not AviSynth with a C interface
  (``avs_create_script_environment`` is missing).

  ``library_path``: UTF-8 file name or path. NULL means ``avisynth`` (Windows),
  ``libavisynth.dylib`` (macOS), ``libavisynth.so`` (others).

- ``AVS_Library* avs_load_library_ex(const char* library_path, char* error, size_t error_size)``

  The same, and optionally tells why it failed: ``error`` (may be NULL) receives a NUL-terminated
  diagnostic in at most ``error_size`` bytes, including the terminator; an empty string on success.
  ``AVSC_LOADER_ERROR_SIZE`` is a suggested buffer size, not a limit on the path or full message length.
  Windows paths and system messages are UTF-8; truncation preserves complete characters.
  On Windows common failures are explained without assuming a unique cause for ambiguous OS errors:

  - path is a directory; file or dependency not found (DLL search path for a bare name)
  - a DLL it needs is missing, e.g. the Visual C++ runtime (the file itself is there)
  - invalid library image or incompatible architecture (32/64 bit or CPU architecture mismatch)
  - not AviSynth with a C interface, or an invalid UTF-8 file name

  e.g. ``avisynth: invalid library image or incompatible architecture for this 64 bit process (error 193)``.
  Other systems give the ``dlerror()`` text in its original encoding.

- ``void avs_free_library(AVS_Library* library)``

  Releases the owned library reference and frees the struct returned by ``avs_load_library`` or
  ``avs_attach_library``. Do not call it on caller-owned plugin storage.

- ``AVS_ScriptEnvironment* avs_library_create_script_environment(AVS_Library* library)``

  For hosts. Creates the environment with the highest version known by both the loaded DLL and
  the ``avisynth_c.h`` the loader was compiled with. Two versions are involved: the header's
  ``AVISYNTH_INTERFACE_VERSION`` (compile time, what the caller understands, the first request;
  rebuilding with a newer header raises it) and the loaded DLL's interface version (run time,
  requests above it are refused). The request list follows historical API milestones, not distinct
  environment implementations. The first accepted request is kept
  (a refused request is cheap, no core environment is built):
  ``AVISYNTH_INTERFACE_VERSION`` (what the caller understands), 11 (V11..V13 AviSynth: 64 bit
  ``AVS_Value`` types), 9 (V9/V10: working C frame property access), 8 (V8: frame property API entries
  exist but are not reliably usable), 6 (classic, older AviSynth),
  3 (AviSynth 2.5).
  From 11 up to the DLL's interface version the environment is the same, as it is for any version
  below 11; bugfix versions cannot be requested at all. The exact version and bugfix version are
  then queried through the environment, filling the version fields of the struct (below).
  Returns NULL on failure. Delete the result with ``lib->avs_delete_script_environment``.

- ``AVS_Library* avs_attach_library(char* error, size_t error_size)``

  The same as ``avs_load_library_ex``, but acquires a reference to an already-loaded AviSynth
  instead of loading one by name (C++: ``lib.attach()``).
  Plugins should normally use the non-owning ``avs_bind_plugin_library`` below.
  On Windows ``avs_load_library(NULL)`` gives the same instance, elsewhere it opens
  ``libavisynth.so`` from the library search path, which is a problem in two cases: that file name often exists
  only in the -dev packages (the real name is ``libavisynth.so.<interface version>``), and if the host
  loaded AviSynth from another path (portable install, absolute path) it may be a different copy, so the
  plugin works with another AviSynth than the one which loaded it.

  - Windows: the module named ``avisynth`` (a renamed ``AviSynth.dll`` is not found).
  - Linux, BSD: the loaded object named ``libavisynth.so[.N]``, opened again by its full path (same instance).
  - macOS: the loaded image named ``libavisynth[.N].dylib``, the same way.
  - Haiku: not supported.

  Returns NULL if there is none, with the reason in ``error``. Load by name as a fallback if you want:
  ``lib = avs_attach_library(NULL, 0); if (!lib) lib = avs_load_library(path);`` Free the result with
  ``avs_free_library`` (other owners determine whether AviSynth stays loaded).

- ``int avs_bind_plugin_library(AVS_Library* library, char* error, size_t error_size)``

  For plugins (C++: ``lib.bind_plugin()``). Resolves the already-loaded core using the same discovery
  as ``avs_attach_library``, but releases the temporary library reference before returning.
  Returns nonzero on success, zero on failure; optional ``error`` diagnostics are UTF-8.
  The C function fills caller-owned storage, normally ``static AVS_Library lib;``.
  Storage must be zero-initialized or previously bound by this function, never an owning loader
  returned by ``avs_load_library``/``avs_attach_library``. On failure it is cleared;
  on success its ``handle`` remains NULL while the API pointers and fallbacks are usable.
  No allocation is retained and no cleanup is needed; do not call ``avs_free_library`` on this storage.

  Call during plugin initialization, not DLL attach/detach. Keep the storage for all plugin callbacks.
  The host must keep the core loaded until all environments and plugin activity finish.
  The binding does not resolve ambiguity between multiple loaded cores or remove the macOS image
  enumeration restriction: no other thread may load or unload libraries during enumeration.

  Loader operations do not lock internally. The plugin patterns below and in ``example_plugin.cpp``
  serialize binding and version detection with a file-scope atomic flag, then keep the shared table
  and capability fields read-only. Register functions separately for every environment.
  The flag requires no cleanup, TLS or function-local-static initialization guard.
  It does not depend on the optional C11 ``<threads.h>``; C uses ``<stdatomic.h>``, C++ uses ``<atomic>``.
  A failed bind can be retried. The cached binding is not refreshed if the core unloads/reloads while
  the plugin remains mapped.

  MSVC C atomics may require ``/experimental:c11atomics`` in addition to ``/std:c11``.
  The ``v141_xp`` C compiler does not support C11; use a different C compiler or provide a custom guard.
  Its C++ compiler supports the atomic pattern, which does not rely on ``/Zc:threadSafeInit`` or TLS.
  The plugin and its runtime must still be built for the target operating system.

- ``int avs_library_detect_plugin_env(AVS_Library* library, AVS_ScriptEnvironment* env, int called_from_init2)``

  For plugins, called from the plugin init with the ``env`` AviSynth passed. Fills the same version
  fields, and ``has_64bit_values``. ``called_from_init2``: nonzero if AviSynth called
  ``avisynth_c_plugin_init2``; the loader cannot see it, the plugin has to tell. See
  :ref:`Plugins and 64 bit values<c_api_loader_plugins>`.

- ``int avs_library_detect_version(AVS_Library* library, AVS_ScriptEnvironment* env)``

  Fills only the version fields (``interface_version``, ``interface_bugfix``, ``has_at_least_xxx``)
  through ``env``.

- ``int avs_library_env_failed(const AVS_Library* library, AVS_ScriptEnvironment* env)``

  Nonzero if ``avs_create_script_environment`` failed. A refused interface version can return a
  non-NULL C wrapper holding a NULL core pointer. The loader first checks for a NULL wrapper or
  an error from ``avs_get_error``.

  V14 (3.7.6) and later report the refusal through ``avs_get_error``. The loader identifies this
  behavior by directly checking for the ``avs_create_map`` export, independently of the function
  table and fallbacks. For these cores it does not inspect the opaque wrapper's layout; without
  ``avs_get_error`` it treats the wrapper as failed. This assumes future cores retain the marker
  export and V14 error-reporting behavior.

  Older cores may leave the error unset. Only for these does the loader inspect the historical
  wrapper layout and check its first member, the core pointer. Do not call core-dependent functions
  through a failed wrapper.

- ``int avs_library_interface_version(const AVS_Library* library, AVS_ScriptEnvironment* env, int* bugfix)``

  Low level: returns the interface version of the loaded AviSynth (e.g. 6, 8, 11, 14), 0 if unknown;
  ``*bugfix`` (may be NULL) receives the bugfix version (interface 11 with bugfix 1 is "V11.1", 3.7.5).
  ``AVS_AEP_INTERFACE_VERSION`` is asked only from V9, earlier versions did not know it.

- ``int avs_library_set_fallbacks(AVS_Library* library)``

  Since 3.7.7, called during loading, attachment or plugin binding. Fills the missing (NULL) entries which can be
  served without the loaded AviSynth, the same way as the C++ ABI (``avisynth.h``) does. Functions the
  loaded AviSynth has are never replaced. Returns the number of entries filled; a second call
  fills nothing.

  +----------------------------------------------------------+-------------------------------------------------+
  | Missing function                                         | Fallback                                        |
  +==========================================================+=================================================+
  | ``avs_is_444``, ``avs_is_422``, ``avs_is_420``,          | ``avs_is_yv24``, ``avs_is_yv16``,               |
  | ``avs_is_y``, ``avs_is_411``                             | ``avs_is_yv12``, ``avs_is_y8``,                 |
  |                                                          | ``avs_is_yv411``: on classic AviSynth 2.6       |
  |                                                          | the 8 bit formats are the only ones             |
  +----------------------------------------------------------+-------------------------------------------------+
  | ``avs_component_size``, ``avs_bits_per_component``,      | 1, 8, from the pixel type                       |
  | ``avs_num_components``                                   |                                                 |
  +----------------------------------------------------------+-------------------------------------------------+
  | ``avs_is_yuva``, ``avs_is_planar_rgb(a)``,               | false: the format does not exist there          |
  | ``avs_is_rgb48/64``, ``avs_is_440``, ``avs_is_410``,     |                                                 |
  | ``avs_is_ya``; deprecated ``avs_is_yuv4xxp16``,          |                                                 |
  | ``avs_is_yuv4xxps``, ``avs_is_y16``, ``avs_is_y32``      |                                                 |
  +----------------------------------------------------------+-------------------------------------------------+
  | ``avs_is_channel_mask_known``, ``avs_get_channel_mask``, | false, 0, does nothing                          |
  | ``avs_set_channel_mask`` (V10.1)                         |                                                 |
  +----------------------------------------------------------+-------------------------------------------------+
  | ``AVS_Value`` functions of V11: ``avs_val_defined``,     | the same result: there are no 64 bit types      |
  | ``avs_val_is_xxx``, ``avs_get_as_xxx``,                  | before V11. ``avs_get_as_clip`` is              |
  | ``avs_get_array_elt``, ``avs_get_array_size``,           | ``avs_take_clip``. ``avs_set_to_double`` and    |
  | ``avs_set_to_xxx`` except ``avs_set_to_array``           | ``avs_set_to_long`` give float and 32 bit int,  |
  |                                                          | as AviSynth V11+ converts them for a pre-V11    |
  |                                                          | plugin                                          |
  +----------------------------------------------------------+-------------------------------------------------+

  Not filled, these need the loaded AviSynth; check the version fields or the pointer:
  ``avs_set_to_array`` (the deep copy is owned by the core), frame properties,
  ``avs_new_video_frame_p(_a)``, ``avs_video_frame_get/amend_pixel_type``, ``avs_pool_xxx``,
  ``avs_get_var_xxx``, the global lock, ``avs_invoke_try``, ``avs_invoke2/3``, function values,
  ``avs_create_map``. For ``avs_get_cpu_flags_ex`` call ``avs_get_cpu_flags`` when it is NULL
  (the flags above 32 bits are new in V12 anyway).
  Functions which exist but work reliably only from a later version (see the table below) are not
  affected by the fallbacks.

Version fields of ``AVS_Library`` (C) and ``avsc::Library`` (C++), filled by environment creation or
plugin-environment detection. Both variants expose the same released milestones plus the current WIP:

+-----------------------+----------------------------------------------------------------------------+
| Field                 | Meaning                                                                    |
+=======================+============================================================================+
| ``interface_version`` | e.g. 6, 8, 11, 14                                                          |
+-----------------------+----------------------------------------------------------------------------+
| ``interface_bugfix``  | e.g. 1 for V11.1 (3.7.5), V14.1 (3.7.7)                                    |
+-----------------------+----------------------------------------------------------------------------+
| ``has_at_least_v8``   | frame property API entries, ``avs_new_video_frame_p`` (property source);   |
|                       | C API frame property access crashes in V8, use it from V9                  |
+-----------------------+----------------------------------------------------------------------------+
| ``has_at_least_v9``   | C API frame property access works, ``avs_is/make_property_writable``,      |
|                       | ``AVS_AEP_INTERFACE_VERSION/BUGFIX`` queries                               |
+-----------------------+----------------------------------------------------------------------------+
| ``has_at_least_v10``  | ``AVS_VideoFrame`` pixel_type, ``avs_new_video_frame_p(_a)`` with a        |
|                       | non-NULL ``prop_src``, ``avs_prop_get_data``; C filters: an error of the   |
|                       | child clip is passed on                                                    |
+-----------------------+----------------------------------------------------------------------------+
| ``has_at_least_v11``  | the core knows 64 bit ``AVS_Value`` types (double, long); whether this     |
|                       | caller gets them: ``has_64bit_values``                                     |
+-----------------------+----------------------------------------------------------------------------+
| ``has_at_least_v11_1``| 3.7.5: non-x86 64-bit value storage correction                             |
+-----------------------+----------------------------------------------------------------------------+
| ``has_at_least_v14``  | function values, ``avs_invoke2/3(_try)``, ``avs_create_map``;              |
|                       | ``avs_prop_get_xxx`` fixes, ``AVS_MT_NICE_FILTER`` error reporting;        |
|                       | includes global locks, ``avs_get_cpu_flags_ex``, L2 cache size query,      |
|                       | and YUV(A) 4:4:0, 4:1:1, 4:1:0, YA pixel formats                           |
+-----------------------+----------------------------------------------------------------------------+
| ``has_at_least_v14_1``| ``_Range`` is the stored frame property, ``_ColorRange`` its alias (0 and  |
|                       | 1 swapped). Before, AviSynth knew only ``_ColorRange``                     |
+-----------------------+----------------------------------------------------------------------------+
| ``has_frame_props``   | version >= 9 and all V8/V9 frame property functions present: one check is  |
|                       | enough before using them. False on Win32 3.6.x - 3.7.1, see below.         |
|                       | Later fixes still depend on the version: ``avs_prop_get_data(_size)``      |
|                       | needs V10, ``avs_prop_get_clip`` needs V14. Before V14, pass a non-NULL    |
|                       | error pointer to ``avs_prop_get_xxx``.                                     |
+-----------------------+----------------------------------------------------------------------------+
| ``has_64bit_values``  | this caller gets double and long (int64) values, not converted to float    |
|                       | and 32 bit int. Hosts: environment requested with version >= 11.           |
|                       | Plugins: AviSynth called ``avisynth_c_plugin_init2``.                      |
+-----------------------+----------------------------------------------------------------------------+

V9.1/V9.2 and V12/V13 were development versions, not separate releases, and have no separate flags.
Use ``has_at_least_v10`` for the V9.1/V9.2 fixes and ``has_at_least_v14`` for the V12/V13 additions
on released cores. Exact comparisons use ``interface_version`` and ``interface_bugfix``.

On V11.0 non-x86 64-bit cores, numeric ``AVS_Value`` storage differs from the current header's
inline helpers. Use the core's ``avs_get_as_xxx``/``avs_set_to_xxx`` functions, which handle its
layout. V11.1 changed that storage to match current inline helpers; ``has_64bit_values`` reports
the caller's value support, not this storage revision.

Example (client)::

    #include "avisynth_c_loader.h" // instead of avisynth_c.h, it defines AVSC_NO_DECLSPEC

    AVS_Library* lib = avs_load_library(NULL);
    if (!lib)
      return -1; // AviSynth not found
    // the highest version both sides know: requests the header's version, then 11, 9, 8, 6, 3
    AVS_ScriptEnvironment* env = avs_library_create_script_environment(lib);
    if (!env) {
      avs_free_library(lib);
      return -1;
    }
    if (lib->has_at_least_v9) {
      // frame properties
    }
    if (lib->has_64bit_values) {
      // double and long (int64) values are kept
    }
    if (lib->has_at_least_v14) {
      // function values, avs_invoke2/3
    }
    lib->avs_delete_script_environment(env);
    avs_free_library(lib);

Notes:

- Why the highest version: the version requested in ``avs_create_script_environment`` tells AviSynth
  what the caller understands.

  - A version newer than the loaded DLL's interface version is refused by the core (its ``CreateScriptEnvironment`` returns NULL),
    but ``avs_create_script_environment`` still returns a non-NULL wrapper, holding the NULL core
    pointer; core-dependent calls through it can crash. From 3.7.6 ``avs_get_error`` tells it, e.g.
    "avs_create_script_environment: requested interface version is newer than this AviSynth";
    older versions give no error message. Check it with ``avs_library_env_failed``.
  - Requesting a version below 11, AviSynth converts double values to float and long (int64) values
    to 32 bit int for the caller, even if it supports them itself:

    +----------------------+---------------------------------+---------------------------------------------+
    | ``Eval`` of          | Environment requested with 6    | ``avs_library_create_script_environment``   |
    +======================+=================================+=============================================+
    | ``1.0/3``            | ``'f'`` 0.3333333432674408      | ``'d'`` 0.33333333333333331                 |
    +----------------------+---------------------------------+---------------------------------------------+
    | ``4294967297``       | ``'i'`` 1                       | ``'l'`` 4294967297                          |
    +----------------------+---------------------------------+---------------------------------------------+

- A function existing in the library does not mean it works there: some were fixed in a later
  interface version than the one they appeared in. ``AVSC_LOADER_FUNCTIONS`` in
  ``avisynth_c_loader_common.h`` documents introduction versions and later fixes:

  +-----------------------------------------------------------+----------+--------------------------------------------+
  | Function                                                  | Appeared | Reliably usable from                       |
  +===========================================================+==========+============================================+
  | ``avs_copy_frame_props``, ``avs_prop_set_frame``          | V8       | V9 (3.7.1), crashed in V8                  |
  | ``avs_get_frame_props_ro``, ``avs_get_frame_props_rw``    |          |                                            |
  +-----------------------------------------------------------+----------+--------------------------------------------+
  | ``avs_new_video_frame_p``, ``avs_new_video_frame_p_a``    | V8       | V10 (3.7.3) with a non-NULL ``prop_src``   |
  +-----------------------------------------------------------+----------+--------------------------------------------+
  | ``avs_prop_get_data``                                     | V8       | V10 (3.7.3)                                |
  +-----------------------------------------------------------+----------+--------------------------------------------+
  | ``avs_prop_get_xxx`` with a NULL ``error`` pointer        | V8       | V14, before it a failed read terminated    |
  |                                                           |          | the process: pass a non-NULL pointer       |
  +-----------------------------------------------------------+----------+--------------------------------------------+
  | ``avs_prop_get_clip``                                     | V8       | V14, before it returned an invalid clip    |
  +-----------------------------------------------------------+----------+--------------------------------------------+
  | ``avs_prop_get_clip``, ``avs_prop_get_frame``: ``*error`` | V8       | V14, it was always 0 in V11 (3.7.4, 3.7.5) |
  +-----------------------------------------------------------+----------+--------------------------------------------+
  | ``avs_new_c_filter`` with ``AVS_MT_NICE_FILTER``          | V3       | V14: error reporting of concurrent calls   |
  +-----------------------------------------------------------+----------+--------------------------------------------+
  | ``avs_add_function(_r)``: one ``user_data`` for several   | V3, V11  | V14; on V11 a call could run another       |
  | functions, e.g. NULL                                      |          | function of the plugin, see below          |
  +-----------------------------------------------------------+----------+--------------------------------------------+
  | ``avs_set_to_array`` with NULL ``src``, size > 0          | V11      | V14                                        |
  +-----------------------------------------------------------+----------+--------------------------------------------+

- AviSynth+ 3.7.4 and 3.7.5 (V11, V11.1): a C plugin's functions could get mixed up, a script call ran
  the callback of another function of the same plugin. It happened when several functions were
  registered with the same ``user_data``: NULL, or on x64 any shared pointer. The core stored each
  registration with ``SaveString``, whose cache compared entries with ``strncmp``, which stops at the
  first zero byte: the comparison ended inside ``user_data`` and never reached the callback. Only
  Windows builds made with MSVC or clang-cl are affected (the official ones): the C++ standard library
  of the MinGW, Linux and macOS builds compares the full hash of two entries before comparing them, and
  the hash covers all bytes. Fixed in 3.7.6 (commit 8a4edc62).
  The loader works around it on V11/V11.1, on any build (it cannot tell them apart): its
  ``avs_add_function`` and ``avs_add_function_r`` register a small function of the loader in between, with its own
  ``user_data`` for each plugin function; when the script calls it, it calls the plugin's function with
  the plugin's own ``user_data``. Without the loader: give each function a ``user_data`` pointer of its
  own whose lowest byte is nonzero and differs from the others.
- Win32 (32 bit) AviSynth+ 3.6.x - 3.7.1 exported the V8 C API functions (frame properties,
  ``avs_subframe_planar_a``, ...) only with decorated ``__stdcall`` names, e.g.
  ``_avs_get_frame_props_ro@8``: the 32 bit ``avisynth.def`` did not list them. The loader looks
  up the plain names, so these entries are NULL there. 3.7.1 is already V9, so the interface version
  alone does not tell whether the frame property functions can be called; ``has_frame_props`` checks
  the pointers as well. ``avs_get_env_property`` (V8) is NULL there too, so the loader finds V9 by
  polling ``avs_check_version``, without the bugfix version. Fixed in 3.7.2 (#276). Checked on the
  export tables of the official release DLLs and with the loader:

  +-----------------+-----------+------------------------------------------------------------+
  | Release         | Interface | Frame property exports                                     |
  +=================+===========+============================================================+
  | 3.6.x, 3.7.0    | V8        | x86, x86 XP: decorated only (NULL in the loader); not      |
  |                 |           | usable anyway, C frame property access crashes in V8       |
  +-----------------+-----------+------------------------------------------------------------+
  | 3.7.1, 3.7.1a   | V9        | x86, x86 XP: decorated only (NULL in the loader),          |
  |                 |           | ``has_frame_props`` is false                               |
  +-----------------+-----------+------------------------------------------------------------+
  | 3.7.2 and later | V9 and up | x86, x86 XP: plain names                                   |
  +-----------------+-----------+------------------------------------------------------------+
  | all versions    |           | x64, x64 XP, ARM64, Linux, macOS: plain names (no          |
  |                 |           | ``__stdcall`` decoration on these platforms)               |
  +-----------------+-----------+------------------------------------------------------------+
- Plugins: use ``avs_bind_plugin_library`` (C++ ``lib.bind_plugin()``) to resolve the already-loaded
  core without retaining a library reference. Default-name loading on POSIX may fail or load another
  copy. Platform restrictions and ambiguity between multiple loaded cores still apply.
- Paths: on Windows the loader sizes and allocates its UTF-16 path buffer dynamically, without a
  fixed ``MAX_PATH`` limit. Windows long-path rules still apply; the caller must supply any required
  ``\\?\`` prefix, or use a long-path-aware host with the required system settings.
- Include the loader only in source files built without special instruction set flags (``-mavx2``,
  ``-msse4.1``, ``/arch:AVX2``, ...): its ``inline`` functions are emitted into every file that uses them, and
  the linker keeps one copy of each, which may be one compiled with such flags and then executed on a CPU
  without that instruction set. Typically the loader is used in the plugin's or host's interface file.

.. _c_api_loader_plugins:

Plugins and 64 bit values
^^^^^^^^^^^^^^^^^^^^^^^^^

A host chooses by the version it requests in ``avs_create_script_environment``. A C plugin does not
create an environment: AviSynth decides by the entry points the plugin exports.

How it works (3.7.4+, V11+): a non-C++ plugin is probed for ``avisynth_c_plugin_init2`` first
(Win32: also ``_avisynth_c_plugin_init2@4``). If exported, the plugin is 64 bit aware.
Otherwise ``avisynth_c_plugin_init`` is called and the plugin is marked pre-V11.

- ``avisynth_c_plugin_init2`` gets the full environment, ``avisynth_c_plugin_init`` the camouflaged
  pre-V11 one (before 3.7.6 the full one as well). The mark is stored with every function the plugin
  registers in its init (``avs_add_function``).
- When such a function is called:

  - 64 bit aware plugin: the full environment, double and long values are passed as they are.
  - pre-V11 plugin: a camouflaged pre-V11 environment. The function arguments are converted (long to
    32 bit int, double to float), array elements included, also when the function is called at
    runtime (e.g. from ScriptClip); ``avs_invoke`` results through this environment as well (3.7.6).

- AviSynth V10 and below (3.7.3 and older) does not know ``avisynth_c_plugin_init2``, calls only
  ``avisynth_c_plugin_init``, and has no 64 bit values at all.

+--------------------------------------+------------------------------------+----------------------------------+
| Plugin exports                       | AviSynth V10 and below             | AviSynth V11 and above           |
+======================================+====================================+==================================+
| ``avisynth_c_plugin_init`` only      | loads, 32 bit values               | loads, values converted to       |
|                                      |                                    | 32 bit int and float             |
+--------------------------------------+------------------------------------+----------------------------------+
| ``avisynth_c_plugin_init2`` only     | does not load ("cannot be used as  | loads, 64 bit values             |
|                                      | a plugin")                         |                                  |
+--------------------------------------+------------------------------------+----------------------------------+
| both, using a shared initializer     | ``init`` is called, 32 bit values  | ``init2`` is called,             |
| (recommended)                        |                                    | 64 bit values                    |
+--------------------------------------+------------------------------------+----------------------------------+

Pass the entry-point flag to a shared initializer and then to ``avs_library_detect_plugin_env``,
which sets ``has_64bit_values``. Initialize shared state under the guard before registering callbacks,
and do not modify it during later environment initialization::

    #include <stdatomic.h>
    #include "avisynth_c_loader.h"

    static AVS_Library lib;
    static atomic_flag init_lock = ATOMIC_FLAG_INIT;

    static const char* init_plugin(AVS_ScriptEnvironment* env, int from_init2)
    {
      while (atomic_flag_test_and_set_explicit(&init_lock, memory_order_acquire)) {}
      if (!lib.avs_create_script_environment) {
        if (!avs_bind_plugin_library(&lib, NULL, 0)) {
          atomic_flag_clear_explicit(&init_lock, memory_order_release);
          return "MyPlugin: AviSynth not loaded";
        }
        avs_library_detect_plugin_env(&lib, env, from_init2);
      }
      atomic_flag_clear_explicit(&init_lock, memory_order_release);

      lib.avs_add_function(env, "MyFilter", "c[x]f", my_filter_create, NULL);
      return "MyPlugin";
    }

    const char* AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment* env)
    {
      return init_plugin(env, 0);
    }

    // AviSynth V11+ calls it first: the plugin's functions get double and long values
    const char* AVSC_CC avisynth_c_plugin_init2(AVS_ScriptEnvironment* env)
    {
      return init_plugin(env, 1);
    }

On Win32 no .def file is needed for the loader: AviSynth+ finds both entry points under any of the names
``name``, ``_name@4`` and ``name@4``. Classic AviSynth 2.6 (``LoadCPlugin``) looks for ``name@4`` and
``name`` only: MinGW g++ exports ``name@4`` (``-Wl,--kill-at``: ``name``), both load there; MSVC exports
``_name@4``, which classic 2.6 does not find unless a .def file adds ``name@4 = _name@4``.

``avs_library_detect_plugin_env`` works through the camouflaged pre-V11 environment that a plugin with only
``avisynth_c_plugin_init`` receives: it forwards ``avs_check_version`` and ``avs_get_env_property`` to the
core, so the version fields show the real version of AviSynth (e.g. 14.1), not 10; only the values passed to
the plugin's functions are converted. ``has_64bit_values`` is what tells the two apart.

Tested with a plugin built both ways, ``LdrTest(x)`` returning the received argument:

+-------------------------+------------------------------------------+------------------------------------------+
| Script                  | Plugin with ``init`` only                | Plugin with ``init2``                    |
+=========================+==========================================+==========================================+
| ``LdrTest(1.0/3)``      | ``'f'`` 0.3333333432674408,              | ``'d'`` 0.33333333333333331,             |
|                         | ``has_64bit_values`` = 0                 | ``has_64bit_values`` = 1                 |
+-------------------------+------------------------------------------+------------------------------------------+
| ``LdrTest(4294967297)`` | ``'i'`` 1                                | ``'l'`` 4294967297                       |
+-------------------------+------------------------------------------+------------------------------------------+

There are no ``l`` and ``d`` letters in the parameter signature of ``avs_add_function``: ``i`` and ``f`` are
the only numeric ones, and from V11 they accept the 64 bit types. A plugin with ``init2`` receives the values as
they are (measured with a probe plugin returning the type letter of its parameter):

+---------------------------+-----------------------------+----------------------------------+
| Script                    | Plugin with ``init`` only   | Plugin with ``init2``            |
+===========================+=============================+==================================+
| ``i`` parameter, ``5``    | ``'i'``                     | ``'i'`` (not widened)            |
+---------------------------+-----------------------------+----------------------------------+
| ``i`` parameter,          | ``'i'`` (cut to 32 bit)     | ``'l'``                          |
| ``3000000000``            |                             |                                  |
+---------------------------+-----------------------------+----------------------------------+
| ``f`` parameter, ``1.5``  | ``'f'``                     | ``'d'``                          |
+---------------------------+-----------------------------+----------------------------------+
| ``f`` parameter, ``2``    | ``'i'``                     | ``'i'`` (an integer stays an     |
|                           |                             | integer, not converted)          |
+---------------------------+-----------------------------+----------------------------------+
| ``f`` parameter,          | ``'i'`` (cut to 32 bit)     | ``'l'``                          |
| ``3000000000``            |                             |                                  |
+---------------------------+-----------------------------+----------------------------------+

So 64 bit mode does not mean every value is 64 bit: the type depends on the value. Do not test the type
letter; read the parameter with ``avs_get_as_long`` or ``avs_get_as_float`` (V11, the loader gives fallbacks
for older AviSynth), which convert any numeric type, and use the 64 bit result type only if
``has_64bit_values``.

Interface versions: 3: AviSynth 2.5; 6: AviSynth 2.6 and AviSynth+ up to 3.5.0; 7: 3.5.1; 8: 3.6.x, 3.7.0;
9: 3.7.1, 3.7.2; 10: 3.7.3; 11: 3.7.4, 3.7.5 (11.1); 14: 3.7.6.

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

- Asd-g's C++ dynamic C API loader and convenience wrappers:

  https://github.com/Asd-g/avisynthplus-c-api-dynamic-loader
  
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
| avs_prop_get_data                   | AVS_Map               | 10 !  |
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
| avs_get_as_clip                     | AVS_Value             | 11    |
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
| avs_val_is_function                 | AVS_Value             | 14    |
+-------------------------------------+-----------------------+-------+
| avs_val_get_type                    | AVS_Value             | 14    |
+-------------------------------------+-----------------------+-------+
| avs_invoke_try                      | AVS_ScriptEnvironment | 14    |
+-------------------------------------+-----------------------+-------+
| avs_invoke2                         | AVS_ScriptEnvironment | 14    |
+-------------------------------------+-----------------------+-------+
| avs_invoke2_try                     | AVS_ScriptEnvironment | 14    |
+-------------------------------------+-----------------------+-------+
| avs_invoke3                         | AVS_ScriptEnvironment | 14    |
+-------------------------------------+-----------------------+-------+
| avs_invoke3_try                     | AVS_ScriptEnvironment | 14    |
+-------------------------------------+-----------------------+-------+
| avs_create_map                      | AVS_Map               | 14    |
+-------------------------------------+-----------------------+-------+
| avs_free_map                        | AVS_Map               | 14    |
+-------------------------------------+-----------------------+-------+
| avs_apply_message                   | AVS_ScriptEnvironment | 14    |
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

.. _c_avs_invoke_try:

avs_invoke_try
--------------

::

    int avs_invoke_try(AVS_ScriptEnvironment *, AVS_Value* result, const char * name,
                       AVS_Value args, const char** arg_names);

v14. No-throw variant of ``avs_invoke`` (C++ ``InvokeTry``).

Return values of the ``_try`` variants (``avs_invoke_try``, ``avs_invoke2_try``, ``avs_invoke3_try``):

- 1: success, the result is in ``*result``, release it with ``avs_release_value``.
- 0 and ``avs_get_error`` returns NULL: the function cannot be called with the given arguments
  (e.g. no such function name). ``*result`` is untouched.
- 0 and ``avs_get_error`` is set: other error, e.g. an error inside the called function.

.. _c_avs_invoke2:

avs_invoke2
-----------

::

    AVS_Value avs_invoke2(AVS_ScriptEnvironment *, AVS_Value implicit_last, const char * name,
                          AVS_Value args, const char** arg_names);

v14. ``avs_invoke`` with a value for the implicit ``last`` (C++ ``Invoke2``); pass ``avs_void`` if none.
The returned value must be released with ``avs_release_value``. On error it returns an error value,
``avs_get_error`` tells the reason.

.. _c_avs_invoke2_try:

avs_invoke2_try
---------------

::

    int avs_invoke2_try(AVS_ScriptEnvironment *, AVS_Value* result, AVS_Value implicit_last,
                        const char * name, AVS_Value args, const char** arg_names);

v14. No-throw variant of ``avs_invoke2`` (C++ ``Invoke2Try``). Return values: see :ref:`avs_invoke_try<c_avs_invoke_try>`.

.. _c_avs_invoke3:

avs_invoke3
-----------

::

    AVS_Value avs_invoke3(AVS_ScriptEnvironment *, AVS_Value implicit_last, AVS_Value func,
                          AVS_Value args, const char** arg_names);

v14. Calls a function value (C++ ``Invoke3``), e.g. a script function passed to a plugin parameter of type ``n``.
``func`` must be a function value (``avs_val_is_function``), otherwise an error value is returned.
Pass ``avs_void`` as ``implicit_last`` if none.
The returned value must be released with ``avs_release_value``. On error it returns an error value,
``avs_get_error`` tells the reason.

The ``AVS_Value`` itself is the function handle, there is no separate function type:
keep it with ``avs_copy_value``, free it with ``avs_release_value``.

::

    // script: ApplyInt(function(x) { return x + 1 }, 5)  # 6
    static AVS_Value AVSC_CC apply_int(AVS_ScriptEnvironment* env, AVS_Value args, void* user_data)
    {
      AVS_Value i = avs_get_array_elt(args, 1);
      AVS_Value fargs;
      avs_set_to_array(&fargs, &i, 1);
      AVS_Value res = avs_invoke3(env, avs_void, avs_get_array_elt(args, 0), fargs, NULL);
      avs_release_value(fargs);
      return res; // result or error value, goes back to the script
    }
    // in avisynth_c_plugin_init2:
    avs_add_function(env, "ApplyInt", "ni", apply_int, NULL);

.. _c_avs_invoke3_try:

avs_invoke3_try
---------------

::

    int avs_invoke3_try(AVS_ScriptEnvironment *, AVS_Value* result, AVS_Value implicit_last,
                        AVS_Value func, AVS_Value args, const char** arg_names);

v14. No-throw variant of ``avs_invoke3`` (C++ ``Invoke3Try``). Return values: see :ref:`avs_invoke_try<c_avs_invoke_try>`.
A ``func`` which is not a function value returns 0 with ``avs_get_error`` set.

.. _c_avs_make_writable:

avs_make_writable
-----------------

See also: :ref:`cplusplus_makewritable`

.. _c_avs_apply_message:

avs_apply_message
-----------------

::

    void avs_apply_message(AVS_ScriptEnvironment *, AVS_VideoFrame * * pvf, const AVS_VideoInfo * vi,
                           const char * message, int size, int textcolor, int halocolor, int bgcolor);

v14. Writes a text message onto the frame (C++ ``ApplyMessage``), e.g. to show an error on the output.
The frame must be writable: call ``avs_make_writable`` first. ``size`` is the font size in GDI units
(pixel size * 8), colors are ``0xRRGGBB`` (converted for YUV formats), ``bgcolor`` is not used.
The message must be UTF-8 encoded on every platform. Unlike the C++ ``ApplyMessage``, the system code
page is never assumed, so the result does not depend on the manifest or the code page of the host process.

::

    // in a C filter's get_frame callback: AVS_VideoFrame* AVSC_CC get_frame(AVS_FilterInfo* fi, int n)
    AVS_VideoFrame* frame = avs_get_frame(fi->child, n);
    avs_make_writable(fi->env, &frame);
    avs_apply_message(fi->env, &frame, &fi->vi, "MyFilter: something went wrong", fi->vi.width / 4, 0xF0F080, 0, 0);
    return frame;

See also: :ref:`ApplyMessage<cplusplus_applymessage>`

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

Use ``fi->env`` in the callbacks, also in ``free_filter``: it lives as long as the filter. The ``env`` received by
the create function (and by ``avisynth_c_plugin_init``) is temporary, valid only during that call: do not store it
for later use. Example with the global lock: :ref:`cplusplus_acquiregloballock`.

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
AVS_Clip can be released with avs_release_clip. The value must be released with ``avs_release_value``.
The inline ``avs_new_value_clip`` calls this API, the same rules apply.
See also the common rules at :ref:`avs_set_to_xxx<c_avs_set_to_xxx>`.

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

``error`` receives one of the ``AVS_GETPROPERROR_xxx`` codes (0: success). It can be NULL, but then
a failed read (missing key, wrong type, index out of range) returns 0/NULL and sets the message
for ``avs_get_error``; pass NULL only for properties known to exist. The same applies to all
``avs_prop_get_xxx`` functions with an ``error`` parameter (the ``_saturated`` ones never set it).
Before 3.7.6 a failed read with NULL ``error`` terminated the process.
See also :ref:`propGetInt<cplusplus_propgetint>`.

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

.. _c_avs_create_map:

avs_create_map
--------------

::

    AVS_Map* avs_create_map(AVS_ScriptEnvironment* p);

v14. Creates a standalone, empty property map, not attached to a frame (C++ ``createMap``).
Use it with the ``avs_prop_xxx`` functions, e.g. as a temporary key-value store.
It must be freed with ``avs_free_map``.

A map is the same container as the frame properties, modelled after VapourSynth's ``VSMap``:
each key holds an array of values of one type (64-bit integer, double, data (string or binary), clip
or frame); a single value is a one-element array. Unlike ``AVS_Value`` arrays, these arrays are flat:
no nesting, no mixed types, no map inside a map, and no function type.

.. _c_avs_free_map:

avs_free_map
------------

::

    void avs_free_map(AVS_ScriptEnvironment* p, AVS_Map* map);

v14. Frees a map created by ``avs_create_map`` (C++ ``freeMap``), with its content: clip and frame
references stored in it are released. Do not call it on a frame's map (``avs_get_frame_props_ro``,
``avs_get_frame_props_rw``), that one is owned by the frame. NULL is allowed.

::

    AVS_Map* m = avs_create_map(env);
    avs_prop_set_int(env, m, "count", 42, AVS_PROPAPPENDMODE_REPLACE);
    int err;
    int64_t count = avs_prop_get_int(env, m, "count", 0, &err);
    avs_free_map(env, m);

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

.. _c_avs_set_to_xxx:

The ``avs_set_to_xxx`` setters are API calls (v11), the counterparts of the ``avs_new_value_xxx``
inline helpers. They construct the value in place, through the pointer.

Common rules:

- The previous content of ``*v`` is overwritten, not released. Use them on a fresh value, or call
  ``avs_release_value`` first if it held a clip, function, array, or a ``double``/``long`` value.
- Resource-owning results must be released with ``avs_release_value``: clip, array, and on 32-bit (x86)
  systems ``double`` and ``long`` (stored in separately allocated memory there; on 64-bit systems the
  release is a no-op for them, portable code calls it anyway). For ``void``, ``bool``, ``int``, ``float``,
  ``string`` and ``error`` the release is a no-op.
- The whole ``AVS_Value`` is initialized, unlike in the inline helpers, which set only the type and
  the data member.

Differences from the inline helpers:

+-----------------------+-----------------------------+--------------------------------------------------------------+
| API (allocates/owns)  | inline helper               | difference                                                   |
+=======================+=============================+==============================================================+
| ``avs_set_to_bool``,  | ``avs_new_value_bool``,     | same result, no resources                                    |
| ``_int``, ``_float``  | ``_int``, ``_float``        |                                                              |
+-----------------------+-----------------------------+--------------------------------------------------------------+
| ``avs_set_to_string``,| ``avs_new_value_string``,   | same: only the pointer is stored, the string is not copied   |
| ``_error``            | ``_error``                  |                                                              |
+-----------------------+-----------------------------+--------------------------------------------------------------+
| ``avs_set_to_double``,| none                        | API only; must be released                                   |
| ``_long``             |                             |                                                              |
+-----------------------+-----------------------------+--------------------------------------------------------------+
| ``avs_set_to_array``  | ``avs_new_value_array``     | API deep copies (must be released); inline only points to    |
|                       |                             | the caller's array (must not be released)                    |
+-----------------------+-----------------------------+--------------------------------------------------------------+
| ``avs_set_to_clip``   | ``avs_new_value_clip``      | same: the inline one calls the API, both add a reference,    |
|                       |                             | both must be released                                        |
+-----------------------+-----------------------------+--------------------------------------------------------------+
| ``avs_set_to_void``   | ``avs_void`` constant       | same result                                                  |
+-----------------------+-----------------------------+--------------------------------------------------------------+

.. _c_avs_set_to_void:

avs_set_to_void
---------------

::

    void avs_set_to_void(AVS_Value*);

Sets an undefined (``'v'``) value, like assigning the ``avs_void`` constant. Release is a no-op.

.. _c_avs_set_to_error:

avs_set_to_error
----------------

::

    void avs_set_to_error(AVS_Value*, const char* v0);

Sets an error value. Only the pointer is stored, the message is not copied: it must outlive the value
(use a string literal or ``avs_save_string``). Release is a no-op.
Returned from a filter function (``avs_add_function`` callback), the error is thrown as a script error.

.. _c_avs_set_to_bool:

avs_set_to_bool
---------------

::

    void avs_set_to_bool(AVS_Value*, int v0);

Any nonzero ``v0`` is stored as true. Release is a no-op.

.. _c_avs_set_to_int:

avs_set_to_int
--------------

::

    void avs_set_to_int(AVS_Value*, int v0);

32-bit integer. Release is a no-op.

.. _c_avs_set_to_string:

avs_set_to_string
-----------------

::

    void avs_set_to_string(AVS_Value*, const char* v0);

Only the pointer is stored, the string is not copied: it must outlive the value. For a string that
lives as long as the script environment, store a copy made by ``avs_save_string``. Release is a no-op.

.. _c_avs_set_to_float:

avs_set_to_float
----------------

::

    void avs_set_to_float(AVS_Value*, float v0);

32-bit float. Release is a no-op.

.. _c_avs_set_to_double:

avs_set_to_double
-----------------

::

    void avs_set_to_double(AVS_Value*, double d);

64-bit floating point. No inline counterpart. Must be released with ``avs_release_value``:
on 32-bit (x86) systems the value is stored in separately allocated memory.

.. _c_avs_set_to_long:

avs_set_to_long
---------------

::

    void avs_set_to_long(AVS_Value*, int64_t l);

64-bit integer. No inline counterpart. Must be released with ``avs_release_value``:
on 32-bit (x86) systems the value is stored in separately allocated memory.

.. _c_avs_set_to_array:

avs_set_to_array
----------------

::

    void avs_set_to_array(AVS_Value*, AVS_Value* src, int size);

Creates an array value from ``size`` elements of ``src``. The array and its elements are deep copied
(nested arrays as well, clips and functions get an additional reference), so the result must be
released with ``avs_release_value``. ``src`` itself is not touched, release its elements separately
if they own resources.

The inline helper ``avs_new_value_array`` is the non-allocating alternative::

    AVS_Value avs_new_value_array(AVS_Value * v0, int size);

It allocates nothing, only points to the caller's array of ``size`` elements:

- do not call ``avs_release_value`` on it, that would free the caller's array;
- elements are not copied or reference counted, the array must outlive the use of the returned value.

Typical use is assembling ``args`` for ``avs_invoke``, ``avs_invoke2`` or ``avs_invoke3``:
the core makes its own copy for the call.

Example, calling ``StackHorizontal(clip, clip2)``, where ``clip`` and ``clip2`` are existing clip values.
Both ways start from a plain C array of the arguments. It holds simple struct copies, never release it.
Non-allocating way::

    AVS_Value src[2] = { clip, clip2 };
    AVS_Value res = avs_invoke(env, "StackHorizontal", avs_new_value_array(src, 2), NULL);
    // no avs_release_value for the array
    avs_release_value(res); // invoke's value must be released

The same through the API::

    AVS_Value src[2] = { clip, clip2 };
    AVS_Value args;
    avs_set_to_array(&args, src, 2);
    AVS_Value res = avs_invoke(env, "StackHorizontal", args, NULL);
    avs_release_value(args); // deep copy, must be released
    avs_release_value(res); // invoke's value must be released

.. _c_avs_new_value_array:

avs_new_value_array
-------------------

::

    AVS_Value avs_new_value_array(AVS_Value * v0, int size);

Inline (non-API) helper from ``avisynth_c.h``, not an exported function: it is compiled into the plugin.
Non-allocating array value, do not release it. See the description and examples
at :ref:`avs_set_to_array<c_avs_set_to_array>`.

.. _c_avs_get_as_xxx:

The ``avs_get_as_xxx`` getters and the array accessors below are API calls (v11), the
non-inline counterparts of the ``avs_as_xxx`` inline helpers.

Releasing: except ``avs_get_as_clip``, they allocate nothing. They return a plain C value
(``int``, ``int64_t``, ``double``) or a pointer into the source ``AVS_Value``, which stays owned by it.
Nothing returned by them must be released.

What must be released with ``avs_release_value`` is the *source* ``AVS_Value`` itself, when it owns
resources: a clip, a function, an array created by ``avs_set_to_array`` (or returned by ``avs_invoke``),
and 64-bit ``double`` and ``long`` values. On 32-bit (x86) systems ``double`` and ``long`` are stored
in separately allocated memory; on 64-bit systems they are stored in the ``AVS_Value`` directly,
releasing them is a no-op there, but portable code releases them anyway. Release the source only after
the last use of the pointers obtained from it.

.. _c_avs_get_as_bool:

avs_get_as_bool
---------------

::

    int avs_get_as_bool(AVS_Value v);

Returns 1 or 0. Nothing to release.

.. _c_avs_get_as_clip:

avs_get_as_clip
---------------

::

    AVS_Clip* avs_get_as_clip(AVS_Value v, AVS_ScriptEnvironment* env);

Like ``avs_take_clip``: the only getter which allocates. Returns a new ``AVS_Clip`` and increases
the clip's reference count by one; release it with ``avs_release_clip``. The source ``AVS_Value``
can be released independently. See also :ref:`c_avs_take_clip`.

.. _c_avs_get_as_int:

avs_get_as_int
--------------

::

    int avs_get_as_int(AVS_Value v);

Returns a 32-bit integer, also from a 64-bit ``long`` value (truncated). Nothing to release.

.. _c_avs_get_as_long:

avs_get_as_long
---------------

::

    int64_t avs_get_as_long(AVS_Value v);

Returns a 64-bit integer, from an ``int`` or ``long`` value. The returned value is a plain
``int64_t``, nothing to release; a source ``long`` value must be released (see above).

.. _c_avs_get_as_string:

avs_get_as_string
-----------------

::

    const char* avs_get_as_string(AVS_Value v);

Returns the string pointer stored in the value, not a copy. Do not free it.
Strings coming from Avisynth (script, ``avs_save_string``) live as long as the script environment.

.. _c_avs_get_as_float:

avs_get_as_float
----------------

::

    double avs_get_as_float(AVS_Value v);

Returns a ``double``, from an ``int``, ``long``, ``float`` or ``double`` value. The returned value is
a plain ``double``, nothing to release; a source ``double`` or ``long`` value must be released (see above).

.. _c_avs_get_as_error:

avs_get_as_error
----------------

::

    const char* avs_get_as_error(AVS_Value v);

Returns the error message of an error value, NULL if the value is not an error. Pointer, not a copy,
do not free it.

.. _c_avs_get_as_array:

avs_get_as_array
----------------

::

    const AVS_Value* avs_get_as_array(AVS_Value v);

Returns a pointer to the elements of an array value. Borrowed: owned by the array value, do not release
it or its elements; it is valid until the array value is released.

.. _c_avs_get_array_elt:

avs_get_array_elt
-----------------

::

    AVS_Value avs_get_array_elt(AVS_Value v, int index);

Returns an element of an array value (or the value itself if it is not an array).
A plain struct copy, no reference counting: do not release it. Use ``avs_copy_value`` on it
to keep the element after the array value is released (and release that copy later).

.. _c_avs_get_array_size:

avs_get_array_size
------------------

::

    int avs_get_array_size(AVS_Value v);

Returns the number of elements of an array value. Nothing to release.
Call it on array values only (``avs_val_is_array``); a non-array value returns 1, but debug builds assert.

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

.. _c_avs_val_is_function:

avs_val_is_function
-------------------

v14. Returns 1 if the value is a function (``'n'`` type). See :ref:`avs_invoke3<c_avs_invoke3>`.

.. _c_avs_val_get_type:

avs_val_get_type
----------------

::

    int avs_val_get_type(AVS_Value v);

v14. Returns the type of the value as one of the ``AVS_VALUE_TYPE_xxx`` constants (C++ ``AVSValue::GetType``)::

    AVS_VALUE_TYPE_UNDEFINED = 'v', // void, undefined value
    AVS_VALUE_TYPE_BOOL = 'b',      // boolean
    AVS_VALUE_TYPE_INT = 'i',       // 32 bit integer
    AVS_VALUE_TYPE_LONG = 'l',      // 64 bit integer
    AVS_VALUE_TYPE_FLOAT = 'f',     // 32 bit floating point
    AVS_VALUE_TYPE_DOUBLE = 'd',    // 64 bit floating point
    AVS_VALUE_TYPE_STRING = 's',    // string (const char *)
    AVS_VALUE_TYPE_CLIP = 'c',      // clip
    AVS_VALUE_TYPE_FUNCTION = 'n',  // function, call it with avs_invoke3 (v14)
    AVS_VALUE_TYPE_ARRAY = 'a',     // array of AVS_Value
    AVS_VALUE_TYPE_ERROR = 'e'      // error message string, C interface only

Use it instead of reading the ``type`` field of ``AVS_Value`` directly.

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

$Date: 2026/11/10 21:00:00 $
