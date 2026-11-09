// AviSynth+ C API dynamic loader (header only, Windows, Linux, macOS, BSD, Haiku)
// Copyright 2026 AviSynth+ project
// Same license and linking exception as avisynth_c.h.

// Sample loader for using the C API without linking avisynth.lib/libavisynth:
// loads AviSynth at runtime and fills a struct with the address of every C API function.
// Use it as is, or copy and trim it for your project.
//
// Usage (host, e.g. an encoder or player):
//   #include "avisynth_c_loader.h" // instead of avisynth_c.h, it defines AVSC_NO_DECLSPEC
//
//   AVS_Library* lib = avs_load_library(NULL); // NULL: default library name, see below
//   if (!lib) { ... AviSynth not found ... }
//   // the highest version both sides know: requests V14, then 11, 8, 6, 3
//   AVS_ScriptEnvironment* env = avs_library_create_script_environment(lib);
//   if (!env) { ... }
//   if (lib->has_at_least_v9) { ... frame properties ... }
//   if (lib->has_64bit_values) { ... double and long (int64) values are kept ... }
//   if (lib->has_at_least_v14) { ... function values, avs_invoke2/3 ... }
//   ...
//   lib->avs_delete_script_environment(env);
//   avs_free_library(lib);
//
// Usage (C plugin): export both entry points, avisynth_c_plugin_init2 forwarding to
// avisynth_c_plugin_init (AviSynth V10 and below knows only the latter):
//   static AVS_Library* lib;
//   static int from_init2;
//   const char* AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment* env) {
//     lib = avs_load_library(NULL);
//     if (!lib) return "...";
//     avs_library_detect_plugin_env(lib, env, from_init2); // the same fields, plus has_64bit_values
//     lib->avs_add_function(env, ...);
//     return "...";
//   }
//   const char* AVSC_CC avisynth_c_plugin_init2(AVS_ScriptEnvironment* env) {
//     from_init2 = 1; // AviSynth V11+ found it: the plugin's functions get double and long values
//     return avisynth_c_plugin_init(env);
//   }
//   (Win32 .def: avisynth_c_plugin_init2@4 = _avisynth_c_plugin_init2@4 as well)
//
// Creating the environment yourself: for a version newer than the loaded DLL's interface version
// (the AVISYNTH_INTERFACE_VERSION compiled into that avisynth.dll/libavisynth, not the one of the
// included avisynth_c.h), the core refuses the request (its CreateScriptEnvironment returns NULL),
// but avs_create_script_environment still returns a non-NULL wrapper, holding a NULL core pointer.
// From 3.7.6 avs_get_error tells the refusal, older versions give no error message. Any call
// through it crashes; check it with avs_library_env_failed. A fixed low version is no solution
// either: below 11 AviSynth converts double values to float and long (int64) values to 32 bit int
// for the caller.
//
// - avs_load_library returns NULL only if the library cannot be loaded.
// - Every function missing from the loaded library (e.g. an older AviSynth) is NULL in the struct.
//   Check the pointer before calling a function newer than the AviSynth versions you support.
// - A function existing in a library does not mean it works there: some were fixed in a later
//   interface version than the one they appeared in. The comments below give for each function the
//   interface version it appeared in, and where it differs, the version from which it is usable
//   reliably. See also "API functions fixed later than they appeared" in avisynth_c.h.
//   The has_at_least_xxx fields of the struct give the important steps, interface_version and
//   interface_bugfix the exact version.
//   Interface versions: 3: AviSynth 2.5, 6: AviSynth 2.6 and AviSynth+ up to 3.5.0; 7: 3.5.1;
//   8: 3.6.x, 3.7.0; 9: 3.7.1, 3.7.2; 10: 3.7.3; 11: 3.7.4, 3.7.5 (11.1); 14: 3.7.6; 14.1: 3.7.7.
//   "AVS+": exists in every AviSynth+ (not in classic AviSynth).
// - Win32 (32 bit) AviSynth+ 3.6.x - 3.7.1 exported the V8 functions only with decorated names
//   (_avs_get_frame_props_ro@8), they are NULL there. Fixed in 3.7.2.
// - The AVSC_INLINE helpers of avisynth_c.h which call into the API (e.g. avs_new_video_frame,
//   avs_get_pitch) are not available with AVSC_NO_DECLSPEC: call the API function through lib->.
// - Plugins: AviSynth is already loaded by the time the plugin is, avs_load_library(NULL) gets the
//   same instance on Windows. On other systems pass the path of the running AviSynth if it is not
//   in the default library search path.

#ifndef AVISYNTH_C_LOADER_H
#define AVISYNTH_C_LOADER_H

#if defined(__AVISYNTH_C__) && !defined(AVSC_NO_DECLSPEC)
#error "avisynth_c.h was included without AVSC_NO_DECLSPEC: include avisynth_c_loader.h first"
#endif

#ifndef AVSC_NO_DECLSPEC
#define AVSC_NO_DECLSPEC
#endif
#include "avisynth_c.h"

#include <stdlib.h>
#ifdef AVS_WINDOWS
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// The list of all C API functions: X(name), grouped by the interface version they appeared in.
#define AVSC_LOADER_FUNCTIONS(X) \
  /* V3: AviSynth 2.5 */ \
  X(avs_add_function) \
  X(avs_at_exit) \
  X(avs_bit_blt) \
  X(avs_check_version) \
  X(avs_clip_get_error) \
  X(avs_copy_clip) \
  X(avs_copy_value) \
  X(avs_copy_video_frame) \
  X(avs_create_script_environment) \
  X(avs_function_exists) \
  X(avs_get_audio) \
  X(avs_get_cpu_flags) \
  X(avs_get_frame) \
  X(avs_get_parity) \
  X(avs_get_var) \
  X(avs_get_version) \
  X(avs_get_video_info) \
  X(avs_invoke) \
  X(avs_make_writable) \
  X(avs_new_c_filter) /* AVS_MT_NICE_FILTER C filters: error reporting reliable from V14 */ \
  X(avs_new_video_frame_a) \
  X(avs_release_clip) \
  X(avs_release_value) \
  X(avs_release_video_frame) \
  X(avs_save_string) \
  X(avs_set_cache_hints) \
  X(avs_set_global_var) \
  X(avs_set_memory_max) \
  X(avs_set_to_clip) \
  X(avs_set_var) \
  X(avs_set_working_dir) \
  X(avs_subframe) \
  X(avs_take_clip) \
  X(avs_vsprintf) \
  X(avs_sprintf) \
  /* V6: AviSynth 2.6 */ \
  X(avs_delete_script_environment) \
  X(avs_subframe_planar) \
  X(avs_get_error) \
  X(avs_is_yv24) \
  X(avs_is_yv16) \
  X(avs_is_yv12) \
  X(avs_is_yv411) \
  X(avs_is_y8) \
  X(avs_is_color_space) \
  X(avs_get_plane_width_subsampling) \
  X(avs_get_plane_height_subsampling) \
  X(avs_bits_per_pixel) \
  X(avs_bytes_from_pixels) \
  X(avs_row_size) \
  X(avs_bmp_size) \
  X(avs_get_pitch_p) \
  X(avs_get_row_size_p) \
  X(avs_get_height_p) \
  X(avs_get_read_ptr_p) \
  X(avs_is_writable) \
  X(avs_get_write_ptr_p) \
  /* AVS+ */ \
  X(avs_num_components) \
  X(avs_component_size) \
  X(avs_bits_per_component) \
  X(avs_is_444) \
  X(avs_is_422) \
  X(avs_is_420) \
  X(avs_is_y) \
  X(avs_is_yuva) \
  X(avs_is_planar_rgb) \
  X(avs_is_planar_rgba) \
  X(avs_is_rgb48) \
  X(avs_is_rgb64) \
  /* AVS+, deprecated */ \
  X(avs_is_yuv444p16) \
  X(avs_is_yuv422p16) \
  X(avs_is_yuv420p16) \
  X(avs_is_y16) \
  X(avs_is_yuv444ps) \
  X(avs_is_yuv422ps) \
  X(avs_is_yuv420ps) \
  X(avs_is_y32) \
  /* V8: frame properties. Frame property access is usable from V9 (crashed in V8) */ \
  X(avs_subframe_planar_a) \
  X(avs_copy_frame_props) /* usable from V9 */ \
  X(avs_get_frame_props_ro) /* usable from V9 */ \
  X(avs_get_frame_props_rw) /* usable from V9 */ \
  X(avs_prop_num_keys) \
  X(avs_prop_get_key) \
  X(avs_prop_num_elements) \
  X(avs_prop_get_type) \
  /* avs_prop_get_xxx: before V14 pass a non-NULL error pointer (NULL: process terminated on a failed read) */ \
  X(avs_prop_get_int) \
  X(avs_prop_get_float) \
  X(avs_prop_get_data) /* usable from V10 */ \
  X(avs_prop_get_data_size) \
  X(avs_prop_get_clip) /* usable from V14 (returned an invalid AVS_Clip) */ \
  X(avs_prop_get_frame) /* V11: *error was always 0, check the result for NULL */ \
  X(avs_prop_delete_key) \
  X(avs_prop_set_int) \
  X(avs_prop_set_float) \
  X(avs_prop_set_data) \
  X(avs_prop_set_clip) \
  X(avs_prop_set_frame) /* usable from V9 */ \
  X(avs_prop_get_int_array) \
  X(avs_prop_get_float_array) \
  X(avs_prop_set_int_array) \
  X(avs_prop_set_float_array) \
  X(avs_clear_map) \
  X(avs_new_video_frame_p) /* non-NULL prop_src usable from V10 */ \
  X(avs_new_video_frame_p_a) /* non-NULL prop_src usable from V10 */ \
  X(avs_get_env_property) /* AVS_AEP_INTERFACE_VERSION and AVS_AEP_INTERFACE_BUGFIX from V9 */ \
  X(avs_pool_allocate) \
  X(avs_pool_free) \
  X(avs_get_var_try) \
  X(avs_get_var_bool) \
  X(avs_get_var_int) \
  X(avs_get_var_double) \
  X(avs_get_var_string) \
  X(avs_get_var_long) \
  /* V9 */ \
  X(avs_is_property_writable) \
  X(avs_make_property_writable) \
  /* V10 */ \
  X(avs_video_frame_get_pixel_type) \
  X(avs_video_frame_amend_pixel_type) \
  /* V10.1 */ \
  X(avs_is_channel_mask_known) \
  X(avs_set_channel_mask) \
  X(avs_get_channel_mask) \
  /* V11: 64 bit AVS_Value types */ \
  X(avs_set_to_void) \
  X(avs_set_to_error) \
  X(avs_set_to_bool) \
  X(avs_set_to_int) \
  X(avs_set_to_string) \
  X(avs_set_to_float) \
  X(avs_set_to_double) \
  X(avs_set_to_long) \
  X(avs_set_to_array) /* NULL src with nonzero size: usable from V14 */ \
  X(avs_get_as_bool) \
  X(avs_get_as_clip) \
  X(avs_get_as_int) \
  X(avs_get_as_long) \
  X(avs_get_as_string) \
  X(avs_get_as_float) \
  X(avs_get_as_error) \
  X(avs_get_as_array) \
  X(avs_get_array_elt) \
  X(avs_get_array_size) \
  X(avs_val_defined) \
  X(avs_val_is_clip) \
  X(avs_val_is_bool) \
  X(avs_val_is_int) \
  X(avs_val_is_long_strict) \
  X(avs_val_is_float) \
  X(avs_val_is_floatf_strict) \
  X(avs_val_is_string) \
  X(avs_val_is_array) \
  X(avs_val_is_error) \
  X(avs_prop_get_int_saturated) \
  X(avs_prop_get_float_saturated) \
  X(avs_prop_get_data_type_hint) \
  X(avs_prop_set_data_h) \
  X(avs_add_function_r) \
  /* V12 */ \
  X(avs_get_cpu_flags_ex) \
  X(avs_acquire_global_lock) \
  X(avs_release_global_lock) \
  /* V13: new pixel formats */ \
  X(avs_is_440) \
  X(avs_is_410) \
  X(avs_is_411) \
  X(avs_is_ya) \
  /* V14 */ \
  X(avs_val_is_function) \
  X(avs_val_get_type) \
  X(avs_invoke_try) \
  X(avs_invoke2) \
  X(avs_invoke2_try) \
  X(avs_invoke3) \
  X(avs_invoke3_try) \
  X(avs_create_map) \
  X(avs_free_map) \
  X(avs_apply_message)

typedef struct AVS_Library AVS_Library;

struct AVS_Library {
  void* handle; // HMODULE on Windows, dlopen handle elsewhere

  // Interface version of the loaded AviSynth and the feature steps.
  // Filled by avs_library_create_script_environment (hosts) or avs_library_detect_plugin_env (plugins),
  // all 0 before.
  int interface_version;  // e.g. 6, 8, 11, 14
  int interface_bugfix;   // e.g. 1 for V11.1 (3.7.5)
  int has_at_least_v8;    // frame property API entries, avs_new_video_frame_p (frame property source).
                          // C API frame property access crashes in V8, use it from V9.
  int has_at_least_v9;    // C API frame property access works, avs_is/make_property_writable,
                          // AVS_AEP_INTERFACE_VERSION/BUGFIX queries
  int has_at_least_v9_1;  // kept for compatibility: V9.1 and V9.2 were test builds only, in releases
  int has_at_least_v9_2;  // these are the same as has_at_least_v10
  int has_at_least_v10;   // AVS_VideoFrame pixel_type (avs_video_frame_get/amend_pixel_type),
                          // avs_new_video_frame_p(_a) with a non-NULL prop_src, avs_prop_get_data,
                          // C filters: an error of the child clip (fi->child->error) is passed on
  int has_at_least_v11;   // the core knows 64 bit AVS_Value types (double, long). Whether this caller
                          // gets them: has_64bit_values
  int has_at_least_v12;   // avs_acquire/release_global_lock, avs_get_cpu_flags_ex, AVS_AEP_CACHESIZE_L2
  int has_at_least_v13;   // YUV(A) 4:4:0, 4:1:1, 4:1:0, YA pixel formats
  int has_at_least_v14;   // function values, avs_invoke2/3(_try), avs_create_map; avs_prop_get_xxx fixes
                          // (NULL error pointer, avs_prop_get_clip), AVS_MT_NICE_FILTER error reporting
  int has_at_least_v14_1; // frame properties: _Range is the stored key, _ColorRange its alias (0 and 1 swapped).
                          // Before, AviSynth knew only _ColorRange

  // This caller gets 64 bit AVS_Value types: double and long (int64) values are kept, not converted
  // to float and 32 bit int.
  // Hosts: the environment was requested with version >= 11 (avs_library_create_script_environment).
  // Plugins: AviSynth called avisynth_c_plugin_init2 (avs_library_detect_plugin_env).
  // Not the same as has_at_least_v11: a V11+ core converts the values for a pre-V11 caller.
  int has_64bit_values;

#define AVSC_LOADER_DECLARE(name) name##_func name;
  AVSC_LOADER_FUNCTIONS(AVSC_LOADER_DECLARE)
#undef AVSC_LOADER_DECLARE
};

// Loads AviSynth and all C API functions, missing ones are NULL.
// library_path: UTF-8 file name or path; NULL: "avisynth" (Windows), "libavisynth.dylib" (macOS),
// "libavisynth.so" (others).
// Returns NULL if the library cannot be loaded. Free the result with avs_free_library.
AVSC_INLINE AVS_Library* avs_load_library(const char* library_path)
{
  AVS_Library* library = (AVS_Library*)calloc(1, sizeof(AVS_Library)); // zero filled: missing entries are NULL
  if (library == NULL)
    return NULL;

#ifdef AVS_WINDOWS
  {
    wchar_t wpath[MAX_PATH];
    if (library_path == NULL)
      library_path = "avisynth";
    // library_path is UTF-8. For a path in the ANSI code page use CP_ACP here instead, or call
    // LoadLibraryA directly (it takes the process code page, UTF-8 with an activeCodePage manifest).
    if (MultiByteToWideChar(CP_UTF8, 0, library_path, -1, wpath, MAX_PATH) == 0) {
      free(library);
      return NULL;
    }
    library->handle = (void*)LoadLibraryW(wpath);
  }
#else
  if (library_path == NULL) {
#ifdef AVS_MACOS
    library_path = "libavisynth.dylib";
#else
    library_path = "libavisynth.so";
#endif
  }
  library->handle = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
#endif
  if (library->handle == NULL) {
    free(library);
    return NULL;
  }

  // the address is copied through void* : the POSIX way of converting dlsym's result to a function pointer
#ifdef AVS_WINDOWS
#define AVSC_LOADER_LOAD(name) { FARPROC proc = GetProcAddress((HMODULE)library->handle, #name); *(void**)&library->name = *(void**)&proc; }
#else
#define AVSC_LOADER_LOAD(name) { void* proc = dlsym(library->handle, #name); *(void**)&library->name = proc; }
#endif
  AVSC_LOADER_FUNCTIONS(AVSC_LOADER_LOAD)
#undef AVSC_LOADER_LOAD

  return library;
}

AVSC_INLINE void avs_free_library(AVS_Library* library)
{
  if (library == NULL)
    return;
#ifdef AVS_WINDOWS
  FreeLibrary((HMODULE)library->handle);
#else
  dlclose(library->handle);
#endif
  free(library);
}

// Returns the interface version of the loaded AviSynth (e.g. 6, 8, 11, 14), 0 if unknown.
// bugfix (may be NULL) receives the bugfix version (V9 and above, otherwise 0).
// E.g. interface 11 with bugfix 1 is "V11.1" (3.7.5).
// avs_get_env_property with AVS_AEP_INTERFACE_VERSION is only asked from V9, earlier versions did not know it.
AVSC_INLINE int avs_library_interface_version(const AVS_Library* library, AVS_ScriptEnvironment* env, int* bugfix)
{
  int version = 0;
  int fix = 0;
  if (library->avs_check_version != NULL) {
    if (library->avs_get_env_property != NULL && library->avs_check_version(env, 9) == 0) {
      version = (int)library->avs_get_env_property(env, AVS_AEP_INTERFACE_VERSION);
      fix = (int)library->avs_get_env_property(env, AVS_AEP_INTERFACE_BUGFIX);
    }
    else {
      // below v9 we can only poll
      int version_probe;
      for (version_probe = 8; version_probe >= 1; version_probe--) {
        if (library->avs_check_version(env, version_probe) == 0) {
          version = version_probe;
          break;
        }
      }
    }
  }
  if (bugfix != NULL)
    *bugfix = fix;
  return version;
}

// Fills interface_version, interface_bugfix and the has_at_least_xxx steps of the library struct.
AVSC_INLINE void avs_library_set_version(AVS_Library* library, int version, int bugfix)
{
#define AVSC_LOADER_AT_LEAST(v, b) (version > (v) || (version == (v) && bugfix >= (b)))
  library->interface_version = version;
  library->interface_bugfix = bugfix;
  library->has_at_least_v8 = AVSC_LOADER_AT_LEAST(8, 0);
  library->has_at_least_v9 = AVSC_LOADER_AT_LEAST(9, 0);
  library->has_at_least_v9_1 = AVSC_LOADER_AT_LEAST(9, 1);
  library->has_at_least_v9_2 = AVSC_LOADER_AT_LEAST(9, 2);
  library->has_at_least_v10 = AVSC_LOADER_AT_LEAST(10, 0);
  library->has_at_least_v11 = AVSC_LOADER_AT_LEAST(11, 0);
  library->has_at_least_v12 = AVSC_LOADER_AT_LEAST(12, 0);
  library->has_at_least_v13 = AVSC_LOADER_AT_LEAST(13, 0);
  library->has_at_least_v14 = AVSC_LOADER_AT_LEAST(14, 0);
  library->has_at_least_v14_1 = AVSC_LOADER_AT_LEAST(14, 1);
#undef AVSC_LOADER_AT_LEAST
}

// Detects the version of the loaded AviSynth through env.
// Fills interface_version, interface_bugfix and the has_at_least_xxx steps; returns the version.
AVSC_INLINE int avs_library_detect_version(AVS_Library* library, AVS_ScriptEnvironment* env)
{
  int bugfix;
  int version = avs_library_interface_version(library, env, &bugfix);
  avs_library_set_version(library, version, bugfix);
  return version;
}

// Plugins: call from the plugin init with the env AviSynth passed. Fills the version fields like
// avs_library_detect_version, and has_64bit_values; returns the version.
// called_from_init2: nonzero if AviSynth called avisynth_c_plugin_init2. The loader cannot see it,
// the plugin has to tell, see the example at the top.
// How AviSynth decides (3.7.4+): it looks for avisynth_c_plugin_init2 first; if it is exported,
// the plugin's functions get double and long values. Otherwise it calls avisynth_c_plugin_init and
// converts these values to float and 32 bit int for the plugin's functions. Older AviSynth (V10 and
// below) knows only avisynth_c_plugin_init and has no 64 bit values at all.
AVSC_INLINE int avs_library_detect_plugin_env(AVS_Library* library, AVS_ScriptEnvironment* env, int called_from_init2)
{
  int version = avs_library_detect_version(library, env);
  library->has_64bit_values = called_from_init2 && version >= 11;
  return version;
}

// Returns nonzero if avs_create_script_environment failed. For a version newer than the loaded
// DLL's interface version, the core refuses the request (its CreateScriptEnvironment returns NULL),
// but the C wrapper returned by avs_create_script_environment is still non-NULL: it holds the NULL
// core pointer; any call through it crashes.
// - From 3.7.6 (V14) avs_get_error reports the refusal.
// - Older versions give no error message, the only sign of the refusal is that NULL core pointer,
//   the first member of the opaque struct: its layout { core, error } has been the same since the
//   first C interface (2003).
AVSC_INLINE int avs_library_env_failed(const AVS_Library* library, AVS_ScriptEnvironment* env)
{
  if (env == NULL)
    return 1;
  if (library->avs_get_error != NULL && library->avs_get_error(env) != NULL)
    return 1;
  {
    // mirror of the core-only struct AVS_ScriptEnvironment { IScriptEnvironment* env; const char* error; }
    typedef struct { const void* core; const char* error; } AVS_ScriptEnvironment_Layout;
    return ((const AVS_ScriptEnvironment_Layout*)env)->core == NULL;
  }
}

// Hosts: creates the script environment with the highest interface version known by both
// the loaded DLL and the avisynth_c.h this loader was compiled with, and fills the version fields
// of the library struct.
// Two different versions are involved:
// - the header's AVISYNTH_INTERFACE_VERSION (compile time): what this caller understands, the first
//   request. Rebuilding with a newer avisynth_c.h raises it automatically; then extend the
//   has_at_least_xxx steps and the function list if needed. An avisynth_c.h older than V14 does not
//   compile with this loader (missing function types).
// - the loaded DLL's interface version (run time): requests above it are refused.
// The requested version matters: it tells AviSynth what the caller understands.
// Below 11 AviSynth converts double values to float and long (int64) values to 32 bit int,
// even if it supports them itself.
// Only the steps which make a difference are requested, the first accepted one is kept;
// a refused request is cheap (no environment is built):
// - AVISYNTH_INTERFACE_VERSION: what this caller understands
// - 11: a V11..V13 AviSynth: 64 bit AVS_Value types. Any version from 11 up to the DLL's interface version
//   gives the same environment.
// - 8: a V8..V10 AviSynth (frame properties, avs_new_video_frame_p). The environment is the same as
//   with 6, the step only follows the has_at_least_v8 feature level.
// - 6 (classic): older AviSynth. Any version below 11 gives the same environment; bugfix versions
//   cannot be requested at all.
// - 3: AviSynth 2.5
// The exact version and the has_at_least_xxx steps are then queried through the environment.
// Returns NULL on failure. Delete the result with lib->avs_delete_script_environment.
AVSC_INLINE AVS_ScriptEnvironment* avs_library_create_script_environment(AVS_Library* library)
{
  static const int request[] = { AVISYNTH_INTERFACE_VERSION, 11, 8, AVISYNTH_INTERFACE_CLASSIC_VERSION, 3 };
  int i;
  if (library->avs_create_script_environment == NULL)
    return NULL;
  for (i = 0; i < (int)(sizeof(request) / sizeof(request[0])); i++) {
    AVS_ScriptEnvironment* env = library->avs_create_script_environment(request[i]);
    if (!avs_library_env_failed(library, env)) {
      avs_library_detect_version(library, env); // the DLL's exact version, also if the DLL is newer than the header
      library->has_64bit_values = request[i] >= 11; // accepted only by a V11+ core
      return env;
    }
    if (env != NULL && library->avs_delete_script_environment != NULL)
      library->avs_delete_script_environment(env); // frees the wrapper only
  }
  return NULL;
}

#endif // AVISYNTH_C_LOADER_H
