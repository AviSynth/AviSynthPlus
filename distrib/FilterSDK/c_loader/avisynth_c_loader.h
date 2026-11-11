// AviSynth+ C API dynamic loader, C variant
// (headers only, no source file to build; Windows, Linux, macOS, BSD, Haiku).
// Needs avisynth_c_loader_common.h next to it.
// For C++ use avisynth_c_loader.hpp.
//
// Copyright 2026 AviSynth+ project
// Same license and linking exception as avisynth_c.h.

// What it is for: using the AviSynth C API without linking avisynth.lib/libavisynth.
// Loads AviSynth at runtime for hosts, or attaches to an already-loaded AviSynth for plugins
// (subject to the platform restrictions documented below). Both variants resolve C API function
// pointers, provide fallbacks for older cores, and detect the interface version through an environment.
// avs_load_library/avs_attach_library own a library reference; avs_bind_plugin_library keeps only function pointers.
//
// Two variants sharing the loading, binding and compatibility handling:
//
// 1. avisynth_c_loader.h:
//    C11, also compiles as C++.
//    A struct, AVS_Library, and free functions. For C code, or C++ code which prefers the C style.
//
// 2. avisynth_c_loader.hpp:
//    C++17.
//    A class, avsc::Library, which releases any owned library reference in its destructor.
//    For C++ hosts and plugins using the C API, the usual case.
//
// Their common part (function list, fallbacks, version query, platform calls) is avisynth_c_loader_common.h.
//
// Samples, not installed AviSynth headers (avisynth_c_loader.h was installed in 3.7.6):
// FilterSDK/c_loader in the source tree, Windows packages and installer SDK;
// share/doc/avisynth/examples/c_loader in a CMake install. Copy the chosen loader header and
// avisynth_c_loader_common.h into your project, as they are or trimmed. avisynth_c.h and its avs/
// folder must be on the include path; any avisynth_c.h from V14 (3.7.6) on works.
//
// Usage (host, e.g. an encoder or player):
//   #include "avisynth_c_loader.h" // instead of avisynth_c.h, it defines AVSC_NO_DECLSPEC
//
//   AVS_Library* lib = avs_load_library(NULL); // NULL: default library name, see below
//   if (!lib) { ... AviSynth not found ... }
//   // or, with the reason of a failure (optional):
//   char error[AVSC_LOADER_ERROR_SIZE];
//   AVS_Library* lib = avs_load_library_ex(NULL, error, sizeof(error));
//   if (!lib) { printf("%s\n", error); ... }  // e.g. "avisynth: a DLL it needs is missing (...)"
//   // Negotiate the environment version; see AVSC_LOADER_ENV_REQUESTS.
//   AVS_ScriptEnvironment* env = avs_library_create_script_environment(lib);
//   if (!env) { ... }
//   if (lib->has_frame_props) { ... avs_prop_xxx, avs_get_frame_props_ro ... } // V9 and the functions are there
//   if (lib->has_64bit_values) { ... double and long (int64) values are kept ... }
//   if (lib->has_at_least_v14) { ... function values, avs_invoke2/3 ... }
//   ...
//   lib->avs_delete_script_environment(env);
//   avs_free_library(lib);
//
// Usage (C11 plugin): export both entry points through a shared initializer
// (AviSynth V10 and below knows only avisynth_c_plugin_init):
//   #include <stdatomic.h>
//   static AVS_Library lib; // caller-owned storage, no allocation or retained library reference
//   static atomic_flag init_lock = ATOMIC_FLAG_INIT;
//   static const char* init_plugin(AVS_ScriptEnvironment* env, int from_init2) {
//     while (atomic_flag_test_and_set_explicit(&init_lock, memory_order_acquire)) {}
//     if (!lib.avs_create_script_environment) {
//       if (!avs_bind_plugin_library(&lib, NULL, 0)) {
//         atomic_flag_clear_explicit(&init_lock, memory_order_release);
//         return "...";
//       }
//       avs_library_detect_plugin_env(&lib, env, from_init2); // initialize once, then read-only
//     }
//     atomic_flag_clear_explicit(&init_lock, memory_order_release);
//     lib.avs_add_function(env, ...);
//     return "...";
//   }
//   const char* AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment* env) {
//     return init_plugin(env, 0);
//   }
//   const char* AVSC_CC avisynth_c_plugin_init2(AVS_ScriptEnvironment* env) {
//     return init_plugin(env, 1); // the plugin's functions get double and long values
//   }
//
// Notes:
// - Initialization: avs_load_library/avs_load_library_ex and avs_attach_library return NULL on failure.
//   Binding requires avs_create_script_environment. Optional diagnostics: avs_load_library_ex and
//   avs_attach_library accept an error buffer. Release successful results with avs_free_library.
//   avs_bind_plugin_library returns nonzero on success and uses caller-owned storage; do not free it.
// - Synchronization: AVS_Library does not lock internally. The plugin pattern serializes binding and
//   version detection, then leaves shared fields read-only; registration still runs for every env.
//   Its C11 atomic_flag needs no cleanup, TLS or optional threads.h support; older compilers need
//   their own guard. A failed bind can be retried. Core unload/reload does not refresh the cached binding.
// - Missing functions: fallbacks are installed where possible; other entries remain NULL.
//   Examples: avs_is_444 aliases avs_is_yv24 on classic 2.6; V11 AVS_Value accessors have legacy
//   fallbacks. See AVSC_LOADER_FALLBACKS. Fallbacks (added in 3.7.7) make pointer presence unsuitable
//   for version detection; use interface_version/interface_bugfix and has_at_least_xxx.
// - API compatibility: require both the function pointer and its usable interface version.
//   has_frame_props checks both for the V8/V9 frame property set (one check is enough there).
//   AVSC_LOADER_FUNCTIONS documents introduction versions
//   and later fixes; see also "API functions fixed later than they appeared" in avisynth_c.h.
// - Version flags: both variants expose released milestones plus the current WIP, not the
//   development steps V9.1/V9.2/V12/V13. Exact comparisons use interface_version/interface_bugfix.
// - Environment creation: prefer avs_library_create_script_environment for version negotiation.
//   Direct avs_create_script_environment calls require avs_library_env_failed: unsupported requests
//   can return a non-NULL wrapper with a NULL core. V14+ reports this through avs_get_error; older
//   cores may leave the error unset. Do not call core-dependent functions through a failed wrapper.
// - 64-bit values: hosts must request V11+; plugins must export avisynth_c_plugin_init2.
//   Otherwise the core converts double/long values to float/int. has_64bit_values describes the
//   caller's value support, not just the core version.
// - Plugin attachment: use avs_bind_plugin_library, not default-name loading, which may fail or select
//   another copy on POSIX. It releases its temporary library reference before returning. See
//   avs_loader_attach for platform limits and macOS concurrency requirements; multiple cores remain ambiguous.
// - Lifetime: release values, clips, frames and host-created environments before freeing the loader.
//   Keep the plugin's storage for its callback lifetime. The host must keep the core loaded until all
//   environments and plugin activity finish. Caller-owned plugin storage needs no cleanup.
//   Do not free an owning loader during DLL detach, since it calls FreeLibrary.
// - Plugin exports (Win32): AviSynth+ accepts name, _name@4 and name@4 for both C init entry points.
//   Classic 2.6 accepts only name@4 and name (g++ default and -Wl,--kill-at; not MSVC _name@4 without
//   a .def alias); see example_plugin.cpp.
// - API exports (Win32): 3.6.x through 3.7.1 exported V8 functions only under decorated names
//   (e.g. _avs_get_frame_props_ro@8); this loader leaves those entries NULL, so has_frame_props is
//   false even on the V9 3.7.1. Fixed in 3.7.2.
// - Helpers: AVSC_NO_DECLSPEC disables avisynth_c.h inline helpers that call the linked API.
//   Use lib-> function pointers, e.g. avs_new_video_frame_a and avs_get_pitch_p.
// - Linkage: loader functions are static inline in C, inline in C++.
//
// Interface versions: V3: classic 2.5; V6: classic 2.6 and AviSynth+ through 3.5.0; V7: 3.5.1;
// V8: 3.6.x/3.7.0; V9: 3.7.1/3.7.2; V10: 3.7.3; V11: 3.7.4; V11.1: 3.7.5;
// V14: 3.7.6; V14.1: 3.7.7 WIP. "AVS+" in the function list means all AviSynth+ versions.

#ifndef AVISYNTH_C_LOADER_H
#define AVISYNTH_C_LOADER_H

#if defined(__AVISYNTH_C__) && !defined(AVSC_NO_DECLSPEC)
#error "avisynth_c.h was included without AVSC_NO_DECLSPEC: include avisynth_c_loader.h first"
#endif

#ifndef AVSC_NO_DECLSPEC
#define AVSC_NO_DECLSPEC
#endif
#include "avisynth_c.h"
#include "avisynth_c_loader_common.h"

typedef struct AVS_Library AVS_Library;

struct AVS_Library {
  void* handle; // owned HMODULE/dlopen handle; NULL for a caller-owned plugin binding

  // Detected when binding by directly looking up the V14-only avs_create_map export.
  // Independent of the function table (which users may trim) and any fallbacks.
  int has_env_creation_error_reporting;

  // Interface version of the loaded AviSynth and the feature steps.
  // Filled by avs_library_create_script_environment (hosts) or avs_library_detect_plugin_env (plugins),
  // all 0 before.
  int interface_version;   // e.g. 6, 8, 11, 14
  int interface_bugfix;    // e.g. 1 for V11.1 (3.7.5) and V14.1 (3.7.7)
  int has_at_least_v8;     // 3.6.x, 3.7.0: frame property API entries, avs_new_video_frame_p.
                           // Do not use for C frame property access: it crashes in V8; require V9.
  int has_at_least_v9;     // 3.7.1, 3.7.2: C frame property access fixed, avs_is/make_property_writable,
                           // AVS_AEP_INTERFACE_VERSION/BUGFIX queries
  int has_at_least_v10;    // 3.7.3: includes the V9.1/V9.2 development fixes.
                           // AVS_VideoFrame pixel_type (avs_video_frame_get/amend_pixel_type),
                           // avs_new_video_frame_p(_a) with a non-NULL prop_src, avs_prop_get_data,
                           // C filters: an error of the child clip (fi->child->error) is passed on
  int has_at_least_v11;    // 3.7.4: 64-bit AVS_Value types (double, long) and related C API additions.
                           // Use the core's numeric accessors: V11.0 non-x86 64-bit storage differs
                           // from current inline helpers. Caller value support: has_64bit_values.
  int has_at_least_v11_1;  // 3.7.5: aligns non-x86 64-bit AVS_Value storage with current inline helpers.
  // V12/V13: WIP before V14; not set.
  int has_at_least_v14;    // 3.7.6: includes the V12/V13 development additions:
                           // avs_acquire/release_global_lock, avs_get_cpu_flags_ex, AVS_AEP_CACHESIZE_L2;
                           // YUV(A) 4:4:0, 4:1:1, 4:1:0, YA pixel formats.
                           // Function values, avs_invoke2/3(_try), avs_create_map; avs_prop_get_xxx fixes
                           // (NULL error pointer, avs_prop_get_clip), AVS_MT_NICE_FILTER error reporting
  int has_at_least_v14_1;  // 3.7.7 WIP: frame properties: _Range is the stored key, _ColorRange its alias
                           // (0 and 1 swapped). Before, AviSynth knew only _ColorRange

  // has_at_least_v9 and all V8/V9 frame property functions present (AVSC_LOADER_FRAME_PROP_FUNCTIONS):
  // a single check before using them. 0 on Win32 3.6.x-3.7.1 (decorated-only exports) and on V8.
  // Later fixes still depend on the version:
  // - avs_prop_get_data and avs_prop_get_data_size work from V10 (has_at_least_v10),
  // - avs_prop_get_clip from V14 (has_at_least_v14)
  // - before V14 pass a non-NULL error pointer to avs_prop_get_xxx.
  int has_frame_props;

  // The above version-dependent fields are all 0 until the version is detected

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

// ** Missing function fallback handling.**
// Fills the missing (NULL) entries which can be served without the loaded AviSynth, the same way as
// the C++ ABI (avisynth.h) does: see AVSC_LOADER_FALLBACKS in avisynth_c_loader_common.h.
// E.g. make avs_is_444 or avs_bits_per_component usable against classic AviSynth 2.6,
// or avs_is_440 before Avisynth 3.7.6.
// Called during binding. Returns the number of entries filled (a second call fills nothing).
AVSC_LOADER_INLINE int avs_library_set_fallbacks(AVS_Library* library)
{
  int filled = 0;
#define AVSC_LOADER_FALLBACK(name, fn) \
  if (library->name == NULL) { library->name = (fn); filled++; }
#define AVSC_LOADER_ALIAS(name, older) \
  AVSC_LOADER_FALLBACK(name, library->older != NULL ? library->older : avs_loader_fallback_false)
#define AVSC_LOADER_ALIAS_IF_EXISTS(name, older) \
  if (library->older != NULL) AVSC_LOADER_FALLBACK(name, library->older)
  AVSC_LOADER_FALLBACKS(AVSC_LOADER_FALLBACK, AVSC_LOADER_ALIAS, AVSC_LOADER_ALIAS_IF_EXISTS)
#undef AVSC_LOADER_ALIAS_IF_EXISTS
#undef AVSC_LOADER_ALIAS
#undef AVSC_LOADER_FALLBACK
  return filled;
}

// Fills caller-owned storage from a live handle; does not take ownership of or close the handle.
// Storage must not own a library reference. Clears it on failure. name: for the error text.
AVSC_LOADER_INLINE int avs_library_bind_handle(AVS_Library* library, void* handle,
  const char* name, char* error, size_t error_size)
{
  memset(library, 0, sizeof(*library));
#define AVSC_LOADER_LOAD(name) *(void**)&library->name = avs_loader_symbol(handle, #name);
  AVSC_LOADER_FUNCTIONS(AVSC_LOADER_LOAD)
#undef AVSC_LOADER_LOAD
  if (library->avs_create_script_environment == NULL) {
    avs_loader_not_avisynth_error(name, error, error_size);
    memset(library, 0, sizeof(*library));
    return 0;
  }
  library->has_env_creation_error_reporting = avs_loader_symbol(handle, "avs_create_map") != NULL;
  avs_library_set_fallbacks(library);
  return 1;
}

// Allocates the owning loader for avs_load_library_ex and avs_attach_library.
// Takes over the handle, including closing it on failure.
AVSC_LOADER_INLINE AVS_Library* avs_library_from_handle(void* handle, const char* name, char* error, size_t error_size)
{
  AVS_Library* library = (AVS_Library*)malloc(sizeof(AVS_Library));
  if (library == NULL) {
    if (error != NULL && error_size > 0)
      avs_loader_append_error(error, error_size, "out of memory");
    avs_loader_close(handle);
    return NULL;
  }
  if (!avs_library_bind_handle(library, handle, name, error, error_size)) {
    avs_loader_close(handle);
    free(library);
    return NULL;
  }
  library->handle = handle;
  return library;
}

// Loads AviSynth and all C API functions. Missing ones get a fallback where possible
// (avs_library_set_fallbacks), the others are NULL.
// library_path: UTF-8 file name or path; NULL: "avisynth" (Windows), "libavisynth.dylib" (macOS),
// "libavisynth.so" (others).
// error (optional, may be NULL): receives the reason of a failure as UTF-8 text, at most error_size
// bytes with the terminating zero (AVSC_LOADER_ERROR_SIZE is enough); an empty string on success. E.g.
// "avisynth: invalid library image or incompatible architecture for this 64 bit process (error 193)".
// Returns NULL if the library cannot be loaded, or if it is not AviSynth with a C interface.
// Free the result with avs_free_library.
AVSC_LOADER_INLINE AVS_Library* avs_load_library_ex(const char* library_path, char* error, size_t error_size)
{
  void* handle = avs_loader_open(library_path, error, error_size);
  if (handle == NULL)
    return NULL;
  return avs_library_from_handle(handle, library_path, error, error_size);
}

// Like avs_load_library_ex, but acquires a reference to an already-loaded AviSynth instead of loading one by name.
// Plugins should normally use the non-owning avs_bind_plugin_library below.
// On POSIX avs_load_library(NULL) loads "libavisynth.so" from the library search path, which may not exist
// (it is usually in the -dev packages only) or may be another copy than the running one, and the plugin
// would work with another AviSynth. On Windows both give the same instance.
// Windows: the module named "avisynth" (a renamed AviSynth.dll is not found). Linux, BSD, macOS: the loaded
// library named libavisynth.so[.N] / libavisynth[.N].dylib. Haiku: not supported.
// Returns NULL if there is none; load it by name as a fallback if you want:
//   lib = avs_attach_library(NULL, 0); if (!lib) lib = avs_load_library(path);
// Free with avs_free_library; other owners determine whether AviSynth stays loaded.
AVSC_LOADER_INLINE AVS_Library* avs_attach_library(char* error, size_t error_size)
{
  void* handle = avs_loader_attach(error, error_size);
  if (handle == NULL)
    return NULL;
  return avs_library_from_handle(handle, "AviSynth", error, error_size);
}

// Plugins: resolve the already-loaded core into caller-owned storage without retaining a reference.
// Call during plugin init, not DLL attach/detach. The host must keep the core loaded throughout plugin
// activity. Storage must be zero-initialized or previously bound by this function, never an owning
// loader from avs_load_library/avs_attach_library. Keep it for all callbacks; no cleanup is needed and
// avs_free_library must not be called on it. Discovery restrictions are the same as avs_attach_library.
// Returns nonzero on success (handle stays NULL); on failure clears the storage and writes error if supplied.
AVSC_LOADER_INLINE int avs_bind_plugin_library(AVS_Library* library, char* error, size_t error_size)
{
  void* handle;
  int bound;
  memset(library, 0, sizeof(*library));
  handle = avs_loader_attach(error, error_size);
  if (handle == NULL)
    return 0;
  bound = avs_library_bind_handle(library, handle, "AviSynth", error, error_size);
  avs_loader_close(handle);
  return bound;
}

// The same as avs_load_library_ex, without the error text.
AVSC_LOADER_INLINE AVS_Library* avs_load_library(const char* library_path)
{
  return avs_load_library_ex(library_path, NULL, 0);
}

// Only for heap-allocated owning loaders returned by avs_load_library/avs_attach_library.
AVSC_LOADER_INLINE void avs_free_library(AVS_Library* library)
{
  if (library == NULL)
    return;
  avs_loader_close(library->handle);
  free(library);
}

// Returns the interface version of the loaded AviSynth (e.g. 6, 8, 11, 14), 0 if unknown.
// bugfix (may be NULL) receives the bugfix version (V9 and above, otherwise 0).
// E.g. interface 11 with bugfix 1 is "V11.1" (3.7.5).
AVSC_LOADER_INLINE int avs_library_interface_version(const AVS_Library* library, AVS_ScriptEnvironment* env, int* bugfix)
{
  return avs_loader_query_version(library->avs_check_version, library->avs_get_env_property, env, bugfix);
}

// Records the exact interface version and sets flags for released milestones plus the current WIP.
// Development steps V9.1/V9.2/V12/V13 have no separate flags.
AVSC_LOADER_INLINE void avs_library_set_version(AVS_Library* library, int version, int bugfix)
{
  library->interface_version = version;
  library->interface_bugfix = bugfix;
  library->has_at_least_v8 = avs_loader_at_least(version, bugfix, 8, 0);
  library->has_at_least_v9 = avs_loader_at_least(version, bugfix, 9, 0);
  library->has_at_least_v10 = avs_loader_at_least(version, bugfix, 10, 0);
  library->has_at_least_v11 = avs_loader_at_least(version, bugfix, 11, 0);
  library->has_at_least_v11_1 = avs_loader_at_least(version, bugfix, 11, 1);
  library->has_at_least_v14 = avs_loader_at_least(version, bugfix, 14, 0);
  library->has_at_least_v14_1 = avs_loader_at_least(version, bugfix, 14, 1);
#define AVSC_LOADER_FRAME_PROP_PRESENT(name) && library->name != NULL
  library->has_frame_props = library->has_at_least_v9 AVSC_LOADER_FRAME_PROP_FUNCTIONS(AVSC_LOADER_FRAME_PROP_PRESENT);
#undef AVSC_LOADER_FRAME_PROP_PRESENT
}

// Detects the version of the loaded AviSynth through env.
// Fills interface_version, interface_bugfix and the has_at_least_xxx steps; returns the version.
AVSC_LOADER_INLINE int avs_library_detect_version(AVS_Library* library, AVS_ScriptEnvironment* env)
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
AVSC_LOADER_INLINE int avs_library_detect_plugin_env(AVS_Library* library, AVS_ScriptEnvironment* env, int called_from_init2)
{
  int version = avs_library_detect_version(library, env);
  library->has_64bit_values = called_from_init2 && version >= AVSC_LOADER_64BIT_VALUES_VERSION;
  return version;
}

// Returns nonzero if avs_create_script_environment failed: see avs_loader_env_failed in
// avisynth_c_loader_common.h (a refused request gives a non-NULL environment with a NULL core pointer).
AVSC_LOADER_INLINE int avs_library_env_failed(const AVS_Library* library, AVS_ScriptEnvironment* env)
{
  return avs_loader_env_failed(library->avs_get_error, env, library->has_env_creation_error_reporting);
}

// Hosts: creates the script environment with the highest interface version known by both
// the loaded DLL and the avisynth_c.h this loader was compiled with, and fills the version fields
// of the library struct. The request order and why: AVSC_LOADER_ENV_REQUESTS in avisynth_c_loader_common.h.
// Returns NULL on failure. Delete the result with lib->avs_delete_script_environment.
AVSC_LOADER_INLINE AVS_ScriptEnvironment* avs_library_create_script_environment(AVS_Library* library)
{
  static const int request[] = AVSC_LOADER_ENV_REQUESTS;
  int i;
  if (library->avs_create_script_environment == NULL)
    return NULL;
  for (i = 0; i < (int)(sizeof(request) / sizeof(request[0])); i++) {
    AVS_ScriptEnvironment* env = library->avs_create_script_environment(request[i]);
    if (!avs_library_env_failed(library, env)) {
      avs_library_detect_version(library, env); // the DLL's exact version, also if the DLL is newer than the header
      library->has_64bit_values = request[i] >= AVSC_LOADER_64BIT_VALUES_VERSION; // accepted only by a V11+ core
      return env;
    }
    if (env != NULL && library->avs_delete_script_environment != NULL)
      library->avs_delete_script_environment(env); // frees the wrapper only
  }
  return NULL;
}

#endif // AVISYNTH_C_LOADER_H
