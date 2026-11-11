// AviSynth+ C API dynamic loader, C++ variant
// (headers only, no source file to build; Windows, Linux, macOS, BSD, Haiku).
// Needs avisynth_c_loader_common.h next to it.
// For C use avisynth_c_loader.h.
//
// Copyright 2026 AviSynth+ project
// Same license and linking exception as avisynth_c.h.

// What it is for: using the AviSynth C API without linking avisynth.lib/libavisynth.
// Loads AviSynth at runtime for hosts, or attaches to an already-loaded AviSynth for plugins
// (subject to the platform restrictions documented below). Both variants resolve C API function
// pointers, provide fallbacks for older cores, and detect the interface version through an environment.
// load()/attach() own a library reference; bind_plugin() keeps only function pointers.
//
// The C API is the usual choice from C++ as well when the plugin or host must not depend on the
// compiler AviSynth was built with (MSVC, MinGW GCC, Clang): its ABI is plain C.
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

// Samples, not installed AviSynth headers (avisynth_c_loader.h was installed in 3.7.6):
// FilterSDK/c_loader in the source tree, Windows packages and installer SDK;
// share/doc/avisynth/examples/c_loader in a CMake install. Copy the chosen loader header and
// avisynth_c_loader_common.h into your project, as they are or trimmed. avisynth_c.h and its avs/
// folder must be on the include path; any avisynth_c.h from V14 (3.7.6) on works.
//
// Usage (host, e.g. an encoder or player):
//   #include "avisynth_c_loader.hpp" // instead of avisynth_c.h, it defines AVSC_NO_DECLSPEC
//
//   avsc::Library lib;          // empty, nothing loaded yet
//   if (!lib.load()) { ... AviSynth not found, lib.error() tells why (optional) ... }
//                               // load(): the default name ("avisynth", "libavisynth.so", ...) from the
//                               // library search path; load("C:/path/AviSynth.dll"): a given file
//   // Negotiate the environment version; see AVSC_LOADER_ENV_REQUESTS.
//   AVS_ScriptEnvironment* env = lib.create_script_environment();
//   if (!env) { ... }
//   if (lib.has_frame_props) { ... avs_prop_xxx, avs_get_frame_props_ro ... } // V9 and the functions are there
//   if (lib.at_least(14, 1)) { ... the same as has_at_least_v14_1 ... }
//   if (lib.has_64bit_values) { ... double and long (int64) values are kept ... }
//   AVS_Value none = avs_new_value_array(nullptr, 0); // "no arguments" (classic 2.6 does not take avs_void)
//   AVS_Value v = lib.avs_invoke(env, "Version", none, nullptr); // the C API, through lib
//   AVS_VideoFrame* f = lib.avs_new_video_frame(env, vi); // helpers of avisynth_c.h are members too
//   ...
//   lib.avs_delete_script_environment(env); // before lib goes out of scope
//
// Usage (C plugin written in C++): export both entry points through a shared initializer
// (AviSynth V10 and below knows only avisynth_c_plugin_init):
//   #include <atomic>
//   // bind_plugin() releases its temporary library reference; only pointers and version/capability fields remain.
//   // The C++ runtime destroys this static object at plugin unload; no explicit cleanup hook is needed.
//   static avsc::Library lib; // destruction clears local state, with no FreeLibrary/dlclose call
//   static std::atomic_flag init_lock = ATOMIC_FLAG_INIT;
//   static const char* init_plugin(AVS_ScriptEnvironment* env, bool from_init2) {
//     while (init_lock.test_and_set(std::memory_order_acquire)) {}
//     if (!lib.loaded()) {
//       if (!lib.bind_plugin()) {
//         init_lock.clear(std::memory_order_release);
//         return "...";
//       }
//       lib.detect_plugin_env(env, from_init2); // initialize once, read-only during callbacks
//     }
//     init_lock.clear(std::memory_order_release);
//     lib.avs_add_function(env, ...);
//     return "...";
//   }
//   extern "C" const char* AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment* env) {
//     return init_plugin(env, false);
//   }
//   extern "C" const char* AVSC_CC avisynth_c_plugin_init2(AVS_ScriptEnvironment* env) {
//     return init_plugin(env, true); // the plugin's functions get double and long values
//   }
//
// Notes:
// - Initialization: construction is empty; load()/attach()/bind_plugin() return false on failure
//   and error() provides diagnostics. Binding requires avs_create_script_environment.
//   loaded()/operator bool report usable binding, not ownership. Library is movable and noncopyable.
// - Synchronization: Library does not lock internally. The plugin pattern serializes binding and
//   version detection, then leaves shared fields read-only; registration still runs for every env.
//   Its file-scope atomic_flag needs no cleanup, TLS or local-static initialization guard.
//   A failed bind can be retried. The cached binding is not refreshed across core unload/reload.
// - Missing functions: fallbacks are installed where possible; other entries remain nullptr.
//   Examples: avs_is_444 aliases avs_is_yv24 on classic 2.6; V11 AVS_Value accessors have legacy
//   fallbacks. See AVSC_LOADER_FALLBACKS. Fallbacks (added in 3.7.7) make pointer presence unsuitable
//   for version detection; use interface_version/interface_bugfix and has_at_least_xxx.
// - API compatibility: require both the function pointer and its usable interface version.
//   has_frame_props checks both for the V8/V9 frame property set (one check is enough there).
//   AVSC_LOADER_FUNCTIONS documents introduction versions
//   and later fixes; see also "API functions fixed later than they appeared" in avisynth_c.h.
// - Version flags: both variants expose released milestones plus the current WIP, not the
//   development steps V9.1/V9.2/V12/V13. Exact comparisons use interface_version/interface_bugfix.
// - Environment creation: prefer create_script_environment() for version negotiation.
//   Direct avs_create_script_environment calls require env_failed(): unsupported requests can return
//   a non-NULL wrapper with a NULL core. V14+ reports this through avs_get_error; older cores may leave
//   the error unset. Do not call core-dependent functions through a failed wrapper.
// - 64-bit values: hosts must request V11+; plugins must export avisynth_c_plugin_init2.
//   Otherwise the core converts double/long values to float/int. has_64bit_values describes the
//   caller's value support, not just the core version.
// - Plugin attachment: use bind_plugin(), not default-name loading, which may fail or select another
//   copy on POSIX. It releases its temporary library reference before returning. See avs_loader_attach
//   for platform limits and macOS concurrency requirements; multiple loaded cores remain ambiguous.
// - Lifetime: release values, clips, frames and host-created environments before unload()/destruction.
//   Keep the plugin's Library for its callback lifetime. The host must keep the core loaded until all
//   environments and plugin activity finish. A non-owning Library can be destroyed during DLL detach:
//   it makes no FreeLibrary/dlclose call. Do not destroy an owning Library during DLL detach.
// - Plugin exports (Win32): AviSynth+ accepts name, _name@4 and name@4 for both C init entry points.
//   Classic 2.6 accepts only name@4 and name (g++ default and -Wl,--kill-at; not MSVC _name@4 without
//   a .def alias); see example_plugin.cpp.
// - API exports (Win32): 3.6.x through 3.7.1 exported V8 functions only under decorated names
//   (e.g. _avs_get_frame_props_ro@8); this loader leaves those entries nullptr, so has_frame_props
//   is false even on the V9 3.7.1. Fixed in 3.7.2.
// - Helpers: AVSC_NO_DECLSPEC disables avisynth_c.h inline helpers that call the linked API.
//   Library supplies member equivalents, e.g. lib.avs_new_video_frame and lib.avs_get_pitch.
// - Linkage: loader functions are inline; common C-compatible functions use AVSC_LOADER_INLINE.
//
// Interface versions: V3: classic 2.5; V6: classic 2.6 and AviSynth+ through 3.5.0; V7: 3.5.1;
// V8: 3.6.x/3.7.0; V9: 3.7.1/3.7.2; V10: 3.7.3; V11: 3.7.4; V11.1: 3.7.5;
// V14: 3.7.6; V14.1: 3.7.7 WIP. "AVS+" in the function list means all AviSynth+ versions.

#ifndef AVISYNTH_C_LOADER_HPP
#define AVISYNTH_C_LOADER_HPP

#if defined(__AVISYNTH_C__) && !defined(AVSC_NO_DECLSPEC)
#error "avisynth_c.h was included without AVSC_NO_DECLSPEC: include avisynth_c_loader.hpp first"
#endif

#ifndef AVSC_NO_DECLSPEC
#define AVSC_NO_DECLSPEC
#endif
#include "avisynth_c.h"
#include "avisynth_c_loader_common.h"

#include <cstring>
#include <utility>

namespace avsc {

// The public data of Library: the version fields and the C API function pointers.
// A separate struct only to make moving simple, use Library.
struct LibraryData {
  // Detected when binding by directly looking up the V14-only avs_create_map export.
  // Independent of the function table (which users may trim) and any fallbacks.
  bool has_env_creation_error_reporting = false;
  // has_at_least_v9 and all V8/V9 frame property functions present (AVSC_LOADER_FRAME_PROP_FUNCTIONS):
  // a single check before using them. False on Win32 3.6.x-3.7.1 (decorated-only exports) and on V8.
  // Later fixes still depend on the version:
  // - avs_prop_get_data and avs_prop_get_data_size work from V10 (has_at_least_v10),
  // - avs_prop_get_clip from V14 (has_at_least_v14)
  // - before V14 pass a non-NULL error pointer to avs_prop_get_xxx.
  bool has_frame_props = false;

  // Interface version of the loaded AviSynth and the feature steps.
  // Filled by create_script_environment (hosts) or detect_plugin_env (plugins), all 0 before.
  int interface_version = 0;       // e.g. 6, 8, 11, 14
  int interface_bugfix = 0;        // e.g. 1 for V11.1 (3.7.5) and V14.1 (3.7.7)
  bool has_at_least_v8 = false;    // 3.6.x, 3.7.0: frame property API entries, avs_new_video_frame_p.
                                   // Do not use for C frame property access: it crashes in V8; require V9.
  bool has_at_least_v9 = false;    // 3.7.1, 3.7.2: C frame property access fixed, avs_is/make_property_writable,
                                   // AVS_AEP_INTERFACE_VERSION/BUGFIX queries
  bool has_at_least_v10 = false;   // 3.7.3: includes the V9.1/V9.2 development fixes.
                                   // AVS_VideoFrame pixel_type (avs_video_frame_get/amend_pixel_type),
                                   // avs_new_video_frame_p(_a) with a non-NULL prop_src, avs_prop_get_data,
                                   // C filters: an error of the child clip (fi->child->error) is passed on
  bool has_at_least_v11 = false;   // 3.7.4: 64-bit AVS_Value types (double, long) and related C API additions.
                                   // Use the core's numeric accessors: V11.0 non-x86 64-bit storage differs
                                   // from current inline helpers. Caller value support: has_64bit_values.
  bool has_at_least_v11_1 = false; // 3.7.5: aligns non-x86 64-bit AVS_Value storage with current inline helpers.
  // V12/V13: WIP before V14; not set.
  bool has_at_least_v14 = false;   // 3.7.6: includes the V12/V13 development additions:
                                   // avs_acquire/release_global_lock, avs_get_cpu_flags_ex, AVS_AEP_CACHESIZE_L2;
                                   // YUV(A) 4:4:0, 4:1:1, 4:1:0, YA pixel formats.
                                   // Function values, avs_invoke2/3(_try), avs_create_map; avs_prop_get_xxx fixes
                                   // (NULL error pointer, avs_prop_get_clip), AVS_MT_NICE_FILTER error reporting
  bool has_at_least_v14_1 = false; // 3.7.7 WIP: frame properties: _Range is the stored key, _ColorRange its alias
                                   // (0 and 1 swapped). Before, AviSynth knew only _ColorRange

  // This caller gets 64 bit AVS_Value types: double and long (int64) values are kept, not converted
  // to float and 32 bit int.
  // Hosts: the environment was requested with version >= 11 (create_script_environment).
  // Plugins: AviSynth called avisynth_c_plugin_init2 (detect_plugin_env).
  // Not the same as has_at_least_v11: a V11+ core converts the values for a pre-V11 caller.
  bool has_64bit_values = false;

  // The C API functions, e.g. lib.avs_invoke(...). nullptr if missing and there is no fallback.
#define AVSC_LOADER_DECLARE(name) name##_func name = nullptr;
  AVSC_LOADER_FUNCTIONS(AVSC_LOADER_DECLARE)
#undef AVSC_LOADER_DECLARE
};

class Library : public LibraryData {
public:
  // Empty: call load(), attach() or bind_plugin() and check its result.
  Library() noexcept = default;

  ~Library() { unload(); }

  Library(const Library&) = delete;
  Library& operator=(const Library&) = delete;

  Library(Library&& other) noexcept { take(other); }
  Library& operator=(Library&& other) noexcept
  {
    if (this != &other) {
      unload();
      take(other);
    }
    return *this;
  }

  // Loads AviSynth and all C API functions; a library loaded before is unloaded first.
  // Missing functions get a fallback where possible (AVSC_LOADER_FALLBACKS in avisynth_c_loader_common.h),
  // the others are nullptr.
  // library_path: UTF-8 file name or path; nullptr: "avisynth" (Windows), "libavisynth.dylib" (macOS),
  // "libavisynth.so" (others).
  // Returns false if the library cannot be loaded, or if it is not AviSynth with a C interface
  // (no avs_create_script_environment); the Library is empty then, error() tells why.
  bool load(const char* library_path = nullptr) noexcept
  {
    unload();
    handle_ = avs_loader_open(library_path, error_, sizeof(error_));
    if (handle_ == nullptr)
      return false;
    return bind(library_path);
  }

  // Like load(), but acquires a reference to an already-loaded AviSynth instead of loading one by name.
  // Plugins should normally use the non-owning bind_plugin() below.
  // On POSIX load() with the default name opens "libavisynth.so" from the library search path, which may
  // not exist (it is usually in the -dev packages only) or may be another copy than the running one, and
  // the plugin would work with another AviSynth. On Windows both give the same instance.
  // Windows: the module named "avisynth" (a renamed AviSynth.dll is not found). Linux, BSD, macOS: the
  // loaded library named libavisynth.so[.N] / libavisynth[.N].dylib. Haiku: not supported.
  // Returns false if there is none (error() tells); load it by name as a fallback if you want:
  //   if (!lib.attach() && !lib.load(path)) { ... }
  // unload() releases this reference; other owners determine whether AviSynth stays loaded.
  bool attach() noexcept
  {
    unload();
    handle_ = avs_loader_attach(error_, sizeof(error_));
    if (handle_ == nullptr)
      return false;
    return bind("AviSynth");
  }

  // Plugins: resolve the already-loaded core without retaining a library reference.
  // Call during plugin init, not DLL attach/detach. The host must keep the core loaded throughout
  // plugin activity. On success handle() is nullptr, but loaded() is true and the API remains usable.
  // Discovery restrictions are the same as attach(). Any previous binding is released first.
  bool bind_plugin() noexcept
  {
    if (!attach())
      return false;
    avs_loader_close(handle_);
    handle_ = nullptr;
    return true;
  }

  // Optional: the reason of the last failed load/attach/bind as UTF-8 text, "" after success. E.g.
  // "avisynth: not found in the DLL search path, or a DLL it needs is missing
  // (e.g. the Visual C++ runtime) (error 126)" or
  // "avisynth: invalid library image or incompatible architecture for this 64 bit process (error 193)".
  // Other systems: the dlerror() text.
  const char* error() const noexcept { return error_; }

  // Releases any owned reference and clears the binding. A plugin binding makes no OS close call.
  // Delete the script environments created through it first.
  void unload() noexcept
  {
    avs_loader_close(handle_);
    handle_ = nullptr;
    static_cast<LibraryData&>(*this) = LibraryData();
  }

  bool loaded() const noexcept { return avs_create_script_environment != nullptr; }
  explicit operator bool() const noexcept { return loaded(); }

  // Owned HMODULE on Windows, dlopen handle elsewhere; nullptr for an empty or plugin binding.
  void* handle() const noexcept { return handle_; }

  // Hosts: creates the script environment with the highest interface version known by both the
  // loaded AviSynth and the avisynth_c.h this was compiled with, and fills the version fields.
  // The request order and why: AVSC_LOADER_ENV_REQUESTS in avisynth_c_loader_common.h.
  // Returns nullptr on failure. Delete the result with avs_delete_script_environment.
  AVS_ScriptEnvironment* create_script_environment() noexcept
  {
    static constexpr int request[] = AVSC_LOADER_ENV_REQUESTS;
    if (avs_create_script_environment == nullptr)
      return nullptr;
    for (int version : request) {
      AVS_ScriptEnvironment* env = avs_create_script_environment(version);
      // FIXME: AVSC_LOADER_ENV_REQUESTS granularity is too coarse.
      if (!env_failed(env)) {
        detect_version(env); // the exact version, also if the loaded AviSynth is newer than the header
        has_64bit_values = version >= AVSC_LOADER_64BIT_VALUES_VERSION; // accepted only by a V11+ core
        return env;
      }
      if (env != nullptr && avs_delete_script_environment != nullptr)
        avs_delete_script_environment(env); // frees the wrapper only
    }
    return nullptr;
  }

  // Plugins: call from the plugin init with the env AviSynth passed. Fills the version fields, and
  // has_64bit_values; returns the interface version.
  // called_from_init2: true if AviSynth called avisynth_c_plugin_init2; the loader cannot see it,
  // the plugin has to tell, see the example at the top.
  int detect_plugin_env(AVS_ScriptEnvironment* env, bool called_from_init2) noexcept
  {
    int version = detect_version(env);
    has_64bit_values = called_from_init2 && version >= AVSC_LOADER_64BIT_VALUES_VERSION;
    return version;
  }

  // Detects the version of the AviSynth behind env, fills the version fields; returns the interface version.
  int detect_version(AVS_ScriptEnvironment* env) noexcept
  {
    int bugfix = 0;
    int version = avs_loader_query_version(avs_check_version, avs_get_env_property, env, &bugfix);
    set_version(version, bugfix);
    return version;
  }

  // Records the exact interface version and sets flags for released milestones plus the current WIP.
  // Development steps V9.1/V9.2/V12/V13 have no separate flags.
  void set_version(int version, int bugfix) noexcept
  {
    interface_version = version;
    interface_bugfix = bugfix;
    has_at_least_v8 = at_least(8);
    has_at_least_v9 = at_least(9);
    has_at_least_v10 = at_least(10);
    has_at_least_v11 = at_least(11);
    has_at_least_v11_1 = at_least(11, 1);
    has_at_least_v14 = at_least(14);
    has_at_least_v14_1 = at_least(14, 1);
#define AVSC_LOADER_FRAME_PROP_PRESENT(name) && name != nullptr
    has_frame_props = has_at_least_v9 AVSC_LOADER_FRAME_PROP_FUNCTIONS(AVSC_LOADER_FRAME_PROP_PRESENT);
#undef AVSC_LOADER_FRAME_PROP_PRESENT
  }

  // True if the detected interface version is at least version.bugfix
  bool at_least(int version, int bugfix = 0) const noexcept
  {
    return avs_loader_at_least(interface_version, interface_bugfix, version, bugfix) != 0;
  }

  // True if avs_create_script_environment failed: a refused request gives a non-NULL environment
  // with a NULL core pointer, see avs_loader_env_failed in avisynth_c_loader_common.h.
  bool env_failed(AVS_ScriptEnvironment* env) const noexcept
  {
    return avs_loader_env_failed(avs_get_error, env, has_env_creation_error_reporting) != 0;
  }

  // ---- Helpers --------------------------------------------------------------------------------------
  // The inline helper functions of avisynth_c.h which call into the API are not defined with
  // AVSC_NO_DECLSPEC (the loader needs it), because they would call the linked API. These members are
  // their counterparts: the same names and arguments, working through the pointers of this Library, so
  // porting code written for the linked C API is a matter of adding "lib.". The C loader has no such
  // helpers on purpose: there the call through lib-> is one short line each.
  // They use old functions (V3/V6) only, so they work with every AviSynth.

  // A new frame for vi with the default alignment (AVS_FRAME_ALIGN). From V8 avs_new_video_frame_p_a
  // (with a frame property source) is the better choice. avs_new_frame is the older name of the same.
  AVS_VideoFrame* avs_new_video_frame(AVS_ScriptEnvironment* env, const AVS_VideoInfo* vi) const noexcept
  {
    return avs_new_video_frame_a(env, vi, AVS_FRAME_ALIGN);
  }
  AVS_VideoFrame* avs_new_frame(AVS_ScriptEnvironment* env, const AVS_VideoInfo* vi) const noexcept
  {
    return avs_new_video_frame_a(env, vi, AVS_FRAME_ALIGN);
  }

  // An AVS_Value holding the clip (a new reference, like avs_set_to_clip). Free it with avs_release_value.
  AVS_Value avs_new_value_clip(AVS_Clip* clip) const noexcept
  {
    AVS_Value v;
    avs_set_to_clip(&v, clip);
    return v;
  }

  // True if the two formats are the same; the 4:2:0 8 bit planar YV12 and I420 count as one.
  int avs_is_same_colorspace(const AVS_VideoInfo* x, const AVS_VideoInfo* y) const noexcept
  {
    return (x->pixel_type == y->pixel_type) || (avs_is_yv12(x) && avs_is_yv12(y));
  }

  // The default plane (AVS_DEFAULT_PLANE: Y, or the packed RGB/YUY2 data) of a frame. Deprecated names in
  // avisynth_c.h (AVSC_ALLOW_DEPRECATED), kept here as they are short: the plane-aware avs_get_xxx_p
  // functions are the general ones.
  int avs_get_pitch(const AVS_VideoFrame* f) const noexcept { return avs_get_pitch_p(f, AVS_DEFAULT_PLANE); }
  int avs_get_row_size(const AVS_VideoFrame* f) const noexcept { return avs_get_row_size_p(f, AVS_DEFAULT_PLANE); }
  int avs_get_height(const AVS_VideoFrame* f) const noexcept { return avs_get_height_p(f, AVS_DEFAULT_PLANE); }
  const BYTE* avs_get_read_ptr(const AVS_VideoFrame* f) const noexcept { return avs_get_read_ptr_p(f, AVS_DEFAULT_PLANE); }
  BYTE* avs_get_write_ptr(AVS_VideoFrame* f) const noexcept { return avs_get_write_ptr_p(f, AVS_DEFAULT_PLANE); }

  // Alternative names of avs_release_video_frame and avs_copy_video_frame.
  void avs_release_frame(AVS_VideoFrame* f) const noexcept { avs_release_video_frame(f); }
  AVS_VideoFrame* avs_copy_frame(AVS_VideoFrame* f) const noexcept { return avs_copy_video_frame(f); }

  // The CPU flags as 64 bit value (AVS_CPUF_xxx), on every AviSynth. avs_get_cpu_flags_ex exists from V12
  // only; before that this returns the 32 bit avs_get_cpu_flags result, zero extended (the flags above
  // 32 bits are new in V12 anyway). So no nullptr check is needed. (The name has no avs_ prefix: that
  // one is the function pointer, which is nullptr before V12.)
  int64_t get_cpu_flags_ex(AVS_ScriptEnvironment* env) const noexcept
  {
    if (avs_get_cpu_flags_ex != nullptr)
      return avs_get_cpu_flags_ex(env);
    return (int64_t)(unsigned int)avs_get_cpu_flags(env);
  }

private:
  void* handle_ = nullptr;
  char error_[AVSC_LOADER_ERROR_SIZE] = {};

  void take(Library& other) noexcept
  {
    handle_ = std::exchange(other.handle_, nullptr);
    static_cast<LibraryData&>(*this) = std::exchange(static_cast<LibraryData&>(other), LibraryData());
    std::memcpy(error_, other.error_, sizeof(error_));
    other.error_[0] = '\0';
  }

  // After handle_ is set: gets all functions, checks that it is AviSynth, adds the fallbacks.
  // name: for the error text. Unloads and returns false if it is not AviSynth.
  bool bind(const char* name) noexcept
  {
#define AVSC_LOADER_LOAD(fn) fn = reinterpret_cast<fn##_func>(avs_loader_symbol(handle_, #fn));
    AVSC_LOADER_FUNCTIONS(AVSC_LOADER_LOAD)
#undef AVSC_LOADER_LOAD
    if (avs_create_script_environment == nullptr) {
      avs_loader_not_avisynth_error(name, error_, sizeof(error_));
      unload();
      return false;
    }
    // Flag is set simply after the existence of a V14-new function.
    has_env_creation_error_reporting = avs_loader_symbol(handle_, "avs_create_map") != nullptr;
    set_fallbacks();
    return true;
  }

  // ** Missing function fallback handling.**
  // Fills the missing (nullptr) entries which can be served without the loaded AviSynth, the same way as
  // the C++ ABI (avisynth.h) does: see AVSC_LOADER_FALLBACKS in avisynth_c_loader_common.h.
  // E.g. make avs_is_444 or avs_bits_per_component usable against classic AviSynth 2.6,
  // or avs_is_440 before Avisynth 3.7.6.
  void set_fallbacks() noexcept
  {
#define AVSC_LOADER_FALLBACK(name, fn) \
    if (name == nullptr) name = (fn);
#define AVSC_LOADER_ALIAS(name, older) \
    AVSC_LOADER_FALLBACK(name, older != nullptr ? older : avs_loader_fallback_false)
#define AVSC_LOADER_ALIAS_IF_EXISTS(name, older) \
    if (older != nullptr) AVSC_LOADER_FALLBACK(name, older)
    AVSC_LOADER_FALLBACKS(AVSC_LOADER_FALLBACK, AVSC_LOADER_ALIAS, AVSC_LOADER_ALIAS_IF_EXISTS)
#undef AVSC_LOADER_ALIAS_IF_EXISTS
#undef AVSC_LOADER_ALIAS
#undef AVSC_LOADER_FALLBACK
  }
};

} // namespace avsc

#endif // AVISYNTH_C_LOADER_HPP
