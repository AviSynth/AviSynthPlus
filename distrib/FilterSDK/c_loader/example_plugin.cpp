// Example C plugin written in C++ with avisynth_c_loader.hpp: the smallest one which shows
// - the loader in a plugin (no AviSynth.lib / libavisynth linked),
// - the two entry points: avisynth_c_plugin_init2 (AviSynth V11+, 3.7.4+) and avisynth_c_plugin_init,
// - serialized initialization, followed by read-only access from plugin callbacks,
// - 64 bit values: double and long (int64) reach the plugin only through init2.
// For a clip filter see the SimpleSample plugins of the FilterSDK documentation.
//
// The script functions:
//   LoaderSampleInfo()  string: interface version, the entry point AviSynth called, and the type
//                       Pi() comes back with through avs_invoke ("double" through init2, else "float")
//   LoaderSampleTwice(n) n * 2. Through init2 n may be a long (e.g. 3000000000) and the result is a
//                       long; through init AviSynth converts n to a 32 bit int first.
//
// The parameter letters are the same for both entry points: "i" and "f" (there are no "l" and "d"
// letters). From V11 an "i" parameter accepts a long and an "f" parameter a double as well; whether
// the plugin sees them as 64 bit depends only on the entry point AviSynth called. Even then the type
// depends on the value: 5 stays an int ('i'), 3000000000 is a long ('l'), 1.5 a double ('d'), and an
// integer given to an "f" parameter stays an integer. So do not test the type letter: read the
// parameter with avs_get_as_long / avs_get_as_float, which convert any numeric type.
//
// Build, e.g.:
//   Windows x64: g++ -std=c++17 -shared -I<avisynth include dir> example_plugin.cpp -o LoaderSample.dll
//                cl /std:c++17 /EHsc /LD /I<avisynth include dir> example_plugin.cpp /Fe:LoaderSample.dll
//   Win32 (x86): the entry points are __stdcall and exported decorated: MSVC as _name@4, g++ as name@4
//                (with -Wl,--kill-at: name). AviSynth+ finds any of name, _name@4 and name@4, so nothing
//                more is needed. Classic AviSynth 2.6 (LoadCPlugin) looks for name@4 and name only: the
//                g++ builds load there, an MSVC build needs a .def file with the line
//                  avisynth_c_plugin_init@4 = _avisynth_c_plugin_init@4
//                (classic 2.6 does not know init2, no alias needed for it).
//   Linux:       g++ -std=c++17 -shared -fPIC -I<avisynth include dir> example_plugin.cpp -o libloadersample.so -ldl
// Script:
//   LoadPlugin("LoaderSample.dll")
//   Subtitle(LoaderSampleInfo() + ", Twice(3000000000) = " + String(LoaderSampleTwice(3000000000)))

#include <cstdio>
#include <atomic>
#include "avisynth_c_loader.hpp"

// bind_plugin() releases its temporary library reference; only pointers and version/capability fields remain.
// The C++ runtime destroys this static object at plugin unload; no explicit cleanup hook is needed.
// Destruction clears local state, with no FreeLibrary/dlclose call. The host owns the core's lifetime.
static avsc::Library lib;

// File-scope atomic initialization needs no TLS or thread-safe local-static initialization.
static std::atomic_flag init_lock = ATOMIC_FLAG_INIT;
// Set with the table before any function is registered; read-only afterward.
static bool from_init2 = false;

// LoaderSampleInfo()
static AVS_Value AVSC_CC loader_sample_info(AVS_ScriptEnvironment* env, AVS_Value args, void* user_data)
{
  (void)args; (void)user_data;

  // avs_invoke results follow the same rule as the parameters: Pi() is a double through init2,
  // a float (converted by AviSynth) through init.
  AVS_Value expr, pi;
  lib.avs_set_to_string(&expr, "Pi()");
  pi = lib.avs_invoke(env, "Eval", expr, nullptr);
  const char* pi_type = lib.avs_val_is_floatf_strict(pi) ? "float" : "double";
  lib.avs_release_value(pi); // an invoke result is owned by the caller

  char text[200];
  std::snprintf(text, sizeof(text), "AviSynth interface %d.%d, called through %s, 64 bit values: %s, Pi() is %s",
    lib.interface_version, lib.interface_bugfix, from_init2 ? "init2" : "init",
    lib.has_64bit_values ? "yes" : "no", pi_type);

  // The returned string must outlive this call: let AviSynth keep a copy.
  AVS_Value result;
  lib.avs_set_to_string(&result, lib.avs_save_string(env, text, -1));
  return result;
}

// LoaderSampleTwice(int n)
static AVS_Value AVSC_CC loader_sample_twice(AVS_ScriptEnvironment* env, AVS_Value args, void* user_data)
{
  (void)env; (void)user_data;
  AVS_Value n = lib.avs_get_array_elt(args, 0);
  AVS_Value result;
  if (lib.has_64bit_values) {
    lib.avs_set_to_long(&result, lib.avs_get_as_long(n) * 2); // n may be a long, the result is one
  }
  else {
    // 32 bit only: AviSynth already cut a long n to int (3000000000 arrives as -1294967296).
    // Multiplied in 64 bit (no int overflow), the result is cut to 32 bit as well.
    const int64_t twice = (int64_t)lib.avs_get_as_int(n) * 2;
    lib.avs_set_to_int(&result, (int)twice);
  }
  return result;
}

static const char* init_plugin(AVS_ScriptEnvironment* env, bool called_from_init2)
{
  while (init_lock.test_and_set(std::memory_order_acquire)) {}

  // The AviSynth which is loading the plugin, on every system. (load() would open "libavisynth.so" from
  // the library search path elsewhere than Windows: it may not exist or may be another copy.)
  if (!lib.loaded()) {
    if (!lib.bind_plugin()) {
      init_lock.clear(std::memory_order_release);
      return "LoaderSample: AviSynth is not loaded"; // lib.error() would tell why
    }
    lib.detect_plugin_env(env, called_from_init2);
    from_init2 = called_from_init2;
  }
  init_lock.clear(std::memory_order_release);

  lib.avs_add_function(env, "LoaderSampleInfo", "", loader_sample_info, nullptr);
  lib.avs_add_function(env, "LoaderSampleTwice", "i", loader_sample_twice, nullptr);
  return "LoaderSample: C plugin example for the C API loader";
}

// AviSynth V10 and below, or a plugin without init2: 64 bit values are converted to 32 bit for it.
AVSC_EXPORT const char* AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment* env)
{
  return init_plugin(env, false);
}

// AviSynth V11+ (3.7.4+) looks for this first. When it is exported, the plugin's functions get the 64
// bit values as they are (parameters, avs_invoke results). Both entry points use the same initializer.
AVSC_EXPORT const char* AVSC_CC avisynth_c_plugin_init2(AVS_ScriptEnvironment* env)
{
  return init_plugin(env, true);
}
