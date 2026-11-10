# AviSynth+ C API dynamic loader (sample)

Loads AviSynth (`AviSynth.dll`, `libavisynth.so`, `libavisynth.dylib`) at run time, without linking
`AviSynth.lib`/`libavisynth`, and gives access to every C API function. Windows, Linux, macOS, BSD, Haiku.

- Functions missing from an older AviSynth get a fallback where one is possible (e.g. `avs_is_444` is
  `avs_is_yv24` on classic AviSynth 2.6, the V11 `AVS_Value` functions work on older versions); the others
  are NULL.
- It tells the interface version of the loaded AviSynth (`has_at_least_v9` ... `has_at_least_v14_1`) and
  whether the caller gets 64 bit values (`has_64bit_values`).
- Hosts: it creates the script environment with the highest version both sides know.
- Plugins: non-owning binding resolves the already-loaded core without retaining a library reference.
  The host keeps the core loaded until all environments and plugin activity finish.

This is a sample, not one of the installed AviSynth headers (in 3.7.6 `avisynth_c_loader.h` was installed
with them). Copy the files into your project, as they are or trimmed.

| File | |
|---|---|
| `avisynth_c_loader.hpp` | C++17: class `avsc::Library`, releases any owned reference in its destructor. Hosts use `load()`, plugins use `bind_plugin()`. |
| `avisynth_c_loader.h` | C11, compiles as C++ too: hosts use `avs_load_library`/`avs_free_library`; plugins use `avs_bind_plugin_library` with static `AVS_Library` storage, no cleanup needed. |
| `avisynth_c_loader_common.h` | Needed by both: function list, fallbacks, version query, platform calls. |
| `example_host.cpp`, `example_host.c` | Minimal hosts, one for each. |
| `example_plugin.cpp` | Minimal C plugin in C++: the loader in a plugin, the `avisynth_c_plugin_init2` / `avisynth_c_plugin_init` entry points and 64 bit values (long, double). For a clip filter see the SimpleSample plugins of the FilterSDK documentation. |

Requirements: `avisynth_c.h` and its `avs/` folder on the include path, from V14 (3.7.6) on. On Linux link
with `-ldl` (glibc before 2.34).

The plugin patterns serialize initialization with a file-scope C11/C++ atomic flag, then leave the
shared table and capability fields read-only. No TLS, optional C11 `<threads.h>`, or synchronization
cleanup is needed. MSVC C atomics may require `/experimental:c11atomics`; the `v141_xp` C compiler
does not support C11, but its C++ compiler supports the C++ atomic pattern. The cached binding is not
refreshed if the core unloads/reloads while the plugin remains mapped. XP still requires a compatible
SDK and runtime; the initialization guard itself uses only lock-free atomic operations.

The details (versions, fallbacks, failed environments, plugin usage) are in the comments of the headers
and in the documentation: FilterSDK, C API, "Dynamic loading".

Where to find it: `distrib/FilterSDK/c_loader` in the source tree; `FilterSDK/c_loader` in the Windows
packages and in the SDK of the Windows installer; `share/doc/avisynth/examples/c_loader` after a CMake
install.
