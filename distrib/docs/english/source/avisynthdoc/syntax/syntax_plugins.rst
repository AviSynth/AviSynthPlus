
AviSynth Syntax - Plugins
=========================

.. toctree::
    :maxdepth: 3

.. contents:: Table of contents

With these functions you can add external functions to AviSynth or
get/set autoload directories.

In general, where applicable:

- ``utf8`` (optional, boolean) - when ``true``, the supplied directory or file name 
  string is interpreted as UTF-8. When ``false``, on Windows the string is interpreted
  as ANSI (system code page) and converted internally to UTF-8 before being used.
- Default behaviour:

  * On non-Windows platforms (POSIX, macOS, Linux) the system uses UTF-8 by default,
    so the function behaves as UTF-8-only and the ``utf8`` parameter is effectively
    transparent (default === ``true``).
  * On Windows the historical default is ANSI. If you do not pass ``utf8`` the
    function will behave in the legacy ANSI manner (default === ``false``). To pass
    a UTF-8 encoded directory path from a script on Windows, pass ``utf8=true``.
  * When the Windows process is built or launched with UTF-8 manifest (making the
    process use UTF-8 as the default ANSI code page), the behaviour is transparent
    UTF-8 and supplying ``utf8`` is not required.

Manual plugin loading
---------------------

LoadPlugin
~~~~~~~~~~
::

    LoadPlugin ("filename" [, ...] [, bool utf8=false])

Loads one or more external avisynth plugins (DLLs).

Plugins can be either C or C++ plugins, they are autodetected.

Windows historically uses ANSI (system code page) for script string parameters,
so scripts that need to supply UTF-8 paths must indicate this explicitly. The
``utf8`` parameter allows a script to unambiguously tell the function that the
provided filename path is UTF-8 encoded. On UTF-8-native systems (POSIX or Windows
with UTF-8 manifest) the conversion step is unnecessary and thus transparent, the
parameter is ignored.


LoadVirtualDubPlugin
~~~~~~~~~~~~~~~~~~~~
::

    LoadVirtualDubPlugin ("filename", "filtername", preroll)

This loads a plugin written for VirtualDub. "filename" is the name of the
.vdf file. After calling this function, the filter will be known as
"filtername" in avisynth. 

Old VirtualDub filters only supports RGB32, but then some came with YUV support
(a newer API). The most compatible way of using them is using ``ConvertToRGB32``
(``ConvertToRGB`` won't suffice for an RGB24 source).

Some filters output depends on previous frames; for those preroll should be
set to at least the number of frames the filter needs to pre-process to fill
its buffers and/or updates its internal variables.


LoadVFAPIPlugin (removed, not part of AviSynth+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    LoadVFAPIPlugin ("filename", "filtername")

This allows you to use VFAPI plugins (TMPGEnc import plugins).


LoadCPlugin
~~~~~~~~~~~
Load_Stdcall_Plugin
~~~~~~~~~~~~~~~~~~~
::

    LoadCPlugin ("filename" [, ...])
    Load_Stdcall_Plugin ("filename" [, ...])

In Avisynth+ these function were kept for script compatibility reasons.

LoadPlugin can load any type of (C or C++) plugins by automatically detecting 
their types.

Loads so called Avisynth C-plugins (DLLs).
Load_Stdcall_Plugin() is an alias for LoadCPlugin().
C-plugins are created on pure C language and use special "AviSynth C API"
(unlike ordinary Avisynt plugins which are created with MS C++). C-plugins
must be loaded with LoadCPlugin() or Load_Stdcall_Plugin().

Kevin provides a LoadCPlugin.dll that overloads the LoadCPlugin() verb to
support plugins compiled using the C subroutine calling sequence, use
Load_Stdcall_Plugin() to load stdcall calling sequence plugins when using
Kevins version. Advice: keep these plugins outside your auto plugin loading
directory to prevent crashes. `[discussion]`_ `[AVISynth C API (by
kevina20723)]`_


Autoload helper functions
-------------------------

AddAutoloadDir
~~~~~~~~~~~~~~
::

    AddAutoloadDir (string "directory", bool toFront = true, bool utf8 = false)

This function appends an extra directory to the autoload directory list. The plugins
are searched in the order the directories appear in the list. Setting the optional 
``toFront`` parameter to false will put the directory at the end of 
the list (the lowest priority place).

Windows historically uses ANSI (system code page) for script string parameters,
so scripts that need to supply UTF-8 paths must indicate this explicitly. The
``utf8`` parameter allows a script to unambiguously tell the function that the
provided directory path is UTF-8 encoded.

On UTF-8-native systems (POSIX or Windows with UTF-8 manifest) the conversion step
is unnecessary and thus transparent, the parameter is ignored.

``directory`` parameter can contain macros. These are expanded to their actual values
during the operation.

- Folder names maintained by Avisynth instance:

  * ``SCRIPTDIR`` from Avisynth variable ``$ScriptDirUtf8$``
  * ``MAINSCRIPTDIR`` from Avisynth variable ``$MainScriptDirUtf8$``
  * ``PROGRAMDIR`` the actual folder of the host executable 
  
- Windows specific: registry-backed macros that can contain plugin folder paths 

  - x86 (Intel 32 or 64 bit builds)

    * ``USER_PLUS_PLUGINS`` from ``HKCU\Software\Avisynth\PluginDir+`` (or GNU C Build: ``HKCU\Software\Avisynth\PluginDir+GCC``)
    * ``MACHINE_PLUS_PLUGINS`` from ``HKLM\Software\Avisynth\PluginDir+`` (or GNU C Build:  ``HKLM\Software\Avisynth\PluginDir+GCC``)
    * ``USER_CLASSIC_PLUGINS`` from ``HKCU\Software\Avisynth\PluginDir2_5`` (GNU C Build: not used, removed)
    * ``MACHINE_CLASSIC_PLUGINS`` from ``HKLM\Software\Avisynth\PluginDir2_5`` (GNU C Build: not used, removed)
  
  - other (e.g. ARM)
  
    * ``USER_PLUS_PLUGINS`` from ``HKCU\Software\Avisynth\PluginDir+``
    * ``MACHINE_PLUS_PLUGINS`` from ``HKLM\Software\Avisynth\PluginDir+``
    * ``USER_CLASSIC_PLUGINS`` from ``HKCU\Software\Avisynth\PluginDir2_5``
    * ``MACHINE_CLASSIC_PLUGINS`` from ``HKLM\Software\Avisynth\PluginDir2_5``

  A macro whose registry entry does not exist is removed (the folder is not added).


ClearAutoloadDirs
~~~~~~~~~~~~~~~~~
::

    ClearAutoloadDirs

Removes everything from the plugin autoload directory list, making a clean 
start for your custom environment.

ListAutoloadDirs
~~~~~~~~~~~~~~~~
::

    ListAutoloadDirs(bool utf8 = false)

Returns a LF (0x0A, \\n) separated list of the currently set autoload directories.
The multiline string can be displayed with "Text" or "Subtitle" directly.

Windows historically uses ANSI (system code page) for script string parameters,
so scripts that need to obtain path in UTF-8 must indicate this explicitly. The
``utf8`` parameter allows a script to unambiguously tell the function that the
expected directory path list is UTF-8 encoded. On UTF-8-native systems (POSIX or Windows
with UTF-8 manifest) the conversion step is unnecessary and thus transparent, the
parameter is ignored.

::

    # This works for ANSI-only hosts as well
    AddAutoLoadDir("d:\20251116_Utf8pathImportError_Утка Πάπια\", utf8=true)
    AddAutoLoadDir("MAINSCRIPTDIR\myplugins")
    AddAutoLoadDir("SCRIPTDIR\myplugins2")
    AddAutoLoadDir("PROGRAMDIR\extraplugins")
    AddAutoLoadDir("MACHINE_PLUS_PLUGINS\myplugins3")
    SubTitle(ListAutoLoadDirs(utf8=true), align=7, size=20, utf8=true)

AutoloadPlugins
~~~~~~~~~~~~~~~
::

    AutoloadPlugins

Initiates plugin autoloading, if it did not happened so far.


Plugin autoload and name precedence (Historical, Avisynth v2)
-------------------------------------------------------------

It is possible to put all plugins and script files with user-defined
functions or (global) variables in a directory from where all files with the
extension .AVSI (**v2.08, v2.5**, the type was .AVS in **v2.05-2.07**) and
.DLL are loaded at startup, unloaded and then loaded dynamically as the
script needs them.

.AVSI scripts in this directory should only contain function definitions and
global variables, no main processing section (else strange errors may occur),
it also is not recommended to put other files in that directory.

The directory is stored in the registry (the registry key has changed for
**v2.5**). You can use double-clicking a .REG-file with the following lines
to set the path (of course inserting your actual path):
::

    REGEDIT4

    [HKEY_LOCAL_MACHINE\SOFTWARE\Avisynth]
    "plugindir2_5"="c:\\program files\\avisynth 2.5\\plugins"

The order in which function names take precedence is as follows:
::

    user-defined function (always have the highest priority)
       plugin-function (have higher priority than built-in functions, they will override a built-in function)
          built-in function

Inside those groups the function loaded at last takes precedence, there is no
error in a namespace conflict.


Plugin autoload and conflicting function names (v2.55)
------------------------------------------------------

Starting from v2.55 there is DLLName_function() support. The problem is that
two plugins can have different functions which are named the same. To call
the needed one, DLLName_function() support is added. It auto-generates the
additional names both for auto-loaded plugins and for plugins loaded with
LoadPlugin.

**Some examples:**

::

    # using fielddeinterlace from decomb510.dll
    AviSource("D:\captures\jewel.avi")
    decomb510_fielddeinterlace(blend=false)

Suppose you have  the plugins mpeg2dec.dll and mpeg2dec3.dll in your auto
plugin dir, and you want to load a d2v file with mpeg2dec.dll (which outputs
YUY2):

::

    # using mpeg2source from mpeg2dec.dll
    mpeg2dec_mpeg2source("F:\From_hell\from_hell.d2v")

or with mpeg2dec3.dll (which outputs YV12):

::

    # using mpeg2source from mpeg2dec3.dll
    mpeg2dec3_mpeg2source("F:\From_hell\from_hell.d2v")

Changelog
~~~~~~~~~
+----------------+------------------------------------------------------------+
| Version        | Changes                                                    |
+================+============================================================+
| Avisynth 3.7.6 | | Added utf8 parameter to AddAutoLoadDir, mention folder   |
|                |   macros in documentation                                  |
|                | | Added utf8 parameter to ListAutoLoadDirs                 |
|                | | Added utf8 parameter to LoadPlugin                       |
+----------------+------------------------------------------------------------+

$Date: 2025/11/18 11:38:00 $

.. _[discussion]: http://forum.doom9.org/showthread.php?s=&threadid=58840
.. _[AVISynth C API (by kevina20723)]:
    http://kevin.atkinson.dhs.org/avisynth_c/
