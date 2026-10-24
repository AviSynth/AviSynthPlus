@echo off
setlocal EnableDelayedExpansion

rem ============================================================
rem  AviSynth+ Dev Build Script  (Visual Studio 2026)
rem
rem  Produces a set of developer binary builds for testing and
rem  distribution to beta testers.  This is NOT an official
rem  release (no installer, no compiled HTML docs).
rem
rem  Up to 9 configurations are built in sequence:
rem    1. v141_xp  Win32  -- Windows XP 32-bit (SSE2, v141_xp toolset)   [optional]
rem    2. v141_xp  x64    -- Windows XP 64-bit (v141_xp toolset)        [optional]
rem    3. ClangCL  Win32  -- 32-bit, Clang/LLVM front-end
rem    4. ClangCL  x64    -- 64-bit, Clang/LLVM front-end
rem    5. v145     Win32  -- 32-bit, current MSVC (VS 2026 native)
rem    6. v145     x64    -- 64-bit, current MSVC (VS 2026 native)
rem    7. v143     x64    -- 64-bit, Win7-compat MSVC (VS 2022 toolset)
rem    8. v143     Win32  -- 32-bit, Win7-compat MSVC (VS 2022 toolset)
rem    9. MinGW    ARM64  -- Windows on ARM64 (aarch64), cross-compiled
rem
rem  The two v141_xp builds (and their folder/dependency setup) are
rem  controlled by the BUILD_XP switch below.  The v141_xp toolset is
rem  not part of VS2026 any more -- it is only available in VS2022 --
rem  so the XP builds are configured with the VS 2022 generator.
rem
rem  The ARM64 build is controlled by the BUILD_ARM64 switch below; it
rem  needs the llvm-mingw toolchain (see LLVM_MINGW_ROOT in the
rem  DETECTION block below).
rem
rem  The version, the build date and the revision number are NOT set by
rem  hand -- they are detected (see the CONFIGURATION block).
rem
rem  USAGE: Run from any directory -- paths are absolute.
rem         Optionally pass a build tag:  build_release.bat beta
rem ============================================================


rem ============================================================
rem  CONFIGURATION  -- update before each dev build run
rem ============================================================

rem  Which variants to build.  Set a switch to 0 to skip that group --
rem  handy when iterating on a single variant, since the XP build alone
rem  takes almost an hour.  The groups are:
rem    BUILD_XP     the two v141_xp (Windows XP) builds
rem    BUILD_CLANG  the two ClangCL builds
rem    BUILD_V145   the two v145 (current MSVC) builds
rem    BUILD_WIN7   the two v143 (Windows 7) builds
rem    BUILD_ARM64  the MinGW ARM64 (aarch64) build
rem  Only the variants that are switched on get their destination folder,
rem  their dependency checks and a build slot.
rem
rem  The v141_xp toolset was removed from VS2026, so the XP builds need
rem  the VS2022 installation; on a VS2026-only machine they cannot be run.
rem  Every switch can also be set from the environment before the run, e.g.
rem      set BUILD_CLANG=0 && set BUILD_V145=0 && build_release.bat
rem  to build a single variant.
if not defined BUILD_XP    set "BUILD_XP=0"
if not defined BUILD_CLANG set "BUILD_CLANG=1"
if not defined BUILD_V145  set "BUILD_V145=1"
if not defined BUILD_WIN7  set "BUILD_WIN7=1"

rem  The Windows-on-ARM64 (aarch64) build.  It is on by default, but it
rem  needs the llvm-mingw toolchain (see LLVM_MINGW_ROOT in the DETECTION
rem  block below): when that is missing, the variant is skipped with a
rem  warning -- unless the switch was set explicitly, in which case the
rem  missing toolchain is an error.
set _OVR_BUILD_ARM64=%BUILD_ARM64%
if not defined BUILD_ARM64 set "BUILD_ARM64=1"

rem  Build identification
rem  -----------------------------------------------------------------------
rem  AVS_VER / BUILD_DATE / BUILD_RNUM are derived automatically -- there is
rem  nothing to edit here before a build:
rem
rem    AVS_VER     major.minor.bugfix, read from avs_core\core\version.h.in
rem                (the same source the top-level CMakeLists.txt uses), so
rem                the release folder name can never disagree with the
rem                binaries inside it
rem    BUILD_DATE  the day the build runs (YYYYMMDD, local time)
rem    BUILD_RNUM  the number of commits in the repository (git rev-list
rem                --count HEAD) -- the same counter the built DLLs report
rem                as AVS_SEQREV
rem
rem  BUILD_TAG stays manual: it labels the build series (dev/beta/rc...).
rem
rem  Every one of the four can be overridden -- an already defined
rem  variable wins over the detected value, and the tag can also be given
rem  on the command line:
rem
rem      set BUILD_TAG=beta
rem      build_release.bat
rem  or
rem      build_release.bat beta
rem
rem  The values used are echoed at the start of the run.
rem  -----------------------------------------------------------------------
if not defined BUILD_TAG set "BUILD_TAG=tst"

rem  Remember anything that was set in the environment, so the detected
rem  values do not overwrite it.
set _OVR_AVS_VER=%AVS_VER%
set _OVR_BUILD_DATE=%BUILD_DATE%
set _OVR_BUILD_RNUM=%BUILD_RNUM%

rem  Command line tag (build_release.bat beta)
if not "%~1"=="" set "BUILD_TAG=%~1"

rem  Compiler version labels used in destination subfolder names.
rem
rem  MSVC_VER  and CLANG_VER (current v145/ClangCL toolchain) are
rem  auto-detected below via vswhere -- the installed toolchain moves
rem  forward on its own (e.g. LLVM 20.1.x -> 22.1.3) and hardcoding
rem  them here just goes stale silently.
rem
rem  WIN7_MSVC_VER and WIN7_TV stay manual: they identify the older,
rem  deliberately pinned v143 (VS 2022-era) toolset used for Windows 7
rem  compatibility (see BUILD v143 section below), which is NOT the
rem  toolchain vswhere reports as "latest" -- there is no reliable way
rem  to ask vswhere for "the v143 toolset's own version" specifically.
rem  Update these only when the pinned v143 toolset itself is changed.
rem  WIN7_MSVC_VER : cl.exe version of the v143 toolset (e.g. "19.44")
rem  WIN7_TV       : second label for the win7 folder, the VS 2022
rem                  toolset minor version (e.g. "17.14")
set WIN7_MSVC_VER=19.44
set WIN7_TV=17.14

rem  Locate the Visual Studio installation (vswhere ships with VS since 2017).
rem  Note: paths under "Program Files (x86)" contain literal parentheses,
rem  which break cmd.exe's paren-matching inside if(...) blocks if expanded
rem  with %...% there -- so every reference inside such a block below uses
rem  delayed expansion (!...!) instead.
set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found at: !VSWHERE!
    echo        Cannot auto-detect MSVC/Clang toolchain versions.
    exit /b 1
)
for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do set VS_INSTALL=%%I
if not defined VS_INSTALL (
    echo ERROR: vswhere could not locate a Visual Studio installation.
    exit /b 1
)

rem  MSVC_VER: derived from the default VC++ Tools version file.
rem  VCToolsVersion "14.NN.xxxxx" maps 1:1 to cl.exe's reported "19.NN" version.
set VCTOOLS_VER_FILE=%VS_INSTALL%\VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt
if not exist "%VCTOOLS_VER_FILE%" (
    echo ERROR: VCToolsVersion file not found: !VCTOOLS_VER_FILE!
    exit /b 1
)
set /p VCTOOLS_VER=<"%VCTOOLS_VER_FILE%"
for /f "tokens=1,2 delims=." %%A in ("%VCTOOLS_VER%") do set MSVC_VER=19.%%B
if not defined MSVC_VER (
    echo ERROR: could not parse MSVC_VER from: !VCTOOLS_VER!
    exit /b 1
)

rem  CLANG_VER: query the bundled clang-cl.exe directly -- works without
rem  a Developer Command Prompt, since we use its full vswhere-derived path.
rem  Output is redirected to a temp file rather than piped through a
rem  quoted-path-with-spaces command inside for /f, which cmd.exe mangles.
set CLANGCL_EXE=%VS_INSTALL%\VC\Tools\Llvm\x64\bin\clang-cl.exe
if not exist "%CLANGCL_EXE%" set CLANGCL_EXE=%VS_INSTALL%\VC\Tools\Llvm\bin\clang-cl.exe
if not exist "%CLANGCL_EXE%" (
    echo ERROR: clang-cl.exe not found under: !VS_INSTALL!\VC\Tools\Llvm
    exit /b 1
)
set CLANGCL_VER_TMP=%TEMP%\avs_build_release_clangver.txt
"%CLANGCL_EXE%" --version > "%CLANGCL_VER_TMP%" 2>&1
set "CLANG_VER_LINE="
for /f "usebackq delims=" %%L in ("%CLANGCL_VER_TMP%") do if not defined CLANG_VER_LINE set "CLANG_VER_LINE=%%L"
del "%CLANGCL_VER_TMP%" 2>nul
set "CLANG_VER="
if "!CLANG_VER_LINE:~0,13!"=="clang version" for /f "tokens=3" %%V in ("!CLANG_VER_LINE!") do set CLANG_VER=%%V
if not defined CLANG_VER (
    echo ERROR: could not parse CLANG_VER from "!CLANGCL_EXE!" --version output: !CLANG_VER_LINE!
    exit /b 1
)

echo   Detected MSVC_VER=%MSVC_VER%  CLANG_VER=%CLANG_VER%  (VS_INSTALL=%VS_INSTALL%)

rem  LLVM_MINGW: the cross compiler used for the Windows-on-ARM64 builds.
rem  The aarch64 builds are done with the MinGW-style toolchain, NOT with
rem  MSVC: the C++ ABI of an MSVC-built core and a MinGW-built plugin
rem  differ, so a single toolchain family is used for a whole variant.
rem
rem  llvm-mingw (https://github.com/mstorsjo/llvm-mingw) is an x86_64
rem  hosted toolchain that targets aarch64-w64-mingw32, so no ARM64
rem  hardware is needed to build for Windows on ARM64.  It produces
rem  MinGW-w64 ABI binaries -- the same calling conventions, name
rem  mangling and standard library behaviour as an x86_64 MinGW build.
rem
rem  The distribution also contains an aarch64-w64-mingw32-g++.exe, but
rem  that is the very same clang in its GCC-compatible driver mode, so
rem  building the variant twice would produce the same code.
rem
rem  Use the UCRT flavour of llvm-mingw (the "-ucrt-" build): it links
rem  against ucrtbase.dll and the api-ms-win-crt-* forwarders, i.e. the
rem  CRT of Windows 10/11 -- the same one the MSVC build uses.
rem  Update the path below when the toolchain is updated.
if not defined LLVM_MINGW_ROOT set "LLVM_MINGW_ROOT=C:\avsplus_build_deps\llvm-mingw-20261020-ucrt-x86_64"
set LLVM_MINGW_CLANG=%LLVM_MINGW_ROOT%\bin\aarch64-w64-mingw32-clang++.exe
set LLVM_MINGW_VER=
if exist "%LLVM_MINGW_CLANG%" (
    set "LM_VER_TMP=%TEMP%\avs_build_release_llvmmingwver.txt"
    "%LLVM_MINGW_CLANG%" --version > "!LM_VER_TMP!" 2>&1
    set "LM_VER_LINE="
    for /f "usebackq delims=" %%L in ("!LM_VER_TMP!") do if not defined LM_VER_LINE set "LM_VER_LINE=%%L"
    del "!LM_VER_TMP!" 2>nul
    if defined LM_VER_LINE for /f "tokens=3" %%V in ("!LM_VER_LINE!") do set LLVM_MINGW_VER=%%V
)
if defined LLVM_MINGW_VER (
    echo   Detected LLVM_MINGW_VER=%LLVM_MINGW_VER%
    echo   llvm-mingw root: %LLVM_MINGW_ROOT%
) else (
    echo   llvm-mingw aarch64 toolchain not found at: %LLVM_MINGW_ROOT%
    rem  The ARM64 variant is on by default, so a missing toolchain only
    rem  means "skip it" -- unless the switch was set explicitly, then it
    rem  is an error the user has to know about.
    if defined _OVR_BUILD_ARM64 (
        echo ERROR: BUILD_ARM64 was set explicitly, but the llvm-mingw
        echo        toolchain is missing at: %LLVM_MINGW_ROOT%
        echo        Download the UCRT build from
        echo        https://github.com/mstorsjo/llvm-mingw/releases and
        echo        update LLVM_MINGW_ROOT, or unset BUILD_ARM64.
        goto :fatal_exit
    )
    set "BUILD_ARM64=0"
    echo   WARNING: skipping the ARM64 variant -- no llvm-mingw toolchain.
)

rem  Plugin dependency paths
rem  -----------------------------------------------------------------------
rem  These point to pre-built MSVC-compatible (static or import) libraries.
rem  MinGW/msys2 .a files CANNOT be used here -- wrong ABI.
rem  msys2's pkg-config.exe on PATH is OK and used by cmake for detection,
rem  but the actual libs must come from the locations below.
rem
rem  DevIL SDK  (DLL-based, pre-built x86/x64):
rem    Download: http://downloads.sourceforge.net/openil/DevIL-Windows-SDK-1.8.0.zip
rem    Extract to: %DEVIL_SDK%  (note: folder name has spaces)
rem    DevIL.dll must be distributed alongside ImageSeq.dll in plugins\
rem
rem  SoundTouch (static, built from source with cmake --target install):
rem    Source:  https://codeberg.org/soundtouch/soundtouch
rem    See also: avisynth-build-win\helpers\makesoundtouch.bat
rem    Two toolset variants are built:
rem      v141_xp -- used by the v141_xp builds (only needed if BUILD_XP=1)
rem      v143    -- used by all other builds
set DEVIL_SDK=C:\avsplus_build_deps\DevIL Windows SDK
set DEPS_XP_X86=C:\avsplus_build_deps\xp-x86-32
set DEPS_XP_X64=C:\avsplus_build_deps\xp-x86-64
set DEPS_V143_X86=C:\avsplus_build_deps\v143-x86-32
set DEPS_V143_X64=C:\avsplus_build_deps\v143-x86-64
rem  aarch64 (Windows on ARM64) SoundTouch, built with the MinGW toolchain
set DEPS_ARM64=C:\avsplus_build_deps\mingw-arm64

rem  Paths
rem  SOURCE_ROOT: the repo root, derived as the parent of this script's own
rem  folder (this .bat always lives in <repo root>\avisynth-build-win\), so it
rem  works no matter what the repo root folder is named locally (e.g.
rem  "AviSynthPlus" vs "AviSynthPlusPF") or what directory it's run from.
for %%I in ("%~dp0..") do set SOURCE_ROOT=%%~fI
set BUILD_ROOT=%SOURCE_ROOT%\avisynth-build-win
set DEST_ROOT=%~dp0current-dev-release
set SEVENZIP=C:\Program Files\7-Zip\7z.exe

rem  Parallel build jobs (set to your CPU thread count)
set BUILD_JOBS=8

rem ============================================================
rem  BUILD IDENTIFICATION  -- detected, nothing to edit here
rem ============================================================
rem  build_info.cmake collects the same information the top-level
rem  CMakeLists.txt uses for the DLLs themselves:
rem    * the version number from avs_core\core\version.h.in
rem    * the revision counter from git (rev-list --count HEAD)
rem    * the build date
rem  It writes a small .cmd file with plain "set" commands, which is
rem  called right after, so the values end up as plain variables.
set BUILD_INFO_CMD=%BUILD_ROOT%\build_info.cmd
echo   Detecting build identification...
cmake -D REPO="%SOURCE_ROOT%" ^
      -D SRC="%SOURCE_ROOT%\avs_core\core\version.h.in" ^
      -D OUT="%BUILD_INFO_CMD%" ^
      -P "%~dp0build_info.cmake"
if errorlevel 1 (
    echo ERROR: could not determine the build identification.
    echo        Is git installed and is %SOURCE_ROOT% a git repository?
    exit /b 1
)
call "%BUILD_INFO_CMD%"

rem  Environment/command line overrides win over the detected values.
if not "%_OVR_AVS_VER%"==""    set "AVS_VER=%_OVR_AVS_VER%"
if not "%_OVR_BUILD_DATE%"=="" set "BUILD_DATE=%_OVR_BUILD_DATE%"
if not "%_OVR_BUILD_RNUM%"=="" set "BUILD_RNUM=%_OVR_BUILD_RNUM%"

if not defined AVS_VER (
    echo ERROR: the version number could not be determined.
    exit /b 1
)
if not defined BUILD_RNUM (
    echo ERROR: the revision number could not be determined.
    exit /b 1
)

rem ============================================================
rem  DERIVED  -- do not edit below unless the folder scheme changes
rem ============================================================

set BUILD_NAME=%AVS_VER%_%BUILD_DATE%_%BUILD_TAG%_r%BUILD_RNUM%
set DEST=%DEST_ROOT%\%BUILD_NAME%
set ORIGIN=%BUILD_ROOT%\release-sample

rem  Destination subfolder names
set DIR_XP_X86=x86-xp
set DIR_XP_X64=x64-xp
set DIR_CLANG_X86=x86-%MSVC_VER%-clangcl-llvm-%CLANG_VER%
set DIR_CLANG_X64=x64-%MSVC_VER%-clangcl-llvm-%CLANG_VER%
set DIR_V145_X86=x86-%MSVC_VER%
set DIR_V145_X64=x64-%MSVC_VER%
set DIR_WIN7_X64=x64-win7-%WIN7_MSVC_VER%-%WIN7_TV%
set DIR_WIN7_X86=x86-win7-%WIN7_MSVC_VER%-%WIN7_TV%
set DIR_ARM64=arm64-mingw-clang-%LLVM_MINGW_VER%

rem  Isolated CMake build directories
set BLD_XP_X86=%BUILD_ROOT%\bld_xp_x86
set BLD_XP_X64=%BUILD_ROOT%\bld_xp_x64
set BLD_CLANG_X86=%BUILD_ROOT%\bld_clang_x86
set BLD_CLANG_X64=%BUILD_ROOT%\bld_clang_x64
set BLD_V145_X86=%BUILD_ROOT%\bld_v145_x86
set BLD_V145_X64=%BUILD_ROOT%\bld_v145_x64
set BLD_WIN7_X64=%BUILD_ROOT%\bld_win7_x64
set BLD_WIN7_X86=%BUILD_ROOT%\bld_win7_x86
set BLD_ARM64=%BUILD_ROOT%\bld_lm_arm64

rem  Build counter / total (for the "[BUILD n/N]" progress messages)
rem  -- counts only the variants that are switched on
set TOTAL_BUILDS=0
if "%BUILD_XP%"=="1"    set /a TOTAL_BUILDS+=2
if "%BUILD_CLANG%"=="1" set /a TOTAL_BUILDS+=2
if "%BUILD_V145%"=="1"  set /a TOTAL_BUILDS+=2
if "%BUILD_WIN7%"=="1"  set /a TOTAL_BUILDS+=2
if "%BUILD_ARM64%"=="1" set /a TOTAL_BUILDS+=1
set BUILDNUM=0

rem ============================================================

echo.
echo ========================================================
echo  AviSynth+ Dev Build : %BUILD_NAME%
echo  Destination         : %DEST%
echo  Origin folder       : %ORIGIN%
echo  ---------------------------------------------
echo  Version             : %AVS_VER%  (avs_core\core\version.h.in)
echo  Revision            : r%BUILD_RNUM%  (git: %GIT_BRANCH%, %GIT_DESCRIBE%)
echo  Build date          : %BUILD_DATE%  (last commit: %COMMIT_DATE%)
echo ========================================================
echo.

rem  Sanity checks
if not exist "%SOURCE_ROOT%\CMakeLists.txt" (
    echo ERROR: CMakeLists.txt not found at %SOURCE_ROOT%
    exit /b 1
)
if not exist "%ORIGIN%" (
    echo ERROR: release-sample folder not found: %ORIGIN%
    echo        Please populate it -- see %ORIGIN%\README.txt for details.
    exit /b 1
)
rem  DevIL is only needed by the ImageSeq plugin, which is not part of
rem  the ARM64 variant.
if "%BUILD_XP%"=="1" goto :check_devil
if "%BUILD_CLANG%"=="1" goto :check_devil
if "%BUILD_V145%"=="1" goto :check_devil
if "%BUILD_WIN7%"=="1" goto :check_devil
goto :skip_devil_check
:check_devil
if not exist "%DEVIL_SDK%\lib\x64\Release\DevIL.lib" (
    echo ERROR: DevIL SDK not found at: %DEVIL_SDK%
    echo        Download DevIL-Windows-SDK-1.8.0.zip from SourceForge and extract to:
    echo        C:\avsplus_build_deps\DevIL Windows SDK
    exit /b 1
)
:skip_devil_check
if "%BUILD_XP%"=="1" (
    if not exist "%DEPS_XP_X64%\lib\SoundTouch.lib" (
        echo ERROR: SoundTouch v141_xp x64 not found at: %DEPS_XP_X64%\lib\SoundTouch.lib
        echo        Run helpers\makesoundtouch.bat to build SoundTouch.
        goto :fatal_exit
    )
    if not exist "%DEPS_XP_X86%\lib\SoundTouch.lib" (
        echo ERROR: SoundTouch v141_xp x86 not found at: %DEPS_XP_X86%\lib\SoundTouch.lib
        echo        Run helpers\makesoundtouch.bat to build SoundTouch.
        goto :fatal_exit
    )
)
rem  The ClangCL, v145 and v143 builds all use the v143 SoundTouch libraries
if "%BUILD_CLANG%"=="1" goto :check_soundtouch_v143
if "%BUILD_V145%"=="1" goto :check_soundtouch_v143
if "%BUILD_WIN7%"=="1" goto :check_soundtouch_v143
goto :skip_soundtouch_check
:check_soundtouch_v143
if not exist "%DEPS_V143_X64%\lib\SoundTouch.lib" (
    echo ERROR: SoundTouch v143 x64 not found at: %DEPS_V143_X64%\lib\SoundTouch.lib
    echo        Run helpers\makesoundtouch.bat to build SoundTouch.
    exit /b 1
)
if not exist "%DEPS_V143_X86%\lib\SoundTouch.lib" (
    echo ERROR: SoundTouch v143 x86 not found at: %DEPS_V143_X86%\lib\SoundTouch.lib
    echo        Run helpers\makesoundtouch.bat to build SoundTouch.
    exit /b 1
)
:skip_soundtouch_check
rem  The llvm-mingw toolchain itself was already checked in the DETECTION
rem  block (missing toolchain -> the variant is skipped there).  Only the
rem  runtime DLLs are checked here, since without them the ARM64 build
rem  would produce DLLs that cannot start on the target machine.
if "%BUILD_ARM64%"=="1" (
    if not exist "%LLVM_MINGW_ROOT%\aarch64-w64-mingw32\bin\libc++.dll" (
        echo ERROR: the aarch64 runtime of llvm-mingw was not found under
        echo        %LLVM_MINGW_ROOT%\aarch64-w64-mingw32\bin
        goto :fatal_exit
    )
)

if exist "%DEST%" (
    echo WARNING: Destination already exists: %DEST%
    echo          Existing files will be overwritten.  Press Ctrl+C to abort.
    pause > nul
)


rem ============================================================
rem  STEP 0  Kill any lingering build processes
rem  (leftover MSBuild.exe from a previous run holds .obj file
rem   locks that block rmdir on the bld_* directories)
rem ============================================================
echo [STEP 0] Stopping any lingering MSBuild processes...
taskkill /f /im MSBuild.exe  2>nul
taskkill /f /im cl.exe       2>nul
taskkill /f /im link.exe     2>nul


rem ============================================================
rem  STEP 1  Create destination folder structure
rem ============================================================
echo [STEP 1] Creating destination folder structure...

rem  Only the variants that are switched on get a destination folder
set DIRLIST=
if "%BUILD_XP%"=="1"    set "DIRLIST=%DIRLIST% %DIR_XP_X86% %DIR_XP_X64%"
if "%BUILD_CLANG%"=="1" set "DIRLIST=%DIRLIST% %DIR_CLANG_X86% %DIR_CLANG_X64%"
if "%BUILD_V145%"=="1"  set "DIRLIST=%DIRLIST% %DIR_V145_X86% %DIR_V145_X64%"
if "%BUILD_WIN7%"=="1"  set "DIRLIST=%DIRLIST% %DIR_WIN7_X64% %DIR_WIN7_X86%"
if "%BUILD_ARM64%"=="1" set "DIRLIST=%DIRLIST% %DIR_ARM64%"

for %%D in (%DIRLIST%) do (
    mkdir "%DEST%\%%D\plugins" 2>nul
    mkdir "%DEST%\%%D\c_api"  2>nul
)

rem  XP builds carry an empty system\ folder (legacy placeholder)
if "%BUILD_XP%"=="1" (
    mkdir "%DEST%\%DIR_XP_X86%\system" 2>nul
    mkdir "%DEST%\%DIR_XP_X64%\system" 2>nul
)
mkdir "%DEST%\include\avs"         2>nul


rem ============================================================
rem  STEP 2  Copy license files and readme to build root
rem ============================================================
echo [STEP 2] Copying license and documentation files...

copy /y "%SOURCE_ROOT%\distrib\gpl.txt"                   "%DEST%\" > nul
copy /y "%SOURCE_ROOT%\distrib\OFL.TXT"                   "%DEST%\" > nul
copy /y "%SOURCE_ROOT%\distrib\lgpl_for_used_libs.txt"    "%DEST%\" > nul
copy /y "%SOURCE_ROOT%\distrib\Readme\readme_history.txt" "%DEST%\" > nul


rem ============================================================
rem  BUILD  v141_xp  Win32  (Windows XP, 32-bit, SSE2)  [optional]
rem
rem  The v141_xp toolset is only in VS2022, so this build uses the
rem  VS 2022 generator.
rem ============================================================
if "%BUILD_XP%"=="0" (
    echo.
    echo [SKIP] v141_xp Win32 -- BUILD_XP=0, skipping...
    goto :skip_xp_x86
)
set /a BUILDNUM+=1
echo.
echo [BUILD %BUILDNUM%/%TOTAL_BUILDS%] v141_xp Win32 -- XP-compatible 32-bit, SSE2...

if exist "%BLD_XP_X86%" rmdir /s /q "%BLD_XP_X86%" 2>nul

cmake -S "%SOURCE_ROOT%" -B "%BLD_XP_X86%" ^
  -G "Visual Studio 17 2022" -A Win32 -T "v141_xp" ^
  -DMSVC_CPU_ARCH:string="SSE2" ^
  -DWINXP_SUPPORT:bool=on ^
  -DBUILD_DIRECTSHOWSOURCE:bool=off ^
  -DENABLE_PLUGINS:bool=on ^
  -DENABLE_INTEL_SIMD:bool=ON ^
  -DBUILD_IMAGESEQ:bool=on ^
  -DIL_LIBRARIES="%DEVIL_SDK%\lib\x86\Release\DevIL.lib" ^
  -DILU_LIBRARIES="%DEVIL_SDK%\lib\x86\Release\ILU.lib" ^
  -DIL_INCLUDE_DIR="%DEVIL_SDK%\include\IL" ^
  -DBUILD_TIMESTRETCH:bool=on ^
  -DSOUNDTOUCH_LINK_LIBRARIES="%DEPS_XP_X86%\lib\SoundTouch.lib" ^
  -DSOUNDTOUCH_INCLUDE_DIRS="%DEPS_XP_X86%\include\soundtouch"
if errorlevel 1 ( echo ERROR: CMake configure failed for v141_xp Win32 & exit /b 1 )

cmake --build "%BLD_XP_X86%" --config Release -- /maxcpucount:%BUILD_JOBS%
if errorlevel 1 ( echo ERROR: Build failed for v141_xp Win32 & exit /b 1 )

echo   Assembling %DIR_XP_X86%...
set OUT=%BLD_XP_X86%\Output
copy /y "%OUT%\AviSynth.dll"               "%DEST%\%DIR_XP_X86%\"          > nul
copy /y "%OUT%\c_api\AviSynth.lib"         "%DEST%\%DIR_XP_X86%\c_api\"    > nul
copy /y "%OUT%\plugins\ConvertStacked.dll" "%DEST%\%DIR_XP_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\Shibatch.dll"       "%DEST%\%DIR_XP_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\VDubFilter.dll"     "%DEST%\%DIR_XP_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\ImageSeq.dll"       "%DEST%\%DIR_XP_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\TimeStretch.dll"    "%DEST%\%DIR_XP_X86%\plugins\"  > nul
copy /y "%DEVIL_SDK%\lib\x86\Release\DevIL.dll"    "%DEST%\%DIR_XP_X86%\plugins\"  > nul
copy /y "%ORIGIN%\x86\plugins\DirectShowSource.dll" "%DEST%\%DIR_XP_X86%\plugins\" > nul
if exist "%ORIGIN%\x86\plugins\imageseq-xp-legacy" (
    xcopy /e /y /i /q "%ORIGIN%\x86\plugins\imageseq-xp-legacy" "%DEST%\%DIR_XP_X86%\plugins\imageseq-xp-legacy" > nul
)

:skip_xp_x86


rem ============================================================
rem  BUILD  v141_xp  x64  (Windows XP, 64-bit)  [optional]
rem
rem  The v141_xp toolset is only in VS2022, so this build uses the
rem  VS 2022 generator.
rem ============================================================
if "%BUILD_XP%"=="0" (
    echo.
    echo [SKIP] v141_xp x64 -- BUILD_XP=0, skipping...
    goto :skip_xp_x64
)
set /a BUILDNUM+=1
echo.
echo [BUILD %BUILDNUM%/%TOTAL_BUILDS%] v141_xp x64 -- XP-compatible 64-bit...

if exist "%BLD_XP_X64%" rmdir /s /q "%BLD_XP_X64%" 2>nul

cmake -S "%SOURCE_ROOT%" -B "%BLD_XP_X64%" ^
  -G "Visual Studio 17 2022" -A x64 -T "v141_xp" ^
  -DWINXP_SUPPORT:bool=on ^
  -DBUILD_DIRECTSHOWSOURCE:bool=off ^
  -DENABLE_PLUGINS:bool=on ^
  -DENABLE_INTEL_SIMD:bool=ON ^
  -DBUILD_IMAGESEQ:bool=on ^
  -DIL_LIBRARIES="%DEVIL_SDK%\lib\x64\Release\DevIL.lib" ^
  -DILU_LIBRARIES="%DEVIL_SDK%\lib\x64\Release\ILU.lib" ^
  -DIL_INCLUDE_DIR="%DEVIL_SDK%\include\IL" ^
  -DBUILD_TIMESTRETCH:bool=on ^
  -DSOUNDTOUCH_LINK_LIBRARIES="%DEPS_XP_X64%\lib\SoundTouch.lib" ^
  -DSOUNDTOUCH_INCLUDE_DIRS="%DEPS_XP_X64%\include\soundtouch"
if errorlevel 1 ( echo ERROR: CMake configure failed for v141_xp x64 & exit /b 1 )

cmake --build "%BLD_XP_X64%" --config Release -- /maxcpucount:%BUILD_JOBS%
if errorlevel 1 ( echo ERROR: Build failed for v141_xp x64 & exit /b 1 )

echo   Assembling %DIR_XP_X64%...
set OUT=%BLD_XP_X64%\Output
copy /y "%OUT%\AviSynth.dll"               "%DEST%\%DIR_XP_X64%\"          > nul
copy /y "%OUT%\c_api\AviSynth.lib"         "%DEST%\%DIR_XP_X64%\c_api\"    > nul
copy /y "%OUT%\plugins\ConvertStacked.dll" "%DEST%\%DIR_XP_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\Shibatch.dll"       "%DEST%\%DIR_XP_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\VDubFilter.dll"     "%DEST%\%DIR_XP_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\ImageSeq.dll"       "%DEST%\%DIR_XP_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\TimeStretch.dll"    "%DEST%\%DIR_XP_X64%\plugins\"  > nul
copy /y "%DEVIL_SDK%\lib\x64\Release\DevIL.dll"    "%DEST%\%DIR_XP_X64%\plugins\"  > nul
copy /y "%ORIGIN%\x64\plugins\DirectShowSource.dll" "%DEST%\%DIR_XP_X64%\plugins\" > nul
if exist "%ORIGIN%\x64\plugins\imageseq-xp-legacy" (
    xcopy /e /y /i /q "%ORIGIN%\x64\plugins\imageseq-xp-legacy" "%DEST%\%DIR_XP_X64%\plugins\imageseq-xp-legacy" > nul
)

:skip_xp_x64


rem ============================================================
rem  BUILD  ClangCL  Win32  (32-bit, Clang/LLVM)
rem ============================================================
if "%BUILD_CLANG%"=="0" (
    echo.
    echo [SKIP] ClangCL Win32 -- BUILD_CLANG=0, skipping...
    goto :skip_clang_x86
)
set /a BUILDNUM+=1
echo.
echo [BUILD %BUILDNUM%/%TOTAL_BUILDS%] ClangCL Win32 -- 32-bit...

if exist "%BLD_CLANG_X86%" rmdir /s /q "%BLD_CLANG_X86%" 2>nul

cmake -S "%SOURCE_ROOT%" -B "%BLD_CLANG_X86%" ^
  -G "Visual Studio 18 2026" -A Win32 -T "ClangCL" ^
  -DWINXP_SUPPORT:bool=off ^
  -DBUILD_DIRECTSHOWSOURCE:bool=off ^
  -DENABLE_PLUGINS:bool=on ^
  -DENABLE_INTEL_SIMD:bool=ON ^
  -DBUILD_IMAGESEQ:bool=on ^
  -DIL_LIBRARIES="%DEVIL_SDK%\lib\x86\Release\DevIL.lib" ^
  -DILU_LIBRARIES="%DEVIL_SDK%\lib\x86\Release\ILU.lib" ^
  -DIL_INCLUDE_DIR="%DEVIL_SDK%\include\IL" ^
  -DBUILD_TIMESTRETCH:bool=on ^
  -DSOUNDTOUCH_LINK_LIBRARIES="%DEPS_V143_X86%\lib\SoundTouch.lib" ^
  -DSOUNDTOUCH_INCLUDE_DIRS="%DEPS_V143_X86%\include\soundtouch"
if errorlevel 1 ( echo ERROR: CMake configure failed for ClangCL Win32 & exit /b 1 )

cmake --build "%BLD_CLANG_X86%" --config Release -- /maxcpucount:%BUILD_JOBS%
if errorlevel 1 ( echo ERROR: Build failed for ClangCL Win32 & exit /b 1 )

echo   Assembling %DIR_CLANG_X86%...
set OUT=%BLD_CLANG_X86%\Output
copy /y "%OUT%\AviSynth.dll"               "%DEST%\%DIR_CLANG_X86%\"          > nul
copy /y "%OUT%\c_api\AviSynth.lib"         "%DEST%\%DIR_CLANG_X86%\c_api\"    > nul
copy /y "%OUT%\plugins\ConvertStacked.dll" "%DEST%\%DIR_CLANG_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\Shibatch.dll"       "%DEST%\%DIR_CLANG_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\VDubFilter.dll"     "%DEST%\%DIR_CLANG_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\ImageSeq.dll"       "%DEST%\%DIR_CLANG_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\TimeStretch.dll"    "%DEST%\%DIR_CLANG_X86%\plugins\"  > nul
copy /y "%DEVIL_SDK%\lib\x86\Release\DevIL.dll"    "%DEST%\%DIR_CLANG_X86%\plugins\"  > nul
copy /y "%ORIGIN%\x86\plugins\DirectShowSource.dll" "%DEST%\%DIR_CLANG_X86%\plugins\" > nul

:skip_clang_x86


rem ============================================================
rem  BUILD  ClangCL  x64  (64-bit, Clang/LLVM)
rem ============================================================
if "%BUILD_CLANG%"=="0" (
    echo.
    echo [SKIP] ClangCL x64 -- BUILD_CLANG=0, skipping...
    goto :skip_clang_x64
)
set /a BUILDNUM+=1
echo.
echo [BUILD %BUILDNUM%/%TOTAL_BUILDS%] ClangCL x64 -- 64-bit...

if exist "%BLD_CLANG_X64%" rmdir /s /q "%BLD_CLANG_X64%" 2>nul

cmake -S "%SOURCE_ROOT%" -B "%BLD_CLANG_X64%" ^
  -G "Visual Studio 18 2026" -A x64 -T "ClangCL" ^
  -DWINXP_SUPPORT:bool=off ^
  -DBUILD_DIRECTSHOWSOURCE:bool=off ^
  -DENABLE_PLUGINS:bool=on ^
  -DENABLE_INTEL_SIMD:bool=ON ^
  -DBUILD_IMAGESEQ:bool=on ^
  -DIL_LIBRARIES="%DEVIL_SDK%\lib\x64\Release\DevIL.lib" ^
  -DILU_LIBRARIES="%DEVIL_SDK%\lib\x64\Release\ILU.lib" ^
  -DIL_INCLUDE_DIR="%DEVIL_SDK%\include\IL" ^
  -DBUILD_TIMESTRETCH:bool=on ^
  -DSOUNDTOUCH_LINK_LIBRARIES="%DEPS_V143_X64%\lib\SoundTouch.lib" ^
  -DSOUNDTOUCH_INCLUDE_DIRS="%DEPS_V143_X64%\include\soundtouch"
if errorlevel 1 ( echo ERROR: CMake configure failed for ClangCL x64 & exit /b 1 )

cmake --build "%BLD_CLANG_X64%" --config Release -- /maxcpucount:%BUILD_JOBS%
if errorlevel 1 ( echo ERROR: Build failed for ClangCL x64 & exit /b 1 )

echo   Assembling %DIR_CLANG_X64%...
set OUT=%BLD_CLANG_X64%\Output
copy /y "%OUT%\AviSynth.dll"               "%DEST%\%DIR_CLANG_X64%\"          > nul
copy /y "%OUT%\c_api\AviSynth.lib"         "%DEST%\%DIR_CLANG_X64%\c_api\"    > nul
copy /y "%OUT%\plugins\ConvertStacked.dll" "%DEST%\%DIR_CLANG_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\Shibatch.dll"       "%DEST%\%DIR_CLANG_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\VDubFilter.dll"     "%DEST%\%DIR_CLANG_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\ImageSeq.dll"       "%DEST%\%DIR_CLANG_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\TimeStretch.dll"    "%DEST%\%DIR_CLANG_X64%\plugins\"  > nul
copy /y "%DEVIL_SDK%\lib\x64\Release\DevIL.dll"    "%DEST%\%DIR_CLANG_X64%\plugins\"  > nul
copy /y "%ORIGIN%\x64\plugins\DirectShowSource.dll" "%DEST%\%DIR_CLANG_X64%\plugins\" > nul

:skip_clang_x64


rem ============================================================
rem  BUILD  v145  Win32  (32-bit, current VS 2026 MSVC)
rem ============================================================
if "%BUILD_V145%"=="0" (
    echo.
    echo [SKIP] v145 Win32 -- BUILD_V145=0, skipping...
    goto :skip_v145_x86
)
set /a BUILDNUM+=1
echo.
echo [BUILD %BUILDNUM%/%TOTAL_BUILDS%] v145 Win32 -- current MSVC 32-bit...

if exist "%BLD_V145_X86%" rmdir /s /q "%BLD_V145_X86%" 2>nul

cmake -S "%SOURCE_ROOT%" -B "%BLD_V145_X86%" ^
  -G "Visual Studio 18 2026" -A Win32 -T "v145" ^
  -DWINXP_SUPPORT:bool=off ^
  -DBUILD_DIRECTSHOWSOURCE:bool=off ^
  -DENABLE_PLUGINS:bool=on ^
  -DENABLE_INTEL_SIMD:bool=ON ^
  -DBUILD_IMAGESEQ:bool=on ^
  -DIL_LIBRARIES="%DEVIL_SDK%\lib\x86\Release\DevIL.lib" ^
  -DILU_LIBRARIES="%DEVIL_SDK%\lib\x86\Release\ILU.lib" ^
  -DIL_INCLUDE_DIR="%DEVIL_SDK%\include\IL" ^
  -DBUILD_TIMESTRETCH:bool=on ^
  -DSOUNDTOUCH_LINK_LIBRARIES="%DEPS_V143_X86%\lib\SoundTouch.lib" ^
  -DSOUNDTOUCH_INCLUDE_DIRS="%DEPS_V143_X86%\include\soundtouch"
if errorlevel 1 ( echo ERROR: CMake configure failed for v145 Win32 & exit /b 1 )

cmake --build "%BLD_V145_X86%" --config Release -- /maxcpucount:%BUILD_JOBS%
if errorlevel 1 ( echo ERROR: Build failed for v145 Win32 & exit /b 1 )

echo   Assembling %DIR_V145_X86%...
set OUT=%BLD_V145_X86%\Output
copy /y "%OUT%\AviSynth.dll"               "%DEST%\%DIR_V145_X86%\"          > nul
copy /y "%OUT%\c_api\AviSynth.lib"         "%DEST%\%DIR_V145_X86%\c_api\"    > nul
copy /y "%OUT%\plugins\ConvertStacked.dll" "%DEST%\%DIR_V145_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\Shibatch.dll"       "%DEST%\%DIR_V145_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\VDubFilter.dll"     "%DEST%\%DIR_V145_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\ImageSeq.dll"       "%DEST%\%DIR_V145_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\TimeStretch.dll"    "%DEST%\%DIR_V145_X86%\plugins\"  > nul
copy /y "%DEVIL_SDK%\lib\x86\Release\DevIL.dll"    "%DEST%\%DIR_V145_X86%\plugins\"  > nul
copy /y "%ORIGIN%\x86\plugins\DirectShowSource.dll" "%DEST%\%DIR_V145_X86%\plugins\" > nul

:skip_v145_x86


rem ============================================================
rem  BUILD  v145  x64  (64-bit, current VS 2026 MSVC)
rem ============================================================
if "%BUILD_V145%"=="0" (
    echo.
    echo [SKIP] v145 x64 -- BUILD_V145=0, skipping...
    goto :skip_v145_x64
)
set /a BUILDNUM+=1
echo.
echo [BUILD %BUILDNUM%/%TOTAL_BUILDS%] v145 x64 -- current MSVC 64-bit...

if exist "%BLD_V145_X64%" rmdir /s /q "%BLD_V145_X64%" 2>nul

cmake -S "%SOURCE_ROOT%" -B "%BLD_V145_X64%" ^
  -G "Visual Studio 18 2026" -A x64 -T "v145" ^
  -DWINXP_SUPPORT:bool=off ^
  -DBUILD_DIRECTSHOWSOURCE:bool=off ^
  -DENABLE_PLUGINS:bool=on ^
  -DENABLE_INTEL_SIMD:bool=ON ^
  -DBUILD_IMAGESEQ:bool=on ^
  -DIL_LIBRARIES="%DEVIL_SDK%\lib\x64\Release\DevIL.lib" ^
  -DILU_LIBRARIES="%DEVIL_SDK%\lib\x64\Release\ILU.lib" ^
  -DIL_INCLUDE_DIR="%DEVIL_SDK%\include\IL" ^
  -DBUILD_TIMESTRETCH:bool=on ^
  -DSOUNDTOUCH_LINK_LIBRARIES="%DEPS_V143_X64%\lib\SoundTouch.lib" ^
  -DSOUNDTOUCH_INCLUDE_DIRS="%DEPS_V143_X64%\include\soundtouch"
if errorlevel 1 ( echo ERROR: CMake configure failed for v145 x64 & exit /b 1 )

cmake --build "%BLD_V145_X64%" --config Release -- /maxcpucount:%BUILD_JOBS%
if errorlevel 1 ( echo ERROR: Build failed for v145 x64 & exit /b 1 )

echo   Assembling %DIR_V145_X64%...
set OUT=%BLD_V145_X64%\Output
copy /y "%OUT%\AviSynth.dll"               "%DEST%\%DIR_V145_X64%\"          > nul
copy /y "%OUT%\c_api\AviSynth.lib"         "%DEST%\%DIR_V145_X64%\c_api\"    > nul
copy /y "%OUT%\plugins\ConvertStacked.dll" "%DEST%\%DIR_V145_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\Shibatch.dll"       "%DEST%\%DIR_V145_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\VDubFilter.dll"     "%DEST%\%DIR_V145_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\ImageSeq.dll"       "%DEST%\%DIR_V145_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\TimeStretch.dll"    "%DEST%\%DIR_V145_X64%\plugins\"  > nul
copy /y "%DEVIL_SDK%\lib\x64\Release\DevIL.dll"    "%DEST%\%DIR_V145_X64%\plugins\"  > nul
copy /y "%ORIGIN%\x64\plugins\DirectShowSource.dll" "%DEST%\%DIR_V145_X64%\plugins\" > nul

:skip_v145_x64


rem ============================================================
rem  BUILD  v143  x64  (64-bit, Windows 7, MSVC 19.44)
rem
rem  The v145 toolset (19.50+) links the CRT against CreateFile2, a
rem  kernel32 entry point introduced in Windows 8 that does not exist
rem  in Windows 7.  v145-built DLLs crash at startup with a missing
rem  entry point error.  MSVC 19.44 (v143) uses CreateFileW instead
rem  and is the highest toolset that still runs on Windows 7.
rem ============================================================
if "%BUILD_WIN7%"=="0" (
    echo.
    echo [SKIP] v143 x64 -- BUILD_WIN7=0, skipping...
    goto :skip_win7_x64
)
set /a BUILDNUM+=1
echo.
echo [BUILD %BUILDNUM%/%TOTAL_BUILDS%] v143 x64 -- Windows 7 compatible, MSVC %WIN7_MSVC_VER%...

if exist "%BLD_WIN7_X64%" rmdir /s /q "%BLD_WIN7_X64%" 2>nul

cmake -S "%SOURCE_ROOT%" -B "%BLD_WIN7_X64%" ^
  -G "Visual Studio 18 2026" -A x64 -T "v143" ^
  -DWINXP_SUPPORT:bool=off ^
  -DBUILD_DIRECTSHOWSOURCE:bool=off ^
  -DENABLE_PLUGINS:bool=on ^
  -DENABLE_INTEL_SIMD:bool=ON ^
  -DBUILD_IMAGESEQ:bool=on ^
  -DIL_LIBRARIES="%DEVIL_SDK%\lib\x64\Release\DevIL.lib" ^
  -DILU_LIBRARIES="%DEVIL_SDK%\lib\x64\Release\ILU.lib" ^
  -DIL_INCLUDE_DIR="%DEVIL_SDK%\include\IL" ^
  -DBUILD_TIMESTRETCH:bool=on ^
  -DSOUNDTOUCH_LINK_LIBRARIES="%DEPS_V143_X64%\lib\SoundTouch.lib" ^
  -DSOUNDTOUCH_INCLUDE_DIRS="%DEPS_V143_X64%\include\soundtouch"
if errorlevel 1 ( echo ERROR: CMake configure failed for v143 x64 & exit /b 1 )

cmake --build "%BLD_WIN7_X64%" --config Release -- /maxcpucount:%BUILD_JOBS%
if errorlevel 1 ( echo ERROR: Build failed for v143 x64 & exit /b 1 )

echo   Assembling %DIR_WIN7_X64%...
set OUT=%BLD_WIN7_X64%\Output
copy /y "%OUT%\AviSynth.dll"               "%DEST%\%DIR_WIN7_X64%\"          > nul
copy /y "%OUT%\c_api\AviSynth.lib"         "%DEST%\%DIR_WIN7_X64%\c_api\"    > nul
copy /y "%OUT%\plugins\ConvertStacked.dll" "%DEST%\%DIR_WIN7_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\Shibatch.dll"       "%DEST%\%DIR_WIN7_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\VDubFilter.dll"     "%DEST%\%DIR_WIN7_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\ImageSeq.dll"       "%DEST%\%DIR_WIN7_X64%\plugins\"  > nul
copy /y "%OUT%\plugins\TimeStretch.dll"    "%DEST%\%DIR_WIN7_X64%\plugins\"  > nul
copy /y "%DEVIL_SDK%\lib\x64\Release\DevIL.dll"    "%DEST%\%DIR_WIN7_X64%\plugins\"  > nul
copy /y "%ORIGIN%\x64\plugins\DirectShowSource.dll" "%DEST%\%DIR_WIN7_X64%\plugins\" > nul

:skip_win7_x64


rem ============================================================
rem  BUILD  v143  Win32  (32-bit, Windows 7, MSVC 19.44)
rem
rem  Same Win7 compatibility rationale as build 7 (v143 x64).
rem  Provides a 32-bit Win7 option without requiring the older
rem  v141_xp toolset.  Users needing XP support should still
rem  use the v141_xp Win32 variant.
rem ============================================================
if "%BUILD_WIN7%"=="0" (
    echo.
    echo [SKIP] v143 Win32 -- BUILD_WIN7=0, skipping...
    goto :skip_win7_x86
)
set /a BUILDNUM+=1
echo.
echo [BUILD %BUILDNUM%/%TOTAL_BUILDS%] v143 Win32 -- Windows 7 compatible, MSVC %WIN7_MSVC_VER%...

if exist "%BLD_WIN7_X86%" rmdir /s /q "%BLD_WIN7_X86%" 2>nul

cmake -S "%SOURCE_ROOT%" -B "%BLD_WIN7_X86%" ^
  -G "Visual Studio 18 2026" -A Win32 -T "v143" ^
  -DWINXP_SUPPORT:bool=off ^
  -DBUILD_DIRECTSHOWSOURCE:bool=off ^
  -DENABLE_PLUGINS:bool=on ^
  -DENABLE_INTEL_SIMD:bool=ON ^
  -DBUILD_IMAGESEQ:bool=on ^
  -DIL_LIBRARIES="%DEVIL_SDK%\lib\x86\Release\DevIL.lib" ^
  -DILU_LIBRARIES="%DEVIL_SDK%\lib\x86\Release\ILU.lib" ^
  -DIL_INCLUDE_DIR="%DEVIL_SDK%\include\IL" ^
  -DBUILD_TIMESTRETCH:bool=on ^
  -DSOUNDTOUCH_LINK_LIBRARIES="%DEPS_V143_X86%\lib\SoundTouch.lib" ^
  -DSOUNDTOUCH_INCLUDE_DIRS="%DEPS_V143_X86%\include\soundtouch"
if errorlevel 1 ( echo ERROR: CMake configure failed for v143 Win32 & exit /b 1 )

cmake --build "%BLD_WIN7_X86%" --config Release -- /maxcpucount:%BUILD_JOBS%
if errorlevel 1 ( echo ERROR: Build failed for v143 Win32 & exit /b 1 )

echo   Assembling %DIR_WIN7_X86%...
set OUT=%BLD_WIN7_X86%\Output
copy /y "%OUT%\AviSynth.dll"               "%DEST%\%DIR_WIN7_X86%\"          > nul
copy /y "%OUT%\c_api\AviSynth.lib"         "%DEST%\%DIR_WIN7_X86%\c_api\"    > nul
copy /y "%OUT%\plugins\ConvertStacked.dll" "%DEST%\%DIR_WIN7_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\Shibatch.dll"       "%DEST%\%DIR_WIN7_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\VDubFilter.dll"     "%DEST%\%DIR_WIN7_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\ImageSeq.dll"       "%DEST%\%DIR_WIN7_X86%\plugins\"  > nul
copy /y "%OUT%\plugins\TimeStretch.dll"    "%DEST%\%DIR_WIN7_X86%\plugins\"  > nul
copy /y "%DEVIL_SDK%\lib\x86\Release\DevIL.dll"    "%DEST%\%DIR_WIN7_X86%\plugins\"  > nul
copy /y "%ORIGIN%\x86\plugins\DirectShowSource.dll" "%DEST%\%DIR_WIN7_X86%\plugins\" > nul

:skip_win7_x86


rem ============================================================
rem  BUILD  MinGW  ARM64  (Windows on ARM64, aarch64)
rem
rem  Cross-compiled right here on the x86_64 host with llvm-mingw:
rem  the toolchain is hosted on x86_64 and targets aarch64, so
rem  neither ARM64 hardware nor ARM64 Windows is needed to produce
rem  these binaries.
rem
rem  This variant uses the MinGW C++ runtime (libc++) and NOT the
rem  MSVC one -- the C++ ABI of an MSVC-built core and a
rem  MinGW-built plugin differ, so the whole ARM64 variant is one
rem  toolchain family.  Do not mix these DLLs with the MSVC-built
rem  ones.
rem
rem  The C runtime is the UCRT (ucrtbase.dll + the api-ms-win-crt-*
rem  forwarders), which is part of Windows 10/11 -- nothing to ship
rem  for it.  libc++.dll and libunwind.dll are NOT part of Windows,
rem  so they are copied next to the DLLs that need them.
rem
rem  ImageSeq is not part of this variant: the prebuilt DevIL SDK
rem  has no ARM64 binaries.  TimeStretch needs the aarch64
rem  SoundTouch build (helpers\makesoundtouch.bat); when that is
rem  missing, the plugin is left out.
rem ============================================================
if "%BUILD_ARM64%"=="1" goto :build_arm64
echo.
echo [SKIP] MinGW ARM64 -- BUILD_ARM64=0, skipping...
goto :after_build_arm64

:build_arm64
set /a BUILDNUM+=1
echo.
echo [BUILD %BUILDNUM%/%TOTAL_BUILDS%] MinGW ARM64 -- Windows on ARM64 (aarch64), llvm-mingw %LLVM_MINGW_VER%...

if exist "%BLD_ARM64%" rmdir /s /q "%BLD_ARM64%" 2>nul

rem  The cross build needs Ninja -- the Visual Studio generator cannot
rem  cross-compile with a different compiler.
set NINJA_EXE=
for %%N in (ninja.exe) do if not defined NINJA_EXE set "NINJA_EXE=%%~$PATH:N"
if not defined NINJA_EXE (
    echo ERROR: ninja.exe not found on PATH -- it is needed for the ARM64 build.
    echo        Install it with "pip install ninja" or set BUILD_ARM64=0.
    exit /b 1
)

rem  TimeStretch needs the aarch64 SoundTouch build
set ARM64_TIMESTRETCH=off
set ARM64_ST_ARGS=
if exist "%DEPS_ARM64%\lib\libSoundTouch.a" (
    set ARM64_TIMESTRETCH=on
    set ARM64_ST_ARGS=-DSOUNDTOUCH_LINK_LIBRARIES=%DEPS_ARM64%\lib\libSoundTouch.a -DSOUNDTOUCH_INCLUDE_DIRS=%DEPS_ARM64%\include\soundtouch
)

cmake -S "%SOURCE_ROOT%" -B "%BLD_ARM64%" ^
  -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE="%~dp0toolchains\llvm-mingw-aarch64.cmake" ^
  -DAVS_LLVM_MINGW_ROOT="%LLVM_MINGW_ROOT%" ^
  -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_DIRECTSHOWSOURCE:bool=off ^
  -DENABLE_PLUGINS:bool=on ^
  -DENABLE_INTEL_SIMD:bool=OFF ^
  -DBUILD_IMAGESEQ:bool=off ^
  -DBUILD_VDUBFILTER:bool=off ^
  -DBUILD_TIMESTRETCH:bool=%ARM64_TIMESTRETCH% %ARM64_ST_ARGS%
if errorlevel 1 ( echo ERROR: CMake configure failed for MinGW ARM64 & exit /b 1 )

cmake --build "%BLD_ARM64%" --parallel %BUILD_JOBS%
if errorlevel 1 ( echo ERROR: Build failed for MinGW ARM64 & exit /b 1 )

echo   Assembling %DIR_ARM64%...
set OUT=%BLD_ARM64%
copy /y "%OUT%\avs_core\AviSynth.dll"        "%DEST%\%DIR_ARM64%\"          > nul
copy /y "%OUT%\avs_core\AviSynth.dll.a"      "%DEST%\%DIR_ARM64%\c_api\"    > nul
copy /y "%OUT%\plugins\ConvertStacked\ConvertStacked.dll" "%DEST%\%DIR_ARM64%\plugins\" > nul
copy /y "%OUT%\plugins\Shibatch\Shibatch.dll"             "%DEST%\%DIR_ARM64%\plugins\" > nul
if exist "%OUT%\plugins\TimeStretch\TimeStretch.dll" (
    copy /y "%OUT%\plugins\TimeStretch\TimeStretch.dll" "%DEST%\%DIR_ARM64%\plugins\" > nul
)
rem  The C++ runtime of the MinGW build is not part of Windows
copy /y "%LLVM_MINGW_ROOT%\aarch64-w64-mingw32\bin\libc++.dll"    "%DEST%\%DIR_ARM64%\"         > nul
copy /y "%LLVM_MINGW_ROOT%\aarch64-w64-mingw32\bin\libunwind.dll"  "%DEST%\%DIR_ARM64%\"         > nul
copy /y "%LLVM_MINGW_ROOT%\aarch64-w64-mingw32\bin\libc++.dll"    "%DEST%\%DIR_ARM64%\plugins\" > nul
copy /y "%LLVM_MINGW_ROOT%\aarch64-w64-mingw32\bin\libunwind.dll"  "%DEST%\%DIR_ARM64%\plugins\" > nul

:after_build_arm64


rem ============================================================
rem  STEP 3  Copy shared include headers
rem          (headers are identical across configurations;
rem           taken from any build that was made in this run)
rem ============================================================
echo.
echo [STEP 3] Copying include headers...
set INC_SRC=
if "%BUILD_WIN7%"=="1" if exist "%BLD_WIN7_X86%\Output\include" set "INC_SRC=%BLD_WIN7_X86%\Output\include"
if not defined INC_SRC if "%BUILD_V145%"=="1" if exist "%BLD_V145_X86%\Output\include" set "INC_SRC=%BLD_V145_X86%\Output\include"
if not defined INC_SRC if "%BUILD_CLANG%"=="1" if exist "%BLD_CLANG_X86%\Output\include" set "INC_SRC=%BLD_CLANG_X86%\Output\include"
if not defined INC_SRC if "%BUILD_XP%"=="1" if exist "%BLD_XP_X86%\Output\include" set "INC_SRC=%BLD_XP_X86%\Output\include"
if defined INC_SRC (
    xcopy /e /y /i /q "%INC_SRC%" "%DEST%\include" > nul
) else (
    rem  Only the cross build was made: it has no Output\include tree,
    rem  so the headers are taken from the source tree.  The two
    rem  generated headers (avs\version.h and avs\arch.h) are left out
    rem  on purpose: they are per-target, and no header of the plugin
    rem  SDK includes them.
    xcopy /e /y /i /q "%SOURCE_ROOT%\avs_core\include" "%DEST%\include" > nul
)


rem ============================================================
rem  STEP 4  Create 7-Zip archive
rem ============================================================
echo.
echo [STEP 4] Creating 7-Zip archive...
if exist "%SEVENZIP%" (
    "%SEVENZIP%" a -mx=5 -mmt=on ^
      "%DEST_ROOT%\Avisynth_%BUILD_NAME%.7z" ^
      "%DEST%" > nul
    if errorlevel 1 (
        echo WARNING: 7-Zip returned an error.  Archive may be incomplete.
    ) else (
        echo   Created: %DEST_ROOT%\Avisynth_%BUILD_NAME%.7z
    )
) else (
    echo   7-Zip not found at: %SEVENZIP%
    echo   Skipping archive.  Install 7-Zip or update the SEVENZIP path above.
)


echo.
echo ========================================================
echo  DEV BUILD COMPLETE
echo  %DEST%
echo ========================================================
echo.

goto :script_done

rem  -----------------------------------------------------------------------
rem  Fatal exit.
rem
rem  "exit /b N" inside a parenthesised block does not reliably set the
rem  script's exit code: cmd.exe loses it when the enclosing block has
rem  further commands after the nested if.  The script still stops, but the
rem  status comes back as 0 -- which hides the failure from anything that
rem  checks it.  The error paths therefore print their message and jump
rem  here, where the exit happens outside of any block.
rem  -----------------------------------------------------------------------
:fatal_exit
exit /b 1

:script_done
endlocal
goto :eof
