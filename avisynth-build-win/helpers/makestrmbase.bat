@echo off
rem ===========================================================================
rem  makestrmbase.bat - builds the DirectShow base classes (strmbase.lib), needed by the
rem  DirectShowSource plugin, from the Windows SDK 7.1 samples, with the v141_xp toolset.
rem
rem  One v141_xp build serves both the XP (v141_xp) and the main (v143) DirectShowSource:
rem  a static library from an older MSVC toolset links into a newer one.
rem
rem  Source : %BASECLASSES_SRC% (Windows SDK 7.1 samples, read only; copied to a work folder)
rem  Work   : %WORK%
rem  Output : %OUT%\x86\strmbase.lib, %OUT%\x64\strmbase.lib  (+ the headers in %OUT%\include)
rem
rem  Needs Visual Studio 2022 with "MSVC v141 - VS 2017 C++ build tools" (v141_xp).
rem  Usage  : makestrmbase.bat
rem ===========================================================================
setlocal EnableExtensions

set "BASECLASSES_SRC=C:\Program Files\Microsoft SDKs\Windows\v7.1\Samples\multimedia\directshow\baseclasses"
set "WORK=C:\avsplus-build-deps\baseclasses"
set "OUT=C:\avsplus_build_deps\strmbase"

if not exist "%BASECLASSES_SRC%\BaseClasses.vcxproj" (
  echo ERROR: base classes sources not found: %BASECLASSES_SRC%
  exit /b 1
)

rem --- MSBuild of Visual Studio 2022 (it has the v141_xp toolset; VS 2026 does not) ---
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS2022="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -version [17.0^,18.0^) -products * -requires Microsoft.VisualStudio.Component.VC.v141.x86.x64 -property installationPath`) do set "VS2022=%%i"
if not defined VS2022 (
  echo ERROR: Visual Studio 2022 with the v141 toolset not found
  exit /b 1
)
set "MSBUILD=%VS2022%\MSBuild\Current\Bin\MSBuild.exe"

rem --- fresh copy of the sources (without the old build outputs) ---
if exist "%WORK%" rmdir /s /q "%WORK%"
mkdir "%WORK%"
copy /y "%BASECLASSES_SRC%\*.cpp" "%WORK%\" > nul
copy /y "%BASECLASSES_SRC%\*.h" "%WORK%\" > nul
copy /y "%BASECLASSES_SRC%\BaseClasses.vcxproj" "%WORK%\" > nul
copy /y "%BASECLASSES_SRC%\BaseClasses.vcxproj.filters" "%WORK%\" > nul

for %%P in (Win32 x64) do (
  echo === strmbase Release_MBCS %%P, v141_xp ===
  "%MSBUILD%" "%WORK%\BaseClasses.vcxproj" /nologo /v:minimal /m ^
    /p:Configuration=Release_MBCS /p:Platform=%%P /p:PlatformToolset=v141_xp /p:TargetName=strmbase ^
    /p:OutDir="%WORK%\out\%%P\\" /p:IntDir="%WORK%\obj\%%P\\"
  if errorlevel 1 (
    echo ERROR: strmbase build failed for %%P
    exit /b 1
  )
)

if not exist "%OUT%\x86" mkdir "%OUT%\x86"
if not exist "%OUT%\x64" mkdir "%OUT%\x64"
if not exist "%OUT%\include" mkdir "%OUT%\include"
copy /y "%WORK%\out\Win32\strmbase.lib" "%OUT%\x86\" > nul || exit /b 1
copy /y "%WORK%\out\x64\strmbase.lib" "%OUT%\x64\" > nul || exit /b 1
copy /y "%WORK%\*.h" "%OUT%\include\" > nul

echo strmbase.lib ready: %OUT%\x86, %OUT%\x64 (headers: %OUT%\include)
endlocal
