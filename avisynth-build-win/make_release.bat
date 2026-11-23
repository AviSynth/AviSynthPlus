@echo off
rem ===========================================================================
rem  make_release.bat - full Windows release build of AviSynth+ (entry point).
rem  The work is done by make_release.ps1 next to this file; see its header for
rem  the variants, stages and layout.  build_release.bat (dev/test build) is
rem  separate and unchanged.
rem
rem  Usage:
rem    make_release.bat                         all default stages: deps build check docs pack
rem    make_release.bat -Stages build,check     selected stages
rem    make_release.bat -Variants x64-v145,arm64 selected variants (internal names: x86-v145 x64-v145
rem                                             x86-v143 x64-v143 x86-xp x64-xp arm64, x86-clangcl x64-clangcl)
rem    make_release.bat -Tag test               name suffix
rem    make_release.bat -ClangCL                also build x86/x64 with clang-cl: it is then the default
rem                                             (x86\ x64\, <name>.exe), MSVC v145 x86-msvc\ x64-msvc\, _msvc
rem  Environment switch: DEPS_BUILD=auto|all|off
rem ===========================================================================
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_release.ps1" %*
exit /b %errorlevel%
