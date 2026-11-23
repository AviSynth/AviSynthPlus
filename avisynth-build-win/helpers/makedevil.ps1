# makedevil.ps1 - builds DevIL and its dependency chain from pinned sources, static, for ImageSeq.
#
# Follows qyot27's recipe (distrib/docs/.../avsplus_external_deps_guide_manual.rst) with the library set
# that the 3.7.6 release ImageSeq actually contains: zlib-ng, xz (lzma), libjpeg-turbo, libpng, jbigkit,
# libdeflate, lerc, zstd, libtiff, libsquish, JasPer, DevIL (no OpenEXR, lcms2, libwebp, libmng).
#
#   -Target msvc-x64 | msvc-x86     MSVC (Visual Studio generator, -T $Toolset, default v143)
#           -Toolset v141_xp        Windows XP (VS 2022 generator by default); XP-specific changes, see "XP" below
#   -Target mingw-arm64             llvm-mingw aarch64 (Ninja, toolchains\llvm-mingw-aarch64.cmake), Windows on ARM64
#   -Clean                          delete the prefix and the build folders first
#
# XP (v141_xp, the VS 2017 compiler), with the current library versions:
#   libdeflate  1.26+ requires C11 / VS 2019 16.8 (#error): libdeflate 1.25 for XP (VS 2015+)
#   JasPer      jas_malloc.c includes sysinfoapi.h and calls GetPhysicallyInstalledSystemMemory (Vista SP1+):
#               replaced with GlobalMemoryStatusEx (XP); the guide comments it out instead (memory size unknown)
#   zstd        its multithreading uses Windows condition variables (Vista+): ZSTD_MULTITHREAD_SUPPORT=OFF
#               (builds without error, but ImageSeq would then not load on XP)
#   runtime     projects with cmake_minimum_required < 3.15 lose /MD with v141_xp (empty RuntimeLibrary ->
#               /MT): CMP0091 NEW, so CMAKE_MSVC_RUNTIME_LIBRARY applies
#
# Sources : C:\avsplus-build-deps\devil-src\<lib>          (shared by all targets, pinned below)
# Builds  : C:\avsplus-build-deps\devil-build\<target>\<lib>
# Prefix  : C:\avsplus_build_deps\devil-<target>           (lib\, include\, deps_stamp.txt)
#
# The stamp file records the pins a prefix was built from; make_release.bat (DEPS_BUILD=auto) rebuilds
# only when it differs from the pins below.
param(
  [Parameter(Mandatory = $true)][ValidateSet('msvc-x64', 'msvc-x86', 'mingw-arm64')][string]$Target,
  [string]$Toolset = 'v143',
  [string]$Generator = 'Visual Studio 18 2026',
  [switch]$Clean,
  [string[]]$Only = @(),           # build only these libraries (iteration); sources are still prepared
  [string]$LlvmMingwRoot = 'C:/avsplus_build_deps/llvm-mingw-20261020-ucrt-x86_64'
)
$ErrorActionPreference = 'Stop'

# ---- pinned sources -----------------------------------------------------------------------------------
# kind 'git': repo + ref (tag or full commit); kind 'tar': url + sha256 (folder name inside the archive)
$Pins = [ordered]@{
  'zlib-ng'       = @{ kind = 'git'; url = 'https://github.com/zlib-ng/zlib-ng'; ref = 'bc98d84fcc1865a842bf1b07dbbc3ac2427a0938' } # develop, contains 4564ba4 (#2485 fix, after 2.3.3)
  'xz'            = @{ kind = 'git'; url = 'https://github.com/tukaani-project/xz'; ref = 'v5.8.4' }
  'libjpeg-turbo' = @{ kind = 'git'; url = 'https://github.com/libjpeg-turbo/libjpeg-turbo'; ref = '3.2.1' }
  'libpng'        = @{ kind = 'git'; url = 'https://github.com/pnggroup/libpng'; ref = 'v1.6.59' }
  'jbigkit'       = @{ kind = 'git'; url = 'https://github.com/qyot27/jbigkit'; ref = '6ca37b1c608b5b5038a13549fb5022bd48934ae3' } # mingw-w64 branch (CMake build)
  'libdeflate'    = @{ kind = 'git'; url = 'https://github.com/ebiggers/libdeflate'; ref = 'v1.26'; xp = 'never' }
  'libdeflate-xp' = @{ kind = 'git'; url = 'https://github.com/ebiggers/libdeflate'; ref = 'v1.25'; xp = 'only' }  # last for VS 2017 (v141_xp)
  'lerc'          = @{ kind = 'git'; url = 'https://github.com/esri/lerc'; ref = 'v4.3.0' }
  'zstd'          = @{ kind = 'git'; url = 'https://github.com/facebook/zstd'; ref = 'v1.5.7' }
  'libtiff'       = @{ kind = 'git'; url = 'https://gitlab.com/libtiff/libtiff.git'; ref = 'v4.7.2' }
  'libsquish'     = @{ kind = 'tar'; url = 'https://sourceforge.net/projects/libsquish/files/libsquish-1.15.tgz/download'; sha256 = '' ; flat = $true }
  'jasper'        = @{ kind = 'git'; url = 'https://github.com/jasper-software/jasper'; ref = 'version-4.2.9' }
  'DevIL'         = @{ kind = 'git'; url = 'https://github.com/DentonW/DevIL'; ref = '6f3d5e9bc5d173f56060fc8e14bed35c8edcfdcc' } # master
}
$JasperPatchUrl = 'https://gist.githubusercontent.com/qyot27/b362b3e3834485c3e7b7e33e3b8d5049/raw/4fdcfa2b5b516f47d8ce1e967d70877f63c85497/0001-jasper-git.patch'

$SrcRoot   = 'C:\avsplus-build-deps\devil-src'
$IsMsvc    = $Target -like 'msvc-*'
$IsXp      = $IsMsvc -and $Toolset -eq 'v141_xp'
if ($IsXp -and -not $PSBoundParameters.ContainsKey('Generator')) { $Generator = 'Visual Studio 17 2022' }  # VS 2026 has no v141_xp
# the pins this target uses (libdeflate: one of the two)
$Used = @($Pins.Keys | Where-Object { $x = $Pins[$_].xp; -not $x -or ($x -eq 'only' -and $IsXp) -or ($x -eq 'never' -and -not $IsXp) })
$Tag       = if ($IsMsvc) { "$Target-$Toolset" } else { $Target }
$BuildRoot = "C:\avsplus-build-deps\devil-build\$Tag"
$Prefix    = "C:\avsplus_build_deps\devil-$Tag"
$Arch      = if ($Target -eq 'msvc-x64') { 'x64' } elseif ($Target -eq 'msvc-x86') { 'Win32' } else { 'arm64' }
$Toolchain = (Join-Path $PSScriptRoot '..\toolchains\llvm-mingw-aarch64.cmake') -replace '\\', '/'
$PrefixFwd = $Prefix -replace '\\', '/'

function Log($m) { Write-Host "[makedevil $Target] $m" }
function Run($exe, [string[]]$a) {
  & $exe @a
  if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $exe $($a -join ' ')" }
}

# ---- sources ------------------------------------------------------------------------------------------
function Get-Source($name) {
  $p = $Pins[$name]; $dir = Join-Path $SrcRoot $name
  if ($p.kind -eq 'git') {
    if (-not (Test-Path "$dir\.git")) {
      Log "fetch $name @ $($p.ref)"
      New-Item -ItemType Directory -Force $dir | Out-Null
      Run git @('-C', $dir, 'init', '-q')
      Run git @('-C', $dir, 'remote', 'add', 'origin', $p.url)
    }
    $have = (git -C $dir rev-parse -q --verify 'HEAD^{commit}' 2>$null)
    $want = (git -C $dir rev-parse -q --verify "$($p.ref)^{commit}" 2>$null)
    if (-not $want -or $have -ne $want) {
      Run git @('-C', $dir, 'fetch', '-q', '--depth', '1', 'origin', $p.ref)
      Run git @('-C', $dir, 'checkout', '-q', '-f', 'FETCH_HEAD')
    }
    Run git @('-C', $dir, 'reset', '-q', '--hard')     # drop earlier source patches (re-applied below)
    Run git @('-C', $dir, 'clean', '-q', '-fdx')
  } else {
    if (-not (Test-Path $dir)) {
      Log "download $name"
      New-Item -ItemType Directory -Force $dir | Out-Null
      $tgz = "$SrcRoot\$name.tgz"
      Run curl.exe @('-sSL', '-o', $tgz, $p.url)
      $h = (Get-FileHash $tgz -Algorithm SHA256).Hash.ToLower()
      if ($p.sha256 -and $h -ne $p.sha256) { throw "$name sha256 mismatch: $h" }
      Log "$name sha256 $h"
      Run tar @('-xzf', $tgz, '-C', $dir)
    }
  }
  return $dir
}

# ---- CMake build + install into the prefix -------------------------------------------------------------
function Build-CMake($name, $srcSub, [string[]]$opts, $srcName = $name) {
  if ($Only.Count -gt 0 -and $Only -notcontains $name) { return }
  $src = Join-Path (Join-Path $SrcRoot $srcName) $srcSub
  $bld = Join-Path $BuildRoot $name
  Log "build $name"
  if (Test-Path $bld) { Remove-Item -Recurse -Force $bld }
  $common = @("-DCMAKE_INSTALL_PREFIX=$PrefixFwd", "-DCMAKE_PREFIX_PATH=$PrefixFwd",
      '-DCMAKE_POLICY_VERSION_MINIMUM=3.5', '-DBUILD_SHARED_LIBS=OFF')
  if ($IsMsvc) {
    # XP: CMP0091 NEW, else old-policy projects get an empty RuntimeLibrary (= /MT) with v141_xp
    $rt = @('-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL') + $(if ($IsXp) { @('-DCMAKE_POLICY_DEFAULT_CMP0091=NEW') } else { @() })
    Run cmake (@('-S', $src, '-B', $bld, '-G', $Generator, '-A', $Arch, '-T', $Toolset) + $rt + $common + $opts)
    Run cmake @('--build', $bld, '--config', 'Release', '--target', 'install', '--', '-m', '-v:minimal')
  } else {
    # the toolchain file searches libraries only under CMAKE_FIND_ROOT_PATH: the prefix is that root
    Run cmake (@('-S', $src, '-B', $bld, '-G', 'Ninja', "-DCMAKE_TOOLCHAIN_FILE=$Toolchain",
        "-DAVS_LLVM_MINGW_ROOT=$LlvmMingwRoot", '-DCMAKE_BUILD_TYPE=Release', "-DCMAKE_FIND_ROOT_PATH=$PrefixFwd") + $common + $opts)
    Run cmake @('--build', $bld, '--target', 'install')
  }
}

# ---- main ---------------------------------------------------------------------------------------------
if ($Clean) {
  foreach ($d in $Prefix, $BuildRoot) { if (Test-Path $d) { Remove-Item -Recurse -Force $d } }
}
New-Item -ItemType Directory -Force $SrcRoot, $BuildRoot, $Prefix | Out-Null
foreach ($n in $Used) { Get-Source $n | Out-Null }

# zlib-ng: the pinned commit must contain the #2485 fix (VEX AVX-VNNI on MSVC)
& git -C "$SrcRoot\zlib-ng" fetch -q --depth 200 origin $Pins['zlib-ng'].ref 2>$null
& git -C "$SrcRoot\zlib-ng" merge-base --is-ancestor 4564ba4 HEAD 2>$null
if ($LASTEXITCODE -ne 0) { Log 'WARNING: cannot confirm that zlib-ng contains 4564ba4 (#2485 fix)' }

# DevIL source edits (from the guide): no ILUT, static ILU, newer JasPer, UTF-8 ILU error headers
$devil = "$SrcRoot\DevIL"
$cm = "$devil\DevIL\CMakeLists.txt"
(Get-Content $cm) | Where-Object { $_ -notmatch 'add_subdirectory\s*\(\s*src-ILUT' } | Set-Content $cm
$cmu = "$devil\DevIL\src-ILU\CMakeLists.txt"
(Get-Content $cmu) -replace '(add_library\s*\(\s*ILU)\s+SHARED', '$1' | Set-Content $cmu
$patch = "$SrcRoot\0001-jasper-git.patch"
if (-not (Test-Path $patch)) { Run curl.exe @('-sSL', '-o', $patch, $JasperPatchUrl) }
Run git @('-C', $devil, 'apply', '--whitespace=nowarn', $patch)
foreach ($lang in 'french', 'german', 'italian', 'spanish') {
  $h = "$devil\DevIL\src-ILU\include\ilu_error\ilu_err-$lang.h"
  $txt = [IO.File]::ReadAllText($h, [Text.Encoding]::GetEncoding('ISO-8859-1'))
  [IO.File]::WriteAllText($h, $txt, (New-Object Text.UTF8Encoding($false)))
}

# XP: JasPer's total memory query (GetPhysicallyInstalledSystemMemory, Vista SP1+, sysinfoapi.h) ->
# GlobalMemoryStatusEx (XP), so JasPer still knows the memory size (the guide's kludge makes it 0)
if ($IsXp) {
  $jm = "$SrcRoot\jasper\src\libjasper\base\jas_malloc.c"
  $t = [IO.File]::ReadAllText($jm)
  # Sanity checks instead of trusting the pattern: the JasPer source may have changed shape since 4.2.9.
  $incPat = '(?m)^#\s*include <sysinfoapi\.h>\r?\n'
  $bodyPat = '(?s)ULONGLONG mem_size_in_kb;.*?: SIZE_MAX;'
  if ([regex]::Matches($t, $incPat).Count -ne 1) { throw 'JasPer XP patch: expected exactly one sysinfoapi.h include in jas_malloc.c' }
  $bm = [regex]::Matches($t, $bodyPat)
  if ($bm.Count -ne 1) { throw "JasPer XP patch: expected one memory query body, found $($bm.Count) (jas_malloc.c changed?)" }
  # the replaced span must be the small body of the one function: short, and it must hold the Vista+ call
  $span = $bm[0].Value
  if (($span -split "`n").Count -gt 20 -or $span -notmatch 'GetPhysicallyInstalledSystemMemory') { throw 'JasPer XP patch: the matched span is not the GetPhysicallyInstalledSystemMemory block' }
  $t2 = [regex]::Replace($t, $incPat, '')
  $t2 = [regex]::Replace($t2, $bodyPat,
    "MEMORYSTATUSEX ms; /* XP: GlobalMemoryStatusEx instead of GetPhysicallyInstalledSystemMemory (Vista SP1+) */`n`tms.dwLength = sizeof(ms);`n`tif (!GlobalMemoryStatusEx(&ms)) {`n`t`treturn 0;`n`t}`n`treturn (ms.ullTotalPhys < SIZE_MAX) ? JAS_CAST(size_t, ms.ullTotalPhys) : SIZE_MAX;")
  $open = { param($s) ($s.ToCharArray() | Where-Object { $_ -eq '{' }).Count - ($s.ToCharArray() | Where-Object { $_ -eq '}' }).Count }
  if ($t2 -match 'GetPhysicallyInstalledSystemMemory\(' -or $t2 -match 'sysinfoapi\.h' -or
      [regex]::Matches($t2, 'GlobalMemoryStatusEx\(').Count -ne 1 -or (& $open $t2) -ne (& $open $t)) {
    throw 'JasPer XP patch did not apply cleanly (Vista+ call left, or the new call count / brace balance is wrong)'
  }
  [IO.File]::WriteAllText($jm, $t2)
}

$cpu = if ($Arch -eq 'x64') { 'x86_64' } else { 'i686' }
Build-CMake 'zlib-ng'       '' @('-DZLIB_COMPAT=ON', '-DZLIB_ENABLE_TESTS=OFF', '-DWITH_GTEST=OFF', "-DPKGCONFIG_INSTALL_DIR=$PrefixFwd/lib/pkgconfig")
Build-CMake 'xz'            '' @('-DXZ_TOOL_XZ=OFF', '-DXZ_TOOL_XZDEC=OFF', '-DXZ_TOOL_LZMADEC=OFF', '-DXZ_TOOL_LZMAINFO=OFF', '-DXZ_TOOL_SCRIPTS=OFF', '-DXZ_DOC=OFF', '-DXZ_NLS=OFF', '-DBUILD_TESTING=OFF')
Build-CMake 'libjpeg-turbo' '' @('-DENABLE_SHARED=OFF', "-DCMAKE_SYSTEM_PROCESSOR=$cpu", '-DWITH_TURBOJPEG=OFF', '-DWITH_CRT_DLL=ON')  # else /MT
# zlib-ng (ZLIB_COMPAT) static library name: zlibstatic.lib (MSVC), libz.a (MinGW)
$zlibLib = @("$PrefixFwd/lib/zlibstatic.lib", "$PrefixFwd/lib/libz.a", "$PrefixFwd/lib/libzlibstatic.a") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $zlibLib -and ($Only.Count -eq 0 -or $Only -contains 'libpng')) { throw 'zlib-ng static library not found in the prefix' }
Build-CMake 'libpng'        '' @('-DPNG_SHARED=OFF', '-DPNG_TESTS=OFF', '-DPNG_TOOLS=OFF', "-DZLIB_INCLUDE_DIR=$PrefixFwd/include", "-DZLIB_LIBRARY=$zlibLib")
Build-CMake 'jbigkit'       '' @()
Build-CMake 'libdeflate'    '' @('-DLIBDEFLATE_BUILD_SHARED_LIB=OFF', '-DLIBDEFLATE_BUILD_GZIP=OFF', $(if ($IsMsvc) { '-DCMAKE_C_FLAGS_RELEASE=/MD /O2 /DNDEBUG' } else { '-DCMAKE_C_FLAGS_RELEASE=-O2 -DNDEBUG' })) $(if ($IsXp) { 'libdeflate-xp' } else { 'libdeflate' })  # MSVC: its default drops /MD
Build-CMake 'lerc'          '' @()
Build-CMake 'zstd'          'build/cmake' (@('-DZSTD_BUILD_SHARED=OFF', '-DZSTD_BUILD_PROGRAMS=OFF', '-DZSTD_BUILD_TESTS=OFF') + $(if ($IsXp) { @('-DZSTD_MULTITHREAD_SUPPORT=OFF') } else { @() }))  # XP: no condition variables
Build-CMake 'libtiff'       '' @('-Dtiff-docs=OFF', '-Dtiff-tools=OFF', '-Dtiff-tests=OFF', '-Dtiff-contrib=OFF', '-Dwebp=OFF', $(if ($IsMsvc) { '-DCMAKE_C_FLAGS=-DLZMA_API_STATIC /DWIN32 /D_WINDOWS' } else { '-DCMAKE_C_FLAGS=-DLZMA_API_STATIC' }))
# libtiff's installed CMake package does not import its dependencies (Deflate::Deflate etc. unknown to users):
# remove it, so DevIL's find_package(TIFF) uses the plain library (the deps are linked explicitly by ImageSeq)
$tiffcfg = "$Prefix\lib\cmake\tiff"
if (Test-Path $tiffcfg) { Remove-Item -Recurse -Force $tiffcfg }
Build-CMake 'libsquish'     '' @('-DBUILD_SQUISH_WITH_OPENMP=OFF')
Build-CMake 'jasper'        '' (@('-DJAS_ENABLE_SHARED=OFF', '-DJAS_ENABLE_OPENGL=OFF', '-DJAS_ENABLE_DOC=OFF', '-DJAS_ENABLE_PROGRAMS=OFF', '-DALLOW_IN_SOURCE_BUILD=ON', '-DJAS_ENABLE_MULTITHREADING_SUPPORT=OFF') + $(if (-not $IsMsvc) { @('-DJAS_STDC_VERSION=201710L') } else { @() }))  # cross build: cannot run its C version probe
Build-CMake 'DevIL'         'DevIL' @('-DCMAKE_CXX_STANDARD=14')

# every static library must use the DLL C runtime (/MD), like ImageSeq: a /MT library pulls in LIBCMT
# (LNK4098) and mixes two C runtimes in one DLL. MSVC only (llvm-mingw: always the UCRT DLL)
if ($IsMsvc) {
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$dumpbin = Get-ChildItem "$vs\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe" | Select-Object -Last 1
$bad = @()
foreach ($l in Get-ChildItem "$Prefix\lib" -Filter *.lib) {
  $d = & $dumpbin.FullName /directives $l.FullName | Select-String -Pattern 'DEFAULTLIB:"?LIBCMT' -SimpleMatch:$false
  if ($d) { $bad += $l.Name }
}
if ($bad.Count) { throw "static C runtime (/MT) in: $($bad -join ', ')" }
Log 'CRT check: all libraries use the DLL runtime (/MD)'
}
# stamp: what this prefix was built from
$stamp = foreach ($n in $Used) {
  $p = $Pins[$n]
  if ($p.kind -eq 'git') { "$n $($p.url) $((git -C "$SrcRoot\$n" rev-parse HEAD))" } else { "$n $($p.url)" }
}
$stamp += if ($IsMsvc) { "toolset $Toolset $Generator $Arch" } else { "toolset llvm-mingw $LlvmMingwRoot $Arch" }
$stamp | Set-Content "$Prefix\deps_stamp.txt"
Log "done: $Prefix"
