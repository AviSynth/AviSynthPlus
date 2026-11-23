# make_release.ps1 - full Windows release build of AviSynth+ (see make_release.bat, the entry point).
#
# Produces what the 3.7.6 release shipped for Windows (qyot27's packs; distrib/docs/release_automation_plan.md
# §2.1 has the verified layout), from source, dependencies included:
#   variant (internal) toolset                    user-facing name (pack folder, installer)   min. Windows
#   x86-v145, x64-v145  v145 (current VS 2026)    x86\ x64\         <name>.exe (the default)   10
#   with -ClangCL, both modern builds:
#   x86-clangcl, x64-clangcl  ClangCL (VS 2026)   x86\ x64\         <name>.exe (the default)   10
#   x86-v145, x64-v145  v145, MSVC                x86-msvc\ x64-msvc\  <name>_msvc.exe         10
#   x86-v143, x64-v143  v143 14.44 (VS 2026 -T)   x86-win7\ x64-win7\  <name>_win7.exe         7 SP1
#   x86-xp, x64-xp      v141_xp (VS 2022)         x86-xp\ x64-xp\   <name>_xp.exe              XP SP3
#   arm64               llvm-mingw aarch64        arm64\            <name>_arm64.exe           10 on ARM
#   Users see the OS, not the toolset (toolset names only in build folders, log, -Variants).
#   ImageSeq: static DevIL chain (v143 libs, also linked by v145/ClangCL; v141_xp libs for XP; llvm-mingw for ARM64).
#   pack      <name>-filesonly\ (licences, html\, sdk\c_loader\, the folders above) + .7z, <name>-pdb.7z
#   installer (stage 'installer', Inno Setup scripts in distrib\WinInstaller)
#
# build_release.bat (the dev/test build) is independent: own build dirs (bld_*), own output; both share only
# build_info.cmake and the dependency prefixes under C:\avsplus_build_deps.
#
# Usage: make_release.ps1 [-Stages deps,build,check,docs,pack,installer] [-Variants ...] [-Tag name] [-ClangCL]
param(
  [string[]]$Stages = @('deps', 'build', 'check', 'docs', 'pack'),
  [string[]]$Variants = @('x86-v145', 'x64-v145', 'x86-v143', 'x64-v143', 'x86-xp', 'x64-xp', 'arm64'),
  [string]$Tag = '',
  [ValidateSet('auto', 'all', 'off')][string]$DepsBuild = $(if ($env:DEPS_BUILD) { $env:DEPS_BUILD } else { 'auto' }),
  [int]$Jobs = 8,
  [bool]$RefreshRedist = $true,   # download the latest VC_redist.<arch>.exe before the installers
  [switch]$ClangCL                # also build x86/x64 with clang-cl: the default modern build, MSVC v145 becomes _msvc
)
$ErrorActionPreference = 'Stop'

# ======================================================================================================
#  CONFIGURATION (machine specific: the release machine only)
# ======================================================================================================
$SourceRoot   = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$OutRoot      = Join-Path $PSScriptRoot 'release-out'          # final assets
$BuildRoot    = Join-Path $PSScriptRoot 'release-build'        # bld_rel_* build dirs, logs
$DepsRoot     = 'C:\avsplus_build_deps'
$LlvmMingw    = "$DepsRoot\llvm-mingw-20261020-ucrt-x86_64"
$SevenZip     = 'C:\Program Files\7-Zip\7z.exe'
# Inno Setup: 5.6.1 (unicode) for the main + XP installers, as 3.7.6 (5.5.7; 6.0 dropped XP); 6.x for ARM64 (as 3.7.6: 6.3.0)
# https://files.jrsoftware.org/is/5/innosetup-5.6.1-unicode.exe . Without IS5, IS6 is used and the XP installer gets MinVersion 6.1sp1 (not for XP)
$Iscc5        = 'C:\Program Files (x86)\Inno Setup 5\ISCC.exe'
$Iscc         = 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
# VC++ redistributables, in distrib\Prerequisites (keep.me): the Win7 installer gets the frozen 14.44 pair, the
# default (v145) installer the rolling VC_redist.<arch>.exe (refreshed below), the XP one the 14.27 pair (.iss default)
$VcRedistWin7X86 = 'VC_redist_14.44.35211.0.x86.exe'
$VcRedistWin7X64 = 'VC_redist_14.44.35211.0.x64.exe'
$Toolchain    = Join-Path $PSScriptRoot 'toolchains\llvm-mingw-aarch64.cmake'
$Helpers      = Join-Path $PSScriptRoot 'helpers'

# per variant: generator/toolset/arch, the dependency prefixes, and pub = the user-facing folder name
# (the v143 static libraries - DevIL chain, SoundTouch - link into v145 as well)
$Cfg = [ordered]@{
  'x86-v145' = @{ pub = 'x86'; gen = 'Visual Studio 18 2026'; arch = 'Win32'; ts = 'v145'; st = "$DepsRoot\v143-x86-32"; strm = 'x86'; devil = 'msvc-x86-v143'; modern = $true }
  'x64-v145' = @{ pub = 'x64'; gen = 'Visual Studio 18 2026'; arch = 'x64'; ts = 'v145'; st = "$DepsRoot\v143-x86-64"; strm = 'x64'; devil = 'msvc-x64-v143'; modern = $true }
  'x86-clangcl' = @{ pub = 'x86'; gen = 'Visual Studio 18 2026'; arch = 'Win32'; ts = 'clangcl'; st = "$DepsRoot\v143-x86-32"; strm = 'x86'; devil = 'msvc-x86-v143'; modern = $true }
  'x64-clangcl' = @{ pub = 'x64'; gen = 'Visual Studio 18 2026'; arch = 'x64'; ts = 'clangcl'; st = "$DepsRoot\v143-x86-64"; strm = 'x64'; devil = 'msvc-x64-v143'; modern = $true }
  'x86-v143' = @{ pub = 'x86-win7'; gen = 'Visual Studio 18 2026'; arch = 'Win32'; ts = 'v143'; st = "$DepsRoot\v143-x86-32"; strm = 'x86'; devil = 'msvc-x86-v143' }
  'x64-v143' = @{ pub = 'x64-win7'; gen = 'Visual Studio 18 2026'; arch = 'x64'; ts = 'v143'; st = "$DepsRoot\v143-x86-64"; strm = 'x64'; devil = 'msvc-x64-v143' }
  'x86-xp'   = @{ pub = 'x86-xp'; gen = 'Visual Studio 17 2022'; arch = 'Win32'; ts = 'v141_xp'; st = "$DepsRoot\xp-x86-32"; strm = 'x86'; devil = 'msvc-x86-v141_xp'; xp = $true }
  'x64-xp'   = @{ pub = 'x64-xp'; gen = 'Visual Studio 17 2022'; arch = 'x64'; ts = 'v141_xp'; st = "$DepsRoot\xp-x86-64"; strm = 'x64'; devil = 'msvc-x64-v141_xp'; xp = $true }
  'arm64'    = @{ pub = 'arm64'; gen = 'Ninja'; arch = 'arm64'; ts = 'llvm-mingw'; st = "$DepsRoot\mingw-arm64"; devil = 'mingw-arm64' }
}
if ($ClangCL) { $Variants = @(@('x86-clangcl', 'x64-clangcl') + $Variants | Select-Object -Unique) }
foreach ($v in $Variants) { if (-not $Cfg.Contains($v)) { throw "unknown variant '$v' (known: $($Cfg.Keys -join ', '))" } }
# clang-cl in the run: it is the default modern build (x86\ x64\, <name>.exe), MSVC v145 gets the -msvc / _msvc names
$HasClangCL = [bool]($Variants | Where-Object { $_ -like '*-clangcl' })
if ($HasClangCL) { $Cfg['x86-v145'].pub = 'x86-msvc'; $Cfg['x64-v145'].pub = 'x64-msvc' }
$ImageSeqLibsMsvc  = 'DevIL', 'jpeg-static', 'libpng16_static', 'tiff', 'squish', 'jasper', 'zlibstatic', 'lzma', 'jbig', 'deflatestatic', 'Lerc', 'zstd_static'
$ImageSeqLibsMingw = 'libIL', 'libjpeg', 'libpng16', 'libtiff', 'libsquish', 'libjasper', 'libz', 'liblzma', 'libjbig', 'libdeflate', 'libLerc', 'libzstd'

# ======================================================================================================
New-Item -ItemType Directory -Force $OutRoot, $BuildRoot | Out-Null
$Log = Join-Path $BuildRoot ("make_release_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
function Log($m) { $l = "[{0:HH:mm:ss}] $m" -f (Get-Date); Write-Host $l; Add-Content $Log $l }
function Fail($m) { Log "ERROR: $m"; throw $m }
function Run($exe, [string[]]$a, [string]$what) {
  Add-Content $Log ">> $exe $($a -join ' ')"
  # Windows PowerShell 5.1 turns redirected native stderr (e.g. CMake message() without mode) into an
  # error record, which 'Stop' makes fatal. Only the exit code decides here. Lines go to the log as plain
  # text (*>> would write UTF-16 and wrap stderr lines in error records there).
  $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try { & $exe @a 2>&1 | ForEach-Object { "$_" } | Add-Content $Log } finally { $ErrorActionPreference = $eap }
  if ($LASTEXITCODE -ne 0) { Fail "$what failed (exit $LASTEXITCODE), see $Log" }
}

# ---- build identification (same as build_release.bat) -------------------------------------------------
$infoCmd = Join-Path $BuildRoot 'build_info.cmd'
Run cmake @('-D', "REPO=$SourceRoot", '-D', "SRC=$SourceRoot\avs_core\core\version.h.in", '-D', "OUT=$infoCmd",
  '-P', (Join-Path $PSScriptRoot 'build_info.cmake')) 'build_info.cmake'
$Info = @{}
foreach ($l in Get-Content $infoCmd) { if ($l -match '^set "([A-Z_]+)=(.*)"$') { $Info[$matches[1]] = $matches[2] } }
$Name = "AviSynthPlus_$($Info.AVS_VER)_$($Info.BUILD_DATE)" + $(if ($Tag) { "_$Tag" } else { '' })
Log "release $Name  (r$($Info.BUILD_RNUM), $($Info.GIT_DESCRIBE)), stages: $($Stages -join ','), variants: $($Variants -join ',')"

# ======================================================================================================
#  STAGE deps
# ======================================================================================================
# DevIL chain id: msvc-<arch>-<toolset> (makedevil -Target msvc-<arch> -Toolset <toolset>) or mingw-arm64
function Get-DevilPrefix($d) { return "$DepsRoot\devil-$d" }
if ($Stages -contains 'deps') {
  Log '== stage deps'
  if ($DepsBuild -ne 'off') {
    # strmbase (DirectShowSource): v141_xp, serves all MSVC variants
    if ($DepsBuild -eq 'all' -or -not (Test-Path "$DepsRoot\strmbase\x64\strmbase.lib")) {
      Run cmd.exe @('/c', (Join-Path $Helpers 'makestrmbase.bat')) 'makestrmbase'
    }
    # SoundTouch (TimeStretch): five prefixes
    $stMissing = $Cfg.Values | Where-Object { -not (Test-Path "$($_.st)\lib\SoundTouch.lib") -and -not (Test-Path "$($_.st)\lib\libSoundTouch.a") }
    if ($DepsBuild -eq 'all' -or $stMissing) { Run cmd.exe @('/c', (Join-Path $Helpers 'makesoundtouch.bat')) 'makesoundtouch' }
    # DevIL chains (static): v143 (modern + Win7), v141_xp (XP), llvm-mingw (ARM64)
    $chains = $Variants | ForEach-Object { $Cfg[$_].devil } | Sort-Object -Unique
    foreach ($c in $chains) {
      $a = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $Helpers 'makedevil.ps1'))
      if ($c -match '^(msvc-x(?:86|64))-(.+)$') { $a += @('-Target', $matches[1], '-Toolset', $matches[2]) } else { $a += @('-Target', $c) }
      if ($DepsBuild -eq 'all') { $a += '-Clean' }
      $stamp = Join-Path (Get-DevilPrefix $c) 'deps_stamp.txt'
      # auto: rebuild when the stamp is missing or its pins differ from makedevil.ps1's pins
      $need = $DepsBuild -eq 'all' -or -not (Test-Path $stamp)
      if (-not $need) {
        $pins = Select-String (Join-Path $Helpers 'makedevil.ps1') -Pattern "ref = '([^']+)'" | ForEach-Object { $_.Matches[0].Groups[1].Value }
        $have = Get-Content $stamp -Raw
        $need = [bool]($pins | Where-Object { $_ -match '^[0-9a-f]{40}$' -and -not $have.Contains($_) })
      }
      if ($need) { Log "makedevil $c"; Run powershell.exe $a "makedevil $c" } else { Log "DevIL chain $c up to date" }
    }
  }
}

# ======================================================================================================
#  STAGE build
# ======================================================================================================
function Bld($v) { Join-Path $BuildRoot "bld_rel_$v" }
if ($Stages -contains 'build') {
  Log '== stage build'
  foreach ($v in $Variants) {
    $c = $Cfg[$v]; $b = Bld $v
    Log "build $v ($($c.gen), $($c.ts), $($c.arch))"
    if (Test-Path $b) { Remove-Item -Recurse -Force $b }
    $dp = Get-DevilPrefix $c.devil; $dpf = $dp -replace '\\', '/'
    $opt = @('-DENABLE_PLUGINS=ON', '-DBUILD_IMAGESEQ=ON', '-DBUILD_TIMESTRETCH=ON')
    if ($c.gen -eq 'Ninja') {
      $opt += @("-DCMAKE_TOOLCHAIN_FILE=$($Toolchain -replace '\\','/')", "-DAVS_LLVM_MINGW_ROOT=$($LlvmMingw -replace '\\','/')",
        '-DCMAKE_BUILD_TYPE=Release', '-DENABLE_INTEL_SIMD=OFF', '-DBUILD_VDUBFILTER=OFF', '-DBUILD_DIRECTSHOWSOURCE=OFF',
        "-DSOUNDTOUCH_LINK_LIBRARIES=$($c.st -replace '\\','/')/lib/libSoundTouch.a", "-DSOUNDTOUCH_INCLUDE_DIRS=$($c.st -replace '\\','/')/include/soundtouch",
        "-DIL_LIBRARIES=$(($ImageSeqLibsMingw | ForEach-Object { "$dpf/lib/$_.a" }) -join ';')", "-DILU_LIBRARIES=$dpf/lib/libILU.a",
        "-DIL_INCLUDE_DIR=$dpf/include/IL", '-DCMAKE_CXX_FLAGS=-DIL_STATIC_LIB',
        '-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG -g -gcodeview', '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -g -gcodeview',
        '-DAVS_MINGW_RELEASE_LINK_EXTRA=-Wl,--pdb= -Wl,-Xlink=/pdbaltpath:%_PDB%', "-DCMAKE_INSTALL_PREFIX=$((Join-Path $b 'install') -replace '\\','/')")
      Run cmake (@('-S', $SourceRoot, '-B', $b, '-G', 'Ninja') + $opt) "configure $v"
      Run cmake @('--build', $b, '--parallel', "$Jobs") "build $v"
      Run cmake @('--install', $b) "install $v"
    } else {
      $tsArg = if ($c.ts -eq 'clangcl') { 'ClangCL' } else { $c.ts }
      $strmInc = "$DepsRoot\strmbase\include" -replace '\\', '/'
      $strmLib = "$DepsRoot\strmbase\$($c.strm)\strmbase.lib" -replace '\\', '/'
      $opt += @('-DENABLE_INTEL_SIMD=ON', '-DBUILD_DIRECTSHOWSOURCE=ON',
        "-DDSHOWSRC_BASECLASSES_PATH=$strmInc", "-DDSHOWSRC_BASECLASSES_LIB=$strmLib",
        "-DSOUNDTOUCH_LINK_LIBRARIES=$($c.st)\lib\SoundTouch.lib", "-DSOUNDTOUCH_INCLUDE_DIRS=$($c.st)\include\soundtouch")
      # ImageSeq: the static DevIL chain of the variant's toolset (XP: the v141_xp chain), the same way for all
      $opt += @("-DIL_LIBRARIES=$(($ImageSeqLibsMsvc | ForEach-Object { "$dpf/lib/$_.lib" }) -join ';')",
        "-DILU_LIBRARIES=$dpf/lib/ILU.lib", "-DIL_INCLUDE_DIR=$dpf/include/IL",
        '-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /W3 /GR /EHsc -DIL_STATIC_LIB')
      if ($c.xp) {
        $opt += '-DWINXP_SUPPORT=ON'
        if ($c.arch -eq 'Win32') { $opt += '-DMSVC_CPU_ARCH=SSE2' }
      } else {
        $opt += '-DWINXP_SUPPORT=OFF'
      }
      # Release code (/O2 /Ob2) + PDB: /Zi, linked with /DEBUG but keeping /OPT:REF /OPT:ICF (which /DEBUG turns off
      # by default)
      $opt += @('-DCMAKE_CXX_FLAGS_RELEASE=/MD /O2 /Ob2 /DNDEBUG /Zi', '-DCMAKE_C_FLAGS_RELEASE=/MD /O2 /Ob2 /DNDEBUG /Zi',  # /MD: the runtime comes from these flags here (old CMake policy)
        '-DCMAKE_SHARED_LINKER_FLAGS_RELEASE=/DEBUG /OPT:REF /OPT:ICF /INCREMENTAL:NO')
      Run cmake (@('-S', $SourceRoot, '-B', $b, '-G', $c.gen, '-A', $c.arch, '-T', $tsArg) + $opt) "configure $v"
      # /PDBALTPATH:%_PDB% records only the PDB file name in the DLL, not the build folder. Given in the LINK environment
      # variable (read by link.exe and lld-link): through CMake the VS generator escapes the % and the DLL then records
      # "%AviSynth.pdb%" (link.exe) or "%%_PDB%%" (lld-link)
      $env:LINK = '/PDBALTPATH:%_PDB%'
      try { Run cmake @('--build', $b, '--config', 'Release', '--', "/maxcpucount:$Jobs", '/nodeReuse:false', '/v:minimal') "build $v" }
      finally { Remove-Item Env:LINK -ErrorAction SilentlyContinue }
    }
  }
}

# ======================================================================================================
#  STAGE check (binary checks; a failure stops the release)
# ======================================================================================================
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$Dumpbin = (Get-ChildItem "$vs\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe" | Select-Object -Last 1).FullName
$Objdump = "$LlvmMingw\bin\llvm-objdump.exe"
function Get-VariantFiles($v) {
  $b = Bld $v
  if ($Cfg[$v].gen -eq 'Ninja') {
    $i = Join-Path $b 'install'
    return @{ core = "$i\bin\AviSynth.dll"; plugins = @(Get-ChildItem "$i\lib\avisynth" -Filter *.dll | ForEach-Object FullName) }
  }
  return @{ core = "$b\Output\AviSynth.dll"; plugins = @(Get-ChildItem "$b\Output\plugins" -Filter *.dll | ForEach-Object FullName) }
}
# KERNEL32/ADVAPI32 functions that exist only from Vista (or later) on: an XP DLL must not import them
$VistaApis = '^(GetTickCount64|.*SRWLock.*|.*ConditionVariable.*|InitOnce\w+|CreateSymbolicLink\w*|GetFinalPathNameByHandle\w*|' +
  'InitializeCriticalSectionEx|Create(Event|Mutex|Semaphore|WaitableTimer)Ex\w*|GetFileInformationByHandleEx|' +
  'SetFileInformationByHandle|CompareStringEx|LCMapStringEx|GetLocaleInfoEx|\w*LocaleName\w*|' +
  'GetPhysicallyInstalledSystemMemory|QueryFullProcessImageName\w*|GetCurrentProcessorNumber\w*|CancelIoEx|' +
  '\w*Threadpool\w*|GetErrorMode|SetThreadErrorMode|K32\w+|GetDynamicTimeZoneInformation|Fls(Alloc|Free|GetValue|SetValue)|' +
  'GetThreadId|GetProcessIdOfThread|GetSystemTimePreciseAsFileTime|GetLogicalProcessorInformationEx|CreateFile2|RegGetValue\w*)$'
if ($Stages -contains 'check') {
  Log '== stage check'
  $problems = @()
  foreach ($v in $Variants) {
    $f = Get-VariantFiles $v; $all = @($f.core) + $f.plugins
    $expected = if ($Cfg[$v].gen -eq 'Ninja') { 4 } else { 6 }   # ConvertStacked DirectShowSource ImageSeq Shibatch TimeStretch VDubFilter
    if ($f.plugins.Count -ne $expected) { $problems += "$v`: $($f.plugins.Count) plugins, expected $expected" }
    foreach ($dll in $all) {
      if (-not (Test-Path $dll)) { $problems += "$v`: missing $dll"; continue }
      if ($Cfg[$v].gen -eq 'Ninja') {
        $imp = & $Objdump -p $dll | Select-String 'DLL Name: (.+)$' | ForEach-Object { $_.Matches[0].Groups[1].Value }
        $bad = $imp | Where-Object { $_ -match 'libc\+\+|libunwind|libwinpthread|libgcc|libstdc' }
        if ($bad) { $problems += "$v`: $(Split-Path $dll -Leaf) imports $($bad -join ',')" }
      } else {
        $h = & $Dumpbin /headers $dll | Out-String
        $sub = [regex]::Match($h, '([\d.]+) subsystem version').Groups[1].Value
        $want = if ($Cfg[$v].xp) { if ($Cfg[$v].arch -eq 'x64') { '5.02' } else { '5.01' } } else { '6.00' }   # v145: measured below
        if ($sub -ne $want) { $problems += "$v`: $(Split-Path $dll -Leaf) subsystem $sub, expected $want" }
        if ($Cfg[$v].xp) {
          # XP: no import of a Vista+ API (e.g. zstd's multithreading: condition variables). Builds silently, but the
          # DLL then does not load on XP ("entry point not found"). A list of the usual suspects, not complete.
          $imps = & $Dumpbin /imports $dll | Select-String '^\s+[0-9A-F]+\s+(\w+)\s*$' | ForEach-Object { $_.Matches[0].Groups[1].Value }
          $vista = $imps | Where-Object { $_ -match $VistaApis }
          if ($vista) { $problems += "$v`: $(Split-Path $dll -Leaf) imports Vista+ functions: $(($vista | Sort-Object -Unique) -join ', ')" }
        }
      }
    }
    # PDB reference: exactly <dll name>.pdb (no build machine folder, no unexpanded/escaped %_PDB%)
    foreach ($dll in $all) {
      if (-not (Test-Path $dll)) { continue }
      # llvm-readobj prints the CodeView record (PDBFileName) for MSVC, lld-link and llvm-mingw DLLs alike
      $hd = & "$LlvmMingw\bin\llvm-readobj.exe" --coff-debug-directory $dll | Out-String
      $pdbRef = [regex]::Match($hd, 'PDBFileName: (.*)').Groups[1].Value.Trim()
      $want = [IO.Path]::GetFileNameWithoutExtension($dll) + '.pdb'
      if (-not $pdbRef) { $problems += "${v}: $(Split-Path $dll -Leaf) has no PDB reference" }
      elseif ($pdbRef -ne $want) { $problems += "${v}: $(Split-Path $dll -Leaf) records PDB '$pdbRef', expected '$want'" }
    }
  }
  if ($problems) { $problems | ForEach-Object { Log "CHECK: $_" }; Fail "$($problems.Count) check(s) failed" }
  Log 'checks passed'
}

# ======================================================================================================
#  STAGE docs (sphinx HTML, shipped in the files-only pack as html\, as 3.7.6)
# ======================================================================================================
$DocsHtml = "$SourceRoot\distrib\docs\english\build\html"
if ($Stages -contains 'docs') {
  Log '== stage docs'
  Run python @('-m', 'sphinx', '-b', 'html', '-q', "$SourceRoot\distrib\docs\english\source", $DocsHtml) 'sphinx'
}

# ======================================================================================================
#  STAGE pack (files-only tree as 3.7.6 + sdk\c_loader, and the .7z)
# ======================================================================================================
if ($Stages -contains 'pack') {
  Log '== stage pack'
  $P = Join-Path $OutRoot "$Name-filesonly"
  if (Test-Path $P) { Remove-Item -Recurse -Force $P }
  New-Item -ItemType Directory -Force $P | Out-Null
  foreach ($lic in 'gpl.txt', 'lgpl_for_used_libs.txt', 'OFL.TXT') { Copy-Item "$SourceRoot\distrib\$lic" $P }
  if (-not (Test-Path "$DocsHtml\index.html")) { Fail "no HTML docs at $DocsHtml (run stage docs)" }
  Copy-Item $DocsHtml "$P\html" -Recurse
  Copy-Item "$SourceRoot\distrib\FilterSDK\c_loader" "$P\sdk\c_loader" -Recurse -Force
  foreach ($v in $Variants) {
    $b = Bld $v; $d = Join-Path $P $Cfg[$v].pub
    if ($Cfg[$v].gen -eq 'Ninja') {
      Copy-Item "$b\install" $d -Recurse        # unmodified install tree (bin, lib, lib\avisynth, lib\pkgconfig, include, share)
      continue
    }
    New-Item -ItemType Directory -Force "$d\plugins", "$d\c_api", "$d\system" | Out-Null
    Copy-Item "$b\Output\AviSynth.dll" $d
    Copy-Item "$b\avs_core\Release\AviSynth.lib" "$d\c_api"
    # .exp: written by link.exe only (as in 3.7.6), lld-link (clang-cl) does not; not needed to link against AviSynth
    if (Test-Path "$b\avs_core\Release\AviSynth.exp") { Copy-Item "$b\avs_core\Release\AviSynth.exp" "$d\c_api" }
    Copy-Item "$b\Output\include" "$d\include" -Recurse
    Get-ChildItem "$b\Output\plugins" -Filter *.dll | Copy-Item -Destination "$d\plugins"
    # system\ stays empty for all variants (3.7.6: DevIL.dll of the DevIL SDK for XP; now DevIL is static there too)
  }
  # PDBs (separate asset): same layout as the binaries, <variant>\AviSynth.pdb, <variant>\plugins\<plugin>.pdb
  $PP = Join-Path $OutRoot "$Name-pdb"
  if (Test-Path $PP) { Remove-Item -Recurse -Force $PP }
  foreach ($v in $Variants) {
    $f = Get-VariantFiles $v; $b = Bld $v
    foreach ($dll in @($f.core) + $f.plugins) {
      $pdbName = [IO.Path]::GetFileNameWithoutExtension($dll) + '.pdb'
      $pdb = Get-ChildItem $b -Recurse -Filter $pdbName -File | Where-Object { $_.FullName -notmatch '\\(obj|CMakeFiles)\\|\\[^\\]+\.dir\\' } | Select-Object -First 1
      if (-not $pdb) { Fail "${v}: no $pdbName in $b" }
      $pub = $Cfg[$v].pub; $dst = if ($dll -eq $f.core) { "$PP\$pub" } else { "$PP\$pub\plugins" }
      New-Item -ItemType Directory -Force $dst | Out-Null
      Copy-Item $pdb.FullName $dst
    }
  }
  $zp = Join-Path $OutRoot "$Name-pdb.7z"
  if (Test-Path $zp) { Remove-Item -Force $zp }
  Run $SevenZip @('a', '-mx=9', '-mmt=on', $zp, $PP) '7-Zip (pdb)'
  Log ("pdb: {0} ({1:N0} bytes)" -f $zp, (Get-Item $zp).Length)

  $z = Join-Path $OutRoot "$Name-filesonly.7z"
  if (Test-Path $z) { Remove-Item -Force $z }
  Run $SevenZip @('a', '-mx=9', '-mmt=on', $z, $P) '7-Zip'
  Log ("pack: {0} ({1:N0} bytes)" -f $z, (Get-Item $z).Length)
  # which 7-Zip can extract the packs (for the release notes). 7-Zip 23.00+ applies its ARM64 filter (method 0A) to
  # ARM64 DLLs automatically; 7-Zip before 23.00 then reports "Unsupported method" for those files (the 3.7.6 pack had
  # it). Current 7-Zip still runs on XP/2000, so an XP user just needs a current 7-Zip.
  foreach ($arc in $zp, $z) {
    $methods = @(& $SevenZip l -slt $arc | Select-String '^Method = (.+)$' | ForEach-Object { $_.Matches[0].Groups[1].Value -split ' ' } |
      ForEach-Object { ($_ -split ':')[0] } | Where-Object { $_ } | Sort-Object -Unique)
    $need = if ($methods -contains 'RISCV') { '24.03' } elseif ($methods -contains '0A' -or $methods -contains 'ARM64') { '23.00' } else { '9.20' }
    Log ("{0}: methods {1}, extracts with 7-Zip {2} or newer" -f (Split-Path $arc -Leaf), ($methods -join ' '), $need)
  }
}

# ======================================================================================================
#  STAGE installer (Inno Setup; scripts in distrib\WinInstaller, driven by /D defines)
#    main  : x86 + x64 (v143), MinVersion Win7 SP1, the 14.44 VC++ redistributable (as 3.7.6), docs + SDK components
#    xp    : x86-xp + x64-xp, the 14.27 XP redistributable included (script default). Inno Setup 5.6.1 (the last one
#            supporting XP) with MinVer=5.1sp3; without it Inno Setup 6 with 6.1sp1 (6.3+/7.x reject < 6.1): not for XP.
#    arm64 : the cmake --install tree as usr\aarch64-w64-mingw32 (the script's layout), docs + SDK on
# ======================================================================================================
if ($Stages -contains 'installer') {
  Log '== stage installer'
  if (-not (Test-Path $Iscc)) { Fail "ISCC (Inno Setup 6) not found: $Iscc" }
  $HaveIs5 = Test-Path $Iscc5
  if (-not $HaveIs5) { Log "WARNING: Inno Setup 5 not found ($Iscc5): main + XP installers with Inno Setup 6, the XP one will not run on XP" }
  if (-not (Test-Path "$DocsHtml\index.html")) { Fail "no HTML docs at $DocsHtml (run stage docs)" }
  $S = Join-Path $BuildRoot 'installer-stage'
  if (Test-Path $S) { Remove-Item -Recurse -Force $S }
  foreach ($v in $Variants) {
    $b = Bld $v
    if ($Cfg[$v].gen -eq 'Ninja') {
      $t = "$S\arm64\usr\aarch64-w64-mingw32"; New-Item -ItemType Directory -Force (Split-Path $t) | Out-Null
      Copy-Item "$b\install" $t -Recurse
    } else {
      Copy-Item "$b\Output" "$S\$v\Output" -Recurse
    }
  }
  # the rolling redistributable (v145 build) is refreshed before every release (distrib\Prerequisites\keep.me);
  # the 14.44 and 14.27 ones are frozen and kept under their versioned names
  if ($RefreshRedist) {
    foreach ($a in 'x86', 'x64', 'arm64') {
      $dst = "$SourceRoot\distrib\Prerequisites\VC_redist.$a.exe"; $tmp = "$dst.download"
      Run curl.exe @('-sSL', '--fail', '-o', $tmp, "https://aka.ms/vc14/vc_redist.$a.exe") "download VC_redist.$a.exe"
      $sig = Get-AuthenticodeSignature $tmp
      if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') { Fail "VC_redist.$a.exe: signature not valid ($($sig.Status))" }
      Move-Item -Force $tmp $dst
      Log ("VC_redist.{0}.exe refreshed: {1}" -f $a, (Get-Item $dst).VersionInfo.ProductVersion)
    }
  }
  $wi = "$SourceRoot\distrib\WinInstaller"
  $innoJobs = @()
  if ($Variants -contains 'x86-v143' -and $Variants -contains 'x64-v143') {
    # Windows 7 SP1 / 8.x: the frozen 14.44 redistributable
    $innoJobs += @{ iss = 'avisynth+.iss'; out = "${Name}_win7"; is5 = $true; d = @("/DBuildDir32=$S\x86-v143", "/DBuildDir64=$S\x64-v143", '/DMinVer=6.1sp1', '/DWITH_VC_REDIST', "/DVcRedistX86=$VcRedistWin7X86", "/DVcRedistX64=$VcRedistWin7X64") }
  }
  # Windows 10+ (modern builds): the rolling (latest) redistributable, installed as VC_redist.x86/x64.exe (script
  # defaults). clang-cl, when built, is the default <name>.exe and MSVC v145 the <name>_msvc.exe
  $modernJobs = @(@{ ts = 'clangcl'; out = $Name }, @{ ts = 'v145'; out = $(if ($HasClangCL) { "${Name}_msvc" } else { $Name }) })
  foreach ($m in $modernJobs) {
    if ($Variants -contains "x86-$($m.ts)" -and $Variants -contains "x64-$($m.ts)") {
      $innoJobs += @{ iss = 'avisynth+.iss'; out = $m.out; is5 = $true; d = @("/DBuildDir32=$S\x86-$($m.ts)", "/DBuildDir64=$S\x64-$($m.ts)", '/DMinVer=10.0', '/DWITH_VC_REDIST') }
    }
  }
  if ($Variants -contains 'x86-xp' -and $Variants -contains 'x64-xp') {
    $xpMin = if ($HaveIs5) { '5.1sp3' } else { '6.1sp1' }   # IS 6.3+/7.x: MinVersion >= 6.1 (6.2.x accepted 6.0)
    $innoJobs += @{ iss = 'avisynth+.iss'; out = "${Name}_xp"; is5 = $true; d = @("/DBuildDir32=$S\x86-xp", "/DBuildDir64=$S\x64-xp", "/DMinVer=$xpMin") }
  }
  if ($Variants -contains 'arm64') {
    $innoJobs += @{ iss = 'avisynth+_arm64.iss'; out = "${Name}_arm64"; d = @("/DBuildDir64=$S\arm64", '/DWITH_SDK', '/DWITH_DOCS') }
  }
  Push-Location $wi
  try {
    foreach ($j in $innoJobs) {
      $cc = if ($j.is5 -and $HaveIs5) { $Iscc5 } else { $Iscc }
      Log "ISCC $($j.iss) -> $($j.out).exe ($cc)"
      Run $cc (@('/Q', "/O$OutRoot", "/F$($j.out)") + $j.d + @($j.iss)) "ISCC $($j.out)"
      Log ("installer: {0}\{1}.exe ({2:N0} bytes)" -f $OutRoot, $j.out, (Get-Item "$OutRoot\$($j.out).exe").Length)
    }
  } finally { Pop-Location }
}

# ======================================================================================================
#  SHA-256 list of the assets (for the release notes)
# ======================================================================================================
if ($Stages -contains 'pack' -or $Stages -contains 'installer') {
  $assets = Get-ChildItem $OutRoot -File | Where-Object { $_.Name -like "$Name*" -and $_.Extension -in '.7z', '.exe' }
  $assets | ForEach-Object { '{0}  {1}' -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name } |
    Set-Content (Join-Path $OutRoot "$Name-SHA256SUMS.txt")
}
Log "done: $Name"
