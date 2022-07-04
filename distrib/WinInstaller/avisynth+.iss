;!!!Conditional!!!  comment out the following define to ignore VS redistributables
;#define WITH_VC_REDIST
#define WITH_VC_REDIST_XP
;Note 1: do not enable both.
;Note 2: XP: Though Microsoft Visual C++ Redistributable (2015-2022) still installs on older OS's,
;XP support has been ended, an older redistributable (officially 14.27.29114.0) needed for XP
;along with a specially built version of Avisynth+ (v141_xp toolset, /Zc:threadSafeInit-).
;XP version of redistributable must be renamed to VC_redist_14.27.29114.0.x86.exe and VC_redist_14.27.29114.0.x64.exe

;headers and c lib are OK, but documentation is not up to date, anyway we include them
#define WITH_SDK

;this section is not up to date, don't include them
#define WITH_DOCS

;!!!Conditional!!!  comment out the following define to include project web url
;original Avs+ project page is not maintained at the moment (Dec. 2017) and contains very old content
;#define WITH_AVSPLUS_URL

#define AvsName "AviSynth+"
#define AvsFriendlyName "AviSynthPlus"
#define AvsPublisher "The Public"
#define AppId "{AC78780F-BACA-4805-8D4F-AE1B52B7E7D3}"
#define AvsGitURL "https://github.com/AviSynth/AviSynthPlus/"

#ifdef WITH_AVSPLUS_URL
#define AvsWebURL "https://github.com/AviSynth/AviSynthPlus/"
#endif

;There is no specific x86/x64 output directory.
;folders must be created relative to the folder of iss script.
;Build x86 avs+ then copy the Output folder (with the folder itself) here
#define BuildDir32 "x86"
;Build x64 avs+ then copy the Output folder (with the folder itself) here
#define BuildDir64 "x64"
;.\x86\Output\...
;.\x64\Output\...

#ifdef WITH_VC_REDIST_XP
#define VcVersion "Microsoft Visual C++ Redistributable for Visual Studio 2015-2019 (latest XP compatible)"
#else
#define VcVersion "Microsoft Visual C++ Redistributable for Visual Studio 2015-2022"
#endif

#define BuildDate GetFileDateTimeString(AddBackslash(BuildDir32) + "Output\AviSynth.dll", 'yyyy/mm/dd', '-',);

#expr Exec("powershell", "-ExecutionPolicy unrestricted -File update_git_rev.ps1", SourcePath, 1)
#define IniFile AddBackslash(SourcePath) + "git_rev.ini"
#define RevisionNumber  ReadIni(IniFile, "Version", "RevisionNumber" )
#define Revision        ReadIni(IniFile, "Version", "Revision"       )
#define IsRelease       ReadIni(IniFile, "Version", "IsRelease"      )
#define Version         ReadIni(IniFile, "Version", "Version"        )
#define Branch          ReadIni(IniFile, "Version", "Branch"         )

[Setup]
AppId={{#AppId}
AppName={#AvsName}
AppVersion={#Version}.{#RevisionNumber}
#if IsRelease == "True"
  AppVerName={#AvsName} {#Version}
  OutputBaseFilename={#AvsFriendlyName}_{#Version}
#else
  AppVerName={#AvsName} {#Version} r{#RevisionNumber}
  OutputBaseFilename={#AvsFriendlyName}-r{#RevisionNumber}-{#Branch}-{#Revision}
#endif
AppPublisher={#AvsPublisher}
#ifdef WITH_AVSPLUS_URL
AppPublisherURL={#AvsWebURL}
AppSupportURL={#AvsWebURL}/get_started.html
#endif
AppUpdatesURL={#AvsGitURL}/releases
AppReadmeFile={#AvsGitURL}/blob/master/README.rst
VersionInfoVersion={#Version}.{#RevisionNumber}
DefaultDirName={pf}\{#AvsName}
DefaultGroupName={#AvsName}
DisableWelcomePage=no
DisableProgramGroupPage=yes
OutputDir={#BuildDir32}\..
SetupIconFile=..\Icons\Ico\InstIcon.ico
UninstallDisplayIcon=..\Icons\Ico\InstIcon.ico
WizardImageFile=WizardImageBig.bmp
WizardSmallImageFile=WizardImageSmall.bmp
ChangesAssociations=yes
ChangesEnvironment=yes
;Compression=lzma2/max
Compression=lzma2/ultra
SolidCompression=yes
SetupLogging=yes
MinVersion=5.1sp3

[Types]
Name: "compact"; Description: "{cm:CompactInstallation}"
Name: "full"; Description: "{cm:FullInstallation}"
Name: "custom"; Description: "{cm:CustomInstallation}"; Flags: iscustom

[Languages]
Name: "en"; MessagesFile: "Translations\en.isl"; LicenseFile: ..\gpl.txt
Name: "pt_br"; MessagesFile: "Translations\pt_br.isl"; LicenseFile: ..\gpl-pt_br.txt
Name: "cs"; MessagesFile: "Translations\cs.isl"; LicenseFile: ..\gpl-cs.txt
Name: "fr"; MessagesFile: "Translations\fr.isl"; LicenseFile: ..\gpl-fr.txt
Name: "de"; MessagesFile: "Translations\de.isl"; LicenseFile: ..\gpl-de.txt
;Name: "gr"; MessagesFile: "compiler:Languages\Greek.isl"
Name: "it"; MessagesFile: "Translations\it.isl"; LicenseFile: ..\gpl-it.txt
Name: "ja"; MessagesFile: "Translations\ja.isl"; LicenseFile: ..\gpl-ja.txt
Name: "pl"; MessagesFile: "Translations\pl.isl"; LicenseFile: ..\gpl-pl.txt
Name: "pt"; MessagesFile: "Translations\pt.isl"; LicenseFile: ..\gpl-pt.txt
Name: "ru"; MessagesFile: "Translations\ru.isl"; LicenseFile: ..\gpl-ru.txt

[Components]
Name: "main"; Description: "{cm:CmpMain,{#AvsName}}"; Types: full compact custom; Flags: fixed
Name: "main\avs32"; Description: "{#AvsName} (x86)"; Types: full compact custom
Name: "main\avs64"; Description: "{#AvsName} (x64)"; Types: full compact custom; Check: IsWin64

#ifdef WITH_DOCS
Name: "docs"; Description: "{cm:CmpDocs}";
Name: "docs\enall"; Description: "{cm:CmpDocsEn}"; Types: full; Languages: not en
Name: "docs\en"; Description: "{cm:CmpDocsEn}"; Types: full compact custom; Languages: en
;Name: "docs\cs"; Description: "{cm:CmpDocsCs}"; Types: full compact custom; Languages: cs
;Name: "docs\de"; Description: "{cm:CmpDocsDe}"; Types: full compact custom; Languages: de
;Name: "docs\fr"; Description: "{cm:CmpDocsFr}"; Types: full compact custom; Languages: fr
;Name: "docs\it"; Description: "{cm:CmpDocsIt}"; Types: full compact custom; Languages: it
;Name: "docs\ja"; Description: "{cm:CmpDocsJa}"; Types: full compact custom; Languages: ja
;Name: "docs\pl"; Description: "{cm:CmpDocsPl}"; Types: full compact custom; Languages: pl
;Name: "docs\pt"; Description: "{cm:CmpDocsPt}"; Types: full compact custom; Languages: pt pt_br
;Name: "docs\ru"; Description: "{cm:CmpDocsRu}"; Types: full compact custom; Languages: ru
#endif

Name: "associations"; Description: "{cm:SelectAssoc}"; Types: full custom
Name: "associations\openwithnotepad"; Description: "{cm:SelectAssocNotepadOpen}"; Types: full custom
Name: "associations\shellnew"; Description: "{cm:SelectAssocAddShellNew}"; Types: full custom
Name: "associations\mplayer"; Description: "{cm:SelectAssocMplayer}"; Types: full custom; Check: ExistsMPlayer2
Name: "associations\wmplayer"; Description: "{cm:SelectAssocWMplayer}"; Types: full custom; Check: ExistsWMPlayer

Name: "examples"; Description: "{cm:CmpDocsExamples}"; Types: full compact custom
#ifdef WITH_SDK
Name: "sdk"; Description: "{cm:CmpSdk,{#AvsName}}"; Types: full custom
#endif

Name: "avsmig"; Description: "{cm:CmpMig}"; Types: full compact custom; Flags: fixed; Check: IsLegacyAvsInstalled('32')
Name: "avsmig\uninst"; Description: "{cm:CmpMigUninstall,{#AvsName}}"; Flags: fixed exclusive
Name: "avsmig\backup"; Description: "{cm:CmpMigBackup}"; Flags: fixed exclusive

Name: "custplug"; Description: "{cm:CmpCustomizePluginPaths}"; Types: custom 

[Dirs]
Name: "{code:GetAvsDirsPlus|Plug32}"; Components: main\avs32
Name: "{code:GetAvsDirsPlus|PlugPlus32}"; Components: main\avs32
Name: "{code:GetAvsDirsPlus|Plug64}"; Components: main\avs64
Name: "{code:GetAvsDirsPlus|PlugPlus64}"; Components: main\avs64
 
[Files]
Source: "{code:GetAvsDirsLegacy|Plug32}\*"; DestDir:{code:GetAvsDirsPlus|Plug32}; Components: avsmig\uninst; ExternalSize: 0; Check: IsValidPluginMigration('32'); AfterInstall: WipeLegacyPluginDirs('32'); Flags: external recursesubdirs createallsubdirs uninsneveruninstall onlyifdoesntexist skipifsourcedoesntexist   
Source: "{code:GetAvsDirsLegacy|Plug64}\*"; DestDir:{code:GetAvsDirsPlus|Plug64}; Components: avsmig\uninst; ExternalSize: 0; Check: IsValidPluginMigration('64'); AfterInstall: WipeLegacyPluginDirs('64'); Flags: external recursesubdirs createallsubdirs uninsneveruninstall 64bit onlyifdoesntexist skipifsourcedoesntexist  

Source: "{sys}\AviSynth.dll"; DestDir:{code:GetAvsDirsLegacy|Prog}\PlusBackup\sys32; Components: avsmig\backup; ExternalSize: 0; Flags: external onlyifdoesntexist skipifsourcedoesntexist uninsneveruninstall   
Source: "{sys}\DevIL.dll"; DestDir:{code:GetAvsDirsLegacy|Prog}\PlusBackup\sys32; Components: avsmig\backup; ExternalSize: 0; Flags: external onlyifdoesntexist skipifsourcedoesntexist uninsneveruninstall   
Source: "{sys}\AviSynth.dll"; DestDir:{code:GetAvsDirsLegacy|Prog}\PlusBackup\sys64; Components: avsmig\backup; ExternalSize: 0; Flags: 64bit external onlyifdoesntexist skipifsourcedoesntexist uninsneveruninstall; Check: IsWin64
Source: "{sys}\DevIL.dll"; DestDir:{code:GetAvsDirsLegacy|Prog}\PlusBackup\sys64; Components: avsmig\backup; ExternalSize: 0; Flags: 64bit external onlyifdoesntexist skipifsourcedoesntexist uninsneveruninstall; Check: IsWin64  
Source: "{commonprograms}\AviSynth 2.5\*"; DestDir:{code:GetAvsDirsLegacy|Prog}\PlusBackup\StartMenu; Components: avsmig\backup; ExternalSize: 0; AfterInstall: WipeLegacyStartMenu; Flags: external onlyifdoesntexist skipifsourcedoesntexist uninsneveruninstall  

Source: "..\gpl*.txt"; DestDir: "{app}\License"; Components: main; Flags: ignoreversion
Source: "..\lgpl_for_used_libs.txt"; DestDir: "{app}\License"; Components: main; Flags: ignoreversion

;Source: "..\Readme\readme.txt"; DestDir: "{app}"; Components: main; Flags: ignoreversion
Source: "..\Readme\readme_history.txt"; DestDir: "{app}"; Components: main; Flags: ignoreversion

Source: "{#BuildDir32}\Output\AviSynth.dll"; DestDir:{sys}; Components: main\avs32; Flags: 32bit ignoreversion 
Source: "{#BuildDir32}\Output\System\DevIL.dll"; DestDir:{sys}; Components: main\avs32; Flags: 32bit ignoreversion 
Source: "{#BuildDir32}\Output\Plugins\*.dll"; DestDir:{code:GetAvsDirsPlus|PlugPlus32}; Components: main\avs32; Flags: ignoreversion 
Source: "..\ColorPresets\*"; DestDir:{code:GetAvsDirsPlus|PlugPlus32}; Components: main\avs32; Flags: ignoreversion 
#ifdef WITH_VC_REDIST_XP
Source: "..\Prerequisites\VC_redist_14.27.29114.0.x86.exe"; DestDir: {app}; Components: main\avs32; Flags: deleteafterinstall; Check: IncludeVcRedist()
#endif
#ifdef WITH_VC_REDIST
;get latest from https://www.visualstudio.com/downloads/
Source: "..\Prerequisites\VC_redist.x86.exe"; DestDir: {app}; Components: main\avs32; Flags: deleteafterinstall; Check: IncludeVcRedist()
#endif

Source: "{#BuildDir64}\Output\AviSynth.dll"; DestDir:{sys}; Components: main\avs64; Flags: 64bit ignoreversion 
Source: "{#BuildDir64}\Output\System\DevIL.dll"; DestDir:{sys}; Components: main\avs64; Flags: 64bit ignoreversion 
Source: "{#BuildDir64}\Output\Plugins\*.dll"; DestDir:{code:GetAvsDirsPlus|PlugPlus64}; Components: main\avs64; Flags: ignoreversion 
Source: "..\ColorPresets\*"; DestDir:{code:GetAvsDirsPlus|PlugPlus64}; Components: main\avs64; Flags: ignoreversion
#ifdef WITH_VC_REDIST_XP
Source: "..\Prerequisites\VC_redist_14.27.29114.0.x64.exe"; DestDir: {app}; Components: main\avs64; Flags: deleteafterinstall; Check: IncludeVcRedist()
#endif
#ifdef WITH_VC_REDIST
;get latest from https://www.visualstudio.com/downloads/
Source: "..\Prerequisites\VC_redist.x64.exe"; DestDir: {app}; Components: main\avs64; Flags: deleteafterinstall; Check: IncludeVcRedist()
#endif

#ifdef WITH_DOCS
;don't forget to render .rst sources into html-format by issuing 'make html'. E.g. in the ..\docs\english\
;You needed a working python sphinx-build. https://www.sphinx-doc.org/
Source: "..\docs\*.css"; DestDir: "{app}\docs"; Components: docs; Flags: ignoreversion
;Source: "..\docs\czech\*"; DestDir: "{app}\docs\Czech"; Components: docs\cs; Flags: ignoreversion recursesubdirs 
Source: "..\docs\english\build\html\*"; DestDir: "{app}\docs\English"; Components: docs\en docs\enall; Flags: ignoreversion recursesubdirs 
;Source: "..\docs\french\*"; DestDir: "{app}\docs\French"; Components: docs\fr; Flags: ignoreversion recursesubdirs 
;Source: "..\docs\german\*"; DestDir: "{app}\docs\German"; Components: docs\de; Flags: ignoreversion recursesubdirs 
;Source: "..\docs\italian\*"; DestDir: "{app}\docs\Italian"; Components: docs\it; Flags: ignoreversion recursesubdirs 
;Source: "..\docs\japanese\*"; DestDir: "{app}\docs\Japanese"; Components: docs\ja; Flags: ignoreversion recursesubdirs 
;Source: "..\docs\polish\*"; DestDir: "{app}\docs\Polish"; Components: docs\pl; Flags: ignoreversion recursesubdirs 
;Source: "..\docs\portugese\*"; DestDir: "{app}\docs\Portuguese"; Components: docs\pt; Flags: ignoreversion recursesubdirs 
;Source: "..\docs\russian\*"; DestDir: "{app}\docs\Russian"; Components: docs\ru; Flags: ignoreversion recursesubdirs 
#endif

#ifdef WITH_SDK
Source: "..\FilterSDK\*"; DestDir: "{app}\FilterSDK"; Components: sdk; Flags: ignoreversion recursesubdirs
Source: "..\..\avs_core\include\*"; DestDir: "{app}\FilterSDK\include"; Components: sdk; Flags: ignoreversion recursesubdirs
Source: "{#BuildDir32}\Output\c_api\*"; DestDir: "{app}\FilterSDK\lib\x86"; Components: sdk; Flags: ignoreversion recursesubdirs
Source: "{#BuildDir64}\Output\c_api\*"; DestDir: "{app}\FilterSDK\lib\x64"; Components: sdk; Flags: ignoreversion recursesubdirs
#endif

Source: "..\Examples\*"; DestDir: "{app}\Examples"; Components: examples; Flags: recursesubdirs 

[Registry]
Root: HKLM32; Subkey: "Software\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778}"; Flags: uninsdeletekey; Components: main\avs32
Root: HKLM32; Subkey: "Software\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778}"; ValueName: ""; ValueType: string; ValueData: "{#AvsName}"; Components: main\avs32
Root: HKLM32; Subkey: "Software\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778}\InProcServer32"; ValueName: ""; ValueType: string; ValueData: "AviSynth.dll"; Components: main\avs32
Root: HKLM32; Subkey: "Software\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778}\InProcServer32"; ValueName: "ThreadingModel"; ValueType: string; ValueData: "Apartment"; Components: main\avs32

Root: HKLM64; Subkey: "Software\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778}"; Flags: uninsdeletekey; Check:IsWin64; Components: main\avs64
Root: HKLM64; Subkey: "Software\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778}"; ValueName: ""; ValueType: string; ValueData: "{#AvsName}"; Check:IsWin64; Components: main\avs64
Root: HKLM64; Subkey: "Software\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778}\InProcServer32"; ValueName: ""; ValueType: string; ValueData: "AviSynth.dll"; Check:IsWin64; Components: main\avs64
Root: HKLM64; Subkey: "Software\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778}\InProcServer32"; ValueName: "ThreadingModel"; ValueType: string; ValueData: "Apartment"; Check:IsWin64; Components: main\avs64

Root: HKLM32; Subkey: "Software\Classes\Media Type\Extensions\.avs"; Flags: uninsdeletekey; Components: main\avs32
Root: HKLM32; Subkey: "Software\Classes\Media Type\Extensions\.avs"; ValueName: "Source Filter"; ValueType: string; ValueData: "{{D3588AB0-0781-11CE-B03A-0020AF0BA770}"; Components: main\avs32
Root: HKLM64; Subkey: "Software\Classes\Media Type\Extensions\.avs"; Flags: uninsdeletekey; Check:IsWin64; Components: main\avs64
Root: HKLM64; Subkey: "Software\Classes\Media Type\Extensions\.avs"; ValueName: "Source Filter"; ValueType: string; ValueData: "{{D3588AB0-0781-11CE-B03A-0020AF0BA770}"; Check:IsWin64; Components: main\avs64

Root: HKLM; Subkey: "Software\Classes\.avs"; ValueName: ""; ValueType: string; ValueData: "avsfile"; Flags: uninsdeletekey; Components: main
Root: HKLM; Subkey: "Software\Classes\.avsi"; ValueName: ""; ValueType: string; ValueData: "avs_auto_file"; Flags: uninsdeletekey; Components: main

Root: HKLM; Subkey: "Software\Classes\.avs\OpenWithList\notepad.exe"; Flags: uninsdeletekey; Components: associations\openwithnotepad
Root: HKLM; Subkey: "Software\Classes\avsfile\OpenWithList\notepad.exe"; Flags: uninsdeletekey; Components: associations\openwithnotepad
Root: HKLM; Subkey: "Software\Classes\.avsi\OpenWithList\notepad.exe"; Flags: uninsdeletekey; Components: associations\openwithnotepad
Root: HKLM; Subkey: "Software\Classes\avs_auto_file\OpenWithList\notepad.exe"; Flags: uninsdeletekey; Components: associations\openwithnotepad
Root: HKLM; Subkey: "Software\Classes\.avs\ShellNew"; ValueName: "NullFile"; ValueType: string; ValueData: ""; Flags: uninsdeletekey; Components: associations\shellnew
Root: HKLM; Subkey: "Software\Classes\avsfile\ShellNew"; ValueName: "NullFile"; ValueType: string; ValueData: ""; Flags: uninsdeletekey; Components: associations\shellnew
Root: HKLM; Subkey: "Software\Classes\avsfile\shell\play\command"; ValueName: ""; ValueType: string; ValueData: """{pf}\Windows Media Player\mplayer2.exe"" /Play ""%L"""; Flags: uninsdeletekey; Components: associations\mplayer; Check: ExistsMPlayer2
Root: HKLM; Subkey: "Software\Classes\avsfile\shell\play\command"; ValueName: ""; ValueType: string; ValueData: """{pf}\Windows Media Player\wmplayer.exe"" ""%1"""; Flags: uninsdeletekey; Components: associations\wmplayer; Check: ExistsWMPlayer

Root: HKLM; Subkey: "Software\Classes\avsfile"; ValueName: ""; ValueType: string; ValueData: "{cm:FileTypeDescAvs,{#AvsName}}"; Components: main; Flags: uninsdeletekey
Root: HKLM; Subkey: "Software\Classes\avsfile\DefaultIcon"; ValueName: ""; ValueType: string; ValueData: "{sys}\AviSynth.dll,0"; Components: main\avs32
Root: HKLM; Subkey: "Software\Classes\avsfile\DefaultIcon"; ValueName: ""; ValueType: string; ValueData: "{win}\system32\AviSynth.dll,0"; Components: main\avs64 and not main\avs32 
Root: HKLM; Subkey: "Software\Classes\avs_auto_file"; ValueName: ""; ValueType: string; ValueData: "{cm:FileTypeDescAvsi,{#AvsName}}"; Components: main; Flags: uninsdeletekey
Root: HKLM; Subkey: "Software\Classes\avs_auto_file\DefaultIcon"; ValueName: ""; ValueType: string; ValueData: "{sys}\AviSynth.dll,1"; Components: main\avs32
Root: HKLM; Subkey: "Software\Classes\avs_auto_file\DefaultIcon"; ValueName: ""; ValueType: string; ValueData: "{win}\system32\AviSynth.dll,1"; Components: main\avs64 and not main\avs32

Root: HKLM32; Subkey: "Software\Classes\AVIFile\Extensions\AVS"; ValueName: ""; ValueType: string; ValueData: "{{E6D6B700-124D-11D4-86F3-DB80AFD98778}"; Components: main\avs32; Flags: uninsdeletekey
Root: HKLM64; Subkey: "Software\Classes\AVIFile\Extensions\AVS"; ValueName: ""; ValueType: string; ValueData: "{{E6D6B700-124D-11D4-86F3-DB80AFD98778}"; Components: main\avs64; Flags: uninsdeletekey; Check:IsWin64

Root: HKLM32; Subkey: "Software\AviSynth"; ValueName:""; ValueType: string; ValueData: "{app}"; Components: main\avs32; Flags: uninsdeletevalue uninsdeletekeyifempty
Root: HKLM32; Subkey: "Software\AviSynth"; ValueName:"plugindir2_5"; ValueType: string; ValueData: "{code:GetAvsDirsPlus|Plug32}"; Components: main\avs32; Flags: uninsdeletevalue
Root: HKLM32; Subkey: "Software\AviSynth"; ValueName:"plugindir+"; ValueType: string; ValueData: "{code:GetAvsDirsPlus|PlugPlus32}"; Components: main\avs32; Flags: uninsdeletevalue

Root: HKLM64; Subkey: "Software\AviSynth"; ValueName:""; ValueType: string; ValueData: "{app}"; Components: main\avs64; Flags: uninsdeletevalue uninsdeletekeyifempty; Check:IsWin64
Root: HKLM64; Subkey: "Software\AviSynth"; ValueName:"plugindir2_5"; ValueType: string; ValueData: "{code:GetAvsDirsPlus|Plug64}"; Components: main\avs64; Check:IsWin64; Flags: uninsdeletevalue
Root: HKLM64; Subkey: "Software\AviSynth"; ValueName:"plugindir+"; ValueType: string; ValueData: "{code:GetAvsDirsPlus|PlugPlus64}"; Components: main\avs64; Check:IsWin64; Flags: uninsdeletevalue

#ifdef WITH_SDK
;Set SDK Environment Variable
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueName: "AVISYNTH_SDK_PATH"; ValueType: string; ValueData: "{app}\FilterSDK"; Components: sdk; Flags: uninsdeletevalue
#endif
;Delete Legacy AVS Install Entry
Root: HKLM32; Subkey: "Software\Microsoft\Windows\CurrentVersion\Uninstall\AviSynth"; Flags: deletekey; Components: avsmig\backup
;Add entry for legacy AviSynth Program Folder
Root: HKLM32; Subkey: "Software\AviSynth"; ValueName: "LegacyDir"; ValueType: string; ValueData: "{code:GetAvsDirsLegacy|Prog}"; Components: avsmig\backup;

[UninstallDelete]
Type: files; Name: "{app}\Setup Log*.txt"

[Run]
Filename: "{app}\VC_redist.x86.exe"; Parameters: "/q /norestart"; Components: main\avs32 ;Description: "{#VcVersion} (x86)"; StatusMsg: "{cm:InstallStatusRuntime,{#VcVersion},x86}"; Check: IncludeVcRedist()
Filename: "{app}\VC_redist.x64.exe"; Parameters: "/q /norestart"; Components: main\avs64 ;Description: "{#VcVersion} (x64)"; StatusMsg: "{cm:InstallStatusRuntime,{#VcVersion},x64}"; Check: IncludeVcRedist()

[Ini]
;Backup legacy AviSynth registry entries to .reg file
#define BReg "{code:GetAvsDirsLegacy|Prog}\PlusBackup\PlusBackup.reg"
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778}"; Key: "@"; String: "{code:GetKey|HKLM32^Software\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778%7d}"; Components: avsmig\backup  
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778}\InProcServer32"; Key: "@"; String: "{code:GetKey|HKLM32^Software\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778%7d\InProcServer32}"; Components: avsmig\backup  
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778}\InProcServer32"; Key: """ThreadingModel"""; String: "{code:GetKey|HKLM32^Software\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778%7d\InProcServer32^ThreadingModel}"; Components: avsmig\backup  
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM64}\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778}"; Key: "@"; String: "{code:GetKey|HKLM64^Software\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778%7d}"; Components: avsmig\backup; Check:IsLegacyAvsInstalled('64')  
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM64}\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778}\InProcServer32"; Key: "@"; String: "{code:GetKey|HKLM64^Software\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778%7d\InProcServer32}"; Components: avsmig\backup; Check:IsLegacyAvsInstalled('64')
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM64}\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778}\InProcServer32"; Key: """ThreadingModel"""; String: "{code:GetKey|HKLM64^Software\Classes\CLSID\{{E6D6B700-124D-11D4-86F3-DB80AFD98778%7d\InProcServer32^ThreadingModel}"; Components: avsmig\backup; Check:IsLegacyAvsInstalled('64')

Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}\Classes\Media Type\Extensions\.avs"; Key: "@"; String: "{code:GetKey|HKLM32^Software\Classes\Media Type\Extensions\.avs}"; Components: avsmig\backup  
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}\Classes\Media Type\Extensions\.avs"; Key: """Source Filter"""; String: "{code:GetKey|HKLM32^Software\Classes\Media Type\Extensions\.avs^Source Filter}"; Components: avsmig\backup  

Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}\Classes\.avs"; Key: "@"; String: "{code:GetKey|HKLM32^Software\Classes\.avs}"; Components: avsmig\backup  
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}\Classes\.avsi"; Key: "@"; String: "{code:GetKey|HKLM32^Software\Classes\.avsi}"; Components: avsmig\backup  
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}\Classes\avsfile"; Key: "@"; String: "{code:GetKey|HKLM32^Software\Classes\avsfile}"; Components: avsmig\backup  
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}\Classes\avsfile\DefaultIcon"; Key: "@"; String: "{code:GetKey|HKLM32^Software\Classes\avsfile\DefaultIcon}"; Components: avsmig\backup  
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}\Classes\avs_auto_file"; Key: "@"; String: "{code:GetKey|HKLM32^Software\Classes\avs_auto_file}"; Components: avsmig\backup  
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}\Classes\avs_auto_file\DefaultIcon"; Key: "@"; String: "{code:GetKey|HKLM32^Software\Classes\avs_auto_file\DefaultIcon}"; Components: avsmig\backup

Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}\Classes\AVIFile\Extensions\AVS"; Key: "@"; String: "{code:GetKey|HKLM32^Software\Classes\AVIFile\Extensions\AVS}"; Components: avsmig\backup
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM64}\Classes\AVIFile\Extensions\AVS"; Key: "@"; String: "{code:GetKey|HKLM64^Software\Classes\AVIFile\Extensions\AVS}"; Components: avsmig\backup; Check:IsLegacyAvsInstalled('64')

Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}\AviSynth"; Key: "@"; String: "{code:GetAvsDirsLegacyFmt|Prog}"; Components: avsmig\backup
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}\AviSynth"; Key: """plugindir2_5"""; String: "{code:GetAvsDirsLegacyFmt|Plug32}"; Components: avsmig\backup
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM64}\AviSynth"; Key: "@"; String: "{code:GetAvsDirsLegacyFmt|Prog}"; Components: avsmig\backup; Check:IsLegacyAvsInstalled('64')
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM64}\AviSynth"; Key: """plugindir2_5"""; String: "{code:GetAvsDirsLegacyFmt|Plug64}"; Components: avsmig\backup; Check:IsLegacyAvsInstalled('64')

#define LIns "\Microsoft\Windows\CurrentVersion\Uninstall\AviSynth"
;Backup legacy AviSynth installer registry entries
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}{#LIns}"; Key: """DisplayIcon"""; String: "{code:GetKey|HKLM32^Software{#LIns}^DisplayIcon}"; Components: avsmig\backup
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}{#LIns}"; Key: """DisplayName"""; String: "{code:GetKey|HKLM32^Software{#LIns}^DisplayName}"; Components: avsmig\backup
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}{#LIns}"; Key: """DisplayVersion"""; String: "{code:GetKey|HKLM32^Software{#LIns}^DisplayVersion}"; Components: avsmig\backup
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}{#LIns}"; Key: """Publisher"""; String: "{code:GetKey|HKLM32^Software{#LIns}^Publisher}"; Components: avsmig\backup
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}{#LIns}"; Key: """UninstallString"""; String: "{code:GetKey|HKLM32^Software{#LIns}^UninstallString}"; Components: avsmig\backup
Filename: "{#BReg}"; Section:"{code:GetRegSoft|HKLM32}{#LIns}"; Key: """URLInfoAbout"""; String: "{code:GetKey|HKLM32^Software{#LIns}^URLInfoAbout}"; Components: avsmig\backup; AfterInstall: FixRegFile

[Code]
type
  TAvsDirs = record
    IsSet: Boolean;
    Prog: String;
    Plug32: String;
    PlugPlus32:String;
    Plug64: String;
    PlugPlus64: String;
  end;

var
  PluginPage: TInputDirWizardPage;
  MigrationPage: TInputOptionWizardPage;
  AvsDirsPlus, AvsDirsDefault, AvsDirsReg: TAvsDirs;
  MigrationState: (miNotStarted, miPickedBackup, miPickedUninst, miFailedUninst);

const
  AVSUNINST_YES = 6;
  AVSUNINST_OK = 2;
  LogIndent = '--- ';

procedure LogVal(Prefix, Val: String);
begin
  Log(Prefix + ': ' + Val);
end;

procedure LogValInd(Prefix, Val: String);
begin
  Log(LogIndent + Prefix + ': ' + Val);
end;

procedure LogAvsDirectories(AvsDir: TAvsDirs);
begin
  with AvsDir do begin 
    LogValInd('Program',Prog);
    LogValInd('Plugins',Plug32);
    LogValInd('Plugins+',PlugPlus32);
    LogValInd('Plugins64',Plug64);
    LogValInd('Plugins64+',PlugPlus64);
  end;
end;

// Directory handling                                                              
procedure SetAvsDirsDefault;
begin
  if not AvsDirsDefault.IsSet then begin
    AvsDirsDefault.Plug32 := '\plugins';
    AvsDirsDefault.PlugPlus32 := '\plugins+';
    AvsDirsDefault.Plug64 := '\plugins64';
    AvsDirsDefault.PlugPlus64 := '\plugins64+';
    AvsDirsDefault.IsSet := True;
  end;
end;

procedure SetAvsDirsReg;
begin
  if not AvsDirsReg.IsSet then begin
    RegQueryStringValue(HKLM32, 'Software\AviSynth', '', AvsDirsReg.Prog);
    RegQueryStringValue(HKLM32, 'Software\AviSynth', 'PluginDir2_5', AvsDirsReg.Plug32);
    RegQueryStringValue(HKLM32, 'Software\AviSynth', 'PluginDir+', AvsDirsReg.PlugPlus32);
    AvsDirsReg.IsSet := True; 
    if IsWin64 then begin
      RegQueryStringValue(HKLM64, 'Software\AviSynth', 'PluginDir2_5', AvsDirsReg.Plug64);
      RegQueryStringValue(HKLM64, 'Software\AviSynth', 'PluginDir+', AvsDirsReg.PlugPlus64);
    end;
  end;
end;

procedure SetAvsDirsPlus;
begin
  SetAvsDirsDefault();
  SetAvsDirsReg();

  if not AvsDirsPlus.IsSet or (AvsDirsPlus.Prog <> WizardDirValue()) then begin
    with AvsDirsPlus do begin
      Prog := WizardDirValue();
      if DirExists(AvsDirsReg.Plug32) then 
        Plug32 := AvsDirsReg.Plug32
      else Plug32 := AvsDirsPlus.Prog + AvsDirsDefault.Plug32;

      if DirExists(AvsDirsReg.PlugPlus32) then 
        PlugPlus32 := AvsDirsReg.PlugPlus32
      else PlugPlus32 := AvsDirsPlus.Prog + AvsDirsDefault.PlugPlus32;

      if DirExists(AvsDirsReg.Plug64) then 
        Plug64 := AvsDirsReg.Plug64
      else Plug64 := AvsDirsPlus.Prog + AvsDirsDefault.Plug64;

      if DirExists(AvsDirsReg.PlugPlus64) then 
        PlugPlus64 := AvsDirsReg.PlugPlus64
      else PlugPlus64 := AvsDirsPlus.Prog + AvsDirsDefault.PlugPlus64;    
    end;
    AvsDirsPlus.IsSet := True;
  end;

  if IsComponentSelected('avsmig\uninst') then begin
    AvsDirsPlus.Plug32 := AvsDirsPlus.Prog + AvsDirsDefault.Plug32;
    AvsDirsPlus.Plug64 := AvsDirsPlus.Prog + AvsDirsDefault.Plug64;
  end else if IsComponentSelected('avsmig\backup') then begin
    if DirExists(AvsDirsReg.Plug32) then
      AvsDirsPlus.Plug32 := AvsDirsReg.Plug32;
    if DirExists(AvsDirsReg.Plug64) then  
      AvsDirsPlus.Plug64 := AvsDirsReg.Plug64; 
  end;
end;

procedure UpdatePluginDirPage(Mode: String);
begin
  if Mode = 'read' then begin
    PluginPage.Values[0] := AvsDirsPlus.Plug32;
    PluginPage.Values[1] := AvsDirsPlus.PlugPlus32;
    PluginPage.Values[2] := AvsDirsPlus.Plug64;
    PluginPage.Values[3] := AvsDirsPlus.PlugPlus64;
  end else if Mode = 'write' then begin
    AvsDirsPlus.Plug32 := PluginPage.Values[0];
    AvsDirsPlus.PlugPlus32 := PluginPage.Values[1];
    AvsDirsPlus.Plug64 := PluginPage.Values[2];
    AvsDirsPlus.PlugPlus64 := PluginPage.Values[3];
  end;
end;

// Helper functions

function BoolToStr(Param: Boolean): String;
begin
  if Param then
    Result := 'Yes'
  else
    Result := 'No';
end;

procedure CloseAvsUninstDialog(BtnId: WORD);
var
  Wnd: HWND;
  WM_COMMAND: Longint;
  BN_CLICKED: Longint;
  WParam: Longint;
  UninstTimeout: Integer;
  WndTitle: String;

begin
  WM_COMMAND := $0111;
  BN_CLICKED := 245;
  UninstTimeout := 10000;
  case GetUILanguage() of
    $0409,$0809,$0109,$0C09: WndTitle := 'AviSynth Uninstall';
    $0405: WndTitle := 'Odinstalovat AviSynth';
    $0407,$0807,$0C07: WndTitle := 'AviSynth Deinstallation';
    $040C,$080C,$0C0C,$140C: WndTitle := 'Désinstallation de AviSynth';
    $0410: WndTitle := 'Disinstallazione di AviSynth';
    $0411: WndTitle := 'AviSynth アンインストール';
    $0816: WndTitle := 'Desinstalação de AviSynth';
    $0416: WndTitle := 'Desinstalação do AviSynth';
    $0419: WndTitle := 'Удаление AviSynth';
    $0408: WndTitle := 'Απεγκατάσταση του ''AviSynth''';
  else WndTitle := 'AviSynth Uninstall';
  end;

  repeat
    Sleep(300);
    Wnd := FindWindowByWindowName(WndTitle);
    UninstTimeout := UninstTimeout - 500;  
  until (Wnd <> 0) or (UninstTimeout <= 0);
  if Wnd <> 0 then begin
    WParam := (BN_CLICKED shl 16) or BtnId;
    SendMessage(Wnd, WM_COMMAND, WParam, Wnd);
  end;
end;

function IsSamePath(Path1, Path2: String): Boolean;
var 
  Path1Sanitized, Path2Sanitized: String;
begin
  Path1Sanitized := Lowercase(RemoveBackslashUnlessRoot(Trim(Path1)));
  Path2Sanitized := Lowercase(RemoveBackslashUnlessRoot(Trim(Path2)));
Result := Path1Sanitized = Path2Sanitized
end;

procedure Explode(var Dest: TArrayOfString; Text: String; Separator: String);
var
	i: Integer;
begin
	i := 0;
	repeat
		SetArrayLength(Dest, i+1);
		if Pos(Separator,Text) > 0 then	begin
			Dest[i] := Copy(Text, 1, Pos(Separator, Text)-1);
			Text := Copy(Text, Pos(Separator,Text) + Length(Separator), Length(Text));
			i := i + 1;
		end else begin
			 Dest[i] := Text;
			 Text := '';
		end;
	until Length(Text)=0;
end;

function FmtRegFileString(Param: String): String;
begin
  StringChangeEx(Param,'\','\\', True);
  StringChangeEx(Param,'"','\"', True);
  Result := '"' + Param + '"';
end;

function IsAvsPlusInstalled: Boolean;
begin
  Result := RegValueExists(HKLM32,'Software\Microsoft\Windows\CurrentVersion\Uninstall\{#AppId}_is1', 'UninstallString');
end;

function IsLegacyAvsInstalled(Param: String): Boolean;
begin
  case Param of
    '32': Result := DirExists(AvsDirsReg.Prog) and not IsAvsPlusInstalled;
    '64': Result := IsWin64 and DirExists(AvsDirsReg.Plug64) and not IsAvsPlusInstalled;
  end;
end;

procedure SetInputs(ID: Integer; State: Boolean);
begin
  PluginPage.Edits[ID].Enabled := State
  PluginPage.PromptLabels[ID].Enabled := State
  PluginPage.Buttons[ID].Enabled := State
end;

procedure SetMigrationComponent();
var
  Index: Integer;
begin
  Index := WizardForm.ComponentsList.Items.IndexOf(FmtMessage(CustomMessage('CmpMigUninstall'),['{#AvsName}']));
  if Index <> -1 then
    WizardForm.ComponentsList.Checked[Index] := MigrationState = miPickedUninst;
  
  Index := WizardForm.ComponentsList.Items.IndexOf(FmtMessage(CustomMessage('CmpMigBackup'),['{#AvsName}']));
  if Index <> -1 then
    WizardForm.ComponentsList.Checked[Index] := MigrationState = miPickedBackup;
end;


// Event functions

function InitializeSetup(): Boolean;
begin
  SetAvsDirsReg();
  SetAvsDirsDefault();
  Result := True
end;

procedure InitializeWizard;
begin
  WizardForm.Bevel.Visible := False;
  WizardForm.Bevel1.Visible := False;
  WizardForm.MainPanel.Color := $a35460;
  WizardForm.MainPanel.Font.Color := $fcfcfc;
  WizardForm.PageNameLabel.Font.Color := $fcfcfc;
  //WizardForm.InnerPage.Color := $ffffff;
  WizardForm.WelcomePage.Color := $fcfcfc;
  WizardForm.ReadyMemo.ScrollBars:= ssVertical;
 
  

  MigrationPage := CreateInputOptionPage(wpSelectDir, CustomMessage('MigPageCaption'), FmtMessage(CustomMessage('MigPageDescription'),['{#AvsName}']),
                   FmtMessage(CustomMessage('MigPageSubCaption'),['{#AvsName}',AvsDirsReg.Prog]), True, False)
  
  MigrationPage.Add(FmtMessage(CustomMessage('MigPageOptionBackup'),['{#AvsName}']))
  MigrationPage.Add(FmtMessage(CustomMessage('MigPageOptionUninstall'),['{#AvsName}']))
  MigrationPage.SelectedValueIndex := 0;

  PluginPage := CreateInputDirPage(wpSelectComponents,
    FmtMessage(CustomMessage('PlugPageCaption'),['{#AvsName}']),FmtMessage(CustomMessage('PlugPageDescription'),['{#AvsName}']),
    FmtMessage(CustomMessage('PlugPageSubCaption'),['{#AvsName}']), true, '');
  PluginPage.Add(FmtMessage(CustomMessage('PlugPagePlugDirLegacy'),['32']));
  PluginPage.Add(FmtMessage(CustomMessage('PlugPagePlugDirPlus'),['32','{#AvsName}']));  
  PluginPage.Add(FmtMessage(CustomMessage('PlugPagePlugDirLegacy'),['64']));
  PluginPage.Add(FmtMessage(CustomMessage('PlugPagePlugDirPlus'),['64','{#AvsName}']));
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  I: Integer;
begin
  if CurPageID = wpSelectDir then begin
    SetAvsDirsPlus()
    Result := True

  end else if CurpageID = MigrationPage.ID then begin
    if MigrationPage.SelectedValueIndex = 1 then 
      MigrationState := miPickedUninst
    else
      MigrationState := miPickedBackup;
    SetMigrationComponent()
    SetAvsDirsPlus()
    Result := True

  end else if CurPageID = wpSelectComponents then begin
    SetAvsDirsPlus();
    UpdatePluginDirPage('read');
    if IsComponentSelected('main\avs64') then begin                                                       
      if IsComponentSelected('avsmig\backup') and IsLegacyAvsInstalled('64') then
        SetInputs(2,false)
      else SetInputs(2,true);
      SetInputs(3,true);
    end else begin
      SetInputs(2,false);
      SetInputs(3,false);
    end;
    if IsComponentSelected('main\avs32') then begin
      if IsComponentSelected('avsmig\backup') and DirExists(AvsDirsReg.Plug32) then
        SetInputs(0,false)
      else SetInputs(0,true);
      SetInputs(1,true);
    end else begin
      SetInputs(0,false);
      SetInputs(1,false);
    end;
    Result := True

  end else if CurPageID = PluginPage.ID then begin
    UpdatePluginDirPage('write');
    Result := True;
  end else
    Result := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  LogFilePath: String;

begin
  if CurStep = ssInstall then begin
    LogVal('Installing components', WizardSelectedComponents(false));
    LogVal('Legacy x86 AviSynth installed', BoolToStr(IsLegacyAvsInstalled('32')));
    LogVal('Legacy x64 AviSynth installed', BoolToStr(IsLegacyAvsInstalled('64')));
    LogVal('AviSynth+ installed', BoolToStr(IsAvsPlusInstalled));
    Log('RegDirectories:');
    LogAvsDirectories(AvsDirsReg);
    Log('AvsPlusDirectories:');
    LogAvsDirectories(AvsDirsPlus);

    if MigrationState = miPickedUninst then begin
      Log('Uninstalling legacy AviSynth');
      if Exec(AvsDirsReg.Prog+'\Uninstall.exe', '/S', '', SW_SHOW, ewWaitUntilIdle, ResultCode) then begin
        CloseAvsUninstDialog(AVSUNINST_YES)
        CloseAvsUninstDialog(AVSUNINST_OK)
      end else begin
        MigrationState := miFailedUninst;
         SuppressibleMsgBox(FmtMessage(CustomMessage('MigPageUninstallFailed'),[SysErrorMessage(ResultCode)]), mbError, MB_OK, -1);
      end;
    end;
    end else if CurStep = ssDone then begin
    LogFilePath := ExpandConstant('{log}');
    FileCopy(LogFilePath, WizardDirValue + '\' + ExtractFileName(LogFilePath) , false); 
  end;
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  if (PageID = MigrationPage.ID) and not IsLegacyAvsInstalled('32') then
    Result := True                    
  else if (PageID = PluginPage.ID) and not IsComponentSelected('custplug') then
    Result := True
  else Result := False
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AvsDirLegacyUninst: String;
  ResultCode: Integer;
begin
  if (CurUninstallStep = usPostUninstall) and RegQueryStringValue(HKLM32, 'Software\AviSynth', 'LegacyDir', AvsDirLegacyUninst) then begin
    RenameFile(AvsDirLegacyUninst + '\PlusBackup\sys32\AviSynth.dll', ExpandConstant('{sys}\AviSynth.dll'))
    RenameFile(AvsDirLegacyUninst + '\PlusBackup\sys32\DevIL.dll', ExpandConstant('{sys}\DevIL.dll'));
    RenameFile(AvsDirLegacyUninst + '\PlusBackup\StartMenu', ExpandConstant('{commonprograms}\AviSynth 2.5'));
    if IsWin64 then begin
      EnableFsRedirection(False);
      { Since the installer runs in x86 mode we need redirection to be disabled here for 2 reasons:
          1. To make [sys] expand to the System32, not SysWoW64.
          2. To run the correct x64 regedit.exe on x64 systems to import our backup regfile.
             Using the x86 regedit.exe would cause HKLM\Software\AviSynth to be redirected to HKLM\Software\WoW6432Node\AviSynth,
             which would potentially cause the x86 plugins folder value to be overwritten by its x64 counterpart.
      }
      RenameFile(AvsDirLegacyUninst + '\PlusBackup\sys64\AviSynth.dll', ExpandConstant('{sys}\AviSynth.dll')); 
      RenameFile(AvsDirLegacyUninst + '\PlusBackup\sys64\DevIL.dll', ExpandConstant('{sys}\DevIL.dll'));
    end;
    if Exec(ExpandConstant('{win}\regedit.exe'), '/s "'+ AvsDirLegacyUninst +'\PlusBackup\PlusBackup.reg"', '', SW_SHOW, ewWaitUntilTerminated, ResultCode) then begin
      DelTree(AvsDirLegacyUninst + '\PlusBackup', True, True, True);
      RegDeleteValue(HKLM32, 'Software\AviSynth', 'LegacyDir');
    end else
       SuppressibleMsgBox(FmtMessage(CustomMessage('BackupRestoreFailed'),[SysErrorMessage(ResultCode)]), mbError, MB_OK,-1);
    EnableFsRedirection(True);
  end;  
end;

// Checks and {code ...} snippets 
Procedure FixRegFile;
var
  FileContent: Array of String;
  RegFile: String;
begin
  RegFile := AvsDirsReg.Prog + '\PlusBackup\PlusBackup.reg';
  LoadStringsFromFile(RegFile, FileContent);
  SaveStringToFile(RegFile, 'Windows Registry Editor Version 5.00', False);
  SaveStringsToFile(RegFile, FileContent, True);
end;

function GetKey(Param: String): String;
var
  RootKey: Integer;
  Value, ValueName: String;
  Params: Array of String;
begin
  Explode(Params, Param, '^');
  case Params[0] of
    'HKLM32': RootKey := HKLM32;
    'HKLM64': RootKey := HKLM64;
    'HKCU32': RootKey := HKCU32;
    'HKCU64': RootKey := HKCU64;
  end;
  if GetArrayLength(Params) = 3 then ValueName  := Params[2] else ValueName := ''; 
  RegQueryStringValue(RootKey, Params[1], ValueName, Value);
  Result := FmtRegFileString(Value)
end;

function GetAvsDirsLegacy(Param: String): String;
begin
    case Param of
      'Prog': Result := AvsDirsReg.Prog;
      'Plug32': Result := AvsDirsReg.Plug32;
      'Plug64': Result := AvsDirsReg.Plug64;
    end;
end;

function GetAvsDirsLegacyFmt(Param: String): String;
begin
  Result := FmtRegFileString(GetAvsDirsLegacy(Param));
end;

function GetAvsDirsPlus(Param: String): String;
begin
    case Param of 
      'Plug32': Result := AvsDirsPlus.Plug32;                       
      'PlugPlus32': Result := AvsDirsPlus.PlugPlus32;
      'Plug64': Result := AvsDirsPlus.Plug64;
      'PlugPlus64': Result := AvsDirsPlus.PlugPlus64;
    end;
end;

function GetRegSoft(Param: String): String;
begin
  if IsWin64 then begin
    case Param of
      'HKLM32': Result := 'HKEY_LOCAL_MACHINE\Software\Wow6432Node';
      'HKLM64': Result := 'HKEY_LOCAL_MACHINE\Software';
      'HKCU32': Result := 'HKEY_CURRENT_USER\Software\Wow6432Node';
      'HKCU64': Result := 'HKEY_CURRENT_USER\Software';
    end;
  end else begin
    case Param of
      'HKLM32': Result := 'HKEY_LOCAL_MACHINE\Software';
      'HKCU32': Result := 'HKEY_CURRENT_USER\Software';
    end;
  end;
end;

function IsValidPluginMigration(Param: String): Boolean;
begin
  case Param of
    '32': Result := IsLegacyAvsInstalled('32') and DirExists(AvsDirsReg.Plug32) and not IsSamePath(AvsDirsReg.Plug32,AvsDirsPlus.Plug32);
    '64': Result := IsLegacyAvsInstalled('64') and not IsSamePath(AvsDirsReg.Plug64,AvsDirsPlus.Plug64);
  end;
end;

procedure WipeLegacyPluginDirs(Param: String);
begin
  case Param of
    '32': DelTree(AvsDirsReg.Plug32, True, True, False);
    '64': DelTree(AvsDirsReg.Plug64, True, True, False);
  end;
  LogVal('Wiping legacy AviSynth Plugin Directory',Param);
  RemoveDir(AvsDirsReg.Prog);
end;

procedure WipeLegacyStartMenu;
begin
  DelTree(ExpandConstant('{commonprograms}\AviSynth 2.5'), True, True, False);
end;

function IncludeVcRedist(): boolean;
begin
  #ifdef WITH_VC_REDIST
  Result := True;
  #else
  Result := False;
  #endif
end;

function ExistsMPlayer2(): boolean;
begin
  Result := FileExists(ExpandConstant('{pf}\Windows Media Player\mplayer2.exe'));
end;

function ExistsWMPlayer(): boolean;
begin
  Result := FileExists(ExpandConstant('{pf}\Windows Media Player\wmplayer.exe'));
end;
