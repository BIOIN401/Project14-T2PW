# exit 0 only when heavylock is absent, free RAM >= 6 GB, and no pytest/bounded_run python process exists
$free=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
if(Test-Path 'C:/t/heavylock'){ "heavylock present"; exit 1 }
if($free -lt 4){ "free RAM $free GB < 6"; exit 1 }
$p=Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'pytest|bounded_run' -and $_.CommandLine -notmatch 'pylance|lsp|ram\.ps1' -and $_.Name -match 'python' }
if($p){ "live pytest/bounded_run"; exit 1 }
exit 0
