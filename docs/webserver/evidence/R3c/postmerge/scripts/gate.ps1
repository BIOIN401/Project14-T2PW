$me=$PID
$p=Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'pytest|bounded_run' -and $_.CommandLine -notmatch 'pylance|lsp|gate\.ps1' -and $_.CommandLine -notmatch 'ws-R3c-post' -and $_.Name -match 'python' }
if($p){ $p | ForEach-Object { "$($_.ProcessId) $($_.CommandLine.Substring(0,[Math]::Min(160,$_.CommandLine.Length)))" }; exit 1 } else { exit 0 }
