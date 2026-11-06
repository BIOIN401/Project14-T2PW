$n = @(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.Name -match 'python' -and $_.CommandLine -match 'pytest|bounded_run' -and $_.CommandLine -notmatch 'get_pytest_options' }).Count
Write-Output $n
