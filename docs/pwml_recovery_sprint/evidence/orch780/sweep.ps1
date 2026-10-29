# ORCH-780 orphan sweep: list surviving python.exe whose command line names this lane (never LSP servers).
$mine = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'orch780|ORCH-780' -and $_.CommandLine -notmatch 'lsp_server' }
if ($mine) { $mine | ForEach-Object { "SURVIVOR pid=$($_.ProcessId) $($_.CommandLine)" } } else { "orphan sweep: 0 ORCH-780 python processes alive" }
"heavy lock: " + (Test-Path C:/t/heavylock/holder.json)
