<#
Open an SSH tunnel to a vLLM server on the lab GPU machine, so VLLM_BASE_URL can stay at
its default (http://127.0.0.1:8000/v1). Leave this window open for the length of the run.

  .\scripts\vllm_tunnel.ps1 -RemotePort 8000

Equivalent to:  ssh -N -L 8000:127.0.0.1:<RemotePort> lab-gpu

"lab-gpu" is a Host alias in your own ~/.ssh/config (HostName, User, IdentityFile). No
credentials live here. Then, in the shell that runs T2PW:

  $env:T2PW_LLM_PROVIDER = "vllm"
  $env:VLLM_MODEL = "<model the server serves>"
#>
param(
    [Parameter(Mandatory = $true)][int]$RemotePort,
    [int]$LocalPort = 8000,
    [string]$SshHost = "lab-gpu"
)
ssh -N -L "${LocalPort}:127.0.0.1:${RemotePort}" $SshHost
