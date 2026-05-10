<#
Clean local-only files produced while developing or running Mr. Mythical: SimC Factory.

Default behavior removes disposable caches and scratch files only. Expensive or
stateful generated outputs require explicit switches.

Examples:
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/cleanup_workspace.ps1
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/cleanup_workspace.ps1 -TerraformCache
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/cleanup_workspace.ps1 -GeneratedData -WhatIf
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$Caches,
    [switch]$Scratch,
    [switch]$TerraformCache,
    [switch]$WebJobs,
    [switch]$GeneratedData,
    [switch]$ModelArtifacts,
    [switch]$AllGenerated
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$AnyExplicitSwitch = $Caches -or $Scratch -or $TerraformCache -or $WebJobs -or $GeneratedData -or $ModelArtifacts -or $AllGenerated
$RunDefaultCleanup = -not $AnyExplicitSwitch

function Remove-WorkspacePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $FullPath = (Resolve-Path -LiteralPath $Path).Path
    if ($PSCmdlet.ShouldProcess($FullPath, "Remove")) {
        Remove-Item -LiteralPath $FullPath -Recurse -Force
    }
}

function Remove-WorkspaceItems {
    param(
        [AllowEmptyCollection()]
        [object[]]$Items
    )

    foreach ($Item in $Items) {
        if ($null -eq $Item) {
            continue
        }
        if ($PSCmdlet.ShouldProcess($Item.FullName, "Remove")) {
            Remove-Item -LiteralPath $Item.FullName -Recurse -Force
        }
    }
}

if ($RunDefaultCleanup -or $Caches) {
    Remove-WorkspaceItems -Items @(Get-ChildItem -Force -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue)
    Remove-WorkspaceItems -Items @(Get-ChildItem -Force -Recurse -Directory -Filter ".pytest_cache" -ErrorAction SilentlyContinue)
    Remove-WorkspaceItems -Items @(Get-ChildItem -Force -Recurse -Directory -Filter ".ruff_cache" -ErrorAction SilentlyContinue)
    Remove-WorkspaceItems -Items @(
        Get-ChildItem -Force -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -in ".pyc", ".pyo" }
    )
}

if ($RunDefaultCleanup -or $Scratch) {
    Remove-WorkspaceItems -Items @(Get-ChildItem -Force -File -Filter "tmp_*.py" -ErrorAction SilentlyContinue)
    Remove-WorkspacePath -Path "=3.10"
}

if ($TerraformCache -or $AllGenerated) {
    Remove-WorkspacePath -Path "terraform/.terraform"
}

if ($WebJobs -or $AllGenerated) {
    Remove-WorkspacePath -Path "local/web_jobs"
}

if ($GeneratedData -or $AllGenerated) {
    Remove-WorkspacePath -Path "distributed_shards"
    Remove-WorkspacePath -Path "training_data"
    Remove-WorkspacePath -Path "spec_profiles"
    Remove-WorkspacePath -Path "all_specs_training_data.csv"
    Remove-WorkspacePath -Path "profile_metadata.json"
    Remove-WorkspacePath -Path "local/profile_metadata.json"
    Remove-WorkspacePath -Path "local/generator_mismatch_log.jsonl"
}

if ($ModelArtifacts -or $AllGenerated) {
    Remove-WorkspacePath -Path "local/nn_website_model"
    Remove-WorkspacePath -Path "local/experiment_results"
    Remove-WorkspacePath -Path "sagemaker/best_hyperparameters.json"
}

Write-Host "Cleanup complete. Run with -WhatIf first when using generated-data switches."