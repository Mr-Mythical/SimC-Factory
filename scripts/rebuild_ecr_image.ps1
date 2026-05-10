$ErrorActionPreference = "Stop"

# Rebuild the SimC Batch worker Docker image and push to ECR.
# Windows PowerShell equivalent of scripts/rebuild_ecr_image.sh.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

$AwsRegion = if ($env:AWS_REGION) { $env:AWS_REGION } elseif ($env:AWS_DEFAULT_REGION) { $env:AWS_DEFAULT_REGION } else { "eu-north-1" }
$EcrRepositoryName = if ($env:ECR_REPOSITORY_NAME) { $env:ECR_REPOSITORY_NAME } else { "simc-batch-worker" }
$EcrRepo = $env:ECR_REPO
if (-not $EcrRepo) {
    $AwsAccountId = $env:AWS_ACCOUNT_ID
    if (-not $AwsAccountId) {
        $AwsCliForAccount = Get-Command aws -ErrorAction SilentlyContinue
        if (-not $AwsCliForAccount) {
            throw "Set ECR_REPO or AWS_ACCOUNT_ID before running this script."
        }
        $AwsAccountId = (& aws sts get-caller-identity --query Account --output text --region $AwsRegion).Trim()
    }
    $EcrRepo = "$AwsAccountId.dkr.ecr.$AwsRegion.amazonaws.com/$EcrRepositoryName"
}
$WorkerDir = if ($env:WORKER_DIR) { $env:WORKER_DIR } else { Join-Path $ProjectRoot "worker" }

$EcrRegistry = $EcrRepo.Substring(0, $EcrRepo.LastIndexOf('/'))
$LocalTag = "simc-batch-worker:latest"
$RemoteTag = "$EcrRepo`:latest"

Write-Host "=== SimC Batch Worker - ECR Image Rebuild ==="
Write-Host "  ECR repo:   $EcrRepo"
Write-Host "  Region:     $AwsRegion"
Write-Host "  Dockerfile: $WorkerDir/Dockerfile"
Write-Host ""

Write-Host "--- Step 1/5: Pulling simulationcraftorg/simc:latest ---"
& docker pull simulationcraftorg/simc:latest
Write-Host ""

Write-Host "--- Step 2/5: Building worker image ---"
& docker build --pull --no-cache -t $LocalTag $WorkerDir
Write-Host ""

Write-Host "--- Step 3/5: Verifying SimC version ---"
try {
    & docker run --rm --entrypoint /usr/local/bin/simc $LocalTag --version
} catch {
    # Keep behavior aligned with the bash script (best-effort version print).
    Write-Host "SimC version check failed, continuing..."
}
Write-Host ""

Write-Host "--- Step 4/5: Logging in to ECR ---"
$AwsCli = Get-Command aws -ErrorAction SilentlyContinue
if ($AwsCli) {
    $Password = & aws ecr get-login-password --region $AwsRegion
    if (-not $Password) {
        throw "aws ecr get-login-password returned empty output."
    }
} else {
    Write-Host "aws CLI not found. Falling back to boto3 for ECR auth token..."
    $VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    $PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

    # If static credentials are present, prefer them and ignore credential_process config.
    $HasStaticCreds = -not [string]::IsNullOrWhiteSpace($env:AWS_ACCESS_KEY_ID) -and -not [string]::IsNullOrWhiteSpace($env:AWS_SECRET_ACCESS_KEY)
    if ($HasStaticCreds) {
        $env:AWS_CONFIG_FILE = ""
        $env:AWS_PROFILE = ""
    }

    # If using Roles Anywhere, ensure signing helper path exists before invoking boto3.
    if (-not $HasStaticCreds) {
        $SigningHelper = $env:AWS_SIGNING_HELPER
        if (-not [string]::IsNullOrWhiteSpace($SigningHelper) -and -not (Test-Path $SigningHelper)) {
            throw "AWS_SIGNING_HELPER points to '$SigningHelper' but the file does not exist. Install aws_signing_helper there or update AWS_SIGNING_HELPER in .env."
        }
    }

    $TmpPy = [System.IO.Path]::GetTempFileName() + ".py"
    $PyCode = @"
import base64
import json
import sys

import boto3

region = sys.argv[1]
resp = boto3.client('ecr', region_name=region).get_authorization_token()
auth = resp['authorizationData'][0]
token = auth['authorizationToken']
decoded = base64.b64decode(token).decode('utf-8')
_user, password = decoded.split(':', 1)
print(json.dumps({'password': password, 'proxy': auth.get('proxyEndpoint', '')}))
"@
    Set-Content -Path $TmpPy -Value $PyCode -Encoding UTF8

    try {
        $AuthJson = & $PythonExe $TmpPy $AwsRegion 2>&1
    } finally {
        Remove-Item -Path $TmpPy -ErrorAction SilentlyContinue
    }
    if (-not $AuthJson) {
        throw "Failed to retrieve ECR auth token via boto3. Ensure boto3 is installed and AWS credentials are configured."
    }
    $AuthObj = $AuthJson | ConvertFrom-Json
    $Password = $AuthObj.password

    if (-not $Password) {
        throw "boto3 returned an empty ECR password. Check AWS credentials and ECR permissions."
    }
}
$Password | & docker login --username AWS --password-stdin $EcrRegistry
Write-Host ""

Write-Host "--- Step 5/5: Pushing to ECR ---"
& docker tag $LocalTag $RemoteTag
& docker push $RemoteTag
Write-Host ""

$Digest = & docker inspect --format='{{index .RepoDigests 0}}' $RemoteTag 2>$null
if (-not $Digest) {
    $Digest = "unknown"
}

Write-Host "=== Done ==="
Write-Host "  Pushed:  $RemoteTag"
Write-Host "  Digest:  $Digest"
Write-Host ""
Write-Host "Note: Existing Batch jobs will pick up the new image on their next run."
Write-Host "No Terraform changes needed - the job definition already references :latest."
