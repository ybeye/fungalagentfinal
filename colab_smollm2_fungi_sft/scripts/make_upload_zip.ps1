param(
    [string]$OutputZip = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

if ([string]::IsNullOrWhiteSpace($OutputZip)) {
    $OutputZip = Join-Path $ProjectRoot "smollm2_fungi_colab_bundle.zip"
}

$OutputZip = [System.IO.Path]::GetFullPath($OutputZip)
if (Test-Path -LiteralPath $OutputZip) {
    Remove-Item -LiteralPath $OutputZip -Force
}

$stageParent = Join-Path ([System.IO.Path]::GetTempPath()) ("smollm2_colab_zip_" + [System.Guid]::NewGuid().ToString("N"))
$stageRoot = Join-Path $stageParent (Split-Path $ProjectRoot -Leaf)
New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null

try {
    Get-ChildItem -LiteralPath $ProjectRoot -Force |
        Where-Object { $_.FullName -ne $OutputZip } |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $stageRoot -Recurse -Force
        }

    Compress-Archive -Path $stageRoot -DestinationPath $OutputZip -CompressionLevel Optimal
    Write-Host "Wrote $OutputZip"
}
finally {
    if (Test-Path -LiteralPath $stageParent) {
        Remove-Item -LiteralPath $stageParent -Recurse -Force
    }
}
