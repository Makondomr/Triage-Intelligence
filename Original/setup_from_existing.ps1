$ErrorActionPreference = 'Stop'
$target = Join-Path $PSScriptRoot 'models\triagegeist_lightgbm_research.joblib'
if (Test-Path $target) { Write-Host "MODEL READY: $target"; exit 0 }
$locations = @(
 (Join-Path $PSScriptRoot '..\Kenya_Triage_AI_Integration\models\triagegeist_lightgbm_research.joblib'),
 (Join-Path $PSScriptRoot '..\Triage_Intelligence_Worker_Fix\models\triagegeist_lightgbm_research.joblib'),
 (Join-Path $PSScriptRoot '..\Triage_Intelligence_Readable_Complete\models\triagegeist_lightgbm_research.joblib'),
 (Join-Path $PSScriptRoot '..\Triage_Intelligence_Complete_Contrast_Fix\models\triagegeist_lightgbm_research.joblib')
)
foreach ($source in $locations) {
 if (Test-Path $source) { Copy-Item -LiteralPath $source -Destination $target -Force; Write-Host "MODEL COPIED: $target"; exit 0 }
}
Write-Host 'MODEL NOT FOUND. Copy your existing triagegeist_lightgbm_research.joblib into the models folder.'
exit 1
