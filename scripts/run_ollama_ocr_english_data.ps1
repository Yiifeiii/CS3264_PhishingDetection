param(
    [string]$InputDir = "data\english_data",
    [string]$OutputCsv = "artifacts\ollama_ocr\english_data_ocr.csv",
    [string]$Model = "llama3.2-vision",
    [string]$OllamaHost = "http://localhost:11434",
    [int]$TimeoutSeconds = 600,
    [switch]$Recursive = $true,
    [switch]$Resume = $true
)

$ErrorActionPreference = "Stop"

function Test-OllamaServer {
    param([string]$BaseUrl)
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/api/tags" -TimeoutSec 5 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Wait-OllamaServer {
    param(
        [string]$BaseUrl,
        [int]$MaxAttempts = 30
    )
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        if (Test-OllamaServer -BaseUrl $BaseUrl) {
            return
        }
        Start-Sleep -Seconds 2
    }
    throw "Ollama server did not become ready at $BaseUrl."
}

Write-Host "Checking Ollama CLI..."
$ollamaPath = (Get-Command ollama -ErrorAction Stop).Source
Write-Host "Using Ollama at: $ollamaPath"

if (-not (Test-OllamaServer -BaseUrl $OllamaHost)) {
    Write-Host "Ollama server is not running. Starting it in the background..."
    Start-Process -FilePath $ollamaPath -ArgumentList "serve" -WindowStyle Hidden | Out-Null
    Wait-OllamaServer -BaseUrl $OllamaHost
}

Write-Host "Ensuring vision model is available: $Model"
& $ollamaPath pull $Model

$outputDir = Split-Path -Parent $OutputCsv
if ($outputDir) {
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
}

$arguments = @(
    "scripts\ollama_ocr_extract.py",
    "--input", $InputDir,
    "--output", $OutputCsv,
    "--model", $Model,
    "--host", $OllamaHost,
    "--timeout-seconds", $TimeoutSeconds
)

if ($Recursive) {
    $arguments += "--recursive"
}

if ($Resume) {
    $arguments += "--resume"
}

Write-Host "Starting OCR batch..."
Write-Host ("python " + ($arguments -join " "))
python @arguments

Write-Host ""
Write-Host "Done. OCR results saved to: $OutputCsv"
