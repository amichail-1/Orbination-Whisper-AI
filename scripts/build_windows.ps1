param(
    [ValidateSet("cpu", "cuda", "vulkan")]
    [string]$Backend = "cpu",
    [string]$WhisperCppDir = "external\whisper.cpp",
    [string]$Out = "whisperhybrid.exe",
    [string]$Tags = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

if (!(Test-Path $WhisperCppDir)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $WhisperCppDir) | Out-Null
    git clone https://github.com/ggml-org/whisper.cpp.git $WhisperCppDir
}

$BuildDir = Join-Path $WhisperCppDir "build-$Backend"
$Args = @("-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF")
if ($Backend -eq "cuda") { $Args += "-DGGML_CUDA=ON" }
if ($Backend -eq "vulkan") { $Args += "-DGGML_VULKAN=ON" }

cmake -S $WhisperCppDir -B $BuildDir @Args
cmake --build $BuildDir --config Release

New-Item -ItemType Directory -Force -Path "$Root\app\lib", "$Root\app\inc\ggml" | Out-Null
Copy-Item "$WhisperCppDir\include\whisper.h" "$Root\app\inc\" -Force
Copy-Item "$WhisperCppDir\ggml\include\*.h" "$Root\app\inc\ggml\" -Force
Get-ChildItem $BuildDir -Recurse -Include "whisper*.lib","ggml*.lib","*.dll" | Copy-Item -Destination "$Root\app\lib\" -Force

Push-Location "$Root\app"
$env:CGO_ENABLED = "1"
if ($Tags -ne "") {
    go build -tags $Tags -trimpath -ldflags="-s -w" -o $Out .
} else {
    go build -trimpath -ldflags="-s -w" -o $Out .
}
Pop-Location

Write-Host "built app\$Out"
