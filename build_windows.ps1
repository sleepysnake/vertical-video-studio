param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $project

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "无法创建 Python 虚拟环境" }
}

$venvPython = Join-Path $project ".venv\Scripts\python.exe"
& $venvPython -m pip install -r requirements-build.txt
if ($LASTEXITCODE -ne 0) { throw "安装打包依赖失败" }

& $venvPython -m PyInstaller --noconfirm --clean VerticalVideoStudio.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 打包失败" }

$version = & $venvPython -c "from studio.version import __version__; print(__version__)"
if ($LASTEXITCODE -ne 0) { throw "Cannot read application version" }
$distDir = Join-Path $project "dist\VerticalVideoStudio"
Copy-Item -LiteralPath (Join-Path $project "README.md") -Destination (Join-Path $distDir "README.md")
Copy-Item -LiteralPath (Join-Path $project "LICENSE") -Destination (Join-Path $distDir "LICENSE")
New-Item -ItemType Directory -Force -Path (Join-Path $distDir "config") | Out-Null
Copy-Item -LiteralPath (Join-Path $project "config\strategy.json") -Destination (Join-Path $distDir "config\strategy.json")
Copy-Item -LiteralPath (Join-Path $project "config\edit_prompt.txt") -Destination (Join-Path $distDir "config\edit_prompt.txt")
Copy-Item -LiteralPath (Join-Path $project "config\caption_prompt.txt") -Destination (Join-Path $distDir "config\caption_prompt.txt")
Copy-Item -LiteralPath (Join-Path $project "config\review_prompt.txt") -Destination (Join-Path $distDir "config\review_prompt.txt")
Copy-Item -LiteralPath (Join-Path $project "CHANGELOG.md") -Destination (Join-Path $distDir "CHANGELOG.md")
New-Item -ItemType Directory -Force -Path (Join-Path $distDir "docs") | Out-Null
Copy-Item -LiteralPath (Join-Path $project "docs\WORKFLOW.md") -Destination (Join-Path $distDir "docs\WORKFLOW.md")
Copy-Item -LiteralPath (Join-Path $project "docs\nameplate-styles.png") -Destination (Join-Path $distDir "docs\nameplate-styles.png")

$releaseDir = Join-Path $project "release"
New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
$zip = Join-Path $releaseDir "VerticalVideoStudio-$version-Windows.zip"
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip }
& $venvPython (Join-Path $project "scripts\package_windows.py") $distDir $zip
if ($LASTEXITCODE -ne 0) { throw "Windows 包生成失败" }

& $venvPython (Join-Path $project "scripts\smoke_windows_package.py") $zip
if ($LASTEXITCODE -ne 0) { throw "Windows 包在全新解压后未通过启动验证" }

& $venvPython (Join-Path $project "package_source.py")
if ($LASTEXITCODE -ne 0) { throw "源码打包失败" }
Write-Host "完成：$zip"
