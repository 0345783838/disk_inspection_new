$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$run = Join-Path $root '.work/segmentation_deploy_20261001'
$stage = Join-Path $run 'build/main.dist'
$package = Join-Path $root 'APP/DiskInspection/bin/BroSight_Disinspector'
$plugin = Join-Path $package 'plugin'
$release = Join-Path $root 'APP/DiskInspection/bin/x64/Release'
$backup = Join-Path $run 'rollback/app_files'
if (Test-Path -LiteralPath (Join-Path $run 'deployment.json')) {
    throw 'Already deployed; refusing to replace the original backup.'
}
if (Get-Process -Name DiskInspection -ErrorAction SilentlyContinue) {
    throw 'Close DiskInspection before deployment.'
}
function Read-Settings([string] $path) {
    $values = @{}
    foreach ($line in Get-Content -LiteralPath $path) {
        if ($line -match '^\s*([^#=]+?)\s*=\s*(.*?)\s*$') {
            $values[$matches[1]] = $matches[2]
        }
    }
    return $values
}
$settings = Read-Settings (Join-Path $stage 'config/config.env')
$previous = Read-Settings (Join-Path $plugin 'config/config.env')
foreach ($key in $previous.Keys) {
    if ($key -ne 'DISK_SEGMENT_MODEL_PATH' -and $previous[$key] -ne $settings[$key]) {
        throw ('Unexpected existing setting change: ' + $key)
    }
}
if ((Get-FileHash -LiteralPath (Join-Path $plugin 'main.exe')).Hash -ne
    (Get-FileHash -LiteralPath (Join-Path $run 'rollback/plugin/main.exe')).Hash) {
    throw 'Plugin backup is not the current binary.'
}
$stagedExe = [IO.Path]::GetFullPath((Join-Path $stage 'main.exe'))
foreach ($proc in Get-Process -Name main -ErrorAction SilentlyContinue) {
    if ($proc.Path -ne $stagedExe) { throw ('Unexpected running service: ' + $proc.Path) }
    Stop-Process -Id $proc.Id
    $proc.WaitForExit()
}
New-Item -ItemType Directory -Path $backup -Force | Out-Null
$files = @('DiskInspection.exe', 'DiskInspection.pdb',
    'System.Drawing.Primitives.dll', 'System.Runtime.dll',
    'System.Runtime.InteropServices.RuntimeInformation.dll')
$appReport = @()
foreach ($name in $files) {
    $target = Join-Path $package $name
    $exists = Test-Path -LiteralPath $target
    $oldHash = $null
    if ($exists) {
        $oldHash = (Get-FileHash -LiteralPath $target).Hash
        Copy-Item -LiteralPath $target -Destination (Join-Path $backup $name)
    }
    $appReport += [pscustomobject]@{ name=$name; previously_present=$exists; before_sha256=$oldHash }
}
$preserved = @{}
foreach ($name in @('DiskInspection.exe.config', 'user.ini', 'NLog.config')) {
    $path = Join-Path $package $name
    if (Test-Path -LiteralPath $path) {
        Copy-Item -LiteralPath $path -Destination (Join-Path $backup $name)
        $preserved[$name] = (Get-FileHash -LiteralPath $path).Hash
    }
}
foreach ($child in Get-ChildItem -LiteralPath $stage) {
    Copy-Item -LiteralPath $child.FullName -Destination $plugin -Recurse -Force
}
$pluginReport = @()
foreach ($file in Get-ChildItem -LiteralPath $stage -Recurse -File) {
    $relative = $file.FullName.Substring($stage.Length + 1)
    $hash = (Get-FileHash -LiteralPath $file.FullName).Hash
    if ($hash -ne (Get-FileHash -LiteralPath (Join-Path $plugin $relative)).Hash) {
        throw ('Copied plugin file differs: ' + $relative)
    }
    $pluginReport += [pscustomobject]@{ name=$relative; sha256=$hash }
}
foreach ($entry in $appReport) {
    Copy-Item -LiteralPath (Join-Path $release $entry.name) -Destination (Join-Path $package $entry.name) -Force
    $hash = (Get-FileHash -LiteralPath (Join-Path $release $entry.name)).Hash
    if ($hash -ne (Get-FileHash -LiteralPath (Join-Path $package $entry.name)).Hash) {
        throw ('Copied APP file differs: ' + $entry.name)
    }
    $entry | Add-Member -NotePropertyName after_sha256 -NotePropertyValue $hash
}
foreach ($name in $preserved.Keys) {
    if ($preserved[$name] -ne (Get-FileHash -LiteralPath (Join-Path $package $name)).Hash) {
        throw ('User configuration changed: ' + $name)
    }
}
$report = [pscustomobject]@{
    deployed_at=(Get-Date -Format o); package=$package; plugin=$plugin;
    app_files=$appReport; plugin_files=$pluginReport;
    preserved_files=$preserved; segmentation_settings=$settings;
    rollback=(Join-Path $run 'rollback')
}
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $run 'deployment.json') -Encoding UTF8
Write-Output ('Deployed and hash-verified ' + $pluginReport.Count + ' plugin files and ' + $appReport.Count + ' APP files.')
Write-Output ('Camera/PLC/user settings preserved: ' + ($preserved.Keys -join ', '))
