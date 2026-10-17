param([string]$RunDirectory = '.work/detector_deploy_20261001', [string]$BuildDirectory = '')
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$run = [IO.Path]::GetFullPath((Join-Path $root $RunDirectory))
$stage = Join-Path $run 'build/main.dist'
if ($BuildDirectory) { $stage = [IO.Path]::GetFullPath((Join-Path $root $BuildDirectory)) }
if (-not $run.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase) -or
    -not $stage.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Paths must stay within workspace.' }
$package = Join-Path $root 'APP/DiskInspection/bin/BroSight_Disinspector'
$plugin = Join-Path $package 'plugin'
if (Test-Path -LiteralPath (Join-Path $run 'deployment.json')) { throw 'Already deployed.' }
if (Get-Process -Name DiskInspection -ErrorAction SilentlyContinue) { throw 'Close APP before backend deployment.' }
if ((Get-FileHash -LiteralPath (Join-Path $plugin 'main.exe')).Hash -ne
    (Get-FileHash -LiteralPath (Join-Path $run 'rollback/plugin/main.exe')).Hash) { throw 'Runtime backup differs.' }
if ((Get-FileHash -LiteralPath (Join-Path $stage 'config/models/d_p_ha_20260930.onnx')).Hash -ne
    'B04223C1CE295416E4C7E6F7BF6342499BE7B87A81689AC8CEEE06E19C413BBE') { throw 'Wrong candidate model.' }
$exe = [IO.Path]::GetFullPath((Join-Path $plugin 'main.exe'))
foreach ($proc in Get-Process -Name main -ErrorAction SilentlyContinue) {
    if ($proc.Path -ne $exe) { throw ('Unexpected service: ' + $proc.Path) }
    Stop-Process -Id $proc.Id
    $proc.WaitForExit()
}
$preserved = @{}
foreach ($name in @('DiskInspection.exe','DiskInspection.exe.config','user.ini','NLog.config','plugin/license.dat')) {
    $path=Join-Path $package $name
    if (Test-Path -LiteralPath $path) { $preserved[$name]=(Get-FileHash -LiteralPath $path).Hash }
}
foreach ($child in Get-ChildItem -LiteralPath $stage) {
    Copy-Item -LiteralPath $child.FullName -Destination $plugin -Recurse -Force
}
$copied=@()
foreach ($file in Get-ChildItem -LiteralPath $stage -File -Recurse) {
    $relative=$file.FullName.Substring($stage.Length+1)
    $hash=(Get-FileHash -LiteralPath $file.FullName).Hash
    if ($hash -ne (Get-FileHash -LiteralPath (Join-Path $plugin $relative)).Hash) { throw ('Copied file differs: '+$relative) }
    $copied+=[pscustomobject]@{name=$relative;sha256=$hash}
}
foreach ($name in $preserved.Keys) {
    if ($preserved[$name] -ne (Get-FileHash -LiteralPath (Join-Path $package $name)).Hash) { throw ('User file changed: '+$name) }
}
[pscustomobject]@{deployed_at=(Get-Date -Format o);plugin=$plugin;files=$copied;preserved=$preserved;
    rollback=(Join-Path $run 'rollback');point_conf=0.5;point_iou=0.1;detector_roi=@(0,0,1944,2200);
    slot_completion=$true;config_sha256=(Get-FileHash -LiteralPath (Join-Path $plugin 'config/config.env')).Hash} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $run 'deployment.json') -Encoding UTF8
$process=Start-Process -FilePath $exe -WorkingDirectory $plugin -WindowStyle Hidden -PassThru
Write-Output ('Deployed and verified '+$copied.Count+' backend files. PID='+$process.Id)
Write-Output ('APP/settings/license preserved: '+($preserved.Keys -join ', '))
