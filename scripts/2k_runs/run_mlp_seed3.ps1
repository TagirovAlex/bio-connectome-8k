# Запуск одного прогона: mlp seed3.
# Лог: results\logs\2k_mlp_seed3.log
$ErrorActionPreference = "Stop"
Set-Location "C:\Github\bio-connectome-cl"
& "C:\Python311\python.exe" -u "scripts\02_run_2k.py" --arch mlp --seed 3 2>&1 | Tee-Object -FilePath "results\logs\2k_mlp_seed3.log"
exit $LASTEXITCODE