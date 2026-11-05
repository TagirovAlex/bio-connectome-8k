# Запуск одного прогона: mlp seed1 (перезапуск: старый результат был smoke 2048 шагов).
# Лог: results\logs\2k_mlp_seed1.log
$ErrorActionPreference = "Stop"
Set-Location "C:\Github\bio-connectome-cl"
& "C:\Python311\python.exe" -u "scripts\02_run_2k.py" --arch mlp --seed 1 2>&1 | Tee-Object -FilePath "results\logs\2k_mlp_seed1.log"
exit $LASTEXITCODE
