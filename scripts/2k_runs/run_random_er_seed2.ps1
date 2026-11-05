# Запуск одного прогона: random_er seed2.
# Лог: results\logs\2k_random_er_seed2.log
$ErrorActionPreference = "Stop"
Set-Location "C:\Github\bio-connectome-cl"
& "C:\Python311\python.exe" -u "scripts\02_run_2k.py" --arch random_er --seed 2 2>&1 | Tee-Object -FilePath "results\logs\2k_random_er_seed2.log"
exit $LASTEXITCODE