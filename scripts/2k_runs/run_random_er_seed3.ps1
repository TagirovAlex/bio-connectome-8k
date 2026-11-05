# Запуск одного прогона: random_er seed3.
# Лог: results\logs\2k_random_er_seed3.log
$ErrorActionPreference = "Stop"
Set-Location "C:\Github\bio-connectome-cl"
& "C:\Python311\python.exe" -u "scripts\02_run_2k.py" --arch random_er --seed 3 2>&1 | Tee-Object -FilePath "results\logs\2k_random_er_seed3.log"
exit $LASTEXITCODE