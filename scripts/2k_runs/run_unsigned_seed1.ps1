# Запуск одного прогона: unsigned seed1.
# Лог: results\logs\2k_unsigned_seed1.log
$ErrorActionPreference = "Stop"
Set-Location "C:\Github\bio-connectome-cl"
& "C:\Python311\python.exe" -u "scripts\02_run_2k.py" --arch unsigned --seed 1 2>&1 | Tee-Object -FilePath "results\logs\2k_unsigned_seed1.log"
exit $LASTEXITCODE