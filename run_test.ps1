
$ErrorActionPreference = "Continue"
cd "e:\STUDY MATERIAL\Internship\Quantum Computing Internship\microgpt-updated-HQSALcode-main\microgpt-updated-HQSALcode-main"

Write-Host "=== RUNNING microgpt_quantum.py ===" -ForegroundColor Green
python -u microgpt_quantum.py *>&1 | Tee-Object -FilePath full_test_output.txt
Write-Host "=== DONE ===" -ForegroundColor Green

Write-Host "`nContents of full_test_output.txt:" -ForegroundColor Cyan
Get-Content full_test_output.txt
