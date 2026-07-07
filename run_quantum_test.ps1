
# First, cd to the correct directory!
cd "e:\STUDY MATERIAL\Internship\Quantum Computing Internship\microgpt-updated-HQSALcode-main\microgpt-updated-HQSALcode-main"
Write-Host "Running microgpt_quantum.py with num_steps=5 for quick test..."
python -u microgpt_quantum.py 2>&1 | Tee-Object -FilePath quantum_test_output.txt
