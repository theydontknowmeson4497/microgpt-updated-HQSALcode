
import sys
import os

# Save all output to a file
log_file = open("microgpt_quantum_test_log.txt", "w")
sys.stdout = log_file
sys.stderr = log_file

# Now run the train_and_evaluate function
print("=== Starting test ===")
exec(open("microgpt_quantum.py").read())

print("\n=== Test complete ===")
log_file.close()
