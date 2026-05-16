import os
import subprocess
import time
from datetime import datetime
import sys

# CONFIG

# List of experiment scripts
EXPERIMENTS = [

    "run_weighted.py",
    "run_weighted_hybrid.py",
    "run_contrast_weighted.py",
    "run_smote.py",
    "run_contrast_smote.py"
]

# PATH SETUP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "run_all_log.txt")

# HELPER FUNCTION

def log(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# MAIN RUNNER

def run_all():

    start_time = time.time()

    log("\n********************************")
    log(" STARTING ALL EXPERIMENTS")
    log(f"Start Time: {datetime.now()}")
    log("********************************\n")

    for exp in EXPERIMENTS:

        exp_path = os.path.join(BASE_DIR, exp)

        if not os.path.exists(exp_path):
            log(f" Skipping {exp} (file not found)")
            continue

        log("\n******************************")
        log(f" Running: {exp}")
        log(f"Time: {datetime.now()}")
        log("******************************")

        exp_start = time.time()

        try:

            result = subprocess.run(
                [sys.executable, exp_path],
                check=True
            )

            duration = time.time() - exp_start

            log(f" Completed: {exp}")
            log(f" Time taken: {duration:.2f} seconds")

        except subprocess.CalledProcessError as e:
            duration = time.time() - exp_start

            log(f" FAILED: {exp}")
            log(f" Time before failure: {duration:.2f} seconds")
            log(f"Error: {e}")

            continue

    total_time = time.time() - start_time


    log(" ALL EXPERIMENTS FINISHED")
    log(f"End Time: {datetime.now()}")
    log(f"Total Time: {total_time/60:.2f} minutes")



# ENTRY POINT

if __name__ == "__main__":
    run_all()