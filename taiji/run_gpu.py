import subprocess
import time

time.sleep(60)

while True:
    outputs = subprocess.check_output(
        "ps -aux | grep examples | grep -v grep | awk '{print $2}' ",
        shell=True
    ).decode().strip('\n')
    if outputs:
        print("ogb processes:\n{}".format(outputs))
        time.sleep(60*10)
    else:
        print("no processes")
        break

