import time 
import datetime

now = time.time()
time.sleep(2)
end = time.time()
elapsed_time = datetime.timedelta(seconds =end - now)
print(elapsed_time)
