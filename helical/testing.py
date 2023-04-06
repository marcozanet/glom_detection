import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

os.chdir('/Users/marco/yolo/code/helical')
os.system("pytest test_yolo.py")





