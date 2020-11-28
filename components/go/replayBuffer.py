import os
import sys
import threading

#https://stackoverflow.com/questions/11983938/python-appending-to-same-file-from-multiple-threads

def writeToFile(file_name, message, count):
  with open(file_name, 'a', 1) as f:
    for i in range(0, count):
      f.write(message + os.linesep)

if __name__ == '__main__':
  #start a file
  with open('some.txt', 'w') as f:
    f.write('this is the beginning' + os.linesep)
  #make 10 threads write a million lines to the same file at the same time
  threads = []
  for i in range(0, 10):
    #target = name for method that must be executed in thread
    threads.append(threading.Thread(target=writeToFile, args=('some.txt', 'hey im thread %d' % i, 1000000)))
    threads[-1].start()
  for t in threads:
    t.join()
  #check what the heck the file had
  uniq_lines = set()
  with open('some.txt', 'r') as f:
    for l in f:
      uniq_lines.add(l)
  for u in uniq_lines:
    sys.stdout.write(u)