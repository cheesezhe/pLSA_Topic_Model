
import json
import pprint
import time

LOG_FILE='plsa.log'
KEY_WORD_SIZE = 10

OUT_FILE='result.dat'

res = {}

def clean_z(f, line):
  with open(OUT_FILE, 'a') as fout :
    attrs = line.split()
    z = (int)(attrs[2])
    llhood = (float)(attrs[5])
    cost = (float)(attrs[8])

    res[z] = {'z' : z, 'llhood' : llhood, 'cost' : cost}

    fout.write('z = %d llhood = %f cost = %f\n' %(z, llhood, cost))
    topic_list = [] 
    for it in range(z):
      l = f.readline()
      kw_list = []
      for itx in range(KEY_WORD_SIZE):
        l = f.readline()[10:]
        w = l.split()[0]
        p = (float)(l.split()[2])
        kw_list.append({w : p})

      topic_list.append(kw_list)
      
    res[z]['topic'] = topic_list
    #pprint.pprint( res)
    #time.sleep(10)
      
    

def process_log():
  with open(LOG_FILE, 'r') as f:
    while True :
      line = f.readline()
      if not line : 
        break
      
      if line[10:13] == 'z =' :
        clean_z(f, line)

  
  with open('result_verbose.dat', 'w') as f:
    pprint.pprint(res, f)


if __name__ == '__main__' :
  process_log()








