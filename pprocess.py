
import numpy as np

class Pprocess:

  def __init__(self, fname, fsw):
    self.fname = fname
    self.doc_size = 0
    self.sws = []
    self.w2id = {}
    self.w_size = 0
    self.id2w = {}
    self.docs = []
    f = open(fsw,'r')
    for line in f:
    # with open(fsw, 'r') as f :
      self.sws.append(line.strip())


  def __work(self):
    with open(self.fname,'r') as f :
      self.doc_size = (int)(f.readline())
      for line in f :
        line = line.strip()
        self.docs.append(line)
        items = line.split()
        for it in items :
          if it not in self.sws :
            if it not in self.w2id :
              self.w2id[it] = self.w_size 
              self.id2w[self.w_size] = it
              self.w_size += 1

    self.t_d = np.zeros([self.w_size, self.doc_size], dtype = np.int)
  
    for did, doc in enumerate(self.docs) :
      ws = doc.split()
      for w in ws : 
        if w in self.w2id :
          self.t_d[self.w2id[w]][did] += 1

  def get_t_d(self):
    self.__work()
    return self.t_d

  def get_word(self, wid):
    return self.id2w[wid]

if __name__ == '__main__':
  fname = './dst.txt'
  fsw = './stopwords.txt'

  pp = Pprocess(fname, fsw)
  t_d =  pp.get_t_d()
#  for w,d in zip(*t_d.nonzero()) :
#    if t_d[w][d] != 1 :
#      print w,d,t_d[w][d]





