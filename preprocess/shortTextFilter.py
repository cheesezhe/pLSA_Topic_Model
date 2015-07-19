__author__ = 'zhanghe'
class shortTextFilter:
    def __init__(self,min_length=10):
        self.min_length = min_length
    def filterFile(self,src_filename,dst_filename):
        f_dst = open(dst_filename,'a+')
        with open(src_filename,'r') as src:
            for line in src:
                count = len(line.split(" "))
                if count>=self.min_length:
                    f_dst.write(line)

#test short text filter
ftr = shortTextFilter()
ftr.filterFile("../data/src.txt","../data/dst.txt")


