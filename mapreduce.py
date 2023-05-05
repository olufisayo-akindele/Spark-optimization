from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol

class MRDataPreprocessing(MRJob):
    OUTPUT_PROTOCOL = RawValueProtocol
    def mapper(self, _, line):
        line = line.strip().split(",")# remove leading and trailing whitespace
        line.pop(0)
        a=line.pop(len(line)-1)
        line.insert(0,a)
        y=",".join(x for x in line)
        
        yield None,y
    
        
if __name__ == '__main__':
    MRDataPreprocessing.run()