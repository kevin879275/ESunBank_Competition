import threading 
from pathlib import Path
import os
import re
class Stealer:
    def __init__(self,interval=10):
        self.interval=interval
        self.imgs=[]
        self.pause=False
        self.enable=True
        self.saveFolder="./EsunTestData/"
        self.suffix="jpg"
        self.index=self.getFinalI()
    
    def getFinalI(self): #return last epoch num (final training saved)
        p=Path(self.saveFolder)
        if not p.exists():
            p.mkdir(parents=True,exist_ok=True)
            return 0
        files = [int(re.match(f'(\d+)\.{self.suffix}', x).group(1)) for x in filter(lambda x:re.match(f'\d+\.{self.suffix}', x), os.listdir(self.saveFolder))]

        return 0 if not len(files) else max(files)+1
        
    def stop(self):
        self.enable=False
    def start(self):
        self.enable=True
        self.tick()
    def tick(self):
        if not self.enable:
            return None
        t = threading.Timer(self.interval,self.tick)
        t.start()
        if  self.pause or len(self.imgs)<=0:
            return None
        while len(self.imgs)>0:
            i, label = self.imgs.pop(0)
            i.save(f'{self.saveFolder}{label}_{self.index}.{self.suffix}')
            print(f'{self.saveFolder}{self.index}.{self.suffix}')
            self.index = self.index+1


