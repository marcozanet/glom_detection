
from typing import Any
from tqdm import tqdm
from datetime import datetime 
import time
from configurator import Configurator
import sys

class TQDM_LOG(Configurator):
    def __init__(self, iterable, desc:str=None, same_line:bool = True, symbol:str=None, symbol_lane:str=None, reversed:bool=False, ) -> None:
        super().__init__()
        self.iterable=iterable
        self._i =0
        self.tot_blocks = 20
        self.epochs = len(iterable)
        self.tot_time_needed_str = '?'
        self.iter_time_secs = 0
        self.start_time = time.time()
        self.symbol = symbol if symbol is not None else '🁢'
        self.desc = f"{desc}: " if desc is not None else ''
        self.same_line = same_line
        self.symbol_repeated = False
        self.symbol_lane = symbol_lane if symbol_lane is not None else '-'
        self.reversed = reversed
        return
    
    def strformat(self, seconds:int):
        h, m, s  = round(seconds//(60*60)), round(seconds//60), round(seconds)
        output = f"{s}s"
        if (m>0 or h>0): output += f"{m}m"
        if h>0: output += f"{h}h"
        return output
    
    def progress_bar(self):
        i = self._i
        if i==1:
            self.iter_time_secs = time.time() - self.start_time
            tot_time_needed_secs = self.iter_time_secs * (self.epochs-i)
            self.tot_time_needed_str = self.strformat(tot_time_needed_secs)
        msg = '|          |'         
        perc_block = int((i+1)/self.epochs*self.tot_blocks)
        perc_100 = int((i+1)/self.epochs*100)
        iter_time_str = self.strformat(self.iter_time_secs)
        perc_100 = f"{perc_100}%" + ' '*(3-len(str(perc_100)))
        if self.symbol_repeated:
            if self.reversed is False:
                msg = self.desc+ perc_100  + '|' + self.symbol*(perc_block)  + (self.tot_blocks-perc_block)*' ' +  '|' + f" {i+1}/{self.epochs} [{iter_time_str}/{self.tot_time_needed_str}]"
            else:
                msg = self.desc+ perc_100  + '|' + (self.tot_blocks-perc_block)*' ' + self.symbol*(perc_block)  +  +  '|' + f" {i+1}/{self.epochs} [{iter_time_str}/{self.tot_time_needed_str}]"
        else:
            if self.reversed is False:
                msg = self.desc+ perc_100  + '|' + self.symbol_lane*(perc_block-1) + self.symbol  + (self.tot_blocks-perc_block)*' ' +  '|' + f" {i+1}/{self.epochs} [{iter_time_str}/{self.tot_time_needed_str}]"
            else:
                msg = self.desc+ perc_100  + '|' + ' '*(self.tot_blocks-perc_block) + self.symbol  + (perc_block-1)*self.symbol_lane +  '|' + f" {i+1}/{self.epochs} [{iter_time_str}/{self.tot_time_needed_str}]"
        end = '\r' if self.same_line is True else '\n'
        msg  += end
        # print(msg, end=end)
        self.log.info(msg)
        
        return     

    def __iter__(self, ):
        self.stop = len(self.iterable)
        return self
    
    def __next__(self):
        self.progress_bar()
        self._i +=1
        if self._i == self.stop:
            raise StopIteration
        return 
    
    

# print(next(iter(TQDM_LOG(range(10)))))
# for i in tqdm(range(6), desc='boh'):
#     for i in range(100000000):
#         pass
print(sys.version)
for i in TQDM_LOG(range(63), same_line=True, desc='boh', symbol='🚗', reversed=True, symbol_lane='_'):
    for i in range(10000000):
        pass

