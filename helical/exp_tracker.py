import os 
import pandas as pd

class Exp_Tracker():

    def __init__(self, other_params) -> None:
        self.col_names = ['datetime', 'status', 'duration', 'exp_fold', 'dataset', 'batch_size', 'task', 'map_classes', 'epochs', 'tile_size', 'image_size',  'weights', 'device', 'data_folder', 'crossvalidation', 'tot_kfolds', 'cur_kfold', 'note' ]
        self.file_csv = 'exp_tracker.csv'
        self.other_params = other_params

    def reset_tracker(self):

        df = pd.DataFrame(columns=self.col_names)
        df.to_csv(self.file_csv, index=False)

        return

    def update_tracker(self, **kwargs): 

        df1 = pd.read_csv(self.file_csv)
        
        d1 = self.other_params
        d1.update({key:value for key, value in kwargs.items()})
        print(d1)
        d1 = {key:str(value) for key, value in d1.items() if key in self.col_names}
        print(d1)
        df2 = pd.DataFrame(d1, index = [0], )
        # print(self.other_params)
        # print(filtered_dict)
        df = pd.concat([df1, df2], ignore_index=True)
        print(df)

        df.to_csv(self.file_csv, index=False)

        return
    
    def update_status(self, status:str) -> None:
        """ Updates 'status' column in tracker file. """

        df1 = pd.read_csv(self.file_csv)
        df1.iat[-1, 1]= status
        df1.to_csv(self.file_csv, index=False)

        return
    
    def update_duration(self, duration:str) -> None:
        """ Updates 'duration' column in tracker file. """

        df1 = pd.read_csv(self.file_csv)
        df1.iat[-1, 2]= duration
        df1.to_csv(self.file_csv, index=False)

        return
    
    # def write_note(self, note:str):

    #     df1 = pd.read_csv(self.file_csv)
    #     df1.iat[-1, -1]= note
    #     df1.to_csv(self.file_csv, index=False)

    #     return


if __name__ == '__main__':
    tracker = Exp_Tracker(other_params={'h':3, 'gfd':5})
    tracker.reset_tracker()
    # tracker.update_tracker(boh = 'a', dataset='dj',  exp_fold = '3')
    # tracker.update_duration('2h 1m 3s')
    # tracker.update_duration('2h 4m 3s')
    # tracker.write_note('testing')
    # tracker.update_status(status='started')
    # tracker.update_tracker(boh = 'a3', dataset='4', exp_fold = 'exp43')
    # tracker.update_tracker(boh = '432a', dataset='m234uw', exp_fold = '3')
    # tracker.update_tracker(boh = 'a3', dataset='muw', exp_fold = '334')

