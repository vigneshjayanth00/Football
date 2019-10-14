# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:53:23 2019

@author: Home
"""

#import libraries
import numpy as np
import os
import glob
import pandas as pd
directory = os.listdir(r'C:\Users\Home\Documents\Football Python\CSV\Possessions')
os.chdir(r'C:\Users\Home\Documents\Football Python\CSV\Possessions')


extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
possessions = pd.concat([pd.read_csv(f) for f in all_filenames ])

#fill events in blank spaces using ffill function
possessions_final=possessions.ffill(axis=0)

#Top 10 rows
check1=possessions_final.head(100)

check1.to_csv(r'C:\Users\Home\Documents\Football Python\CSV\possessions_check.csv', encoding='utf-8', index=False)

#Creating Team_Player_Info Dataset
Possession=possessions_final[["Game__id","Game__home_team_id","Game__home_team_name","Game__away_team_id","Game__away_team_name","Game__game_date","Game__competition_id",
"Game__competition_name","Game__season_id","Game__season_name","Game__matchday","Game__period_1_start","Game__period_2_start","Game__Status",
"Game__Event__id","Game__Event__event_id","Game__Event__type_id","Game__Event__period_id","Game__Event__min","Game__Event__sec","Game__Event__player_id",
"Game__Event__team_id","Game__Event__outcome","Game__Event__x","Game__Event__y","Game__Event__timestamp","Game__Event__last_modified","Game__Event__version",
"Game__Event__assist","Game__Event__sequence_id","Game__Event__possession_id","Game__Event__Q__id","Game__Event__Q__qualifier_id","Game__Event__Q__value"]]

Possession.columns = ["Game_ID","home_team_id","home_team_name","away_team_id","away_team_name","game_date","competition_id","competition_name",
"season_id","season_name","matchday","period_1_start","period_2_start","Status","Event_id","Event_event","Event_type","Event_period",
"Event_min","Event_sec","Event_player","Event_team","Event_outcome","Event_x","Event_y","Event_timestamp","Event_last","Event_version",
"Event_assist","Event_sequence","Event_possession","Event_Q_id","Event_Q_qualifier","Event_Q_value"]


#Top 10 rows
check1=Possession.head(100)

#Creating x_destination for passes
Possession["x_destination"]=np.where(Possession['Event_Q_qualifier']=='140', Possession['Event_Q_qualifier'], 0)
Possession["y_destination"]=np.where(Possession['Event_Q_qualifier']=='141', Possession['Event_Q_qualifier'], 0)
