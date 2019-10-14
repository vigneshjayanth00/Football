# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:22:33 2019

@author: Home
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:01:54 2019

@author: Home
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:01:23 2019

@author: Home
"""
#libraries
import csv
import xml.etree.ElementTree as et
import numpy as np
import pandas as pd
from datetime import datetime as dt
import os

tree = et.ElementTree(file = r"C:\Users\Home\Documents\Football Python\XML\Possessions\f73-8-2017-918893-possessions.xml")
gameFile = tree.getroot()

gameFile[0].attrib
gameFile[0][0].attrib
gameFile[0][0][0].attrib

#Print a string with the two teams, using %s and the attrib to dynamically fill the string
print ("{} vs {}".format(gameFile[0].attrib["home_team_name"], gameFile[0].attrib["away_team_name"]))


gameFile[0][0].attrib
gameFile[0][0][0].attrib

team_dict = {gameFile[0].attrib["home_team_id"]: gameFile[0].attrib["home_team_name"],
            gameFile[0].attrib["away_team_id"]: gameFile[0].attrib["away_team_name"]}

print(team_dict)

#Create empty lists for the 8 columns we're collecting data for
x_origin = []
y_origin = []
x_destination = []
y_destination = []
outcome = []
minute = []
second = []
half = []
player_id=[]
team = []
event_id= []
Pass_type=[]
sequence=[]
possession_id=[]
pass_type=[]


#Iterate through each game in our file - we only have one
for game in gameFile:

    #Iterate through each event
    for event in game:

        #If the event is a pass (ID = 1)
        if event.attrib.get("type_id")=='1':

            #To the correct list, append the correct attribute using attrib.get()
            x_origin.append(event.attrib.get("x"))
            y_origin.append(event.attrib.get("y"))
            possession_id.append(event.attrib.get("possession_id"))
            sequence.append(event.attrib.get("sequence_id"))
            outcome.append(event.attrib.get("outcome"))
            minute.append(event.attrib.get("min"))
            second.append(event.attrib.get("sec"))
            event_id.append(event.attrib.get("event_id"))
            half.append(event.attrib.get("period_id"))
            player_id.append(event.attrib.get("player_id"))
            team.append(team_dict[event.attrib.get("team_id")])


            #Iterate through each qualifier
            for qualifier in event:

                #If the qualifier is relevant, append the information to the x or y destination lists
                if (qualifier.attrib.get("qualifier_id") == "140"):
                    x_destination.append(qualifier.attrib.get("value"))
                if (qualifier.attrib.get("qualifier_id") == "1"):
                    pass_type.append(qualifier.attrib.get("qualifier_id"))
                if (qualifier.attrib.get("qualifier_id") == "141"):
                    y_destination.append(qualifier.attrib.get("value"))


#Create a list of our 8 columns/lists
column_titles = ["team","half", "min","second","sequence","event_id","player_id","x_origin", "y_origin", "x_destination", "y_destination","possession_id","pass_type","outcome"]

#Use pd.DataFrame to create our table, assign the data in the order of our columns and give it the column titles above
Passes = pd.DataFrame(data=[team,half, minute,second,sequence,event_id,player_id, x_origin, y_origin, x_destination, y_destination,possession_id,pass_type, outcome], index=column_titles)

#Transpose, or flip, the table. Otherwise, our table will run from left to right, rather than top to bottom
Passes_final = Passes.T

#Adding coloumn for attacking or defensive pass
Passes_final.loc[(Passes_final['x_origin']< Passes_final['x_destination']) & (Passes_final['outcome']=='1'), 'Pass Type'] = 'Offensive Pass Won'
Passes_final.loc[(Passes_final['x_origin']< Passes_final['x_destination']) & (Passes_final['outcome']=='0'), 'Pass Type'] = 'Offensive Pass Lost'
Passes_final.loc[Passes_final['x_origin']== Passes_final['x_destination'], 'Pass Type'] = 'Sideway Pass'
Passes_final.loc[Passes_final['x_origin']> Passes_final['x_destination'], 'Pass Type'] = 'Defensive Pass'

#Show us the top 5 rows of the table
final_table.head()