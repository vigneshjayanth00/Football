# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:59:15 2019

@author: Home
"""


from xml.etree import ElementTree
tree = ElementTree.parse(r'C:\Users\Home\Documents\Football Python\XML\Matchresults\srml-8-2017-f918893-matchresults.xml')
root = tree.getroot()

for att in root:
    first = att.find('attval').text
    for subatt in att.find('children'):
        second = subatt.find('attval').text
        print('{},{}'.format(first, second))

matchdata=pd.read_csv(r'C:\Users\Home\Documents\convertcsv.csv')

#Columns with known column isn't required
matchdata = matchdata[matchdata.columns.drop(list(matchdata.filter(regex='Known')))]
#Team Data
HomeTeam = matchdata.iloc[0:1,0:6]
AwayTeam = matchdata.iloc[1:,0:6]
Managers=Player.iloc[:,70:74]

#Player Data
Player = matchdata.iloc[:,6:]

Team1_Player_1=Player.iloc[:1,0:4]
Team2_Player_1=Player.iloc[1:,0:4]
Team1_Player_2=Player.iloc[:1,4:8]
Team2_Player_2=Player.iloc[1:,4:8]
Team1_Player_3=Player.iloc[:1,8:12]
Team2_Player_3=Player.iloc[1:,8:12]
Team1_Player_4=Player.iloc[:1,12:16]
Team2_Player_4=Player.iloc[1:,12:16]
Team1_Player_5=Player.iloc[:1,16:20]
Team2_Player_5=Player.iloc[1:,16:20]
Team1_Player_6=Player.iloc[:1,20:24]
Team2_Player_6=Player.iloc[1:,20:24]
Team1_Player_7=Player.iloc[:1,24:28]
Team2_Player_7=Player.iloc[1:,24:28]
Team1_Player_8=Player.iloc[:1,28:32]
Team2_Player_8=Player.iloc[1:,28:32]
Team1_Player_9=Player.iloc[:1,32:36]
Team2_Player_9=Player.iloc[1:,32:36]
Team1_Player_10=Player.iloc[:1,36:40]
Team2_Player_10=Player.iloc[1:,36:40]
Team1_Player_11=Player.iloc[:1,40:44]
Team2_Player_11=Player.iloc[1:,40:44]
Team1_Player_12=Player.iloc[:1,44:48]
Team2_Player_12=Player.iloc[1:,44:48]
Team1_Player_13=Player.iloc[:1,48:52]
Team2_Player_13=Player.iloc[1:,48:52]
Team1_Player_14=Player.iloc[:1,52:56]
Team2_Player_14=Player.iloc[1:,52:56]
Team1_Player_15=Player.iloc[:1,56:60]
Team2_Player_15=Player.iloc[1:,56:60]
Team1_Player_16=Player.iloc[:1,60:64]
Team2_Player_16=Player.iloc[1:,60:64]
Team1_Player_17=Player.iloc[:1,64:68]
Team2_Player_17=Player.iloc[1:,64:68]
Team1_Player_18=Player.iloc[:1,68:72]
Team2_Player_18=Player.iloc[1:,68:72]

Home_Team_Players = pd.DataFrame(np.concatenate([Team1_Player_1.values,Team1_Player_10.values,Team1_Player_11.values,Team1_Player_12.values,
Team1_Player_13.values,Team1_Player_14.values,Team1_Player_15.values,Team1_Player_16.values,
Team1_Player_17.values,Team1_Player_18.values,Team1_Player_2.values,Team1_Player_3.values,
Team1_Player_4.values,Team1_Player_5.values,Team1_Player_6.values,Team1_Player_7.values,
Team1_Player_8.values,Team1_Player_9.values]), columns=Team1_Player_1.columns)

#Adding team names to the player list
Home_Team_Players["Team"]=HomeTeam['Name']
Home_Team_Players=Home_Team_Players.ffill(axis=0)

Away_team_Players = pd.DataFrame(np.concatenate([Team2_Player_1.values,Team2_Player_10.values,Team2_Player_11.values,Team2_Player_12.values,
Team2_Player_13.values,Team2_Player_14.values,Team2_Player_15.values,Team2_Player_16.values,
Team2_Player_17.values,Team2_Player_18.values,Team2_Player_2.values,Team2_Player_3.values,
Team2_Player_4.values,Team2_Player_5.values,Team2_Player_6.values,Team2_Player_7.values,
Team2_Player_8.values,Team2_Player_9.values]), columns=Team1_Player_1.columns)

#Adding team names to the player list
Away_team_Players["Team"]=AwayTeam['Name']
Away_team_Players=Away_team_Players.ffill(axis=0)
