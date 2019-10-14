# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:53:23 2019

@author: Home
"""

#import libraries
import os
import glob
import pandas as pd
directory = os.listdir(r'C:\Users\Home\Documents\Football Python\CSV\Match Results')


os.chdir(r'C:\Users\Home\Documents\Football Python\CSV\Match Results')


extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
Matchresults = pd.concat([pd.read_csv(f) for f in all_filenames ])

#fill events in blank spaces using ffill function
Matchresults_final=Matchresults.ffill(axis=0)

#Adding Column headers to list
list=Matchresults_final.columns.values.tolist()
check2=Matchresults_final.head(100)

#Creating Competition Dataset
Competition=Matchresults_final[["SoccerDocument__Competition__Country","SoccerDocument__Competition__Name","SoccerDocument__Competition__Stat__#Text","SoccerDocument__Competition__Stat__Type","SoccerDocument__Competition__uID"]]
Competition.columns = ["Country","Name","Stat#Text","StatType","uID"]

#Creating Match Officials Dataset
Match_Officials=Matchresults_final[["SoccerDocument__MatchData__MatchOfficial__OfficialData__OfficialRef__Type","SoccerDocument__MatchData__MatchOfficial__OfficialName__First","SoccerDocument__MatchData__MatchOfficial__OfficialName__Last","SoccerDocument__MatchData__MatchOfficial__uID","SoccerDocument__MatchData__AssistantOfficials__AssistantOfficial__FirstName","SoccerDocument__MatchData__AssistantOfficials__AssistantOfficial__LastName","SoccerDocument__MatchData__AssistantOfficials__AssistantOfficial__Type","SoccerDocument__MatchData__AssistantOfficials__AssistantOfficial__uID"]]
Match_Officials.columns = ["AssistantOfficial_FirstName","AssistantOfficial_LastName","AssistantOfficial_Type","AssistantOfficial_uID","OfficialData_OfficialRef Type","OfficialName_First","OfficialName_Last","uID"]

#Creating Bookings Dataset
Bookings=Matchresults_final[["SoccerDocument__MatchData__Stat__#Text","SoccerDocument__MatchData__Stat__Type","SoccerDocument__MatchData__TeamData__Booking__Card","SoccerDocument__MatchData__TeamData__Booking__CardType","SoccerDocument__MatchData__TeamData__Booking__EventID","SoccerDocument__MatchData__TeamData__Booking__EventNumber","SoccerDocument__MatchData__TeamData__Booking__Min","SoccerDocument__MatchData__TeamData__Booking__Period","SoccerDocument__MatchData__TeamData__Booking__PlayerRef","SoccerDocument__MatchData__TeamData__Booking__Reason","SoccerDocument__MatchData__TeamData__Booking__Sec","SoccerDocument__MatchData__TeamData__Booking__Time","SoccerDocument__MatchData__TeamData__Booking__TimeStamp","SoccerDocument__MatchData__TeamData__Booking__uID"]]
Bookings.columns = ["Stat_#Text","Stat_Type","Booking_Card","Booking_CardType","Booking_EventID","Booking_EventNumber","Booking_Min","Booking_Period","Booking_PlayerRef","Booking_Reason","Booking_Sec","Booking_Time","Booking_TimeStamp","Booking_uID"]

#Creating MatchData_Team Dataset
MatchData_Team=Matchresults_final[["SoccerDocument__MatchData__TeamData__Formation","SoccerDocument__MatchData__TeamData__Goal__Assist__#Text",
"SoccerDocument__MatchData__TeamData__Goal__Assist__PlayerRef","SoccerDocument__MatchData__TeamData__Goal__EventID",
"SoccerDocument__MatchData__TeamData__Goal__EventNumber","SoccerDocument__MatchData__TeamData__Goal__Min",
"SoccerDocument__MatchData__TeamData__Goal__Period","SoccerDocument__MatchData__TeamData__Goal__PlayerRef",
"SoccerDocument__MatchData__TeamData__Goal__Sec","SoccerDocument__MatchData__TeamData__Goal__SecondAssist__#Text",
"SoccerDocument__MatchData__TeamData__Goal__SecondAssist__PlayerRef","SoccerDocument__MatchData__TeamData__Goal__SoloRun",
"SoccerDocument__MatchData__TeamData__Goal__Time","SoccerDocument__MatchData__TeamData__Goal__TimeStamp",
"SoccerDocument__MatchData__TeamData__Goal__Type","SoccerDocument__MatchData__TeamData__Goal__uID",
"SoccerDocument__MatchData__TeamData__MissedPenalty__EventID","SoccerDocument__MatchData__TeamData__MissedPenalty__EventNumber",
"SoccerDocument__MatchData__TeamData__MissedPenalty__Min","SoccerDocument__MatchData__TeamData__MissedPenalty__Period",
"SoccerDocument__MatchData__TeamData__MissedPenalty__PlayerRef","SoccerDocument__MatchData__TeamData__MissedPenalty__Time",
"SoccerDocument__MatchData__TeamData__MissedPenalty__TimeStamp","SoccerDocument__MatchData__TeamData__MissedPenalty__Type",
"SoccerDocument__MatchData__TeamData__MissedPenalty__uID","SoccerDocument__MatchData__TeamData__PlayerLineUp__MatchPlayer__Captain",
"SoccerDocument__MatchData__TeamData__PlayerLineUp__MatchPlayer__Formation_Place",
"SoccerDocument__MatchData__TeamData__PlayerLineUp__MatchPlayer__PlayerRef","SoccerDocument__MatchData__TeamData__PlayerLineUp__MatchPlayer__Position",
"SoccerDocument__MatchData__TeamData__PlayerLineUp__MatchPlayer__ShirtNumber","SoccerDocument__MatchData__TeamData__PlayerLineUp__MatchPlayer__Status",
"SoccerDocument__MatchData__TeamData__PlayerLineUp__MatchPlayer__SubPosition","SoccerDocument__MatchData__TeamData__Score",
"SoccerDocument__MatchData__TeamData__Side","SoccerDocument__MatchData__TeamData__Substitution__EventID",
"SoccerDocument__MatchData__TeamData__Substitution__EventNumber","SoccerDocument__MatchData__TeamData__Substitution__Min",
"SoccerDocument__MatchData__TeamData__Substitution__Period","SoccerDocument__MatchData__TeamData__Substitution__Reason",
"SoccerDocument__MatchData__TeamData__Substitution__Retired","SoccerDocument__MatchData__TeamData__Substitution__Sec",
"SoccerDocument__MatchData__TeamData__Substitution__SubOff","SoccerDocument__MatchData__TeamData__Substitution__SubOn",
"SoccerDocument__MatchData__TeamData__Substitution__SubstitutePosition","SoccerDocument__MatchData__TeamData__Substitution__Time",
"SoccerDocument__MatchData__TeamData__Substitution__TimeStamp","SoccerDocument__MatchData__TeamData__Substitution__uID",
"SoccerDocument__MatchData__TeamData__TeamRef"]]
MatchData_Team.columns = ["Formation","Goal_Assist","Goal_Assist_PlayerRef","Goal_EventID","Goal_EventNumber","Goal_Min","Goal_Period","Goal_PlayerRef",
"Goal_Sec","Goal_SecondAssist","Goal_SecondAssist_PlayerRef","Goal_SoloRun","Goal_Time","Goal_TimeStamp","Goal_Type","Goal_uID",
"MissedPenalty_EventID","MissedPenalty_EventNumber","MissedPenalty_Min","MissedPenalty_Period","MissedPenalty_PlayerRef","MissedPenalty_Time",
"MissedPenalty_TimeStamp","MissedPenalty_Type","MissedPenalty_uID","MatchPlayer_Captain","MatchPlayer_Formation","MatchPlayer_PlayerRef",
"MatchPlayer_Position","MatchPlayer_ShirtNumber","MatchPlayer_Status","MatchPlayer_SubPosition","Score","Side","Substitution_EventID",
"Substitution_EventNumber","Substitution_Min","Substitution_Period","Substitution_Reason","Substitution_Retired","Substitution_Sec","Substitution_SubOff",
"Substitution_SubOn","Substitution_SubstitutePosition","Substitution_Time","Substitution_TimeStamp","Substitution_uID","TeamRef"]

#Stripping 'p' from player ids
MatchData_Team["Player_ID"]=MatchData_Team["MatchPlayer_PlayerRef"].str[1:]


#Creating Team_Player_Info Dataset
Team_Player_Info=Matchresults_final[["SoccerDocument__Team__Country","SoccerDocument__Team__Kit__colour1","SoccerDocument__Team__Kit__colour2","SoccerDocument__Team__Kit__id",
"SoccerDocument__Team__Kit__type","SoccerDocument__Team__Name","SoccerDocument__Team__Player__PersonName__First","SoccerDocument__Team__Player__PersonName__Known",
"SoccerDocument__Team__Player__PersonName__Last","SoccerDocument__Team__Player__Position","SoccerDocument__Team__Player__uID","SoccerDocument__Team__TeamOfficial__PersonName__First",
"SoccerDocument__Team__TeamOfficial__PersonName__Last","SoccerDocument__Team__TeamOfficial__Type","SoccerDocument__Team__TeamOfficial__uID","SoccerDocument__Team__uID"]]
Team_Player_Info.columns = ["Country","Kitcolour1","Kitcolour2","Kitid","Kittype","Team_Name","PersonName_First",
"PersonName_Known","PersonName_Last","Position","uID","PersonName_First","PersonName_Last",
"Type","Team_official_uID","uID"]

#Creating Venue Dataset
Venue=Matchresults_final[["SoccerDocument__Type","SoccerDocument__Venue__Country","SoccerDocument__Venue__Name","SoccerDocument__Venue__uID"]]
Venue.columns = ["Type","Country","Name","uID"]


