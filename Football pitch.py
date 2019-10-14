# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:26:23 2019

@author: Home
"""

import json
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class
from matplotlib.patches import Arc
import io
import matplotlib.image as image
from matplotlib import transforms

pd.set_option("display.max_columns", 110)

r=0 ##Turn this to 1 for the other team in any match
c = 0 if r==1 else 1

#########

with io.open(r"C:\Users\Home\Documents\StatsBomb Open Data\CSV Files\22921.json", 'r', encoding='utf-8-sig') as f:
    obj = json.load(f)

df = json_normalize(obj)

uteams = df["team.name"].unique()
team = df[(df["type.name"]=="Pass") & (df["team.name"]==uteams[r])]

############
fig, ax = plt.subplots()


#########


player_dict = {}
klist = []
xlist = []
ylist = []
playerposlist = []


for player in df.loc[r,"tactics.lineup"]:
	p = player["player"]
	l = player["position"]
	pos = l["name"]
	name = p["name"]
	playerposlist.append(pos)
	klist.append(name)

with open(r"C:\Users\Home\Documents\StatsBomb Open Data\CSV Files\PositionsStatsbomb.json", "r") as f:
	full_dict = json.load(f) ##Check my repo for this json file

for i in playerposlist:
        x,y = full_dict[i]
        xlist.append(x)
        ylist.append(y)

for x,y,z in zip(xlist, ylist, klist):
    entry = {z:[x,y]}
    player_dict.update(entry)

########

def Passer(player):
    """
    Function to take a player's name and then return a Pandas dataframe with the average angles, frequency, and
    average length of all passes attempted by the player
    """
    local_df = df.copy(deep=True)
    local_df = local_df[local_df["type.name"]=="Pass"]
    local_df = local_df[local_df["player.name"]==player]
    local_df = local_df.dropna(axis=1, how="all")

    df1 = local_df[['pass.angle','pass.length']].copy()


    bins = np.linspace(-np.pi,np.pi,24)

    df1['binned'] = pd.cut(local_df['pass.angle'], bins, include_lowest=True, right = True)
    df1["Bin_Mids"] = df1["binned"].apply(lambda x: x.mid)

    A= df1.groupby("Bin_Mids", as_index=False)["pass.length"].mean()
    B= df1.groupby("Bin_Mids", as_index=False)["pass.length"].count()
    A = A.dropna(0)
    B = B[B["pass.length"] != 0]
    A = pd.merge(A,B, on = "Bin_Mids")
    A.columns = ["Bin_Mids", "pass.length", "Frequency"]
    A['Bin_Mids'] = A['Bin_Mids'].astype(np.float64)
    A["Bin_Mids"] = A["Bin_Mids"] * -1

    return A

##########


norm = plt.Normalize(team["pass.length"].min(), 30) ##Change 30 to whatever you want the upper bound for the length of the pass to be in the colormap. Change to "team["pass.length"].max()" for the maximum
cmap = plt.cm.viridis
ar = np.array(team["pass.length"])
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, orientation="horizontal", fraction=0.046, pad=0.04)
cbar.ax.set_xlabel("Average length of passes in a direction", fontstyle = "italic", fontsize = 7)
cbar.ax.xaxis.set_tick_params(color = "xkcd:salmon")
plt.setp(plt.getp(cbar.ax.axes, "xticklabels"), color = "xkcd:salmon")

###########

def plot_inset(width, axis_main, data, x,y):
    """
    Creating axes instances for the sonars to enable plotting by sending x,y coordinates and then plotting the sonars.
    """
    ax_sub= inset_axes(axis_main, width=width, height=width, loc=10,
                       bbox_to_anchor=(x,y),
                       bbox_transform=axis_main.transData,
                       borderpad=0.0, axes_class=get_projection_class("polar"))

    theta = data["Bin_Mids"]
    radii = data["Frequency"]
    length = np.array(data["pass.length"])
    cm = cmap(norm(length))
    bars = ax_sub.bar(theta, radii, width=0.3, bottom=0.0)
    ax_sub.set_xticklabels([])
    ax_sub.set_yticks([])
    ax_sub.yaxis.grid(False)
    ax_sub.xaxis.grid(False)
    ax_sub.spines['polar'].set_visible(False)
    ax_sub.patch.set_facecolor("white")
    ax_sub.patch.set_alpha(0.1)
    for r,bar in zip(cm,bars):
            bar.set_facecolor(r)

########

for player, loc in player_dict.items():
    plot_inset(1.1,ax, data = Passer(player), x = loc[0], y = loc[1])
    ax.text(loc[0]+7, loc[1]-5, player, size = 6, rotation = -90, fontweight = "bold") ##Adding the player names for the sonars

#plot invisible scatter plot for the axes to autoscale
ax.scatter(xlist, ylist, s=1, alpha=0.0)

###############

#Pitch Outline
ax.plot([0,0],[0,80], color="black")
ax.plot([0,120],[80,80], color="black")
ax.plot([120,120],[80,0], color="black")
ax.plot([120,0],[0,0], color="black")

ax.plot([60,60],[0,80], color="black")

#Left Penalty Area
ax.plot([0,18],[18,18], color="black")
ax.plot([18,18],[18,62], color="black")
ax.plot([18,0],[62,62], color="black")

#Right Penalty Area
ax.plot([102,120],[18,18], color="black")
ax.plot([102,102],[18,62], color="black")
ax.plot([102,120],[62,62], color="black")

#6-yard box left
ax.plot([114,120],[30,30], color="black")
ax.plot([114,114],[30,50], color="black")
ax.plot([114,120],[50,50], color="black")

#6-yard box right
ax.plot([0,6],[30,30], color="black")
ax.plot([6,6],[30,50], color="black")
ax.plot([0,6],[50,50], color="black")


    #Prepare Circles
centreCircle = plt.Circle((60,40),9.15,color="black",fill=False)
centreSpot = plt.Circle((60,40),0.8,color="black")
leftPenSpot = plt.Circle((12,40),0.8,color="black")
rightPenSpot = plt.Circle((108,40),0.8,color="black")

    #Draw Circles
ax.add_patch(centreCircle)
ax.add_patch(centreSpot)
ax.add_patch(leftPenSpot)
ax.add_patch(rightPenSpot)

    #Prepare Arcs
leftArc = Arc((12,40),height=18.3,width=18.3,angle=0,theta1=310,theta2=50,color="black")
rightArc = Arc((108,40),height=18.3,width=18.3,angle=0,theta1=130,theta2=230,color="black")

#Goals

ax.plot([-3,0],[36,36],color="black", linewidth=2)
ax.plot([-3,-3],[36,44],color="black", linewidth=2)
ax.plot([-3,0],[44,44],color="black", linewidth=2)

ax.plot([120,123],[36,36],color="black", linewidth=2)
ax.plot([123,123],[36,44],color="black", linewidth=2)
ax.plot([120,123],[44,44],color="black", linewidth=2)

    #Draw Arcs
ax.add_patch(leftArc)
ax.add_patch(rightArc)



    #Tidy Axes and Extra information
ax.axis('off')
ax.text(125, 42, "PASS SONAR: {}".format(uteams[r]), rotation = -90, fontweight = "bold", fontsize = 12)
ax.text(122, 59, "vs {}".format(uteams[c]), rotation = -90, fontweight = "bold", fontsize = 7)
ax.text(1,1,"by Abhishek Sharma\nConcept: Eliot McKinley", rotation= -90, fontsize=4.5, color="k", fontweight="bold")
plt.show()