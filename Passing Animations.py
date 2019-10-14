# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:02:45 2019

@author: Home
"""
#importing libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

#read csv
df = pd.read_excel(r'C:\Users\Home\Documents\Football Python\CSV\Positional Data.xlsx', index_col=(0,1))
dfPlayers = pd.read_excel(r'C:\Users\Home\Documents\Football Python\CSV\Player Data.xlsx', index_col=0)

#defining colors
colors = {'attack': 'gray',
          'defense': '#00529F'}

fps = 20
length = 10

#Drawing the football field
X_SIZE = 105.0
Y_SIZE = 68.0

BOX_HEIGHT = (16.5*2 + 7.32)/Y_SIZE*100
BOX_WIDTH = 16.5/X_SIZE*100

GOAL = 7.32/Y_SIZE*100

GOAL_AREA_HEIGHT = 5.4864*2/Y_SIZE*100 + GOAL
GOAL_AREA_WIDTH = 5.4864/X_SIZE*100

def draw_pitch():
    """Sets up field
    Returns matplotlib fig and axes objects.
    """

    fig = plt.figure(figsize=(X_SIZE/15, Y_SIZE/15))
    fig.patch.set_facecolor('#a8bc95')

    axes = fig.add_subplot(1, 1, 1, facecolor='#a8bc95')

    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)

    axes.set_xlim(0,100)
    axes.set_ylim(0,100)

    axes = draw_patches(axes)

    return fig, axes

def draw_patches(axes):
    plt.xlim([-5,105])
    plt.ylim([-5,105])

    #pitch
    axes.add_patch(plt.Rectangle((0, 0), 100, 100,
                       edgecolor="white", facecolor="none", alpha=1))

    #half-way line
    axes.add_line(plt.Line2D([50, 50], [100, 0],
                    c='w'))

    #penalty areas
    axes.add_patch(plt.Rectangle((100-BOX_WIDTH, (100-BOX_HEIGHT)/2),  BOX_WIDTH, BOX_HEIGHT,
                       ec='w', fc='none'))
    axes.add_patch(plt.Rectangle((0, (100-BOX_HEIGHT)/2),  BOX_WIDTH, BOX_HEIGHT,
                               ec='w', fc='none'))

    #goal areas
    axes.add_patch(plt.Rectangle((100-GOAL_AREA_WIDTH, (100-GOAL_AREA_HEIGHT)/2),  GOAL_AREA_WIDTH, GOAL_AREA_HEIGHT,
                       ec='w', fc='none'))
    axes.add_patch(plt.Rectangle((0, (100-GOAL_AREA_HEIGHT)/2),  GOAL_AREA_WIDTH, GOAL_AREA_HEIGHT,
                               ec='w', fc='none'))

    #goals
    axes.add_patch(plt.Rectangle((100, (100-GOAL)/2),  1, GOAL,
                       ec='w', fc='none'))
    axes.add_patch(plt.Rectangle((0, (100-GOAL)/2),  -1, GOAL,
                               ec='w', fc='none'))


    #halfway circle
    axes.add_patch(Ellipse((50, 50), 2*9.15/X_SIZE*100, 2*9.15/Y_SIZE*100,
                                    ec='w', fc='none'))

    return axes

draw_pitch()

#Pitch-Animation
attackers = dfPlayers[dfPlayers.team=='attack'].index
defenders = dfPlayers[dfPlayers.team=='defense'].index

def draw_frame(t, display_num=True):
    f = int(t*fps)

    fig, ax = draw_pitch()

    dfFrame = df.loc[f]

    for pid in dfFrame.index:
        if pid==0:
            size = 0.6
            color='black'
            edge='black'
        else:
            size = 6
            color='white'
            if dfPlayers.loc[pid]['team'] == 'defense':
                edge=colors['defense']
            else:
                edge=colors['attack']

        ax.add_artist(Ellipse((dfFrame.loc[pid]['x'],
                               dfFrame.loc[pid]['y']),
                              size/X_SIZE*100, size/Y_SIZE*100,
                              edgecolor=edge,
                              linewidth=2,
                              facecolor=color,
                              alpha=1,
                              zorder=20))
        if display_num:
            plt.text(dfFrame.loc[pid]['x']-1,dfFrame.loc[pid]['y']-1.3,str(pid),fontsize=8, color='black', zorder=30)

    return fig, ax, dfFrame

anim = VideoClip(lambda x: mplfig_to_npimage(draw_frame(x)[0]), duration=length)

from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(anim)
import matplotlib.pyplot as plt
voronoi_plot_2d(anim)
plt.show()
#to save the animation to a file, uncomment the next line
anim.to_videofile(r'C:\Users\Home\Documents\working with positional data - version 1.mp4', fps=fps)

def count_players(dfFrame, pid):
    count = dfFrame.join(dfPlayers.team)[dfFrame['x']<=dfFrame.loc[pid]['x']].groupby('team').agg('count').max(axis=1)
    try:
        num_attack = count['attack']
    except KeyError:
        num_attack = 0
    try:
        num_defense = count['defense']
    except KeyError:
        num_defense = 0
    return (num_attack-num_defense)

def draw_area(t):
    fig, ax, dfFrame = draw_frame(t)


    maxX = dfFrame.loc[0]['x']
    superiority = count_players(dfFrame, 0)

    dfAttackers = dfFrame[(dfFrame.index.get_level_values(0).isin(attackers)) & (dfFrame['x']>maxX)]

    for pid, player in dfAttackers.iterrows():
        count = count_players(dfFrame, pid)
        if count>superiority:
            maxX = dfFrame.loc[pid]['x']
            superiority = count

    if superiority<0:
        color='red'
    else:
        color='black'

    plt.text(-5,110,str(superiority),fontsize=25, color=color)


    ax.add_patch(plt.Rectangle((0, 0), maxX, 100,
                       edgecolor="none", facecolor="yellow", alpha=0.1))

    return fig, ax


anim = VideoClip(lambda x: mplfig_to_npimage(draw_area(x)[0]), duration=length)

#to save the animation to a file, uncomment the next line
#anim.to_videofile('working with positional data - version 2.mp4', fps=fps)

#Marking

def closest_player(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist = np.einsum('ij,ij->i', deltas, deltas)
    return dist.argsort()[0], dist[dist.argsort()[0]]

def draw_marking(t):
    fig, ax, dfFrame = draw_frame(t)

    dfAttackers = dfFrame[dfFrame.index.get_level_values(0).isin(attackers)]

    for pid in defenders:
        circle = False
        dfMarking = dfAttackers[dfAttackers['x']>(dfFrame.loc[pid]['x'])]


        if dfMarking.shape[0]>0:
            closest, closest_dist = closest_player(dfFrame.loc[pid].values,
                                                   dfMarking.values)

            if closest_dist<75:
                ax.add_line(plt.Line2D([dfFrame.loc[pid]['x'], dfMarking.iloc[closest]['x']],
                                       [dfFrame.loc[pid]['y'], dfMarking.iloc[closest]['y']],
                                       c='red', zorder=30))
            else:
                circle = True


        else:
            circle = True

        if circle:
            ax.add_artist(Ellipse((dfFrame.loc[pid]['x'],
                                   dfFrame.loc[pid]['y']),
                                  10/X_SIZE*100, 10/Y_SIZE*100,
                                  edgecolor='gray',
                                  linewidth=0,
                                  facecolor='gray',
                                  alpha=0.2,
                                  zorder=20))

    return fig, ax

anim = VideoClip(lambda x: mplfig_to_npimage(draw_marking(x)[0]), duration=length)

#to save the animation to a file, uncomment the next line
#anim.to_videofile('working with positional data - version 3.mp4', fps=fps)

dfFuture = (df.unstack()+df.unstack().diff()*fps).stack()

def draw_passing(t):
    fig, ax, dfFrame = draw_frame(t)

    if ((dfFrame==dfFrame.loc[0]).sum(axis=1)>1).sum()>1:
        f = int(t*fps)
        dfFutureFrame = dfFuture.loc[f].join(dfPlayers.team) if len(dfFuture.loc[f])>0 else dfFrame.join(dfPlayers.team)

        marked_players = []

        for pid in defenders:
            dists = dfFutureFrame[(dfFutureFrame.team=='attack') & (dfFutureFrame.x>=dfFrame.loc[pid].x)
                                 ].apply(lambda x: np.linalg.norm(x[['x', 'y']]-dfFutureFrame.loc[pid][['x', 'y']]), axis=1)

            if dists.min()<12:
                marked_players.append(dists.idxmin())

        for pid in attackers:
            if pid not in marked_players:
                ax.add_line(plt.Line2D([dfFrame.loc[0]['x'], dfFutureFrame.loc[pid]['x']],
                                       [dfFrame.loc[0]['y'], dfFutureFrame.loc[pid]['y']],
                                       c='black', zorder=30))


    return fig, ax

anim = VideoClip(lambda x: mplfig_to_npimage(draw_passing(x)[0]), duration=length)

#to save the animation to a file, uncomment the next line
#anim.to_videofile('working with positional data - version 4.mp4', fps=fps)

#Passing Quality

