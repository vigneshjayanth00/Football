# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:56:50 2020

@author: Home
"""

select_your_match = "984543"

# || 2. Declare test data locations
f7_file = r"C:\Users\Home\Documents\OptaPro Forum20012020\Updated Data\Latest\1sthalf\f7/" + select_your_match + ".xml"
f24_file = r"C:\Users\Home\Documents\OptaPro Forum20012020\Updated Data\Latest\1sthalf\f73/" + select_your_match + ".xml"
metadata_file = r"C:\Users\Home\Documents\OptaPro Forum20012020\Updated Data\Latest\1sthalf\Metadata/" + select_your_match + ".xml"
tracab_file = r"C:\Users\Home\Documents\OptaPro Forum20012020\Updated Data\Latest\1sthalf\Tracking/" + select_your_match + ".dat"



# || 3. Summary Header Print Out
#print("-" * 80)
#string_print_to_edge("OPTA F24 and Tracab Syncing Algorithm", 80)

# || 4. Parse test data
#    convert to loaded files for production)
#blank_line_print()
#string_print_to_edge("Parsing data", 80)
events = parse_f24(f24_file)

# [print(f) for f in events.columns]
# exit()
tracking_meta = parse_tracking_metadata(metadata_file)
tracking = parse_tracab(tracab_file, metadata_file)

# || 5. Augment Tracab Data
#blank_line_print()
#string_print_to_edge("Augmenting data", 80)

## trim tracking data
during_game_time = [is_frame_during_game_time(f, tracking_meta) for f in tracking.frameID]
tracking = tracking[during_game_time]

# numerous augmentations to tracking
tracking['period_id'] = [period_id_calc(f, tracking_meta) for f in tracking.frameID]
tracking = add_player_id(f7_file, tracking)
tracking = add_attacking_direction(tracking, tracking_meta)
tracking = add_ball_xy(tracking)
tracking = add_distance_to_ball(tracking)
tracking = add_distance_to_goals(tracking)
tracking = create_opta_coords(tracking, tracking_meta)
tracking = add_team_in_possession(tracking)
tracking = add_speed_classification(tracking)

# augment event data
events = add_all_time_syncs(events, tracking_meta)
####################################################################################

## check first row in event data to determine which team FCN was, assign the result
## to the 'FCN_team' variable

if events.iloc[0]['home_team_id'] == "2592":
    FCN_team = 1
elif events.iloc[0]['away_team_id'] == "2592":
    FCN_team = 0
else:
    print("ERRRRRROOOOOOOR ************** NOT FCN")

oppo_tracking = tracking[tracking['team'] != FCN_team].reset_index(drop=True)
oppo_tracking = oppo_tracking[oppo_tracking['team'] != 10].reset_index(drop=True)

#############################################################################################
### Make a List of All Frames that 'in-play' and FCN in Possession

## there is a ball_status column within the tracking data with 
## two options ["Alive", "Dead"] - we keep the "Alive" frames only

alive_tracking = oppo_tracking[oppo_tracking['ball_status'] == "Alive"].reset_index(drop=True)

## create a list of the frameIDs that are in-play
all_alive_frames = list(set(alive_tracking['frameID']))

## we will measure every 1/5 second of tracking data to see if 
## there is a low-block during that frame. This reduces the computational
## load of the analysis. The tracking data has 25 frames of data per second
## Let's take every 5th frame of the all_alive_frames
all_alive_frames = all_alive_frames[0::12]

## we then remove all frames where opposition were in possession
frame_options = oppo_tracking[oppo_tracking['frameID'].isin(all_alive_frames)].reset_index(drop=True)[['frameID', 'team_in_possession']]
all_out_possession_frames = frame_options[frame_options['team_in_possession'] == False].reset_index(drop=True).drop_duplicates()['frameID'].reset_index(drop=True)

### Calculate Defensive Centroid
centroid_x = [] # a list to append all the results to
centroid_y = [] # a list to append all the results to


for F in all_out_possession_frames:

    try:

        def_seg = oppo_tracking[oppo_tracking['frameID'] == F].reset_index(drop=True)

        attacking_direction = def_seg.iloc[0]['attacking_direction']

        if attacking_direction == 1:
            centroid_x.append(def_seg.x.mean())
            centroid_y.append(def_seg.y.mean())

        else: 
            centroid_x.append(def_seg.x.mean()*-1)
            centroid_y.append(def_seg.y.mean()*-1)

    except:
        print("FAILEDDDDDDDDD /////////////////////////////////////")

############## New Lists of Centroids within the Zone

xmin = - (tracking_meta['pitch_x'] / 2) * 100
centroid_min = xmin + 1500
centroid_max = xmin + 3000

low_blocks_frameID_centroids = []

for i in range(len(centroid_x)):
    if centroid_x[i] >= centroid_min:
        if centroid_x[i] <= centroid_max:
            low_blocks_frameID_centroids.append(all_out_possession_frames[i])        

#We want to join qualifying low block frames into 'low-'block' sequences, 
#if the next qualifying frame is less than 2 seconds away it is added to the current 
#low-block sequence. A low-block sequence directory can be created with the start and 
#end frames of each distinct low-block sequence. 
gap_threshold = 100 # under 100 frames = 4 seconds 
duration_threshold = 1 # keep sequences with a length above 200 = (8 seconds)
lb_sequence_id = 0

lb_sequence_id_list = [0]

for lb_frame in range(1,len(low_blocks_frameID_centroids)):
        
        if (low_blocks_frameID_centroids[lb_frame] - low_blocks_frameID_centroids[lb_frame-1]) <= gap_threshold:
            
            lb_sequence_id_list.append(lb_sequence_id)
            
        else:
            lb_sequence_id = lb_sequence_id + 1
            lb_sequence_id_list.append(lb_sequence_id)


lb_sequence_info = pd.DataFrame(
    {'frameID': low_blocks_frameID_centroids,
     'lb_sequence_id': lb_sequence_id_list
    })


start_of_seq = []
end_of_seq = [] 

unique_sequenceids = list(set(lb_sequence_id_list))

for uID in unique_sequenceids:
    
    temp = lb_sequence_info[lb_sequence_info['lb_sequence_id'] == uID]
    
    start_of_seq.append(min(temp.frameID))
    end_of_seq.append(max(temp.frameID))

lb_sequence_summary = pd.DataFrame(
    {'frameID_start': start_of_seq,
     'frameID_end': end_of_seq,
     'lb_sequence_id': unique_sequenceids
    })

## add a duration to thee sequence 
lb_sequence_summary['duration'] = lb_sequence_summary['frameID_end'] - lb_sequence_summary['frameID_start']
lb_sequence_summary['match_id'] = select_your_match

lb_sequence_summary = lb_sequence_summary[lb_sequence_summary['duration'] >= duration_threshold].reset_index(drop=True)           
    
########################################################################################
## Low-Block Broken


lb_seq_list = []
playr_list = []
frame_of_break_list = []
breakID_list = []

for RR in list(range(len(lb_sequence_summary))):

#     RR = 19
    ## select the low block sequence 
    lg_seg_idx = RR
#     print(RR)
    lb_select = lb_sequence_summary.iloc[lg_seg_idx]
#     print("/")
    

    ## create a segment of tracking segment
    lb_seg = tracking[(tracking['frameID'].between(lb_select.frameID_start,lb_select.frameID_end + 75)) & (tracking['ball_status'] == "Alive")].reset_index(drop=True)

    ## find defensive line 
    oppo_seg = lb_seg[(lb_seg['team'] != 10) & (lb_seg['team'] != FCN_team)].reset_index(drop=True)

    ## get attacking direction of defensive team 
    att_dir = oppo_seg.iloc[0]['attacking_direction']
    
    ## calculate the defensive line
    frame_list = []
    def_line_list = []

    ## loop through and get the defensive line for each frame
    for fr in list(set(oppo_seg.frameID)):

        frame_list.append(fr) # append the frame 
        temp_seg = oppo_seg[oppo_seg['frameID'] == fr].reset_index(drop=True)

        if att_dir == -1:
            temp_seg = temp_seg.sort_values(by='x', ascending=False).reset_index(drop=True)
        else: 
            temp_seg = temp_seg.sort_values(by='x', ascending=True).reset_index(drop=True)

        def_line_list.append(temp_seg.iloc[1:3]['x'].mean()) # append the defensive last line
    
    ## create a summary dataframe 
    the_view = pd.DataFrame({'frameID': frame_list,'def_line': def_line_list})
    lb_seg['att_dir'] = att_dir # add attacking direction used 
    lb_seg = pd.merge(lb_seg, the_view, on="frameID") # merge together so the defensive line appears as a new column

    ## create the attacking team seg 
    lb_seg2 = lb_seg[(lb_seg['team'] == FCN_team) & (lb_seg['distance_to_ball'] <= 75)].reset_index(drop=True)

    ### work out the rows where the line is broken
    broken = []
    for i in range(len(lb_seg2)):
        row = lb_seg2.iloc[i]
        if row.att_dir == 1:
            if row.x < row.def_line:            
                broken.append(True)   
            else: 
                broken.append(False)            
        else:
            if row.x > row.def_line:
                broken.append(True)
            else: 
                broken.append(False)    
                
    lb_seg2['broken'] = broken ## attach the broken values as a new column to the tracking segment 
    
    broken_frames = list(lb_seg2[lb_seg2['broken'] == True].frameID)
    
    if len(broken_frames) > 0:
    
        frame_of_break_list.append(broken_frames[0])
        breakID_list.append(str(lb_select.lb_sequence_id) + "_" + "1")
        playr_list.append(lb_seg2[(lb_seg2['frameID'] == broken_frames[0]) & (lb_seg2['broken'] == True)].reset_index(drop=True).iloc[0]['player_id'])
        lb_seq_list.append(lb_select.lb_sequence_id)
        for j in range(1,len(broken_frames)):

            if (broken_frames[j] - broken_frames[j-1]) >= 50:
                #start_id = start_id + 1
                #breakID_list.append(str(lb_select.lb_sequence_id) + "_" + str(start_id))
                frame_of_break_list.append(broken_frames[j])
                playr_list.append(lb_seg2[(lb_seg2['frameID'] == broken_frames[j]) & (lb_seg2['broken'] == True)].reset_index(drop=True).iloc[0]['player_id'])
                lb_seq_list.append(lb_select.lb_sequence_id)


break_summary = pd.DataFrame({'frameID': frame_of_break_list, 'player_id':playr_list, 'lb_sequence_id':lb_seq_list})            
break_summary['match_id']=select_your_match



did_it_break = []

for m in list(range(len(lb_sequence_summary))):
    xx = lb_sequence_summary.iloc[m]['lb_sequence_id']

    if xx in list(set(break_summary['lb_sequence_id'])):
        did_it_break.append(True)
    else:
        did_it_break.append(False)
        
lb_sequence_summary['did_it_break'] = did_it_break


##################################################################################
## The Overload Framework
##We look to create overloads in wide areas, this is classified within .... 


ol_sum_start = []
ol_sum_end = []
ol_sum_type = []
ol_sum_side = []
ol_sum_lb_sequence_id = []
ol_sum_overload_id = []
ol_sum_duration = []
 

for iii in range(len(lb_sequence_summary)):
# grab a segment of tracking for the lb_segment
    lg_seg_idx = iii
    lb_select = lb_sequence_summary.iloc[lg_seg_idx]

    lb_seg = tracking[tracking['frameID'].between(lb_select.frameID_start,lb_select.frameID_end)].reset_index(drop=True)

    ball_seg = lb_seg[lb_seg['team'] == 10].reset_index()
    team0_seg = lb_seg[lb_seg['team'] != FCN_team].reset_index()
    team0_seg = team0_seg[team0_seg['team'] != 10].reset_index()
    team1_seg = lb_seg[lb_seg['team'] == FCN_team].reset_index()


    ## work out the wide zones 

    ''' 
    Attcking direction of 1 means the team is defending the goal -x and attacking the goal +x. 
    An attacking direction of -1 means the team is defending the goal +x and attacking the goal -x.
    '''

    lb_att_dir = team1_seg.iloc[0]['attacking_direction']

#     print("att_dir", lb_att_dir)

    centre_circle_diameter = 915
    wide_area_y_width = 2015

    if lb_att_dir == 1:

        wide_left_xmin = 0 + centre_circle_diameter
        wide_left_xmax = (tracking_meta['pitch_x'] * 100)/2
        wide_left_ymin = wide_area_y_width
        wide_left_ymax = (tracking_meta['pitch_y'] * 100)/2

        wide_right_xmin = 0 + centre_circle_diameter
        wide_right_xmax = (tracking_meta['pitch_x'] * 100)/2 
        wide_right_ymin = - (tracking_meta['pitch_y'] * 100)/2 
        wide_right_ymax = -wide_area_y_width


    elif lb_att_dir == -1:

        wide_left_xmin = -(tracking_meta['pitch_x'] * 100)/2
        wide_left_xmax = 0 - centre_circle_diameter
        wide_left_ymin = - (tracking_meta['pitch_y'] * 100)/2 
        wide_left_ymax = -wide_area_y_width

        wide_right_xmin = -(tracking_meta['pitch_x'] * 100)/2
        wide_right_xmax = 0 - centre_circle_diameter
        wide_right_ymin = wide_area_y_width
        wide_right_ymax = (tracking_meta['pitch_y'] * 100)/2

    else:
        print("ERROR WITH ATTACKING DIRECTION")


    frames_list = []
    wide_left_list = []
    wide_right_list = []

    for bf in list(set(ball_seg.frameID)):

        row_ = ball_seg[ball_seg['frameID'] == bf].reset_index(drop=True).iloc[0]

        if (row_['y'] >= wide_left_ymin) and (row_['y'] <= wide_left_ymax):
            if (row_['x'] >= wide_left_xmin) and (row_['x'] <= wide_left_xmax):
                wide_left_list.append(True)
            else: 
                wide_left_list.append(False)
        else: 
            wide_left_list.append(False)

        if (row_['y'] >= wide_right_ymin) and (row_['y'] <= wide_right_ymax):
            if (row_['x'] >= wide_right_xmin) and (row_['x'] <= wide_right_xmax):
                wide_right_list.append(True)
            else: 
                wide_right_list.append(False)
        else: 
            wide_right_list.append(False)

        frames_list.append(bf)

    overload_calcs = pd.DataFrame(
        {'frameID': frames_list,
         'wide_right': wide_right_list,
         'wide_left': wide_left_list
        })


    # check for overload in wide_left 

    overload_calcs_left = overload_calcs[overload_calcs['wide_left'] == True].frameID
    overload_left_list = []
    overload_left_type_list = []

    for f in overload_calcs.frameID:

        if overload_calcs[overload_calcs['frameID'] == f].reset_index(drop=True).iloc[0]['wide_left'] == True:

            fcn = team1_seg[team1_seg['frameID'] == f].reset_index(drop=True)
            oppo = team0_seg[team1_seg['frameID'] == f].reset_index(drop=True)

            ## count FCN players 
            fcn_ = fcn[(fcn['y']  >= wide_left_ymin) & (fcn['y'] <= wide_left_ymax)]
            fcn_ = fcn_[(fcn_['x'] >= wide_left_xmin) & (fcn_['x'] <= wide_left_xmax)]

            ## count FCN players 
            oppo_ = oppo[(oppo['y']  >= wide_left_ymin) & (oppo['y'] <= wide_left_ymax)]
            oppo_ = oppo_[(oppo_['x'] >= wide_left_xmin) & (oppo_['x'] <= wide_left_xmax)]


            if (len(fcn_) >= len(oppo_)):
                if len(fcn_) >= 3:
                    overload_left_list.append(True)
                    overload_left_type_list.append(str(len(fcn_)) + "v" + str(len(oppo_)))
                else: 
                    overload_left_list.append(False)
                    overload_left_type_list.append("")
            else:
                overload_left_list.append(False)
                overload_left_type_list.append("")
        else:
            overload_left_list.append(False)    
            overload_left_type_list.append("")


    overload_calcs['left_overload'] = overload_left_list
    overload_calcs['left_overload_type'] = overload_left_type_list

    ### right
    overload_calcs_right = overload_calcs[overload_calcs['wide_right'] == True].frameID
    overload_right_list = []
    overload_right_type_list = []

    for f in overload_calcs.frameID:

        if overload_calcs[overload_calcs['frameID'] == f].reset_index(drop=True).iloc[0]['wide_right'] == True:

            fcn = team1_seg[team1_seg['frameID'] == f].reset_index(drop=True)
            oppo = team0_seg[team1_seg['frameID'] == f].reset_index(drop=True)

            ## count FCN players 
            fcn_ = fcn[(fcn['y']  >= wide_right_ymin) & (fcn['y'] <= wide_right_ymax)]
            fcn_ = fcn_[(fcn_['x'] >= wide_right_xmin) & (fcn_['x'] <= wide_right_xmax)]

            ## count FCN players 
            oppo_ = oppo[(oppo['y']  >= wide_right_ymin) & (oppo['y'] <= wide_right_ymax)]
            oppo_ = oppo_[(oppo_['x'] >= wide_right_xmin) & (oppo_['x'] <= wide_right_xmax)]


            if (len(fcn_) >= len(oppo_)):
                if len(fcn_) >= 3:
                    overload_right_list.append(True)
                    overload_right_type_list.append(str(len(fcn_)) + "v" + str(len(oppo_)))
                else: 
                    overload_right_list.append(False)
                    overload_right_type_list.append("")
            else:
                overload_right_list.append(False)
                overload_right_type_list.append("")
        else:
            overload_right_list.append(False)    
            overload_right_type_list.append("")


    overload_calcs['right_overload'] = overload_right_list
    overload_calcs['right_overload_type'] = overload_right_type_list
    
    overload_calcs2 = overload_calcs[(overload_calcs['left_overload'] == True) | (overload_calcs['right_overload'] == True)].reset_index(drop=True)
    
    ## combine the left and right valued just to side and type
    side_list = []
    type_list = []
    
    for j in list(range(len(overload_calcs2))):
    
        if overload_calcs2.iloc[j].left_overload == True:
            side_list.append("left")
            type_list.append(overload_calcs2.iloc[j].left_overload_type)
        else:
            side_list.append("right")
            type_list.append(overload_calcs2.iloc[j].right_overload_type)
    
    overload_calcs2['type'] = type_list
    overload_calcs2['side'] =  side_list  
    
    ## reduce the calcs to just the essentials 
    overload_calcs2 = overload_calcs2[['frameID', 'type', 'side']]
    overload_calcs2['lb_sequence_id'] = lb_select.lb_sequence_id
    
    if len(overload_calcs2) > 0:
        ## add the id 
        overload_id = 1
        overload_id_list = [str(lb_select.lb_sequence_id) + "_" + str(1)]

        for j in list(range(1, len(overload_calcs2))):

            if (overload_calcs2.iloc[j].frameID - overload_calcs2.iloc[j-1].frameID) <= 50:
                overload_id_list.append(str(overload_calcs2.iloc[j].lb_sequence_id) + "_" + str(overload_id))
            else:
                overload_id += 1
                overload_id_list.append(str(overload_calcs2.iloc[j].lb_sequence_id) + "_" + str(overload_id))

        overload_calcs2['overload_id'] = overload_id_list
        
    
        for g in list(set(overload_calcs2.overload_id)):  
    
            ol_sum_start.append(overload_calcs2[overload_calcs2['overload_id'] == g]['frameID'].min())
            ol_sum_end.append(overload_calcs2[overload_calcs2['overload_id'] == g]['frameID'].max())
            ol_sum_type.append(overload_calcs2[overload_calcs2['overload_id'] == g].reset_index(drop=True).iloc[0]['type'])
            ol_sum_side.append(overload_calcs2[overload_calcs2['overload_id'] == g].reset_index(drop=True).iloc[0]['side'])
            ol_sum_lb_sequence_id.append(overload_calcs2.iloc[0]['lb_sequence_id'])
            ol_sum_overload_id.append(g)
            ol_sum_duration.append(overload_calcs2[overload_calcs2['overload_id'] == g]['frameID'].max() - overload_calcs2[overload_calcs2['overload_id'] == g]['frameID'].min())


overload_results = pd.DataFrame({'start': ol_sum_start, 
                                'end': ol_sum_end, 
                                'type': ol_sum_type, 
                                'side': ol_sum_side, 
                                'lb_sequence_id': ol_sum_lb_sequence_id, 
                                'strategy_id': ol_sum_overload_id, 
                                'duration': ol_sum_duration})

overload_results = overload_results[overload_results['duration'] > 50].reset_index(drop=True)


overload_results['overload_strategy_id'] = ["ol_" + f for f in overload_results['strategy_id']]
overload_results['strategy_type'] = "overload"

overload_strategy=overload_results.groupby(['lb_sequence_id'])['overload_strategy_id'].count().reset_index()
lb_sequence_summary = pd.merge(lb_sequence_summary, overload_strategy[['lb_sequence_id','overload_strategy_id']], on = 'lb_sequence_id', how = 'left')

##################################################################################
## Sideways Spread

spread_sum_start = []
spread_sum_end = []
spread_sum_lb_sequence_id = []

ball_speed_threshold = 1200
y_change_threshold = 30


for iii in list(range(len(lb_sequence_summary))):


    # for iii in range(len(lb_sequence_summary)):
    # grab a segment of tracking for the lb_segment
#     lg_seg_idx = 4
    lg_seg_idx = iii
    lb_select = lb_sequence_summary.iloc[lg_seg_idx]

    lb_seg = tracking[tracking['frameID'].between(lb_select.frameID_start,lb_select.frameID_end)].reset_index(drop=True)

    ball_seg = lb_seg[lb_seg['team'] == 10].reset_index()
    team0_seg = lb_seg[lb_seg['team'] != FCN_team].reset_index()
    team0_seg = team0_seg[team0_seg['team'] != 10].reset_index()
    team1_seg = lb_seg[lb_seg['team'] == FCN_team].reset_index()


    speeds = [float(f) for f in list(ball_seg.speed)]


    frames_ = []
    ys = []
    y_change = []
    ball_speed = []

    for g in list(range(1, len(ball_seg.y))):

        if float(ball_seg.iloc[g].speed) >= ball_speed_threshold:
            if abs(ball_seg.iloc[g].y - ball_seg.iloc[g-1].y) > y_change_threshold:
    #             print("frameID", ,"y", , "change", , "ball speed", )
                frames_.append(ball_seg.iloc[g].frameID) 
                ys.append(ball_seg.iloc[g].y) 
                y_change.append(abs(ball_seg.iloc[g].y - ball_seg.iloc[g-1].y)) 
                ball_speed.append(float(ball_seg.iloc[g].speed)) 

    switch_calcs = pd.DataFrame({'frameID': frames_, 
                                'y': ys, 
                                'y_change': y_change, 
                                'ball_speed': ball_speed})


    switch_id = 1
    switch_id_list = [1]

    for t in list(range(1, len(switch_calcs))):

        if (switch_calcs.iloc[t].frameID - switch_calcs.iloc[t-1].frameID) <=10:
            switch_id_list.append(switch_id)
        else: 
            switch_id = switch_id + 1
            switch_id_list.append(switch_id)

    switch_calcs['switch_id'] = switch_id_list

    start_list = []
    end_list = []
    speed_list = []
    ychange_list = []
    duration_list = []
    y_start_list = []
    x_start_list = []
    switch_id_list = []

    for idd in list(set(switch_calcs.switch_id)):

        temp = switch_calcs[switch_calcs['switch_id'] == idd].reset_index(drop=True)

        duration = (max(temp.frameID) - min(temp.frameID))
        if duration >= 12:

            y_change = abs(temp.iloc[0].y - temp.iloc[len(temp)-1].y)
            if y_change >= 1500:

                start_y = temp.iloc[0].y
                if -2015 <= start_y <= 2015: 

                    end_y = temp.iloc[len(temp)-1].y
                    if end_y > 2015:

                        spread_sum_start.append(min(temp.frameID))
                        spread_sum_end.append(max(temp.frameID))
                        spread_sum_lb_sequence_id.append(lb_select.lb_sequence_id)

                    elif end_y < -2015:
                        spread_sum_start.append(min(temp.frameID))
                        spread_sum_end.append(max(temp.frameID))
                        spread_sum_lb_sequence_id.append(lb_select.lb_sequence_id)
                        

spread_summary = pd.DataFrame({'start': spread_sum_start, 
                            'end': spread_sum_end, 
                            'lb_sequence_id': spread_sum_lb_sequence_id})   


# spread_summary['break_through'] = break_from_overload
# spread_summary['shot_occured'] = shot_occured
spread_summary['strategy_id'] = list(range(len(spread_summary)))
spread_summary['spread_strategy_id'] = ["sp_" + str(f) for f in spread_summary['strategy_id']]
spread_summary['match_id'] = select_your_match
spread_summary['strategy_type'] = "spread"

#summarizing the results and merging back to the low-block scenarios
spread_strategy=spread_summary.groupby(['lb_sequence_id'])['spread_strategy_id'].count().reset_index()
lb_sequence_summary = pd.merge(lb_sequence_summary, spread_strategy[['lb_sequence_id','spread_strategy_id']], on = 'lb_sequence_id', how = 'left')


#######################################################################################

## Runs


printer = True
all_run_sequence_summary = pd.DataFrame()

for j in list(range(len(lb_sequence_summary))):
           
              #     RR = 19
    ## select the low block sequence 
    lg_seg_idx = j
#     print(RR)
    lb_select = lb_sequence_summary.iloc[lg_seg_idx]
    
#     print(lb_select.lb_sequence_id)

#     print("/")
        
    ## create a segment of tracking segment
    lb_seg = tracking[(tracking['frameID'].between(lb_select.frameID_start,lb_select.frameID_end + 75)) & (tracking['ball_status'] == "Alive")].reset_index(drop=True)

    ## find defensive line 
    oppo_seg = lb_seg[(lb_seg['team'] != 10) & (lb_seg['team'] != FCN_team)].reset_index(drop=True)

    ## get attacking direction of defensive team 
    att_dir = oppo_seg.iloc[0]['attacking_direction']
    
    ## calculate the defensive line
    frame_list = []
    def_line_list = []

    ## loop through and get the defensive line for each frame
    for fr in list(set(oppo_seg.frameID)):

        frame_list.append(fr) # append the frame 
        temp_seg = oppo_seg[oppo_seg['frameID'] == fr].reset_index(drop=True)

        if att_dir == -1:
            temp_seg = temp_seg.sort_values(by='x', ascending=False).reset_index(drop=True)
        else: 
            temp_seg = temp_seg.sort_values(by='x', ascending=True).reset_index(drop=True)

        def_line_list.append(temp_seg.iloc[1:3]['x'].mean()) # append the defensive last line
    
    ## create a summary dataframe 
    the_view = pd.DataFrame({'frameID': frame_list,'def_line': def_line_list})
    lb_seg['att_dir'] = att_dir # add attacking direction used 
    lb_seg = pd.merge(lb_seg, the_view, on="frameID") # merge together so the defensive line appears as a new column

    ## find defensive line 
    fcn_seg = lb_seg[(lb_seg['team'] != 10) & (lb_seg['team'] == FCN_team)].reset_index(drop=True)

#     if printer:
#         print(fcn_seg.head())    
#         printer = False
#     print("list(set(fcn_seg['player_id']))", len(list(set(fcn_seg['player_id']))))
    
    for player in list(set(fcn_seg['player_id'])):
        
#         print(player, "in", lb_select.lb_sequence_id)
        
        player_seg = fcn_seg[fcn_seg['player_id'] == player].reset_index(drop=True)
        player_seg['next_x'] = player_seg['x'].shift(-1)
#         print("player_seg.iloc[0]['attacking_direction']", player_seg.iloc[0]['attacking_direction'])
        if player_seg.iloc[0]['attacking_direction'] == 1:
            player_seg['direction']=np.where((player_seg['x'] < player_seg['next_x']),1,0)
            player_seg['close_to_def_line']=np.where( ( abs(player_seg['def_line'] - player_seg['x']) <= 500),1,0)
            player_seg['in_front_of_line']=np.where( (player_seg['def_line'] > player_seg['x']),1,0)
            player_seg['run_broken_line']=np.where((player_seg['def_line']< player_seg['x']),1,0)
            player_seg['on_ball']=np.where((player_seg['distance_to_ball']<= 75),1,0)
#             player_seg['vertical_gain_5']=np.where( ((player_seg['next_x'] - player_seg['x']) > 5) ,1,0)
            player_seg['vertical_gain']=np.where( ((player_seg['next_x'] - player_seg['x']) > 10) ,1,0)


        elif player_seg.iloc[0]['attacking_direction'] == -1:
            player_seg['direction']=np.where((player_seg['x'] > player_seg['next_x']),1,0)
            player_seg['close_to_def_line']=np.where( (abs(player_seg['def_line'] - player_seg['x']) <= 500),1,0)
            player_seg['in_front_of_line']=np.where( (player_seg['def_line'] < player_seg['x']),1,0)
            player_seg['run_broken_line']=np.where((player_seg['def_line']>player_seg['x']),1,0)
            player_seg['on_ball']=np.where((player_seg['distance_to_ball']<= 75),1,0)
#             player_seg['vertical_gain_5']=np.where( ((player_seg['next_x'] - player_seg['x']) < -5) ,1,0)
            player_seg['vertical_gain']=np.where( ((player_seg['next_x'] - player_seg['x']) < -10) ,1,0)


        speeds = ['bolt_speed_sprinting' , 'high_speed_sprinting', 'low_speed_sprinting']    
        player_seg['run_speed'] =  np.where((player_seg['speed_class'].isin(speeds)),1,0)
#         player_seg['forward_run_5'] = np.where((  (player_seg['Direction'] == 1)  & (player_seg['vertical_gain_5'] == 1) & (player_seg['dist_to_def_line']  == 1) & (player_seg['run_speed'] == 1) ), 1, 0)
        player_seg['forward_run'] = np.where((  (player_seg['direction'] == 1)  & (player_seg['close_to_def_line'] == 1) & (player_seg['vertical_gain'] == 1) & (player_seg['in_front_of_line']  == 1) & (player_seg['run_speed'] == 1) ), 1, 0)
        
#         if printer:
#             print("'frameID', 'x', 'y', 'vertical_gain', 'on_ball', 'forward_run'")
#             print(player_seg[['frameID', 'x', 'vertical_gain',  'forward_run', 'in_front_of_line', 'close_to_def_line', 'def_line']])    
#             printer = False
#         if lb_select.lb_sequence_id == 17:
            
        print_boi = True
        
        if len(list(set(player_seg.forward_run))) > 1:
#             print("*"*80)
            
            ## bridge gaps in runs of 12 frames 
            run_happening = []
            for ii in list(range(0, len(player_seg) - 12)):
                run_happening.append( int( (1 in list(set(player_seg.iloc[ii:ii+12].forward_run))) == True) )
            player_seg['forward_run2'] = run_happening + list(player_seg.iloc[len(player_seg)-12:len(player_seg)].forward_run)


        
            runs_frameIDs = list(player_seg[player_seg['forward_run2'] == 1].frameID)
            
            gap_threshold = 12 # under 100 frames = 4 seconds 
            run_sequence_id = 1

            run_sequence_id_list = [1]

            for run_frame in range(1,len(runs_frameIDs)):

                    if (runs_frameIDs[run_frame] - runs_frameIDs[run_frame-1]) <= gap_threshold:

                        run_sequence_id_list.append(run_sequence_id)

                    else:
                        run_sequence_id = run_sequence_id + 1
                        run_sequence_id_list.append(run_sequence_id)


            run_sequence_info = pd.DataFrame(
                {'frameID': runs_frameIDs,
                 'run_sequence_id': run_sequence_id_list
                })

            
            start__ = []
            end__ = []
            past_line__ = []
            break_line__ = []
            player_id__ = []
            run_id__ = []
            median_y__ = []
            
            for s in list(set(run_sequence_info.run_sequence_id)):
                
                start_frame = run_sequence_info[run_sequence_info['run_sequence_id'] == s].frameID.min()
                end_frame = run_sequence_info[run_sequence_info['run_sequence_id'] == s].frameID.max()

                start__.append(start_frame)
                end__.append(end_frame)
                
                run_seg = player_seg[player_seg['frameID'].between(start_frame, end_frame)]
                median_y__.append(round(run_seg.y.median(),0))

                player_id__.append(player)
                run_id__.append(s)

            
            run_sequence_summary = pd.DataFrame(
                {'start': start__,
                 'end': end__,
                 'player_id' : player_id__,
                 'run_id' :run_id__,
                 'median_y':median_y__
                })
            
            
            in_behind_scope = 5
            on_ball_scope = 100 
            
            
            in_behind__ = []
            on_ball__ = [] 
            
            for r in list(range(len(run_sequence_summary))):
                
                run_info = run_sequence_summary.iloc[r]
                in_behind__.append(1 in set(player_seg[player_seg['frameID'].between(run_info.start, run_info.end + in_behind_scope)].run_broken_line))
                on_ball__.append(1 in set(player_seg[player_seg['frameID'].between(run_info.start, run_info.end + on_ball_scope)].on_ball))   

            run_sequence_summary['in_behind'] = in_behind__
            run_sequence_summary['on_ball'] = on_ball__            
            run_sequence_summary['lb_sequence_id'] = lb_select.lb_sequence_id

            all_run_sequence_summary = all_run_sequence_summary.append(run_sequence_summary)

            
all_run_sequence_summary['match_id'] = select_your_match
all_run_sequence_summary['run_strategy_id'] = ["r_" + str(a) + "_" + str(b) for a,b in zip(all_run_sequence_summary['lb_sequence_id'] , all_run_sequence_summary['run_id'])]
all_run_sequence_summary['strategy_type'] = "run"
all_run_sequence_summary = all_run_sequence_summary.reset_index(drop=True)

run_strategy=all_run_sequence_summary.groupby(['lb_sequence_id'])['run_strategy_id'].count().reset_index()
lb_sequence_summary = pd.merge(lb_sequence_summary, run_strategy[['lb_sequence_id','run_strategy_id']], on = 'lb_sequence_id', how = 'left')


#####################################################################################################

## Pockets

def dist_to_oppo(segment_, p_pos_):

#     segment_['distance_to_oppo'] = segment_[['x', 'y']].sub(np.array( p_pos_[0], p_pos_[1] )).pow(2).sum(1).pow(0.5)
    return(segment_[['x', 'y']].sub(np.array( p_pos_[0], p_pos_[1] )).pow(2).sum(1).pow(0.5))
#     segment_.distance_to_goal1 = trackingdata.distance_to_goal1.round(2)
#     segment_.distance_to_goal2 = trackingdata.distance_to_goal2.round(2)

#     return(segment_)  

xmin = - (tracking_meta['pitch_x'] / 2) * 100
xmax = (tracking_meta['pitch_x'] / 2) * 100


all_pocket_sequence_summary = pd.DataFrame()




for j in list(range(len(lb_sequence_summary))):
              #     RR = 19
    ## select the low block sequence 
    lg_seg_idx = j
#     print(RR)
    lb_select = lb_sequence_summary.iloc[lg_seg_idx]
#     print(j)
#         
    ## create a segment of tracking segment
    lb_seg = tracking[(tracking['frameID'].between(lb_select.frameID_start,lb_select.frameID_end)) & (tracking['ball_status'] == "Alive")].reset_index(drop=True)

    ## find defensive line 
    oppo_seg = lb_seg[(lb_seg['team'] != 10) & (lb_seg['team'] != FCN_team)].reset_index(drop=True)

    ## get attacking direction of defensive team 
    att_dir = oppo_seg.iloc[0]['attacking_direction']
    
    ## calculate the defensive line
    frame_list = []
    def_line_list = []

    ## loop through and get the defensive line for each frame
    for fr in list(set(oppo_seg.frameID)):

        frame_list.append(fr) # append the frame 
        temp_seg = oppo_seg[oppo_seg['frameID'] == fr].reset_index(drop=True)

        if att_dir == -1:
            temp_seg = temp_seg.sort_values(by='x', ascending=False).reset_index(drop=True)
        else: 
            temp_seg = temp_seg.sort_values(by='x', ascending=True).reset_index(drop=True)

        def_line_list.append(temp_seg.iloc[1:3]['x'].mean()) # append the defensive last line
    
    ## create a summary dataframe 
    the_view = pd.DataFrame({'frameID': frame_list,'def_line': def_line_list})
    lb_seg['att_dir'] = att_dir # add attacking direction used 
    lb_seg = pd.merge(lb_seg, the_view, on="frameID") # merge together so the defensive line appears as a new column

    ## find defensive line 
    fcn_seg = lb_seg[(lb_seg['team'] != 10) & (lb_seg['team'] == FCN_team)].reset_index(drop=True)

#     print(fcn_seg.head())    
    
#     print("list(set(fcn_seg['player_id']))", len(list(set(fcn_seg['player_id']))))
    
    fcn_seg['on_ball'] = np.where((fcn_seg['distance_to_ball'] < 75),1,0)
        
    if fcn_seg.attacking_direction[0] == 1:
        fcn_seg['in_pocket_zone'] = np.where((fcn_seg['x'] < fcn_seg['def_line']) &
#                                                 (fcn_seg['x'] > fcn_seg['centroid_x'] - (fcn_seg['def_line'] - fcn_seg['centroid_x'])) & 
                                                (fcn_seg['y'] < 2015) & 
                                                (fcn_seg['x'] > (xmax - 3000)) &
                                                (fcn_seg['x'] < (xmax - 1500)) &
                                                (fcn_seg['y'] > -2015), 1, 0)

    elif fcn_seg.attacking_direction[0] == -1:
        fcn_seg['in_pocket_zone'] = np.where((fcn_seg['x'] > fcn_seg['def_line']) &
#                                                 (fcn_seg['x'] < fcn_seg['centroid_x'] + (abs(fcn_seg['def_line'] - fcn_seg['centroid_x']))) & 
                                                (fcn_seg['y'] < 2015) & 
                                                (fcn_seg['x'] < (xmin + 3000)) &   
                                                (fcn_seg['x'] > (xmin + 1500)) &
                                                (fcn_seg['y'] > -2015), 1, 0)
        
#     xmin 
    
    for player in list(set(fcn_seg[(fcn_seg['on_ball'] == 1) & (fcn_seg['in_pocket_zone'] == 1)]['player_id'])):
        
#         print(player, "in", lb_select.lb_sequence_id)
        
        player_seg = fcn_seg[fcn_seg['player_id'] == player].reset_index(drop=True)
        

        
        frames_to_test = list(player_seg.frameID)
    
        under_pressure_list = []
        centroid__x_ = []
#         centroid__y_ = []
        
    
        for ff in frames_to_test:
            
            temp_player = player_seg[player_seg['frameID'] == ff].reset_index(drop=True)
            p_pos = np.array((temp_player.iloc[0]['x'],temp_player.iloc[0]['y'])) 
            opp_temp = oppo_seg[oppo_seg['frameID'] == ff].reset_index(drop=True)
#             print("-"*10)
#             op_dists = dist_to_oppo(opp_temp[['x','y']], p_pos)
#             print(op_dists)
#             print("-"*10)
            
            op_dists = dist_to_oppo(opp_temp[['x','y']], p_pos)
            
            under_pressure_list.append(sum( np.where((op_dists < 350),1,0)))
            
            centroid__x_.append(round(opp_temp.x.mean(),0))
            
#             print(sum( np.where((op_dists < 350),1,0)), round(oppo_seg[oppo_seg['frameID'] == ff].x.mean(),0), dist_to_oppo(oppo_seg[oppo_seg['frameID'] == ff].reset_index(drop=True)[['x','y']], p_pos).distance_to_oppo)
            
#         for aa in [np.array((a,b)) for a,b in zip(player_seg['x'], player_seg['y'])]:
        
#             op_dists = dist_to_oppo(oppo_seg[oppo_seg['frameID'] == ff].reset_index(drop=True)[['x','y']], aa).distance_to_oppo
#             under_pressure_list.append(sum( np.where((op_dists < 350),1,0)))
            
#             centroid__x_.append(round(oppo_seg[oppo_seg['frameID'] == ff].x.mean(),0))
            
    
    
        player_seg['on_ball'] = np.where((player_seg['distance_to_ball'] < 75),1,0)
        player_seg['under_pressure']= under_pressure_list
        player_seg['centroid_x'] = centroid__x_
#         player_seg['centroid_y'] = centroid__y_
        
        
        ## abs(player_seg['def_line'] - player_seg['centroid_x'])

        
        if player_seg.attacking_direction[0] == 1:
            player_seg['in_pocket_zone'] = np.where((player_seg['x'] < player_seg['def_line']) &
                                                    (player_seg['x'] > player_seg['centroid_x'] - (player_seg['def_line'] - player_seg['centroid_x'])) & 
                                                    (player_seg['x'] > (xmax - 3000)) &
                                                    (player_seg['x'] < (xmax - 1500)) &                                        
                                                    (player_seg['y'] > -2015), 1, 0)
            
        elif player_seg.attacking_direction[0] == -1:
            player_seg['in_pocket_zone'] = np.where((player_seg['x'] > player_seg['def_line']) &
                                                    (player_seg['x'] < player_seg['centroid_x'] + (abs(player_seg['def_line'] - player_seg['centroid_x']))) & 
                                                    (player_seg['x'] < (xmin + 3000)) &   
                                                    (player_seg['x'] > (xmin + 1500)) &                                          
                                                    (player_seg['y'] > -2015), 1, 0)
        
        else:
            print("ERROR")
            
            
        player_seg['in_pocket'] = np.where((player_seg['on_ball'] == 1) & 
                                           (player_seg['under_pressure'] == 0) & 
                                           (player_seg['in_pocket_zone'] == 1), 1,0)
        
        
        if len(set(player_seg['in_pocket'])) > 1:
            
            
            pocket_frameIDs = list(player_seg[player_seg['in_pocket'] == 1].frameID)
            
            gap_threshold = 12 # under 100 frames = 4 seconds 
            pocket_sequence_id = 1

            pocket_sequence_id_list = [1]

            for pocket_frame in range(1,len(pocket_frameIDs)):

                    if (pocket_frameIDs[pocket_frame] - pocket_frameIDs[pocket_frame-1]) <= gap_threshold:

                        pocket_sequence_id_list.append(pocket_sequence_id)

                    else:
                        pocket_sequence_id = pocket_sequence_id + 1
                        pocket_sequence_id_list.append(pocket_sequence_id)


            pocket_sequence_info = pd.DataFrame(
                {'frameID': pocket_frameIDs,
                 'pocket_sequence_id': pocket_sequence_id_list
                })

#             print(pocket_sequence_info)

            start__ = []
            end__ = []
            player_id__ = []
            pocket_id__ = []
            
            for s in list(set(pocket_sequence_info.pocket_sequence_id)):
                
                start_frame = pocket_sequence_info[pocket_sequence_info['pocket_sequence_id'] == s].frameID.min()
                end_frame = pocket_sequence_info[pocket_sequence_info['pocket_sequence_id'] == s].frameID.max()

                start__.append(start_frame)
                end__.append(end_frame)
                
#                 run_seg = player_seg[player_seg['frameID'].between(start_frame, end_frame)]
#                 median_y__.append(round(run_seg.y.median(),0))

                player_id__.append(player)
                pocket_id__.append(s)

            pocket_sequence_summary = pd.DataFrame(
                {'start': start__,
                 'end': end__,
                 'player_id' : player_id__,
                 'pocket_id' :pocket_id__
                })
            
            pocket_sequence_summary['lb_sequence_id'] = lb_select.lb_sequence_id

            
            all_pocket_sequence_summary = all_pocket_sequence_summary.append(pocket_sequence_summary)
#             print(run_sequence_summary)
            
#             for iii in list(range(len(player_seg))):
#                 if player_seg.iloc[iii]['in_pocket'] == 1:
# #                     pass
#                     print(player_seg.iloc[iii].player_id, player_seg.iloc[iii].frameID, player_seg.iloc[iii].in_pocket)


all_pocket_sequence_summary['match_id'] = select_your_match
all_pocket_sequence_summary['pocket_strategy_id'] = ["pok_" + str(a) + "_" + str(b) for a,b in zip(all_pocket_sequence_summary['lb_sequence_id'] , all_pocket_sequence_summary['pocket_id'])]
all_pocket_sequence_summary['strategy_type'] = "central_pocket"
all_pocket_sequence_summary = all_pocket_sequence_summary.reset_index(drop=True)

pocket_strategy=all_pocket_sequence_summary.groupby(['lb_sequence_id'])['pocket_strategy_id'].count().reset_index()
lb_sequence_summary = pd.merge(lb_sequence_summary, pocket_strategy[['lb_sequence_id','pocket_strategy_id']], on = 'lb_sequence_id', how = 'left')

#############################################################################################
## Half-Space Runs


hsr_start = []
hsr_end = []
hsr_player = []
hsr_lb_seq = []

# 1149555
half_space_inner = 916 
half_space_outer = 2016

# all_run_sequence_summary[(all_run_sequence_summary['median_y'].bewteen(916,2016)) | (all_run_sequence_summary['median_y'].bewteen(-916,-2016))]

half_space_runs = all_run_sequence_summary[(all_run_sequence_summary['median_y'].between(916,2016)) | (all_run_sequence_summary['median_y'].between(-2016,-916))].reset_index(drop=True)
# print("LENGTH", len(half_space_runs))
for hsr in list(range(len(half_space_runs))):
    
    hsr_seg = tracking[(tracking['frameID'].between(half_space_runs.iloc[hsr].start-12, half_space_runs.iloc[hsr].end +12)) & tracking['team'] == FCN_team].reset_index(drop=True)
    hsr_seg['on_ball'] = np.where((hsr_seg['distance_to_ball'] < 75), 1, 0)
    hsr_seg = hsr_seg[hsr_seg['on_ball'] == 1].reset_index(drop=True)
    
    if len(hsr_seg) > 0:
        
#         print(half_space_runs.iloc[hsr].median_y)
#         if hsr_seg.iloc[0]['attacking_direction'] == 1:

        if half_space_runs.iloc[hsr].median_y >= half_space_inner:
            if len(hsr_seg[hsr_seg['x']>half_space_outer]) > 0:
                hsr_start.append(half_space_runs.iloc[hsr].start)
                hsr_end.append(half_space_runs.iloc[hsr].end)
                hsr_player.append(half_space_runs.iloc[hsr].player_id)
                hsr_lb_seq.append(half_space_runs.iloc[hsr].lb_sequence_id)

        elif half_space_runs.iloc[hsr].median_y <= -half_space_inner:
            if len(hsr_seg[hsr_seg['x']<-half_space_outer]) > 0:
                hsr_start.append(half_space_runs.iloc[hsr].start)
                hsr_end.append(half_space_runs.iloc[hsr].end)
                hsr_player.append(half_space_runs.iloc[hsr].player_id)
                hsr_lb_seq.append(half_space_runs.iloc[hsr].lb_sequence_id)

hs_run_sequence_summary = pd.DataFrame(
    {'start': hsr_start,
     'end': hsr_end,
     'player_id' : hsr_player,
     'lb_sequence_id':hsr_lb_seq
    })


hs_run_sequence_summary['match_id'] = select_your_match
hs_run_sequence_summary['hs_run_strategy_id'] = ["r_" + str(a) + "_" + str(b) for a,b in zip(hs_run_sequence_summary['lb_sequence_id'] , list(hs_run_sequence_summary.index))]
hs_run_sequence_summary['strategy_type'] = "wide_pocket_half_space_run"
hs_run_sequence_summary = hs_run_sequence_summary.reset_index(drop=True)

hs_run_strategy=hs_run_sequence_summary.groupby(['lb_sequence_id'])['hs_run_strategy_id'].count().reset_index()
lb_sequence_summary = pd.merge(lb_sequence_summary, hs_run_strategy[['lb_sequence_id','hs_run_strategy_id']], on = 'lb_sequence_id', how = 'left')
lb_sequence_summary=lb_sequence_summary.fillna(0)
