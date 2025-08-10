from utils import read_video,save_video
from tracker import PlayerTracker
from ball_track import BallTracker
from court_line_detector import CourtLineDetector
from copy import deepcopy
import pandas as pd
from bbox_utils import measure_distance,convert_pixel_distance_to_meters
from player_stats_drawer_utils import draw_player_stats

DOUBLE_LINE_WIDTH = 10.97

input_video_path="input_video.mp4"
video_frames=read_video(input_video_path)
court_line_detector=CourtLineDetector("keypoints_model.pth")
court_key_points=court_line_detector.predict(video_frames[0])
from mini_court import MiniCourt
mini_court=MiniCourt(video_frames[0])
player_tracker=PlayerTracker(model_path='yolov8s')
ball_detector=BallTracker(model_path='best.pt')

player_detection=player_tracker.detect_frames(video_frames,read_from_stub=True,stub_path=r"stubs/player_detection.pkl")
ball_detection=ball_detector.detect_frames(video_frames,read_from_stub=True,stub_path="stubs/ball_detector.pkl")
ball_detection=ball_detector.interpolate_ball_positions(ball_detection)

player_detection=player_tracker.choose_and_filter_players(court_key_points,player_detection)
ball_shot_frames=ball_detector.get_ball_shot_frames(ball_detection)


player_mini_court_detection,ball_mini_court_detections=mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detection,ball_detection,court_key_points)
player_mini_court_detection,ball_mini_court_detections=mini_court.convert_mini_court_output_to_lists(player_mini_court_detection,ball_mini_court_detections)
player_mini_court_detection,ball_mini_court_detections=mini_court.interpolate_mini_court_detections_pandas(player_mini_court_detection,ball_mini_court_detections)

player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_5_number_of_shots':0,
        'player_5_total_shot_speed':0,
        'player_5_last_shot_speed':0,
        'player_5_total_player_speed':0,
        'player_5_last_player_speed':0,
    } ]
    
for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame-start_frame)/24 # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                           DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        # player who the ball
        player_positions = player_mini_court_detection[start_frame]
        player_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                 ball_mini_court_detections[start_frame][1]))

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 5 else 5
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detection[start_frame][opponent_player_id],
                                                                player_mini_court_detection[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,
                                                                                DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

player_stats_data_df = pd.DataFrame(player_stats_data)
frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
player_stats_data_df = player_stats_data_df.ffill()

player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
player_stats_data_df['player_5_average_shot_speed'] = player_stats_data_df['player_5_total_shot_speed']/player_stats_data_df['player_5_number_of_shots']
player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_5_number_of_shots']
player_stats_data_df['player_5_average_player_speed'] = player_stats_data_df['player_5_total_player_speed']/player_stats_data_df['player_1_number_of_shots']









output_frames=ball_detector.draw_bboxes(video_frames,ball_detection)
output_frames=player_tracker.draw_bboxes(video_frames,player_detection)
output_frames=court_line_detector.draw_keypoints_on_video(output_frames,court_key_points)
output_frames=mini_court.draw_mini_court(output_frames)
output_frames=mini_court.draw_points_on_mini_court(output_frames,player_mini_court_detection)
output_frames=mini_court.draw_points_on_mini_court(output_frames,ball_mini_court_detections,color=(0,220,220))
output_frames=draw_player_stats(output_frames,player_stats_data_df)
save_video(output_frames,"output/video.avi")