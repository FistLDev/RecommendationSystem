import numpy as np
import pandas as pd
from scipy.linalg import svd

def generate_recommendations(user_track_ids, ratings, new_ratings, num_recommendations=10):
    user_preference_vector = create_preference_vector(user_track_ids, ratings.columns)
    user_predicted_ratings = np.dot(new_ratings, user_preference_vector).flatten()

    # Отфильтровываем уже выбранные треки и сортируем оставшиеся
    recommendations = sort_and_filter_tracks(user_predicted_ratings, user_track_ids, ratings.columns)

    return recommendations[:num_recommendations]

def create_preference_vector(user_track_ids, all_track_ids):
    # Вектор предпочтений, где 1 означает, что пользователь выбрал этот трек
    preference_vector = np.zeros(len(all_track_ids))
    track_indices = [list(all_track_ids).index(track_id) for track_id in user_track_ids if track_id in all_track_ids]
    preference_vector[track_indices] = 1
    return preference_vector

def sort_and_filter_tracks(user_predicted_ratings, user_track_ids, all_track_ids):
    track_ratings = {track_id: rating for track_id, rating in zip(all_track_ids, user_predicted_ratings) if track_id not in user_track_ids}
    sorted_tracks = sorted(track_ratings.items(), key=lambda x: x[1], reverse=True)
    return [track[0] for track in sorted_tracks]

# Пример использования:
user_track_ids = ['TRIOREW128F424EAF0', 'TRLNZBD128F935E4D8']
recommendations = generate_recommendations(user_track_ids, ratings, new_ratings, num_recommendations=10)
print(recommendations)
