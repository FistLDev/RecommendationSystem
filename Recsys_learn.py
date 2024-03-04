%pylab inline

import numpy as np
import pandas as pd
import math
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.linalg import svd
from tqdm import tqdm

#%pylab is deprecated, use %matplotlib inline and import the required libraries.
#Populating the interactive namespace from numpy and matplotlib

df_info = pd.read_csv('Music Info.csv')
df_users = pd.read_csv('User Listening History.csv')  # считываем наш набор данных из файлов Music Info.csv и User Listening History.csv

df_info = df_info.dropna(subset=['tags', 'genre'], how='all') #удалаем  строки, содержащие пропущенные значения в столбцах ('tags', 'genre')

# Создаем набор сортированных уникальных жанров и тегов для них
unique_genres = df_info['genre'].dropna().unique()
unique_genres = sorted(unique_genres)

tags_to_genre = {tag.lower(): tag for tag in unique_genres}

# Функция для назначения жанра на основе тегов
def assign_genre_based_on_tags_case_sensitive(tags_str, existing_genres):
    if pd.notna(tags_str) and tags_str != "":
        first_tag = tags_str.split(',')[0].strip().lower()

        if first_tag in existing_genres:
            return existing_genres[first_tag]
    return "Other"
#  Назначаем жанр на основе тегов в пустых ячейках в столбце 'genre'
df_info['genre'] = df_info.apply(lambda row: row['genre'] if pd.notna(row['genre']) and row['genre']
                    in unique_genres else assign_genre_based_on_tags_case_sensitive(row['tags'], tags_to_genre),axis=1)

grouped_users = df_users.groupby(['user_id', 'track_id']).sum().reset_index() # группируем данные по 'user_id' и 'track_id'

df_merged = pd.merge(grouped_users, df_info, on='track_id', how='left') # Объединение датасетов

# Создаем выборку подмножества пользователей
sampled_users = grouped_users['user_id'].drop_duplicates().sample(frac=0.04)
df_sampled = grouped_users[grouped_users['user_id'].isin(sampled_users)]

# Определением числа тестовых образцов
num_test_samples = 10

# Использование groupby с tail и head для получения тестовых и тренировочных наборов данных
test = df_sampled.groupby('user_id').tail(num_test_samples)
train = df_sampled.drop(test.index)

# Вывод размеров тренировочного и тестового наборов
print(train.shape, test.shape)

# Убираем строки с пустыми значениями  в столбце 'name' от тестового и тренировочного набор данных
train = train.dropna(subset=['name'], how='all')
test = test.dropna(subset=['name'], how='all')

# Агрегирует идентификаторы треков для каждого пользователя в обучающего набора данных и сохранение их в interactions
interactions = (
    train
    .groupby('user_id')['track_id'].agg(lambda x: list(x))
    .reset_index()
    .rename(columns={'track_id': 'true_train'})
    .set_index('user_id')
)
# Агрегация идентификаторов треков для каждого пользователя в тестовом набора данных и сохранение их в interactions в  новом столбце 'true_test'
interactions['true_test'] = (
    test
    .groupby('user_id')['track_id'].agg(lambda x: list(x))
)

# заполнение пропусков пустыми списками
interactions.loc[pd.isnull(interactions.true_test), 'true_test'] = [
    [''] for x in range(len(interactions.loc[pd.isnull(interactions.true_test), 'true_test']))]

# Функция расчета точности
def calc_precision(column):
    return (
        interactions
        .apply(
            lambda row:
            len(set(row['true_test']).intersection(
                set(row[column]))) /
            min(len(row['true_test']) + 0.001, 10.0),
            axis=1)).mean()

#Создание таблицы рейтингов на основе обучающего набора данных
ratings = pd.pivot_table(
    train,
    values='playcount',
    index='user_id',
    columns='track_id').fillna(0)

#Сохранаем значение ratings в ratings_m
ratings_m = ratings.values

# Создание разреженной матрицы оценок
sparse_ratings = csr_matrix(ratings)

# Вычисление сходства
similarity_users = cosine_similarity(ratings)

# Вычисление суммарных рейтингов один раз
sum_ratings = np.argsort(ratings_m.sum(axis=0))[::-1]
sorted_columns = ratings.columns[sum_ratings]

# Предварительное вычисление рекомендаций для всех пользователей
all_recommendations = np.array([sorted_columns[~np.in1d(sorted_columns, interactions.iloc[i])][:10]
                                for i in range(len(similarity_users))])

# Создание маски для фильтрации пользователей с ненулевым сходством
mask = (similarity_users > 0).sum(axis=1) > 0

# Применение маски
prediction_user_based = [list(all_recommendations[i]) if mask[i] else [] for i in range(len(similarity_users))]

# Добавление предсказаний в DataFrame
interactions['prediction_user_based'] = prediction_user_based

# Вычислить среднюю точность рекомендаций на основе предсказаний
print(calc_precision('prediction_user_based'))

# Выполнение SVD
# U, sigma, V содержат результаты разложения
U, sigma, V = svd(ratings)

# Создаем матрицу Sigma с учетом сингулярных значений sigma
Sigma = np.zeros((10589, 16157))
Sigma[:10589, :10589] = np.diag(sigma)

# Вычисление новых рейтингов
new_ratings = U.dot(Sigma).dot(V)

# Вычисление суммы квадратов разностей между новыми рейтингами и исходными рейтингами
print(sum(sum((new_ratings - ratings.values) ** 2)))


K = 100
# Установка сингулярных значений после K-го в нули
sigma[K:] = 0

# Создаем матрицу Sigma с учетом сингулярных значений sigma
Sigma = np.zeros((10589, 16157))
Sigma[:10589, :10589] = np.diag(sigma)

# Вычисление новых рейтингов
new_ratings = U.dot(Sigma).dot(V)

# Вычисление суммы квадратов разностей между новыми рейтингами и исходными рейтингами
# Вычисление суммы квадратов разностей между средним значением новых рейтингов и исходными рейтингами
print(sum(sum((new_ratings - ratings.values) ** 2)))
print(sum(sum((ratings.values.mean() - ratings.values) ** 2)))

# Создаем матрицу новых рейтингов
new_ratings = pd.DataFrame(new_ratings, index=ratings.index, columns=ratings.columns)

#
predictions = []

# Проход по каждому пользователю из индекса interactions
for personId in tqdm(interactions.index):

# Извлечение новых рейтингов и сортировка по убиванию
    prediction = (
        new_ratings
        .loc[personId]
        .sort_values(ascending=False)
        .index.values
    )
# Добавление предсказаний в список predictions
    predictions.append(
        list(prediction[~np.in1d(
            prediction,
            interactions.loc[personId, 'true_train'])])[:10])

# Добавление predictions в новый столбец 'prediction_svd' в  interactions
interactions['prediction_svd'] = predictions

# Вычислить среднюю точность предсказаний
calc_precision('prediction_svd')

#Провераем работы модели на примере
#Функция генерации рекомендаций
def generate_recommendations(user_tracks, ratings, new_ratings, num_recommendations=10):
    user_tracks_name = list(user_tracks.split(', '))
    user_track_ids = Music_Info[(Music_Info['name'].isin(user_tracks_name))]['track_id'].tolist()

    user_preference_vector = create_preference_vector(user_track_ids, ratings.columns)

    if isinstance(new_ratings, pd.DataFrame):
        new_ratings = new_ratings.apply(pd.to_numeric, errors='coerce').fillna(0)

    if isinstance(new_ratings, pd.DataFrame):
        new_ratings = new_ratings.values

    try:
        user_predicted_ratings = np.dot(new_ratings, user_preference_vector).flatten()
        # Отфильтровываем уже выбранные треки и сортируем оставшиеся
        recommendations = sort_and_filter_tracks(user_predicted_ratings, user_track_ids, ratings.columns)

        return recommendations[:num_recommendations]
    except TypeError as e:
        print("Произошла ошибка:", e)

# Функция создания вектор предпочтений пользователя
def create_preference_vector(user_track_ids, all_track_ids):
    # Вектор предпочтений, где 1 означает, что пользователь выбрал этот трек
    preference_vector = np.zeros(len(all_track_ids))
    track_indices = [list(all_track_ids).index(track_id) for track_id in user_track_ids if track_id in all_track_ids]
    preference_vector[track_indices] = 1
    return preference_vector

# Функция сортировки и фильтрации треков на основе предсказанных рейтингов и треков
def sort_and_filter_tracks(user_predicted_ratings, user_track_ids, all_track_ids):
    track_ratings = {track_id: rating for track_id, rating in zip(all_track_ids, user_predicted_ratings) if track_id not in user_track_ids}
    sorted_tracks = sorted(track_ratings.items(), key=lambda x: x[1], reverse=True)
    return [track[0] for track in sorted_tracks]

# Пример использования:
user_tracks = ['TRIOREW128F424EAF0', 'TRLNZBD128F935E4D8']
recommendations = generate_recommendations(user_tracks, ratings, new_ratings, num_recommendations=10)
print(recommendations)


# Предположим, что U, sigma и Vt - это ваши матрицы, полученные из SVD
# Сохранение матриц U, Sigma и Vt
np.save('U_matrix.npy', U)
np.save('Sigma_values.npy', sigma)
np.save('Vt_matrix.npy', V)

# Сохранение матриц interactions, ratings и new_ratings
interactions.to_csv('interactions.csv')
ratings.to_csv('ratings.csv')
new_ratings.to_csv('new_ratings.csv')
