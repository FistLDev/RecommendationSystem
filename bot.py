import cfg
import asyncio
import logging 
from aiogram import Bot, Dispatcher
from aiogram.types import Message, ContentType
from aiogram.filters import Command
import numpy as np
import pandas as pd

Загружаем данные для работы модели
ratings = pd.read_csv('ratings.csv')
new_ratings = pd.read_csv('new_ratings.csv')
Music_Info = pd.read_csv('Music_Info.csv')

# Команда старта
# @dp.message_handler(commands=['start'])
async def send_welcome(message: Message, bot: Bot):
    await message.reply("Привет! Напиши мне свои любимые треки, и я порекомендую тебе музыку.")


# Обработка сообщений пользователя
# @dp.message_handler(lambda message: message.Text)
async def recommend_music(message: Message, bot: Bot):
    user_preferences = message.text
# Обработка предпочтений пользователя и получение рекомендаций
    recommendations = generate_recommendations(user_preferences)
    await message.reply(f"Вот что я нашел для тебя: {recommendations}")

 # Используйте модель для генерации рекомендаций на основе ввода пользователя
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

    def create_preference_vector(user_track_ids, all_track_ids):
        # Вектор предпочтений, где 1 означает, что пользователь выбрал этот трек
        preference_vector = np.zeros(len(all_track_ids))
        track_indices = []
        for track_id in user_track_ids:
            if track_id in all_track_ids:
                track_indices.append(list(all_track_ids).index(track_id))
        preference_vector[track_indices] = 1
        return preference_vector

    def sort_and_filter_tracks(user_predicted_ratings, user_track_ids, all_track_ids):
        track_ratings = {track_id: rating for track_id, rating in zip(all_track_ids, user_predicted_ratings) if
                         track_id not in user_track_ids}
        sorted_tracks = sorted(track_ratings.items(), key=lambda x: x[1], reverse=True)
        return [track[0] for track in sorted_tracks]

    def recommendations(user_tracks):
        recommendations_id = generate_recommendations(user_tracks, ratings, new_ratings, num_recommendations=10)
        recommendations_list = Music_Info[(Music_Info['track_id'].isin(recommendations_id))]['name'].tolist()
        return ', '.join(recommendations_list)

    return recommendations(user_tracks)

#  Объявляем асинхронную функцию start
async def start():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - [%(levelname)s] - %(name)s-"
                               "(%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"
                        )
# Создает экземпляр бота, используя токен из настроек cfg.settings
    bot = Bot(token=cfg.settings["TOKEN"])
# Создает диспетчер для обработки входящих сообщений
    dp = Dispatcher()
    dp.message.register(send_welcome, Command(commands=['start']))
    dp.message.register(recommend_music)

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

# Запускает асинхронную функцию start()
if __name__ == '__main__':
    asyncio.run(start())
