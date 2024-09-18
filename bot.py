import cfg
import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.types import Message, InlineKeyboardMarkup, \
    InlineKeyboardButton, CallbackQuery
from aiogram.filters import Command
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# Команда старта
async def send_welcome(message: Message):
    await message.reply(
        "Привет! Напиши мне свои любимые треки, и я порекомендую тебе музыку.")


# Обработка сообщений пользователя
async def recommend_music(message: Message):
    user_preferences = message.text
    recommendations, next_batch_available = generate_recommendations(
        user_preferences)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Прислать еще",
                              callback_data="more_recommendations")]
    ]) if next_batch_available else None

    await message.reply(f"Вот что я нашел для тебя:\n{recommendations}",
                        parse_mode='HTML', reply_markup=keyboard)


# Функция для обработки нажатия кнопки "Прислать еще"
async def more_recommendations(callback_query: CallbackQuery):
    message = callback_query.message
    user_preferences = message.reply_to_message.text
    recommendations, next_batch_available = generate_recommendations(
        user_preferences, offset=10)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Прислать еще",
                              callback_data="more_recommendations")]
    ]) if next_batch_available else None

    await callback_query.message.answer(
        f"Вот еще что я нашел для тебя:\n{recommendations}",
        parse_mode='HTML',
        reply_markup=keyboard)


# Функция генерации рекомендаций на основе модели
def generate_recommendations(user_tracks, offset=0):
    ratings = pd.read_csv('ratings.csv')
    Music_Info = pd.read_csv('Music_Info.csv')
    model = load_model('recommendation_model.h5')  # Загружаем модель

    user_tracks_name = user_tracks.split(', ')
    user_track_ids = Music_Info[Music_Info['name'].isin(user_tracks_name)][
        'track_id'].tolist()

    user_preference_vector = create_preference_vector(user_track_ids,
                                                      ratings.columns)

    # Приведение размерности вектора предпочтений к размерности,
    # ожидаемой моделью
    if user_preference_vector.shape[0] != model.input_shape[1]:
        user_preference_vector = np.resize(user_preference_vector,
                                           (model.input_shape[1],))

    # Предсказание рейтингов для пользователя
    user_predicted_ratings = model.predict(
        user_preference_vector.reshape(1, -1)).flatten()

    recommendations = sort_and_filter_tracks(user_predicted_ratings,
                                             user_track_ids, ratings.columns)
    recommendations_info = Music_Info[
        Music_Info['track_id'].isin(recommendations[offset:offset + 10])][
        ['name', 'artist', 'spotify_preview_url']]

    recommendations_list = [
        f"<a href='{row['spotify_preview_url']}'>{row['name']} - {row['artist']}</a>"
        for _, row in recommendations_info.iterrows()
    ]

    next_batch_available = len(recommendations) > offset + 10
    return '\n'.join(recommendations_list), next_batch_available


# Функция создания вектора предпочтений пользователя
def create_preference_vector(user_track_ids, all_track_ids):
    preference_vector = np.zeros(len(all_track_ids))
    track_indices = [list(all_track_ids).index(track_id) for track_id in
                     user_track_ids if track_id in all_track_ids]
    preference_vector[track_indices] = 1
    return preference_vector


# Функция сортировки и фильтрации треков на основе предсказанных рейтингов
def sort_and_filter_tracks(user_predicted_ratings, user_track_ids,
                           all_track_ids):
    track_ratings = {track_id: rating for track_id, rating in
                     zip(all_track_ids, user_predicted_ratings) if
                     track_id not in user_track_ids}
    sorted_tracks = sorted(track_ratings.items(), key=lambda x: x[1],
                           reverse=True)
    return [track[0] for track in sorted_tracks]


# Основная функция для запуска бота
async def start():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - [%(levelname)s] - %(name)s-"
                               "(%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"
                        )

    bot = Bot(token=cfg.settings["TOKEN"])
    dp = Dispatcher()

    dp.message.register(send_welcome, Command(commands=['start']))
    dp.message.register(recommend_music)
    dp.callback_query.register(more_recommendations,
                               lambda c: c.data == "more_recommendations")

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == '__main__':
    asyncio.run(start())
