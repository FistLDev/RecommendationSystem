import cfg
import asyncio
import logging 
from aiogram import Bot, Dispatcher
from aiogram.types import Message, ContentType
from aiogram.filters import Command


# Команда старта
# @dp.message_handler(commands=['start'])
async def send_welcome(message: Message, bot: Bot):
    await message.reply("Привет! Напиши мне свои любимые жанры или исполнителей, и я порекомендую тебе музыку.")

# Обработка сообщений пользователя
# @dp.message_handler(lambda message: message.Text)
async def recommend_music(message: Message, bot: Bot):
    user_preferences = message.text
    # TODO: Обработка предпочтений пользователя и получение рекомендаций
    recommendations = generate_recommendations(user_preferences)
    await message.reply(f"Вот что я нашел для тебя: {recommendations}")

def generate_recommendations(user_input):
    # TODO: Используйте модель для генерации рекомендаций на основе ввода пользователя
    return "список рекомендованных треков"


async def start():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - [%(levelname)s] - %(name)s-"
                               "(%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"
                        )
    
    bot = Bot(token=cfg.settings["TOKEN"])
    dp = Dispatcher()
    dp.message.register(send_welcome, Command(commands=['start']))
    dp.message.register(recommend_music)

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == '__main__':
    asyncio.run(start())