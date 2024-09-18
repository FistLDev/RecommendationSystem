import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import bot


class TestMusicBot(unittest.TestCase):

    def setUp(self):
        self.model = MagicMock()
        self.model.predict = MagicMock(return_value=np.array([[0.1, 0.4, 0.3, 0.2, 0.5]]))
        self.model.input_shape = (None, 5)

        patcher1 = patch('bot.load_model', return_value=self.model)
        patcher2 = patch('bot.pd.read_csv', side_effect=self.mock_read_csv)

        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)

        self.mock_load_model = patcher1.start()
        self.mock_read_csv = patcher2.start()

    def mock_read_csv(self, file_name):
        if (file_name == 'ratings.csv'):
            return pd.DataFrame(data={'track_id': [1, 2, 3, 4, 5], 'rating': [5, 4, 3, 2, 1]})
        elif (file_name == 'Music_Info.csv'):
            return pd.DataFrame(
                data={'track_id': [1, 2, 3, 4, 5], 'name': ['Track1', 'Track2', 'Track3', 'Track4', 'Track5'],
                      'artist': ['Artist1', 'Artist2', 'Artist3', 'Artist4', 'Artist5'],
                      'spotify_preview_url': ['url1', 'url2', 'url3', 'url4', 'url5']})

    def test_create_preference_vector(self):
        user_track_ids = [1, 3]
        all_track_ids = [1, 2, 3, 4, 5]
        preference_vector = bot.create_preference_vector(user_track_ids, all_track_ids)

        expected_vector = np.array([1, 0, 1, 0, 0])
        np.testing.assert_array_equal(preference_vector, expected_vector)

    @patch('bot.CallbackQuery')
    async def test_more_recommendations(self, mock_callback_query):
        callback_query = MagicMock()
        callback_query.message = MagicMock()
        callback_query.message.reply_to_message = MagicMock()
        callback_query.message.reply_to_message.text = "Track1, Track2"

        await bot.more_recommendations(callback_query)
        callback_query.message.answer.assert_called()

    @patch('bot.Message')
    async def test_recommend_music(self, mock_message):
        message = MagicMock()
        message.text = "Track1, Track2"

        await bot.recommend_music(message)
        message.reply.assert_called()


if __name__ == '__main__':
    unittest.main()
