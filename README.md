# Recommendation System

## Описание
Проект представляет из себя рекомендательную систему для музыкальных произведений.
В качестве данных используется [dataset](https://disk.yandex.ru/d/zJETeW9wldI8eg) с информацией о треках и прослушиваниях пользователей сервиса Spotify.

## Описание табличных данных

### Music Info
* *track_id* (string) - Идентификатор трека; 50683 уникальных значения; 0% пропущенных значений
* *name* (string) - Название трека; 50683 уникальных значения; 0% пропущенных значений
* *artist* (string) - Название исполнителя; 8317 уникальных значения; 0% пропущенных значений
* *spotify_preview_url* (string/url) - Ссылка на композицию; 50620 уникальных значений; 0% пропущенных значений
* *spotify_id* (string) - Идентификатор трека в Spotify; 50674 уникальных значения; 0% пропущенных значений
* *tags* (string) - Списко тегов композиции; 49050 значений; 2% пропущенных значений
* *genre* (string) - Жанр композиции; 12383 значения; 56% пропущенных значений
* *year* (int) -  Год выхода композиции; 50683 значения; 0% пропущенных значений
* *duration_ms* (int) - Длительность композиции в миллисекундах; 50683 значения; 0% пропущенных значений
* *danceability* (decimal) - Коэффициент "танцевальности" композиции; 50683 значения; 0% пропущенных значений
* *energy* (decimal) - Энергичность композиции; 50683 значения; 0% пропущенных значений
* *key* (int) - Ключ; 50683 значения; 0% пропущенных значений
* *loudness* (decimal) - Громкость композиции; 50683 значения; 0% пропущенных значений
* *mode* (int) - Режим; 50683 значения; 0% пропущенных значений
* *speechiness* (decimal) - Коэффициент наполнености композиции текстом; 50683 значения; 0% пропущенных значений
* *acousticness* (decimal) - Коэффициент акустичности композиции; 50683 значения; 0% пропущенных значений
* *instrumentalness* (decimal) - Коэффициент инструментальности композиции; 50683 значения; 0% пропущенных значений
* *liveness* (decimal) -  Коэффициент вероятности исполнения композиции вживую; 122 значения; 0% пропущенных значений
* *valence* (decimal) - Коэффициент "позитивности" композиции; 50683 значений; 0% пропущенных значений
* *tempo* (decimal) - Коэффициент темпа композиции; 50683 значений; 0% пропущенных значений
* *time_signature* (int) - Временная сигнатура композиции; 50683 значения; 0% пропущенных значений



### User Listening History
* *track_id* (string) - Идентификатор трека; 30459 уникальных значения; 0% пропущенных значений
* *user_id* (string) - Идентификатор пользователя; 962037 уникальных значения; 0% пропущенных значений
* *playcount* (int) - Количество прослушиваний; 9100000 значения; 0% пропущенных значений



