# importing libraries
import logging

from aiogram import Bot, Dispatcher, executor, types
from aiogram.utils.markdown import text, bold, italic
from aiogram.types import ParseMode

from msg import *
from config import API_TOKEN

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

quality = 350  # standard quality
flags = {}  # shows whether the client uploaded a content photo. At the start - False

style_model = Net(ngf=128)  # creating MSG Net
style_model.load_state_dict(torch.load('styles.model'), False)  # loading the weights of the trained model


def run_msg(user_id):
    """
    The function gets the user id and sets the style as target
    Then it starts the model and gets the output
    At the end, the cache is deleted to optimize the server operation
    """
    global quality

    content_image = tensor_load_rgb_image('data/content_{}.jpg'.format(user_id),
                                          size=quality, keep_asp=True).unsqueeze(0)
    style = preprocess_batch(tensor_load_rgb_image('data/style_{}.jpg'.format(user_id),
                                                   size=quality).unsqueeze(0))

    style = Variable(style)
    content_image = Variable(preprocess_batch(content_image))
    style_model.set_target(style)

    output = style_model(content_image)
    tensor_save_bgr_image(output.data[0], 'data/output_{}.jpg'.format(user_id), False)

    del content_image
    del style
    del output
    torch.cuda.empty_cache()


@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    global flags
    user_id = message.from_user.id
    flags[user_id] = False
    
    await message.answer(text('Привет!\n', 'Это ', bold('KUDOFF Bot'), '\nС помощью этого бота Вы можете переносить '
                                                                       'стиль с одной фотографии на другую\n',
                              'Например, так:', sep=''),
                         parse_mode=ParseMode.MARKDOWN)

    with open('data/EXAMPLE.jpg', 'rb') as photo:
        await message.answer_photo(photo)

    await message.answer(text('Сначала отправьте фото,', italic('на которое перенесётся стиль')),
                         parse_mode=ParseMode.MARKDOWN)


@dp.message_handler(lambda message: message.text in ['Изменить content-фото', 'Изменить style-фото'])
async def cmd_cancel(message: types.Message):
    global flags
    user_id = message.from_user.id

    await message.answer(text='Отправьте нужное фото')
    flags[user_id] = not flags[user_id]


@dp.message_handler(lambda message: message.text == 'Продолжить')
async def cmd_continue(message: types.Message):
    await message.answer(text('Теперь отправьте фото, ', italic('стиль с которого вы хотите перенести'), sep=''),
                         parse_mode=ParseMode.MARKDOWN)


@dp.message_handler(lambda message: message.text == 'Низкое (5-7 секунд)')
async def low_quality(message: types.Message):
    global quality
    quality = 250  # set low quality

    await message.answer(text='Процесс запущен! Ожидайте...')

    user_id = message.from_user.id
    run_msg(user_id)

    with open('data/output_{}.jpg'.format(user_id), 'rb') as photo:
        await message.answer_photo(photo)


@dp.message_handler(lambda message: message.text == 'Высокое (15-30 секунд)')
async def high_quality(message: types.Message):
    global quality
    quality = 350  # set high quality

    await message.answer(text='Процесс запущен! Ожидайте...')

    user_id = message.from_user.id
    run_msg(user_id)

    with open('data/output_{}.jpg'.format(user_id), 'rb') as photo:
        await message.answer_photo(photo)


@dp.message_handler(lambda message: message.text == 'Запустить алгоритм')
async def choose_quality(message: types.Message):
    poll_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    poll_keyboard.add(types.KeyboardButton(text='Низкое (5-7 секунд)'))
    poll_keyboard.add(types.KeyboardButton(text='Высокое (15-30 секунд)'))

    await message.answer(text='Выберите качество будущего фото:', reply_markup=poll_keyboard)


@dp.message_handler(content_types=types.ContentTypes.PHOTO)
async def get_photos(message):
    global flagы
    poll_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    user_id = message.from_user.id

    if not flags[user_id]:
        await message.photo[-1].download('data/content_{}.jpg'.format(user_id))
        poll_keyboard.add(types.KeyboardButton(text='Продолжить'))
        poll_keyboard.add(types.KeyboardButton(text='Изменить content-фото'))
        await message.answer(text('Если вы выбрали правильное изображение - ', italic('продолжаем'),
                                  '\nИначе измените content-фото', sep=''),
                             parse_mode=ParseMode.MARKDOWN, reply_markup=poll_keyboard)
        flags[user_id] = True
    else:
        await message.photo[-1].download('data/style_{}.jpg'.format(user_id))
        poll_keyboard.add(types.KeyboardButton(text='Запустить алгоритм'))
        poll_keyboard.add(types.KeyboardButton(text='Изменить style-фото'))
        await message.answer(text('Если вы всё выбрали верно - ', italic('запускаем алгоритм'),
                                  '\nИначе измените style-фото', sep=''),
                             parse_mode=ParseMode.MARKDOWN, reply_markup=poll_keyboard)
        flags[user_id] = False


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
