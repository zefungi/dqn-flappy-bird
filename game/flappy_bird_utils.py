import pygame
import sys

"""設置遊戲資源（路徑）"""
def set_sprites_src_path():
    sprites_directory = 'assets/sprites/'
    BASE_PATH = sprites_directory + 'base.png'
    BACKGROUND_PATH = sprites_directory + 'background-black.png'
    PIPE_PATH = sprites_directory + 'pipe-green.png'
    PLAYER_PATH = (
        sprites_directory + 'redbird-upflap.png',
        sprites_directory + 'redbird-midflap.png',
        sprites_directory + 'redbird-downflap.png'
    )
    NUMBERS_PATH = (
        sprites_directory + '0.png',
        sprites_directory + '1.png',
        sprites_directory + '2.png',
        sprites_directory + '3.png',
        sprites_directory + '4.png',
        sprites_directory + '5.png',
        sprites_directory + '6.png',
        sprites_directory + '7.png',
        sprites_directory + '8.png',
        sprites_directory + '9.png'
        )
    return BASE_PATH, BACKGROUND_PATH, PIPE_PATH, PLAYER_PATH, NUMBERS_PATH

"""使用 pygame 載入所有遊戲資源"""
def load():
    BASE_PATH, BACKGROUND_PATH, PIPE_PATH, PLAYER_PATH, NUMBERS_PATH = set_sprites_src_path()

    """ 使用 pygame 載入所有遊戲資源"""
    IMAGES, SOUNDS, HITMASKS = {}, {}, {}
    # ---------- IMAGES ----------
    IMAGES['base'] = pygame.image.load(BASE_PATH).convert_alpha()
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()
    IMAGES['pipe'] = (
        pygame.transform.rotate(pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha()
    )
    IMAGES['player'] =  tuple(pygame.image.load(path).convert_alpha() for path in PLAYER_PATH)
    IMAGES['numbers'] = tuple(pygame.image.load(path).convert_alpha() for path in NUMBERS_PATH)

    # ---------- HITMASKS ----------
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    # ---------- SOUNDS ----------
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    return IMAGES, SOUNDS, HITMASKS

"""取得 flappy bird 的碰撞遮罩"""
def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask
