SCANCODES = {
    int('00', 16): 'unknown',
    int('01', 16): 'escape',
    int('02', 16): '1',
    int('03', 16): '2',
    int('04', 16): '3',
    int('05', 16): '4',
    int('06', 16): '5',
    int('07', 16): '6',
    int('08', 16): '7',
    int('09', 16): '8',
    int('0a', 16): '9',
    int('0b', 16): '0',
    int('0c', 16): 'dash',
    int('0d', 16): 'equals',
    int('0e', 16): 'backspace',
    int('0f', 16): 'tab',
    int('10', 16): 'q',
    int('11', 16): 'w',
    int('12', 16): 'e',
    int('13', 16): 'r',
    int('14', 16): 't',
    int('15', 16): 'y',
    int('16', 16): 'u',
    int('17', 16): 'i',
    int('18', 16): 'o',
    int('19', 16): 'p',
    int('1a', 16): 'left_bracket',
    int('1b', 16): 'right_bracket',
    int('1c', 16): 'enter',
    int('1d', 16): 'control',
    int('1e', 16): 'a',
    int('1f', 16): 's',
    int('20', 16): 'd',
    int('21', 16): 'f',
    int('22', 16): 'g',
    int('23', 16): 'h',
    int('24', 16): 'j',
    int('25', 16): 'k',
    int('26', 16): 'l',
    int('27', 16): 'semicolon',
    int('28', 16): 'quote',
    int('29', 16): 'back_quote',
    int('2a', 16): 'shift',
    int('2b', 16): 'back_slash',
    int('2c', 16): 'z',
    int('2d', 16): 'x',
    int('2e', 16): 'c',
    int('2f', 16): 'v',
    int('30', 16): 'b',
    int('31', 16): 'n',
    int('32', 16): 'm',
    int('33', 16): 'comma',
    int('34', 16): 'period',
    int('35', 16): 'slash',
    int('36', 16): 'shift',
    int('38', 16): 'alt',
    int('39', 16): 'space',
    int('3a', 16): 'caps_lock',
    int('45', 16): 'num_lock',
    int('46', 16): 'scroll_lock',
    int('4a', 16): 'keypad_minus'
}

MODIFIER_KEYS = {'shift','caps_lock'}

def lookup_key(code):
    if type(code) is str:
        code = int(code, 16)

    if code in SCANCODES:
        return SCANCODES[code]
    else:
        print('Warning: unknown scancode:', code)
        return None
