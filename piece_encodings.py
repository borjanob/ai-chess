piece_encodings_by_name = {
    'pawn' : (7,13),
    'bishop' : (9,15),
    'knight' : (8,14),
    'rook' : (10,16),
    'queen' : (11,17),
    'king' : (12,18)
}

piece_encodings_by_number = {
    7 : 'pawn',
    13 : 'pawn',
    9 : 'bishop',
    15 : 'bishop',
    8 : 'knight',
    14 : 'knight',
    10 : 'rook',
    16 : 'rook',
    11 : 'queen',
    17 : 'queen',
    12 : 'king',
    18: 'king'
}
piece_encodings_by_number_white = {
    13 : 'pawn',
    15 : 'bishop',
    14 : 'knight',
    16 : 'rook',
    17 : 'queen',
    18: 'king'
}

piece_encodings_by_number_black = {
    7 : 'pawn',
    9 : 'bishop',
    8 : 'knight',
    10 : 'rook',
    11 : 'queen',
    12 : 'king',
}

piece_nums = {
    'pawn' : 16,
    'bishop' : 4,
    'knight' : 4,
    'rook' : 4,
    'queen' : 2,
    'king' : 2
}

rewards_by_piece = {
    'pawn' : 2,
    'bishop' : 4,
    'knight' : 5,
    'rook' : 7,
    'queen' : 10,
}