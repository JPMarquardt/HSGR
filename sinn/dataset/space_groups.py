class spg_properties():
    def __init__(self):
        self.international_number2space_group = {
            '1': list(range(1, 2)),
            '-1' : list(range(2, 3)),
            '2': list(range(3, 6)),
            'm': list(range(6, 10)),
            '2/m': list(range(10, 16)),
            '222': list(range(16, 25)),
            'mm2': list(range(25, 47)),
            'mmm': list(range(47, 75)),
            '4': list(range(75, 81)),
            '-4': list(range(81, 83)),
            '4/m': list(range(83, 89)),
            '422': list(range(89, 99)),
            '4mm': list(range(99, 111)),
            '-42m': list(range(111, 123)),
            '4/mmm': list(range(123, 143)),
            '3': list(range(143, 147)),
            '-3': list(range(147, 149)),
            '32': list(range(149, 156)),
            '3m': list(range(156, 162)),
            '-3m': list(range(162, 168)),
            '6': list(range(168, 174)),
            '-6': list(range(174, 175)),
            '6/m': list(range(175, 177)),
            '622': list(range(177, 183)),
            '6mm': list(range(183, 187)),
            '-62m': list(range(187, 191)),
            '6/mmm': list(range(191, 195)),
            '23': list(range(195, 200)),
            'm-3': list(range(200, 207)),
            '432': list(range(207, 215)),
            '-43m': list(range(215, 221)),
            'm-3m': list(range(221, 231))
        }

        self.space_group2international_number = {}
        for key, value in self.international_number2space_group.items():
            for v in value:
                self.space_group2international_number[v] = key