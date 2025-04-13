# mapping conditions to specific depths for ec measurements
DEPTH_MAP_EC = {
    'Geen gebrek': 0,
    '0,1 – 0,5 mm': 0.3,
    '0,6 – 1,0 mm': 0.8,
    '1,1 – 1,5 mm': 1.3,
    '1,6 – 2,0 mm': 1.8,
    'Diepte > 2,1 mm': 2.3
}


# all the cracks depths where transitions are created
CRACKS_DEPTHS_EC = [0, 0.3, 0.8, 1.3, 1.8, 2.3]
CRACKS_DEPTHS_US = [i for i in range(3, 55)]
CRACKS_DEPTHS = CRACKS_DEPTHS_EC + CRACKS_DEPTHS_US


