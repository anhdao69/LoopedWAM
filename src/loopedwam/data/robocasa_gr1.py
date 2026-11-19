"""The 24 RoboCasa-GR1 tabletop tasks (starVLA's `fourier_gr1_unified_1000` mixture) and their joint layout."""

DATASET_REPO = 'nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim'
DATASET_REVISION = 'ea7ac0b68f87da62f1e726771bba0fe74300802f'

GR1_TASKS = (
    'PnPBottleToCabinetClose', 'PnPCanToDrawerClose', 'PnPCupToDrawerClose', 'PnPMilkToMicrowaveClose',
    'PnPPotatoToMicrowaveClose', 'PnPWineToCabinetClose',
    'PosttrainPnPNovelFromCuttingboardToBasketSplitA', 'PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA',
    'PosttrainPnPNovelFromCuttingboardToPanSplitA', 'PosttrainPnPNovelFromCuttingboardToPotSplitA',
    'PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA', 'PosttrainPnPNovelFromPlacematToBasketSplitA',
    'PosttrainPnPNovelFromPlacematToBowlSplitA', 'PosttrainPnPNovelFromPlacematToPlateSplitA',
    'PosttrainPnPNovelFromPlacematToTieredshelfSplitA', 'PosttrainPnPNovelFromPlateToBowlSplitA',
    'PosttrainPnPNovelFromPlateToCardboardboxSplitA', 'PosttrainPnPNovelFromPlateToPanSplitA',
    'PosttrainPnPNovelFromPlateToPlateSplitA', 'PosttrainPnPNovelFromTrayToCardboardboxSplitA',
    'PosttrainPnPNovelFromTrayToPlateSplitA', 'PosttrainPnPNovelFromTrayToPotSplitA',
    'PosttrainPnPNovelFromTrayToTieredbasketSplitA', 'PosttrainPnPNovelFromTrayToTieredshelfSplitA')
GR1_FOLDERS = tuple(f'gr1_unified.{t}_GR1ArmsAndWaistFourierHands_1000' for t in GR1_TASKS)
GR1_ENV_NAMES = tuple(f'gr1_unified/{t}_GR1ArmsAndWaistFourierHands_Env' for t in GR1_TASKS)
# The 29 controlled joints in the dataset's 44-D vectors: left arm, right arm, left hand, right hand, waist
# (legs and neck are constant zero and dropped). The simulator reports and accepts the same five groups.
GR1_SELECT = (*range(0, 7), *range(22, 29), *range(7, 13), *range(29, 35), *range(41, 44))
GR1_GROUPS = (('left_arm', 7), ('right_arm', 7), ('left_hand', 6), ('right_hand', 6), ('waist', 3))
