import numpy as np
from robot_mapping import HOME, RobotMapper


def test_neutral_maps_to_home():
    mapper = RobotMapper()
    human = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert mapper.calibrate(human, 0.5)
    np.testing.assert_allclose(mapper.map(human, 0.5), HOME)


def test_arm_mapping_uses_first_five_ranges():
    mapper = RobotMapper()
    human = np.zeros(5)
    mapper.calibrate(human, 0.5)
    moved = human + np.array([70.0, 60.0, 70.0, 70.0, 90.0])
    result = mapper.map(moved, 0.5)
    np.testing.assert_allclose(result[:5], np.array([180, 270, 270, 180, 180]).clip(0, 180))


def test_gripper_is_scalar_mapping():
    mapper = RobotMapper()
    human = np.zeros(5)
    mapper.calibrate(human, 1.0)
    result = mapper.map(human, 0.0)
    assert result.shape == (6,)
    assert result[5] == 180.0
