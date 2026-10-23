from lerobot.robots.encos import EncosDualArm, EncosDualArmConfig
from lerobot.teleoperators.encos_feedback import EncosFeedback, EncosFeedbackConfig


def test_encos_features_are_bimanual_and_stable():
    robot = EncosDualArmConfig(command_mode="disabled")
    assert robot.type == "encos_dual_arm"
    assert robot.command_mode == "disabled"

    teleop = EncosFeedbackConfig()
    assert teleop.type == "encos_feedback"


def test_encos_right_arm_mode_exposes_only_seven_joint_actions(tmp_path):
    robot = EncosDualArmConfig(command_mode="disabled", arm_mode="right", calibration_dir=tmp_path)
    assert robot.arm_mode == "right"

    # Seven joints expose position, velocity and current in observations,
    # while actions contain only the seven target positions.
    right_robot = EncosDualArm(robot)
    assert len(right_robot.observation_features) == 21
    assert len(right_robot.action_features) == 7
    assert all(name.startswith("right_joint_") for name in right_robot.observation_features)
    assert all(name.startswith("right_joint_") for name in right_robot.action_features)

    teleop = EncosFeedbackConfig(arm_mode="right")
    assert teleop.arm_mode == "right"


def test_encos_feedback_is_concrete(tmp_path):
    teleop = EncosFeedback(EncosFeedbackConfig(calibration_dir=tmp_path, arm_mode="right"))
    assert len(teleop.action_features) == 7
    assert teleop.feedback_features == {}


def test_encos_config_rejects_unknown_command_mode():
    try:
        EncosDualArmConfig(command_mode="vr")
    except ValueError as exc:
        assert "command_mode" in str(exc)
    else:
        raise AssertionError("invalid command mode was accepted")


def test_encos_config_rejects_unknown_arm_mode():
    try:
        EncosDualArmConfig(arm_mode="center")
    except ValueError as exc:
        assert "arm_mode" in str(exc)
    else:
        raise AssertionError("invalid arm mode was accepted")
