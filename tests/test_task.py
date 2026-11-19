"""Fast correctness checks; these do not start Minecraft or train a model."""

from dataclasses import replace
import unittest

import numpy as np

from combat_rl.task import (
    Entity, State, FRAME_SIZE, NO_OP_ACTION, action_to_craftground,
    encode_frame, normalize_action, reward,
)


CONFIG = {
    "environment": {"arena_radius": 12, "fov_degrees": 100},
    "reward": {"kill": 1, "wave": 3, "death": -5, "time": -0.001,
               "damage_dealt": 1, "damage_taken": 1.5},
}


class TaskTests(unittest.TestCase):
    def test_restricted_view_has_no_hidden_enemy_features(self):
        front = Entity("front", 0, 0, 4, 20)
        hidden = Entity("hidden", 0, 0, -1, 20)
        visible_only = State(entities=[front])
        with_hidden = replace(visible_only, entities=[hidden, front])
        expected = encode_frame(visible_only, 1, CONFIG, restricted=True)
        np.testing.assert_array_equal(
            encode_frame(with_hidden, 1, CONFIG, restricted=True), expected
        )
        self.assertEqual(expected.shape, (FRAME_SIZE,))
        self.assertEqual(expected.dtype, np.float32)
        self.assertEqual(expected[11], 1)  # first enemy mask
        self.assertTrue(np.all(expected[12:] == 0))
        self.assertEqual(encode_frame(with_hidden, 1, CONFIG)[16], 1)

    def test_yaw_rotation_and_wrapping(self):
        # Looking west (-X), north (-Z) lies to the player's right.
        state = State(yaw=90, entities=[Entity("right", 0, 0, -6, 20)])
        frame = encode_frame(state, 1, CONFIG)
        self.assertAlmostEqual(frame[7], 0.25)
        self.assertAlmostEqual(frame[8], 0)
        self.assertTrue(np.all(encode_frame(state, 1, CONFIG, True)[7:] == 0))
        # -179 and +181 degrees describe the same camera direction.
        south = State(yaw=-179, entities=[Entity("front", 0, 0, -4, 20)])
        np.testing.assert_allclose(
            encode_frame(south, 1, CONFIG, True),
            encode_frame(replace(south, yaw=181), 1, CONFIG, True), atol=1e-7,
        )
        self.assertEqual(encode_frame(south, 1, CONFIG, True)[11], 1)

    def test_reward_uses_stat_deltas_and_survives_counter_reset(self):
        previous = State(kills=2, damage_dealt=40, damage_taken=6)
        current = State(kills=3, damage_dealt=60, damage_taken=10)
        outcome, deltas = reward(previous, current, True, False, CONFIG, False, 4)
        shaped, _ = reward(previous, current, True, False, CONFIG, True, 4)
        self.assertEqual(deltas, {"kills": 1, "damage_dealt": 20, "damage_taken": 4})
        self.assertAlmostEqual(outcome, 3.996)
        self.assertAlmostEqual(shaped - outcome, 1 - 1.5 * 4 / 20)
        after_reset, deltas = reward(current, State(), False, False, CONFIG, True, 0)
        self.assertEqual(after_reset, 0)
        self.assertEqual(deltas, {"kills": 0, "damage_dealt": 0, "damage_taken": 0})
        death, _ = reward(current, current, False, True, CONFIG, True, 0)
        self.assertEqual(death, -5)

    def test_compact_action_mapping_and_validation(self):
        noop = action_to_craftground(NO_OP_ACTION)
        self.assertTrue(all(value == 0 for value in noop.values()))
        action = action_to_craftground([1, 2, 4, 0, 1, 1, 1])
        self.assertTrue(action["forward"] and action["right"] and action["attack"])
        self.assertTrue(action["jump"] and action["sprint"])
        self.assertFalse(action["left"] or action["back"] or action["drop"])
        self.assertEqual(action["camera_yaw"], 30)
        self.assertEqual(action["camera_pitch"], -10)
        self.assertEqual(len(action), 22)
        self.assertEqual(normalize_action(NO_OP_ACTION).shape, (7,))
        with self.assertRaises(ValueError):
            action_to_craftground([0, 0, 5, 1, 0, 0, 0])
        with self.assertRaises(ValueError):
            action_to_craftground([0, 0, 2.5, 1, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
