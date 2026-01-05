"""
Unit tests for StatTracker.

Position mapping (HiveMind internal IDs):
- Position 1: Gold Queen
- Position 2: Blue Queen
- Positions 3-6: Gold workers
- Positions 7-10: Blue workers
"""

import unittest
from stat_tracker import StatTracker


class TestStatTrackerGame1700985(unittest.TestCase):
    """Test K/D stats for game 1700985.

    Expected stats from user verification:
    - Blue Checkers: 4-1 with 2 queen kills
    - Blue Skulls: 1-2
    - Blue Queen: 2-1
    - Gold Queen: 3-3
    """

    @classmethod
    def setUpClass(cls):
        cls.tracker = StatTracker.from_game_id(1700985)

    def test_blue_checkers_position_10(self):
        """Blue Checkers is position 10: 4 kills, 2 deaths, 2 queen kills.

        Deaths: 1 from playerKill (pos 3), 1 from snailEat (pos 9).
        """
        stats = self.tracker.get_stats(10)
        self.assertEqual(stats.kills, 4)
        self.assertEqual(stats.deaths, 2)  # includes snail death
        self.assertEqual(stats.queen_kills, 2)

    def test_blue_skulls_position_8(self):
        """Blue Skulls is position 8: 1 kill, 2 deaths."""
        stats = self.tracker.get_stats(8)
        self.assertEqual(stats.kills, 1)
        self.assertEqual(stats.deaths, 2)

    def test_blue_queen_position_2(self):
        """Blue Queen is position 2: 2 kills, 1 death."""
        stats = self.tracker.get_stats(2)
        self.assertEqual(stats.kills, 2)
        self.assertEqual(stats.deaths, 1)

    def test_gold_queen_position_1(self):
        """Gold Queen is position 1: 3 kills, 3 deaths."""
        stats = self.tracker.get_stats(1)
        self.assertEqual(stats.kills, 3)
        self.assertEqual(stats.deaths, 3)


if __name__ == '__main__':
    unittest.main()
