import unittest
from replaybuffer.experience import Experience

class TestExperience(unittest.TestCase):
    def test_experience_initialization(self):
        state = [1, 2, 3]
        action = 1
        reward = 1.0
        next_state = [4, 5, 6]
        done = False
        
        experience = Experience(state, action, reward, next_state, done)
        
        self.assertEqual(experience.state, state)
        self.assertEqual(experience.action, action)
        self.assertEqual(experience.reward, reward)
        self.assertEqual(experience.next_state, next_state)
        self.assertEqual(experience.done, done)

if __name__ == '__main__':
    unittest.main()