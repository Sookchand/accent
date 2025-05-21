import unittest
import sys

# Import test modules
from test_accent_detector import TestAccentDetector
from test_app import TestApp

def run_tests():
    """
    Run all tests and return the number of failures.
    """
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestAccentDetector))
    test_suite.addTest(unittest.makeSuite(TestApp))
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return the number of failures
    return len(result.failures) + len(result.errors)

if __name__ == "__main__":
    # Run the tests and exit with the appropriate status code
    failures = run_tests()
    sys.exit(failures)
