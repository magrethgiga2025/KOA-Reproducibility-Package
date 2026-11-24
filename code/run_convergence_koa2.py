"""Convergence tracking for the KOA paper: the 9 algorithms of the main comparison only."""
import sys
import track_convergence as tc

tc.ALGORITHMS = ["KOA-II", "NSGA-II", "NSGA-III", "MOEA/D", "RVEA",
                 "AGE-MOEA2", "CMOPSO", "SMS-EMOA", "GDE3"]
if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    tc.main()
