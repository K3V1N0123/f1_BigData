import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "src" / "ingestion"


class SessionLoadTest(unittest.TestCase):
    def test_session_load_runs_from_script_directory(self):
        result = subprocess.run(
            [sys.executable, "session_load.py"],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
