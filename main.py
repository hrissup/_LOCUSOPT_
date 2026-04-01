# Import System Utilities
import sys

# Impoort the main function from CLI
from optimizer.cli import main

# If run directly [From Terminal] __name__ = "__main__"
# If imported __name__ = "main"
if __name__ == "__main__":
    sys.exit(main())
