"""``python -m thin.store``: the store's command line."""

import sys

from thin.runtime.store import main

if __name__ == "__main__":
    sys.exit(main())
