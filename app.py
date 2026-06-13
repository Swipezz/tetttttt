"""
app.py (root)
=============
Entry point untuk Streamlit Cloud deployment.
Streamlit Cloud secara default mencari app.py di root repository.

File ini hanya wrapper yang memanggil streamlit_app/app.py.
"""

import sys
from pathlib import Path

# Pastikan root project ada di sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Import dan jalankan main app
from streamlit_app.app import main

if __name__ == "__main__":
    main()
else:
    # Streamlit Cloud menjalankan file ini langsung (bukan sebagai __main__)
    main()
