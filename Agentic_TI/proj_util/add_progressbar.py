'''
##############################################################
# Created Date: Wednesday, July 16th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


from pathlib import Path
import shutil


def update_mealpy_optimizer():

    path_optimizer_new = Path(__file__).parent / 'optimizer.py'

    try:
        import mealpy
        path_optimizer_old = Path(mealpy.__file__).parent / 'optimizer.py'
        shutil.copy(path_optimizer_new, path_optimizer_old)
        return "Optimizer updated successfully."
    except ImportError:
        print("mealpy is not installed. Please install it to update the optimizer.")
        return "mealpy not found."
