'''
##############################################################
# Created Date: Wednesday, April 30th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


def func_prompt(name: str, description: str):
    # Decorator to set the name and description of a function
    def name_func(func):
        func.name = name
        func.description = description
        return func
    return name_func
