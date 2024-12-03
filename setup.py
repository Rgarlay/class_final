from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requiments(file_path:str)->List[str]:
    '''
    This will return list of requirments
    '''
    requirments = []
    with open(file_path) as file_obj:
        requirments=file_obj.readlines()
        requirments=[word.replace('\n',"") for word in requirments]

        if HYPHEN_E_DOT in requirments:
            requirments.remove(HYPHEN_E_DOT)


    return requirments

setup (
name='bbc_news_classification_project',
version = '0.0.1',
author = 'Ravi',
author_email='ravigarlay@gmail.com',
packages=find_packages(),
install_requires=get_requiments('req.txt') 
)