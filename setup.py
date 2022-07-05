from glob import glob
from os.path import abspath,basename,dirname,join,normpath,relpath


from shutil import rmtree
from setuptools import setup,find_packages
from setuptools import Command

here = normpath(abspath(dirname(__file__)))

class CleanCommand(Command):
    """
    Custom clean command to tidy up the root folder
    """
    CLEAN_FILES = './build ./dist ./*.pyc ./*.tgz ./*.egg-info ./ __pycache__'.split()
    user_uptions =[]

    def initialize_options(self) -> None:
        pass
    def finalize_options(self) -> None:
        pass
    def run(self) -> None:
        global here
        for path_spec in self.CLEAN_FILES:
            abs_paths  = glob(normpath(join(here,path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(here):
                    raise ValueError("%s is not a path inside %s "%(path,here))
                print("removing %s"%relpath(path))


setup(
    cmdclass={'clean':CleanCommand,
           },
    name='MLframework',
    version = '1.0',
    packages=find_packages(),
    py_modules = ['src'],
)
