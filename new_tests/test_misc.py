import mock
import os, sys

# before importing local modules, must add root dir to system path
# capture location of current file (/root/tests/)
myPath = os.path.dirname(os.path.abspath(__file__))
# add to system path the root dir with relative notation: /../ (go up one dir)
sys.path.insert(0, myPath + '/../')

# Now we can import modules from other directory
from core.storage import misc
from main import ROOT_DIR


@mock.patch('core.storage.misc.os')
def test_cd(mock_cd):
    """
    :param mock_cd: mocked object. In this case, we will mock the module cd
    As a reminder, mocking is used to test the functionality of the modules imported in our script. Instead of testing
    the script of its result, we want to make sure the main components of the script is working perfectly and those are
    usually the main python modules that the script uses.
    In this case, we want to make sure that the module "os" is doing what its supposed to do and that is to change the
    working directory
    """
    # change working directory to
    os.chdir(ROOT_DIR)
    # move to dataFiles
    with misc.cd('dataFiles'):
        # Test to see if getcwd gets called
        mock_cd.getcwd.assert_called_once()
        # Test to see if os.path.expanduser gets called with a string(Folder name)
        mock_cd.path.expanduser.assert_called_with('dataFiles')
        # Test to see if os.chdir gets called
        mock_cd.chdir.assert_called_once()