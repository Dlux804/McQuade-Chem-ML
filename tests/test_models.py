# import os, sys
#
#
# # before importing local modules, must add root dir to system path
# # capture location of current file (/root/tests/)
# myPath = os.path.dirname(os.path.abspath(__file__))
# # add to system path the root dir with relative notation: /../ (go up one dir)
# sys.path.insert(0, myPath + '/../')
#
# # Now we can import modules from other directory
# from core import models, misc
# from main import ROOT_DIR
#
#
# # Test models.py
# def test_models_run():
#     """
#     I still need to find a better way to test models.py since it does not return any thing and it's the script that
#     runs everything.
#     Since models is the accumulation of every other models, I think the best way to tackle this for now and to run
#     the script to see if it works or not. No assertion needed
#     """
#     # change working directory to
#     os.chdir(ROOT_DIR)
#     # move to dataFiles
#     with misc.cd('dataFiles'):
#         print('Now in:', os.getcwd())
#         # Calling MlModel to get our class instance
#         model1 = models.MlModel('rf', 'water-energy.csv', 'expt')
#         # Call featurization function
#         model1.featurization([0])
#         # Call run function
#         # model1.run(tune=False)
