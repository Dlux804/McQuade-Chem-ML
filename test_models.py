from models import MlModel


# Test models.py
def test_models_run():
    """
    I still need to find a better way to test models.py since it does not return any thing and it's the script that
    runs everything.
    Since models is the accumulation of every other models, I think the best way to tackle this for now and to run
    the script to see if it works or not. No assertion needed
    """
    # Calling MlModel to get our class instances
    model1 = MlModel('rf', 'water-energy.csv', 'expt')
    # Call featurization function
    model1.featurization([0])
    # Call run function
    model1.run(tune=False)
    # Call store function
    model1.store()