import pickle
last_generation = pickle.load( open( "last_generation.pkl", "rb" ) )
print(last_generation["best"])