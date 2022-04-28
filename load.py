import pickle
infile = open('/app/data/inference_time/data.pkl','rb')
new_dict = pickle.load(infile)
print(new_dict)