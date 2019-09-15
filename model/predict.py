import pickle

# def get_probability(input_vec, final_model_file_name):
def get_probability(input_vec, clf):

    # with open(final_model_file_name, 'rb') as file:
    #     clf = pickle.load(file)
        
    # clf = pickle.load(open(final_model_file_name, 'rb'))
    prob = clf.predict([input_vec])
    return prob[0]



