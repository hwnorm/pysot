import pickle
for tracker in ['deepsort', 'fairmot']:
    vp_dict = pickle.load(open('vp_dict_{}.pkl'.format(tracker), 'rb'))
    pickle.dump(list(vp_dict.keys()), open('vp_list_{}.pkl'.format(tracker), 'wb'))