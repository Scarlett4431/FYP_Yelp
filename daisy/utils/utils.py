import os
import torch
import datetime
import pickle
import numpy as np
import csv
import scipy.sparse as sp
from collections import defaultdict


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

def get_ur(df):
    """
    Method of getting user-rating pairs
    Parameters
    ----------
    df : pd.DataFrame, rating dataframe

    Returns
    -------
    ur : dict, dictionary stored user-items interactions
    """
    ur = defaultdict(set)
    for _, row in df.iterrows():
        ur[int(row['user'])].add(int(row['item']))

    return ur

def get_ir(df):
    """
    Method of getting item-rating pairs
    Parameters
    ----------
    df : pd.DataFrame, rating dataframe

    Returns
    -------
    ir : dict, dictionary stored item-users interactions
    """
    ir = defaultdict(set)
    for _, row in df.iterrows():
        ir[int(row['item'])].add(int(row['user']))

    return ir


import numpy as np
import os
import pickle
import csv



def build_candidates_set_per_user(test_ur, train_ur, items_info, user_bboxes, config, drop_past_inter=True):
    candidates_num = config['cand_num']
    min_size = config['min_size']
    test_bounding_box = config['test_bounding_box']
    train_bounding_box = config['train_bounding_box']
    logger = config['logger']
    # Check if the .pkl file exists
    if test_bounding_box:
        file_name = 'data/yelp/items_within_bb_'+ str(min_size)+  '_test.pkl'
    elif train_bounding_box:
        file_name = 'data/yelp/items_within_bb_'+ str(min_size)+ '_train.pkl'
    if os.path.exists(file_name):
        logger.info("Loading items_within_bb from pickle file.")
        with open(file_name, 'rb') as f:
            items_within_bb = pickle.load(f)
    else:
        logger.info("Pickle file not found. Computing items_within_bb.")
        items_within_bb = {user_id: set() for user_id in user_bboxes}
        for user_id, bb in user_bboxes.items():
            #this user is from the training set, if not the user would not be in items_within_bb
            min_lat, max_lat, min_lon, max_lon = bb
            for item_id, item in items_info.items():
                if min_lat <= item['latitude'] <= max_lat and min_lon <= item['longitude'] <= max_lon:
                    items_within_bb[user_id].add(item_id)
        logger.info("Precomputing user's bounding boxes done")
        # Save the computed items_within_bb to a .pkl file for future use
        with open(file_name, 'wb') as f:
            pickle.dump(items_within_bb, f)
            
    item_num = config['item_num']
    pos_filtered_percentages = []
    neg_filtered_percentages = []
    test_ucands, test_u = [], []
    for u, r in test_ur.items():
        sample_num = candidates_num - len(r) if len(r) <= candidates_num else 0
        if sample_num == 0:
            samples = np.random.choice(list(r), candidates_num)
        else:
            pos_items = list(r) + list(train_ur[u]) if drop_past_inter else list(r)
            neg_items = np.setdiff1d(np.arange(item_num), pos_items)
            neg_samples = np.random.choice(neg_items, size=sample_num)
            samples = np.concatenate((neg_samples, list(r)), axis=None)
        if u in items_within_bb:
            test_pos_samples = list(r)
            test_neg_samples = [item for item in neg_samples if item in items_within_bb[u]]
            pos_filtered_percentage = 100 * (1 - len(test_pos_samples) / len(list(r))) if len(list(r)) > 0 else 0
            neg_filtered_percentage = 100 * (1 - len(test_neg_samples) / len(neg_items)) if len(neg_items) > 0 else 0
            pos_filtered_percentages.append(pos_filtered_percentage)
            neg_filtered_percentages.append(neg_filtered_percentage)
            filtered_samples = np.concatenate((test_neg_samples, test_pos_samples), axis=None)
            if len(filtered_samples) == 0:
                continue
            # Pad the samples array with -1 if it's shorter than candidates_num
            if len(filtered_samples) < candidates_num:
                samples = np.pad(filtered_samples, (0, candidates_num - len(filtered_samples)), constant_values=(-1,))
    
        test_ucands.append([u, samples])
        test_u.append(u)
        
    if test_bounding_box:
        file_path = 'filter_percentage_'+ str(config['min_size']) + '_test.csv'
    elif train_bounding_box:
       file_path = 'filter_percentage_'+ str(config['min_size']) + '_train.csv'

    # Open the file in write mode ('w') and create a csv.writer object
    # newline='' is used to prevent writing extra blank rows in some environments
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Optionally, write headers as the first row
        writer.writerow(['user','pos_filtered_percentage', 'neg_filtered_percentage'])
        
        # Write the contents of both lists to the file, row by row
        for user, item1, item2 in zip(test_ur.keys(),pos_filtered_percentages, neg_filtered_percentages):
            writer.writerow([user,item1, item2])

    # Inform the user that the lists have been written to the file
    print(f"Lists have been written to {file_path}")

    return test_u, test_ucands


def build_candidates_set(test_ur, train_ur, config, drop_past_inter=True):
    """
    method of building candidate items for ranking
    Parameters
    ----------
    test_ur : dict, ground_truth that represents the relationship of user and item in the test set
    train_ur : dict, this represents the relationship of user and item in the train set
    item_num : No. of all items
    cand_num : int, the number of candidates
    drop_past_inter : drop items already appeared in train set

    Returns
    -------
    test_ucands : dict, dictionary storing candidates for each user in test set
    """
    item_num = config['item_num']
    candidates_num = config['cand_num']

    test_ucands, test_u = [], []
    for u, r in test_ur.items():
        sample_num = candidates_num - len(r) if len(r) <= candidates_num else 0
        if sample_num == 0:
            samples = np.random.choice(list(r), candidates_num)
        else:
            pos_items = list(r) + list(train_ur[u]) if drop_past_inter else list(r)
            neg_items = np.setdiff1d(np.arange(item_num), pos_items)
            samples = np.random.choice(neg_items, size=sample_num)
            samples = np.concatenate((samples, list(r)), axis=None)

        test_ucands.append([u, samples])
        test_u.append(u)
    
    return test_u, test_ucands

def get_history_matrix(df, config, row='user', use_config_value_name=False):
    '''
    get the history interactions by user/item
    '''
    logger = config['logger']
    assert row in df.columns, f'invalid name {row}: not in columns of history dataframe'
    uid_name, iid_name  = config['UID_NAME'], config['IID_NAME']
    user_ids, item_ids = df[uid_name].values, df[iid_name].values
    value_name = config['INTER_NAME'] if use_config_value_name else None

    user_num, item_num = config['user_num'], config['item_num']
    values = np.ones(len(df)) if value_name is None else df[value_name].values

    if row == 'user':
        row_num, max_col_num = user_num, item_num
        row_ids, col_ids = user_ids, item_ids
    else: # 'item'
        row_num, max_col_num = item_num, user_num
        row_ids, col_ids = item_ids, user_ids

    history_len = np.zeros(row_num, dtype=np.int64)
    for row_id in row_ids:
        history_len[row_id] += 1

    col_num = np.max(history_len)
    if col_num > max_col_num * 0.2:
        logger.info(f'Max value of {row}\'s history interaction records has reached: {col_num / max_col_num * 100:.4f}% of the total.')

    history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
    history_value = np.zeros((row_num, col_num))
    history_len[:] = 0
    for row_id, value, col_id in zip(row_ids, values, col_ids):
        history_matrix[row_id, history_len[row_id]] = col_id
        history_value[row_id, history_len[row_id]] = value
        history_len[row_id] += 1

    return torch.LongTensor(history_matrix), torch.FloatTensor(history_value), torch.LongTensor(history_len)

def get_inter_matrix(df, config, form='coo'):
    '''
    get the whole sparse interaction matrix
    '''
    logger = config['logger']
    value_field = config['INTER_NAME']
    src_field, tar_field = config['UID_NAME'], config['IID_NAME']
    user_num, item_num = config['user_num'], config['item_num']

    src, tar = df[src_field].values, df[tar_field].values
    data = df[value_field].values

    mat = sp.coo_matrix((data, (src, tar)), shape=(user_num, item_num))

    if form == 'coo':
        return mat
    elif form == 'csr':
        return mat.tocsr()
    else:
        raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented...')

