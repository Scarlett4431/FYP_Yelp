import os
import gc
import re
import json
import requests
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.spatial.distance import cdist
from collections import Counter
from math import radians, cos, sin, asin, sqrt
from daisy.utils.utils import ensure_dir


class RawDataReader(object):
    def __init__(self, config):
        self.src = config['dataset']
        self.ds_path = f"{config['data_path']}{self.src}/"
        self.uid_name = config['UID_NAME']
        self.iid_name = config['IID_NAME']
        self.tid_name = config['TID_NAME']
        self.inter_name = config['INTER_NAME']
        self.logger = config['logger']

        ensure_dir(self.ds_path)
        self.logger.info(f'Current data path is: {self.ds_path}, make sure you put the right raw data into it...')
    

    def get_data(self):
        df = pd.DataFrame()
        if self.src == 'ml-100k':
            fp = f'{self.ds_path}u.data'
            df = pd.read_csv(fp, sep='\t', header=None,
                            names=[self.uid_name, self.iid_name, self.inter_name, self.tid_name], engine='python')
        elif self.src == 'yelp':
            json_file_path = f'{self.ds_path}yelp_academic_dataset_review.json'
            prime = []
            for line in open(json_file_path, 'r', encoding='UTF-8'):
                val = json.loads(line)
                prime.append([val['user_id'], val['business_id'], val['stars'], val['date']])
            df = pd.DataFrame(prime, columns=[self.uid_name, self.iid_name, self.inter_name, self.tid_name])
            df[self.tid_name] = pd.to_datetime(df[self.tid_name])
            del prime
            gc.collect()

        else:
            raise NotImplementedError('Invalid Dataset Error')
        return df

class Preprocessor(object):
    def __init__(self, config):
        """
        Method of loading certain raw data
        Parameters
        ----------
        src : str, the name of dataset
        prepro : str, way to pre-process raw data input, expect 'origin', f'{N}core', f'{N}filter', N is integer value
        binary : boolean, whether to transform rating to binary label as CTR or not as Regression
        pos_threshold : float, if not None, treat rating larger than this threshold as positive sample
        level : str, which level to do with f'{N}core' or f'{N}filter' operation (it only works when prepro contains 'core' or 'filter')

        Returns
        -------
        df : pd.DataFrame, rating information with columns: user, item, rating, (options: timestamp)
        """
        self.src = config['dataset']
        self.prepro = config['prepro']
        self.uid_name = config['UID_NAME']
        self.iid_name = config['IID_NAME']
        self.tid_name = config['TID_NAME']
        self.inter_name = config['INTER_NAME']
        self.binary = config['binary_inter']
        self.pos_threshold = config['positive_threshold']
        self.level = config['level'] # ui, u, i
        self.logger = config['logger']
        self.src = config['dataset']
        self.ds_path = f"{config['data_path']}{self.src}/"

        self.get_pop = True if 'popularity' in config['metrics'] else False

        self.user_num, self.item_num = None, None
        self.item_pop = None
        
    def create_items_info(self):
        """
        Create a dictionary with item integer indexes as keys and their geographic information as values.

        Returns
        -------
        items_info : dict
            A dictionary where each key is an item integer index, and the value is another dictionary with 'latitude' and 'longitude'.
        """
        json_file_path = f'{self.ds_path}yelp_academic_dataset_business.json'
        items_info = {}
        for line in open(json_file_path, 'r', encoding='UTF-8'):
            item = json.loads(line)
            item_token = item['business_id']
            if item_token in self.token_iid:
                item_id = self.token_iid[item_token]
                items_info[item_id] = {'latitude': item['latitude'], 'longitude': item['longitude']}
        return items_info

    def calculate_geometric_median(self, X, eps=1e-5):
        """
        Calculate the geometric median for a set of points.
        """
        y = np.mean(X, 0)

        while True:
            D = cdist(X, [y])
            nonzeros = (D != 0)[:, 0]

            Dinv = 1 / D[nonzeros]
            Dinvs = np.sum(Dinv)
            W = Dinv / Dinvs
            T = np.sum(W * X[nonzeros], 0)

            num_zeros = len(X) - np.sum(nonzeros)
            if num_zeros == 0:
                y1 = T
            elif num_zeros == len(X):
                return y
            else:
                R = (T - y) * Dinvs
                r = np.linalg.norm(R)
                rinv = 0 if r == 0 else num_zeros/r
                y1 = max(0, 1-rinv)*T + min(1, rinv)*y

            if np.linalg.norm(y - y1) < eps:
                return y1

            y = y1

    def create_adjusted_user_bboxes(self, items_info, coverage):
        """
        Create a dictionary mapping each user ID to an adjusted bounding box based on a percentage of locations they have visited.
        """
        # Initial user_locations dictionary to store all locations for each user
        user_locations = {}
        json_file_path = f'{self.ds_path}yelp_academic_dataset_review.json'
        count = 0
        for line in open(json_file_path, 'r', encoding='UTF-8'):
            if count % 100000 == 0:
                print(count)
            count += 1
            review = json.loads(line)
            user_id = review['user_id']
            business_id = review['business_id']
            if business_id is None:
                continue
            business_info = items_info.get(business_id)

            if not business_info:
                continue

            if user_id not in user_locations:
                user_locations[user_id] = []
            user_locations[user_id].append((business_info['latitude'], business_info['longitude']))

        user_bboxes = {}
        for user, locations in user_locations.items():
            locations = np.array(locations)
            if len(locations) <= 1:
                continue  # Need at least 2 locations to adjust bounding box

            # Calculate geometric median
            median_location = self.calculate_geometric_median(locations)

            # Calculate distances from median and sort locations by distance
            distances = np.linalg.norm(locations - median_location, axis=1)
            sorted_indices = np.argsort(distances)
            cutoff_index = int(len(sorted_indices) * coverage)  # Keep only a certain percentage

            # Select locations based on coverage
            selected_locations = locations[sorted_indices[:cutoff_index]]

            # Calculate bounding box for selected locations
            min_lat, max_lat = np.min(selected_locations[:, 0]), np.max(selected_locations[:, 0])
            min_lon, max_lon = np.min(selected_locations[:, 1]), np.max(selected_locations[:, 1])

            user_bboxes[user] = (min_lat, max_lat, min_lon, max_lon)
            
        return user_bboxes
        
    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance in kilometers between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles
        return c * r

    def bounding_box_area(self, min_lat, max_lat, min_lon, max_lon):
        """
        Calculate the area of a bounding box in square kilometers
        """
        # Height of the bounding box in km (distance between the min and max latitudes)
        height = self.haversine((min_lon + max_lon) / 2, min_lat, (min_lon + max_lon) / 2, max_lat)
        
        # Width of the bounding box at min_lat and max_lat, then take the average
        width_min_lat = self.haversine(min_lon, min_lat, max_lon, min_lat)
        width_max_lat = self.haversine(min_lon, max_lat, max_lon, max_lat)
        width = (width_min_lat + width_max_lat) / 2
        
        # Area in square kilometers
        area = width * height
        return area
        
    
    
    def update_bbox(self, bbox, lat, lon, min_size):
        """
        Update the bounding box to include a new point (lat, lon) and ensure it has a minimum size.
        
        :param bbox: Current bounding box as a tuple (min_lat, max_lat, min_lon, max_lon)
        :param lat: Latitude of the new point
        :param lon: Longitude of the new point
        :param min_size: Minimum size of the sides of the bounding box (in degrees)
        :return: Updated bounding box
        """
        min_lat, max_lat, min_lon, max_lon = bbox
        
        # Update the bounding box to include the new point
        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)
        min_lon = min(min_lon, lon)
        max_lon = max(max_lon, lon)
        
        # Check if the bounding box is too small and adjust if necessary
        if max_lat - min_lat < min_size:
            expansion = (min_size - (max_lat - min_lat)) / 2
            min_lat -= expansion
            max_lat += expansion
            
        if max_lon - min_lon < min_size:
            expansion = (min_size - (max_lon - min_lon)) / 2
            min_lon -= expansion
            max_lon += expansion
        
        return min_lat, max_lat, min_lon, max_lon

    def adjust_new_bounding_box(self, gt_lat, gt_lon, width, height):
        """
        Adjust the bounding box to randomly include the ground truth item.

        This function calculates one corner of the new bounding box at a random position
        around the ground truth item and then derives the opposite corner based on the width and height.
        """
        # Define shifts for the new bounding box to randomly include the ground truth item
        shift_lat = np.random.uniform(-height, height)
        shift_lon = np.random.uniform(-width, width)

        # Calculate one corner of the new bounding box
        corner1_lat = gt_lat + shift_lat
        corner1_lon = gt_lon + shift_lon

        # Ensure the bounding box includes the ground truth item by calculating the opposite corner
        corner2_lat = corner1_lat + np.sign(shift_lat) * height
        corner2_lon = corner1_lon + np.sign(shift_lon) * width

        # Return the new bounding box coordinates, ensuring min and max values are properly assigned
        new_min_lat, new_max_lat = min(corner1_lat, corner2_lat), max(corner1_lat, corner2_lat)
        new_min_lon, new_max_lon = min(corner1_lon, corner2_lon), max(corner1_lon, corner2_lon)

        return new_min_lat, new_max_lat, new_min_lon, new_max_lon

    def adjust_new_bounding_box_for_test(self,test_ur, items_info, user_bboxes):
        for u, r in test_ur.items():
            # Skip if r is empty
            if not r:
                continue
            ground_truth_item = next(iter(r))  # Select one item as the ground truth
            ground_truth_location = items_info[ground_truth_item]

            if u in user_bboxes:
                # Get the original bounding box and its dimensions
                min_lat, max_lat, min_lon, max_lon = user_bboxes[u]
                width = max_lon - min_lon
                height = max_lat - min_lat

                # Assuming ground_truth_location has 'latitude' and 'longitude' keys
                gt_lat = ground_truth_location['latitude']
                gt_lon = ground_truth_location['longitude']

                # Adjust the bounding box to randomly include the ground truth item
                new_min_lat, new_max_lat, new_min_lon, new_max_lon = self.adjust_new_bounding_box(gt_lat, gt_lon, width, height)
                user_bboxes[u] = new_min_lat, new_max_lat, new_min_lon, new_max_lon
            return user_bboxes


    
    def create_user_bboxes(self, items_info, total_train_ur, test_ur, min_size, test_bounding_box=False):
        """
        Create a dictionary mapping each user integer index to a bounding box based on the businesses they have interacted with in the training dataset.

        Returns
        -------
        user_bboxes : dict
            A dictionary where keys are user integer indexes and values are tuples representing the bounding box for each user
            (min_lat, max_lat, min_lon, max_lon).
        """
        user_bboxes = {}
        json_file_path = f'{self.ds_path}yelp_academic_dataset_review.json'
        for line in open(json_file_path, 'r', encoding='UTF-8'):
            review = json.loads(line)
            user_token = review['user_id']
            business_token = review['business_id']
            
            if user_token not in self.token_uid or business_token not in self.token_iid:
                continue
            user_id = self.token_uid[user_token]
            business_id = self.token_iid[business_token]
            # Skip if business_id is None or if the user-business interaction is not in the training set
            if business_id is None or user_id not in total_train_ur or business_id not in total_train_ur[user_id]:
                continue

            business_info = items_info.get(business_id)
            if not business_info:
                continue  # Skip businesses with no geographic information

            bbox = user_bboxes.get(user_id, (float('inf'), float('-inf'), float('inf'), float('-inf')))
            min_lat, max_lat, min_lon, max_lon = self.update_bbox(bbox, business_info['latitude'], business_info['longitude'], min_size)
            user_bboxes[user_id] = (min_lat, max_lat, min_lon, max_lon)
        if test_bounding_box:    
            print("adjest the bounding box using test ground truth")     
            user_bboxes = self.adjust_new_bounding_box_for_test(test_ur, items_info, user_bboxes)
        return user_bboxes

        
    def process(self, df):
        df = self.__remove_duplication(df)
        df = self.__reserve_pos(df)
        df = self.__binary_inter(df)
        df = self.__core_filter(df)
        self.user_num, self.item_num = self.__get_stats(df)
        df = self.__category_encoding(df)
        df = self.__sort_by_time(df)
        if self.get_pop:
            self.__get_item_popularity(df)

        self.logger.info(f'Finish loading [{self.src}]-[{self.prepro}] dataset')

        return df

    def __get_item_popularity(self, df):
        self.item_pop = np.zeros(self.item_num)
        pop = df.groupby(self.iid_name).size() / self.user_num
        self.item_pop[pop.index] = pop.values

    def __sort_by_time(self, df):
        df = df.sort_values(self.tid_name).reset_index(drop=True)

        return df

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def __remove_duplication(self, df):
        return df.drop_duplicates([self.uid_name, self.iid_name], keep='last', ignore_index=True)

    def __category_encoding(self, df):
        # encoding user_id and item_id
        self.uid_token = pd.Categorical(df['user']).categories.to_numpy()
        self.iid_token = pd.Categorical(df['item']).categories.to_numpy()
        self.token_uid = {uid: token for token, uid in enumerate(self.uid_token)}
        self.token_iid = {iid: token for token, iid in enumerate(self.iid_token)}
        df['user'] = pd.Categorical(df['user']).codes
        df['item'] = pd.Categorical(df['item']).codes
        with open("token_iid.json", 'w') as f:
            json.dump(self.token_iid, f)
        with open("token_uid.json", 'w') as f:
            json.dump(self.token_uid, f)
        return df

    def __get_stats(self, df):
        user_num = df['user'].nunique()
        item_num = df['item'].nunique()

        return user_num, item_num

    def __get_illegal_ids_by_inter_num(self, df, field, inter_num, min_num):
        ids = set()
        for id_ in df[field].values:
            if inter_num[id_] < min_num:
                ids.add(id_)
        return ids

    def __core_filter(self, df):
        # which type of pre-dataset will use
        if self.prepro == 'origin':
            pass
        elif self.prepro.endswith('filter'):
            pattern = re.compile(r'\d+')
            filter_num = int(pattern.findall(self.prepro)[0])

            tmp1 = df.groupby(['user'], as_index=False)['item'].count()
            tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
            tmp2 = df.groupby(['item'], as_index=False)['user'].count()
            tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
            df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])
            if self.level == 'ui':    
                df = df.query(f'cnt_item >= {filter_num} and cnt_user >= {filter_num}').reset_index(drop=True).copy()
            elif self.level == 'u':
                df = df.query(f'cnt_item >= {filter_num}').reset_index(drop=True).copy()
            elif self.level == 'i':
                df = df.query(f'cnt_user >= {filter_num}').reset_index(drop=True).copy()        
            else:
                raise ValueError(f'Invalid level value: {self.level}')

            df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
            del tmp1, tmp2
            gc.collect()

        elif self.prepro.endswith('core'):
            pattern = re.compile(r'\d+')
            core_num = int(pattern.findall(self.prepro)[0])

            if self.level == 'ui':
                user_inter_num = Counter(df[self.uid_name].values)
                item_inter_num = Counter(df[self.iid_name].values)
                while True:
                    ban_users = self.__get_illegal_ids_by_inter_num(df, 'user', user_inter_num, core_num)
                    ban_items = self.__get_illegal_ids_by_inter_num(df, 'item', item_inter_num, core_num)

                    if len(ban_users) == 0 and len(ban_items) == 0:
                        break

                    dropped_inter = pd.Series(False, index=df.index)
                    user_inter = df[self.uid_name]
                    item_inter = df[self.iid_name]
                    dropped_inter |= user_inter.isin(ban_users)
                    dropped_inter |= item_inter.isin(ban_items)
                    
                    user_inter_num -= Counter(user_inter[dropped_inter].values)
                    item_inter_num -= Counter(item_inter[dropped_inter].values)

                    dropped_index = df.index[dropped_inter]
                    df.drop(dropped_index, inplace=True)

            elif self.level == 'u':
                tmp = df.groupby(['user'], as_index=False)['item'].count()
                tmp.rename(columns={'item': 'cnt_item'}, inplace=True)
                df = df.merge(tmp, on=['user'])
                df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
                df.drop(['cnt_item'], axis=1, inplace=True)
            elif self.level == 'i':
                tmp = df.groupby(['item'], as_index=False)['user'].count()
                tmp.rename(columns={'user': 'cnt_user'}, inplace=True)
                df = df.merge(tmp, on=['item'])
                df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
                df.drop(['cnt_user'], axis=1, inplace=True)
            else:
                raise ValueError(f'Invalid level value: {self.level}')

            gc.collect()

        else:
            raise ValueError('Invalid dataset preprocess type, origin/Ncore/Nfilter (N is int number) expected')
        
        df = df.reset_index(drop=True)

        return df

    def __reserve_pos(self, df):
        # set rating >= threshold as positive samples
        if self.pos_threshold is not None:
            df = df.query(f'rating >= {self.pos_threshold}').reset_index(drop=True)
        return df

    def __binary_inter(self, df):
        # reset rating to interaction, here just treat all rating as 1
        if self.binary:
            df['rating'] = 1.0
        return df
