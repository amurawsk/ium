import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from math import radians, sin, cos, sqrt, atan2
import ast
from collections import Counter
import re
from sklearn.preprocessing import OneHotEncoder


# klasy stworzone na podstawie notatnika data_analysis, który stworzyliśmy w procesie analizy danych
class BathroomsProcessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.bedrooms_median_ = X['bedrooms'].median()
        self.beds_median_ = X['beds'].median()
        self.bathrooms_num_median_ = None
        return self

    def transform(self, X):
        df = X.copy()
        df['bedrooms'] = df['bedrooms'].fillna(self.bedrooms_median_)
        df['beds'] = df['beds'].fillna(self.beds_median_)

        def normalize_half_bath(value):
            if pd.isna(value):
                return value
            value = value.strip().lower()
            if value == 'half-bath':
                return '0.5 bath'
            elif value == 'shared half-bath':
                return '0.5 shared bath'
            elif value == 'private half-bath':
                return '0.5 private bath'
            return value
        
        def extract_bathrooms_num(text):
            if pd.isna(text) or text == 'nan':
                return np.nan
            match = re.match(r'^(\d*\.?\d+)', text)
            if match:
                return float(match.group(1))
            return np.nan
        
        def extract_suffix(text):
            if pd.isna(text) or text == 'nan':
                return np.nan
            match = re.match(r'^\d*\.?\d+\s*(.*)$', text)
            if match:
                return match.group(1).strip()
            return np.nan
        
        def classify_bathroom_type(suffix):
            if pd.isna(suffix):
                return 'unknown'
            suffix = suffix.lower()
            if 'shared' in suffix:
                return 'shared'
            elif 'private' in suffix:
                return 'private'
            else:
                return 'unknown'
        
        df['bathrooms_text'] = df['bathrooms_text'].apply(normalize_half_bath)
        df['bathrooms_text_num'] = df['bathrooms_text'].apply(extract_bathrooms_num)
        df['bathrooms_suffix'] = df['bathrooms_text'].apply(extract_suffix)
        df['bathrooms_num_old'] = pd.to_numeric(df['bathrooms'], errors='coerce')
        df['bathrooms_num'] = df['bathrooms_text_num'].combine_first(df['bathrooms_num_old'])

        if self.bathrooms_num_median_ is None:
            self.bathrooms_num_median_ = df['bathrooms_num'].median()
        df['bathrooms_num'] = df['bathrooms_num'].fillna(self.bathrooms_num_median_)
        df['bathroom_type'] = df['bathrooms_suffix'].apply(classify_bathroom_type)
        
        df.drop(columns=['bathrooms', 'bathrooms_text', 'bathrooms_text_num', 'bathrooms_num_old', 'bathrooms_suffix'], inplace=True)
        return df

class AmenitiesProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=50):
        self.top_n = top_n
    
    def fit(self, X, y=None):
        all_amenities = Counter(item for amenities in X['amenities'] for item in ast.literal_eval(amenities))
        self.top_amenities_ = [item for item, _ in all_amenities.most_common(self.top_n)]
        return self
    
    def transform(self, X):
        df = X.copy()
        def has_amenity(amenities_str, amenity):
            amenities_list = ast.literal_eval(amenities_str)
            return int(amenity in amenities_list)
        
        for amenity in self.top_amenities_:
            df[f'amenity_{amenity}'] = df['amenities'].apply(lambda x: has_amenity(x, amenity))
        df.drop(columns=['amenities'], inplace=True)
        return df

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # zapamiętujemy wartości do skalowania (mean, std)
        self.latitude_mean_ = X['latitude'].mean()
        self.latitude_std_ = X['latitude'].std()
        self.longitude_mean_ = X['longitude'].mean()
        self.longitude_std_ = X['longitude'].std()
        self.amenity_cols_ = [col for col in X.columns if col.startswith("amenity_")]
        return self
    
    def transform(self, X):
        df = X.copy()
        df['instant_bookable'] = df['instant_bookable'].map({'t': 1, 'f': 0})
        df["bathrooms_per_guest"] = df["bathrooms_num"] / df["accommodates"].replace(0, np.nan)
        df["bedrooms_per_guest"] = df["bedrooms"] / df["accommodates"].replace(0, np.nan)
        df["beds_per_guest"] = df["beds"] / df["accommodates"].replace(0, np.nan)
        df["beds_per_bedroom"] = df["beds"] / df["bedrooms"].replace(0, np.nan)
        df["guests_per_bedroom"] = df["accommodates"] / df["bedrooms"].replace(0, np.nan)

        for col in ["bathrooms_per_guest", "bedrooms_per_guest", "beds_per_guest", "beds_per_bedroom", "guests_per_bedroom"]:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            q_low, q_high = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(q_low, q_high).fillna(df[col].median())

        def haversine(lat1, lon1, lat2=41.3870, lon2=2.1701):
            R = 6371
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
            return R * 2 * atan2(sqrt(a), sqrt(1 - a))
        
        df['dist_to_center'] = df.apply(lambda row: haversine(row['latitude'], row['longitude']), axis=1)
        df['lat_scaled'] = (df['latitude'] - self.latitude_mean_) / self.latitude_std_
        df['lon_scaled'] = (df['longitude'] - self.longitude_mean_) / self.longitude_std_
        df['is_group_friendly'] = (df['accommodates'] >= 4).astype(int)
        df["num_amenities"] = df[self.amenity_cols_].sum(axis=1)
        df["has_tv_or_wifi"] = df[["amenity_TV", "amenity_Wifi"]].max(axis=1)
        df["is_suited_for_longterm"] = (
            df[["amenity_Washer", "amenity_Kitchen", "amenity_Dishes and silverware"]].sum(axis=1) >= 2
        ).astype(int)
        return df

class LocationClusterer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=15, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
    
    def fit(self, X, y=None):
        from sklearn.cluster import KMeans
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.kmeans_.fit(X[['latitude', 'longitude']])
        return self
    
    def transform(self, X):
        df = X.copy()
        df['location_cluster'] = self.kmeans_.predict(df[['latitude', 'longitude']])
        return df

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categorical_cols = ['property_type', 'room_type', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'bathroom_type']
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    def fit(self, X, y=None):
        self.encoder.fit(X[self.categorical_cols])
        return self
    
    def transform(self, X):
        df = X.copy()
        encoded = self.encoder.transform(df[self.categorical_cols])
        encoded_cols = self.encoder.get_feature_names_out(self.categorical_cols)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        df = pd.concat([df.drop(columns=self.categorical_cols), encoded_df], axis=1)
        return df
