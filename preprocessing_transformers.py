import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Handle False
class HandleFalseMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = [
            "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
            "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", 
            "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        for column in self.columns:
            X[column] = X[column].fillna('None')
            
        return X


# 2. Handle True
class HandleTrueMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns
        self.categorical_columns = X.select_dtypes(exclude=[np.number]).columns
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # MasVnrArea
        indexes_MasVnrArea_0 = X.query("MasVnrArea == 0").index
        X.loc[indexes_MasVnrArea_0, 'MasVnrType'] = X.loc[indexes_MasVnrArea_0, 'MasVnrType'].fillna('None')
        X['MasVnrType'] = X['MasVnrType'].fillna('CBlock')
        
        # GarageYrBlt
        X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["YearBuilt"])
        
        # Numeric columns
        for col in self.numeric_columns:
            X[col] = X[col].fillna(X[col].median())
        
        # Categorical columns
        for col in self.categorical_columns:
            X[col] = X[col].fillna(X[col].mode()[0])
        
        return X
   
    
# 3. Handle Outliers
class HandleOutliers(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        self.Q1 = X['LotArea'].quantile(0.25)
        self.Q3 = X['LotArea'].quantile(0.75)
        self.IQR = self.Q3 - self.Q1
        self.lower_bound = self.Q1 - 1.5 * self.IQR
        self.upper_bound = self.Q3 + 1.5 * self.IQR
        return self

    def transform(self, X):
        X = X.copy()
        
        X['LotArea'] = X['LotArea'].clip(self.lower_bound, self.upper_bound)
        
        return X


# 4. Feature Engineering
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()

        # BsmtFinSF
        X['BsmtFinSF'] = X['TotalBsmtSF'] - X['BsmtUnfSF']
        
        # HighQualFinSF
        X['HighQualFinSF'] = X['1stFlrSF'] + X['2ndFlrSF'] + X['TotalBsmtSF'] - X['LowQualFinSF']
        
        # TotalBathrooms
        X['TotalBathrooms'] = X['BsmtFullBath'] + X['BsmtHalfBath'] + X['FullBath'] + X['HalfBath']

        # GarageAreaCars
        X['GarageAreaCars'] = X['GarageArea'] * X['GarageCars']
        X = X.drop(columns=['GarageArea', 'GarageCars'])

        # TotalPorch
        X['TotalPorch'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['3SsnPorch'] + X['ScreenPorch']

        # OverallAvg
        X['OverallAvg'] = (X['OverallQual'] + X['OverallCond']) / 2
        
        # TotalInternalArea
        X['TotalInternalArea'] = X['1stFlrSF'] + X['2ndFlrSF'] + X['TotalBsmtSF']
        
        # ApproxExternalArea
        X['ApproxExternalArea'] = X['LotArea'] - X['1stFlrSF']
        
        # AgeWhenSold
        X['AgeWhenSold'] = X['YrSold'] - X['YearBuilt']
        
        # YearsUntilRemodel
        X['YearsUntilRemodel'] = X['YearRemodAdd'] - X['YearBuilt']
        
        return X
   
    
# 5. Log Tranformation
class ColumnsLogTransformation(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.skewed_columns_ = []

    def fit(self, X, y=None):
        # Seleciona colunas numéricas
        num_columns = X.select_dtypes(np.number).drop(columns=['MSSubClass', 'OverallQual', 'OverallCond']).columns
        # Calcula skew
        skew_values = X[num_columns].skew()
        # Armazena colunas com skew acima do threshold
        self.skewed_columns_ = skew_values[skew_values > 1].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.skewed_columns_:
            # Adiciona checagem de valores negativos
            X[col] = np.log1p(X[col].clip(lower=0))
            
        return X


# 6. Type Conversion
class ColumnsTypeConversion(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ordinal_columns = {
            "LotShape": ["IR3", "IR2", "IR1", "Reg"],
            "LandContour": ["Low", "Lvl", "Bnk", "HLS"],
            "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
            "LandSlope": ["Gtl", "Mod", "Sev"],
            "OverallQual": list(range(1, 11)), 
            "OverallCond": list(range(1, 11)),  
            "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
            "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
            "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
            "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
            "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
            "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
            "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
            "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "GarageFinish": ["None", "Unf", "RFn", "Fin"],
            "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "PavedDrive": ["N", "P", "Y"],
            "PoolQC": ["None", "Fa", "TA", "Gd", "Ex"],
            "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # MSSubClass
        X['MSSubClass'] = X['MSSubClass'].astype('object')
        
        # Categorical ordinal
        for col, order in self.ordinal_columns.items():
            X[col] = pd.Categorical(X[col], categories=order, ordered=True).codes
            
        ## 8. Columns Significance
        X = X.drop(columns=['PoolQC', 'PoolArea', 'ExterCond'])
            
        return X


# 7. Encoder Scaler
class EncoderScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformer_ = None

    def fit(self, X, y=None):
        object_cols = X.select_dtypes('object').columns.tolist()
        numeric_cols = [col for col in X.select_dtypes(exclude=['category', 'object']).columns if col != 'SalePrice']

        self.transformer_ = ColumnTransformer(
            transformers=[
                ('ohe_encoder', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False), 
                 object_cols),
                ('scaler', StandardScaler(), 
                 numeric_cols)
            ],
            remainder='passthrough'
        )
        self.transformer_.fit(X)
        
        return self

    def transform(self, X):
       
        return self.transformer_.transform(X)
    
    def get_feature_names_out(self):
        return self.transformer_.get_feature_names_out()
