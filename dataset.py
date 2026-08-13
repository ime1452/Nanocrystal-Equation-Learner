import os
import pandas as pd
import numpy as np
import torch
from scipy.special import inv_boxcox
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler, PowerTransformer


def read_table_file(file_path):
    _, ext = os.path.splitext(str(file_path).lower())
    if ext in {'.xlsx', '.xls'}:
        return pd.read_excel(file_path)
    if ext == '.csv':
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported data file format: {file_path}. Please use .xlsx, .xls, or .csv.")


class BoxCoxTargetScaler:
    """
    Wrapper for a reversible Box-Cox + standardization transformation on the target column.
    Fit only on the training set during training; reuse the same fitted transformer during validation/testing.
    """
    def __init__(self):
        self.transformer = PowerTransformer(method='box-cox', standardize=True)
        self.method = 'box-cox'
        self.is_linear = False
        self.lambdas_ = None
        self.standard_mean_ = None
        self.standard_scale_ = None

    def fit(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        if np.any(values <= 0):
            raise ValueError("Box-Cox target transformation requires all values to be strictly greater than 0.")

        self.transformer.fit(values)
        self.lambdas_ = np.asarray(self.transformer.lambdas_, dtype=np.float64)

        scaler = getattr(self.transformer, '_scaler', None)
        if scaler is not None:
            self.standard_mean_ = np.asarray(scaler.mean_, dtype=np.float64)
            self.standard_scale_ = np.asarray(scaler.scale_, dtype=np.float64)

        return self

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        return self.transformer.transform(values)

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        if self.standard_mean_ is None or self.standard_scale_ is None or self.lambdas_ is None:
            raise ValueError("BoxCoxTargetScaler must be fitted before inverse_transform.")

        boxcox_values = values * self.standard_scale_ + self.standard_mean_
        lambda_value = float(self.lambdas_[0])

        if abs(lambda_value) >= 1e-8:
            boundary = -1.0 / lambda_value
            eps = 1e-8
            if lambda_value > 0:
                boxcox_values = np.maximum(boxcox_values, boundary + eps)
            else:
                boxcox_values = np.minimum(boxcox_values, boundary - eps)

        return inv_boxcox(boxcox_values, self.lambdas_)

class NanoDataset(Dataset):
    """
    Nanocrystal dataset class.
    Loads cleaned and split Excel files from prepare_data.py, along with inorganic and organic feature descriptors,
    and converts them into tensor formats usable by the model.
    """
    def __init__(
        self,
        data_path,
        inorg_path,
        org_path,
        fit_scaler=True,
        scalers=None,
        use_top10_features=True,
        target_transform='box-cox',
    ):
        """
        Load data and perform preprocessing.
        Added RobustScaler for robust standardization of input features (resistant to outliers).
        Added use_top10_features parameter to control whether to filter features based on importance markers.
        """
        # 1. Read the preprocessed experimental data table
        raw = read_table_file(data_path)
        
        # 1.2 Extract operation descriptors (T_inj, T_rea, t) - Speed column removed as required
        ops_cols = ['T_inj', 'T_rea', 't']
        # Force non-numeric content to NaN
        for col in ops_cols:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')
        ops_raw = raw[ops_cols].values.astype(np.float32)
        
        # Use SimpleImputer to fill missing values with median
        from sklearn.impute import SimpleImputer
        if fit_scaler:
            self.ops_imputer = SimpleImputer(strategy='median').fit(ops_raw)
        else:
            self.ops_imputer = scalers[3]
        ops_raw = self.ops_imputer.transform(ops_raw)
        
        # Standardize operation descriptors
        if fit_scaler:
            self.ops_scaler = RobustScaler().fit(ops_raw)
        else:
            self.ops_scaler = scalers[4]
        ops_scaled = self.ops_scaler.transform(ops_raw)
        
        # 1.3 Extract product morphology (shape) and map to continuous 3D features (Circularity, Aspect_Ratio, Vertices)
        shape_series = raw['shape'].fillna('unknown')
        
        mapping_path = os.path.join(os.path.dirname(data_path), 'morphology_mapping.csv')
        try:
            shape_df = pd.read_csv(mapping_path, sep='\t')
            if 'Circularity' not in shape_df.columns:
                shape_df = pd.read_csv(mapping_path, sep=',')
        except Exception:
            shape_df = pd.read_csv(mapping_path)
            
        shape_dict = {}
        for _, row in shape_df.iterrows():
            if pd.notna(row['shape']):
                shape_dict[str(row['shape']).strip()] = [float(row['Circularity']), float(row['Aspect_Ratio']), float(row['Vertices'])]
                
        default_shape = [0.0, 0.0, 0.0]
        shape_features = []
        for s in shape_series:
            s_val = str(s).strip()
            if s_val in shape_dict:
                shape_features.append(shape_dict[s_val])
            else:
                shape_features.append(default_shape)
                
        shape_encoded = np.array(shape_features, dtype=np.float32)
        
        if fit_scaler:
            self.shape_encoder = None
        else:
            self.shape_encoder = scalers[5] if scalers is not None and len(scalers) > 5 else None
            
        self.dim_shape = shape_encoded.shape[1]
        
        # Read inorganic and organic feature descriptor files
        inorg_df = pd.read_csv(inorg_path)
        org_df = pd.read_csv(org_path)
        
        # Filter features based on importance markers
        if use_top10_features:
            feature_recon_path = os.path.join(os.path.dirname(data_path), 'Feature_Reconstruction.csv')
            if os.path.exists(feature_recon_path):
                feature_recon_df = pd.read_csv(feature_recon_path)
                
                # Extract inorganic features (including products and inorganic reactants)
                prod_inorg_features = feature_recon_df[feature_recon_df['Is a product feature?'] == 'yes']['Raw feature name'].tolist()
                react_inorg_features = feature_recon_df[feature_recon_df['Is a inorganic reactant feature?'] == 'yes']['Raw feature name'].tolist()
                
                # Extract organic features
                react_org_features = feature_recon_df[feature_recon_df['Is a organic reactant feature?'] == 'yes']['Raw feature name'].tolist()
                
                # Deduplicate, merge, and sort to ensure absolute consistency in feature order
                top_inorg_features = sorted(list(set(prod_inorg_features + react_inorg_features)))
                top_org_features = sorted(list(set(react_org_features)))
                
                # Filter DataFrames
                inorg_cols_to_keep = ['filename'] + [col for col in top_inorg_features if col in inorg_df.columns]
                org_cols_to_keep = ['filename'] + [col for col in top_org_features if col in org_df.columns]
                
                inorg_df = inorg_df[inorg_cols_to_keep]
                org_df = org_df[org_cols_to_keep]
                print(f"Top 10 feature filtering enabled. Inorganic features: {len(inorg_cols_to_keep)-1}, Organic features: {len(org_cols_to_keep)-1}")
            else:
                print("Feature_Reconstruction.csv not found, using all features.")

        # Convert all infinities to np.nan, then fill with 0 to prevent overflow when converting to float32
        inorg_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        org_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        inorg_df.fillna(0, inplace=True)
        org_df.fillna(0, inplace=True)
        
        # Convert DataFrames to float64 to prevent overflow from extremely large values, then convert to numpy
        inorg_features_raw = inorg_df.drop('filename', axis=1).astype('float64').values
        org_features_raw = org_df.drop('filename', axis=1).astype('float64').values
        
        # Use np.clip to limit extremely large/small values within the safe range of float32, then force conversion to float32
        inorg_features_raw = np.clip(inorg_features_raw, -3e38, 3e38).astype(np.float32)
        org_features_raw = np.clip(org_features_raw, -3e38, 3e38).astype(np.float32)
        
        # Replace infinities (inf) with NaN, then fill NaN with 0 to prevent training crashes
        inorg_features_raw = np.nan_to_num(inorg_features_raw, posinf=0, neginf=0)
        org_features_raw = np.nan_to_num(org_features_raw, posinf=0, neginf=0)
        
        # Clip data again to prevent extreme outliers from destroying mean and variance calculations
        inorg_features_raw = np.clip(inorg_features_raw, -1e6, 1e6)
        org_features_raw = np.clip(org_features_raw, -1e6, 1e6)
        
        # 3. Data standardization (greatly alleviates loss explosion and NaN issues)
        self.target_transform = target_transform
        target_values = raw['size'].values.reshape(-1, 1)

        if fit_scaler:
            self.inorg_scaler = RobustScaler().fit(inorg_features_raw)
            self.org_scaler = RobustScaler().fit(org_features_raw)

            if target_transform == 'box-cox':
                self.y_scaler = BoxCoxTargetScaler().fit(target_values)
            elif target_transform == 'robust':
                self.y_scaler = RobustScaler().fit(target_values)
            else:
                raise ValueError(
                    f"Unsupported target_transform: {target_transform}. "
                    "Please choose from ['box-cox', 'robust']."
                )
        else:
            self.inorg_scaler = scalers[0]
            self.org_scaler = scalers[1]
            self.y_scaler = scalers[2]
            
        inorg_features_scaled = self.inorg_scaler.transform(inorg_features_raw)
        org_features_scaled = self.org_scaler.transform(org_features_raw)
        
        # Build dictionaries with substance names as keys for fast lookup
        self.inorg_features = {row['filename']: inorg_features_scaled[i] for i, row in inorg_df.iterrows()}
        self.org_features = {row['filename']: org_features_scaled[i] for i, row in org_df.iterrows()}
        
        self.dim_inorg = inorg_features_scaled.shape[1]  # Inorganic feature dimension
        self.dim_org = org_features_scaled.shape[1]        # Organic feature dimension
        
        # Initialize lists for storing tensor data
        X_prod_list = []     # Product features
        X_inorg_list = []    # Inorganic reactant feature sets
        M_inorg_list = []    # Inorganic reactant mole amounts
        X_org_list = []      # Organic reactant feature sets
        M_org_list = []      # Organic reactant mole amounts
        Y_list = []          # Target size
        Ops_scaled_list = [] # Standardized operation descriptors
        Ops_raw_list = []    # Raw operation descriptors
        
        # New: Record specific reactant names for later traceability
        self.inorg_reactant_names = []
        self.org_reactant_names = []
        self.product_names = []
        
        # 4. Iterate through each formulation, perform feature matching and classification
        for idx, (_, row) in enumerate(raw.iterrows()):
            prod = row['product']
            # Skip the sample if the product does not exist in the feature library
            if prod not in self.inorg_features:
                continue
                
            # Concatenate product inorganic features with morphology shape features
            prod_feat = np.concatenate([self.inorg_features[prod], shape_encoded[idx]])
            X_prod_list.append(prod_feat)
            Y_list.append(row['size'])
            Ops_scaled_list.append(ops_scaled[idx])
            Ops_raw_list.append(ops_raw[idx])
            
            self.product_names.append(prod)
            
            inorg_reacts = []
            inorg_amts = []
            inorg_names_current = []
            
            org_reacts = []
            org_amts = []
            org_names_current = []
            
            # Iterate through up to 10 reactants (reactant_1 to reactant_10)
            for i in range(1, 11):
                r = row[f'reactant_{i}']
                a = row[f'Amount_{i}']
                if pd.isna(r) or pd.isna(a):
                    continue
                
                # Determine whether the reactant is inorganic or organic, and store in the corresponding list
                if r in self.inorg_features:
                    inorg_reacts.append(self.inorg_features[r])
                    inorg_amts.append(a)
                    inorg_names_current.append(r)
                elif r in self.org_features:
                    org_reacts.append(self.org_features[r])
                    org_amts.append(a)
                    org_names_current.append(r)
                    
            X_inorg_list.append(inorg_reacts)
            M_inorg_list.append(inorg_amts)
            self.inorg_reactant_names.append(inorg_names_current)
            
            X_org_list.append(org_reacts)
            M_org_list.append(org_amts)
            self.org_reactant_names.append(org_names_current)
            
        # Use the maximum possible reactant count of 10 directly to completely avoid tensor dimension
        # inconsistency issues caused by batch or dataset splitting
        self.max_inorg = 10
        self.max_org = 10
        
        # 5. Build PyTorch tensors
        self.X_prod = torch.tensor(np.array(X_prod_list), dtype=torch.float32)
        self.Ops_scaled = torch.tensor(np.array(Ops_scaled_list), dtype=torch.float32)
        self.Ops_raw = torch.tensor(np.array(Ops_raw_list), dtype=torch.float32)
        Y_scaled = self.y_scaler.transform(np.array(Y_list).reshape(-1, 1))
        self.Y = torch.tensor(Y_scaled, dtype=torch.float32)
        
        N = len(X_prod_list)
        # Initialize tensors with Padding (num_samples, max_reactants, feature_dim)
        self.X_inorg = torch.zeros((N, self.max_inorg, self.dim_inorg), dtype=torch.float32)
        self.M_inorg = torch.zeros((N, self.max_inorg, 1), dtype=torch.float32)
        self.X_org = torch.zeros((N, self.max_org, self.dim_org), dtype=torch.float32)
        self.M_org = torch.zeros((N, self.max_org, 1), dtype=torch.float32)
        
        # Fill with actual data; keep remaining parts as 0 (i.e., padding value is 0)
        for i in range(N):
            for j in range(len(X_inorg_list[i])):
                self.X_inorg[i, j] = torch.tensor(X_inorg_list[i][j])
                self.M_inorg[i, j, 0] = M_inorg_list[i][j]
                
            for j in range(len(X_org_list[i])):
                self.X_org[i, j] = torch.tensor(X_org_list[i][j])
                self.M_org[i, j, 0] = M_org_list[i][j]
                
    def __len__(self):
        """Return dataset size"""
        return len(self.Y)
        
    def __getitem__(self, idx):
        """
        Get a single sample.
        Returns: (product features (including shape), inorganic features, inorganic moles,
                  organic features, organic moles, operation descriptors, raw operation descriptors, target size)
        """
        return (self.X_prod[idx], 
                self.X_inorg[idx], self.M_inorg[idx], 
                self.X_org[idx], self.M_org[idx], 
                self.Ops_scaled[idx], self.Ops_raw[idx],
                self.Y[idx])

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ds = NanoDataset(
        os.path.join(BASE_DIR, 'train_data.xlsx'),
        os.path.join(BASE_DIR, 'inorganic_descriptors_vif_filtered.csv'), 
        os.path.join(BASE_DIR, 'organic_descriptors_vif_filtered.csv')
    )
    print(f"Dataset size: {len(ds)}")