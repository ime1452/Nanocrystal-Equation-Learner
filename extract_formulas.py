import os
import torch
import pandas as pd
import numpy as np
import re
import copy
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from model import NanoEQLModel, EQLLayer

OPERATOR_LABELS = [
    'identity',
    'square',
    'sqrt',
    'cbrt',
    'inverse',
    'exp',
    'ln',
    'log10',
]

def evaluate_model(model, loader, y_scaler, device):
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in loader:
            x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled, ops_raw, y = [b.to(device) for b in batch]
            outputs = model(x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled)
            pred = outputs[0]
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())
            
    y_true_all = np.concatenate(y_true_list).reshape(-1, 1)
    y_pred_all = np.concatenate(y_pred_list).reshape(-1, 1)
    
    y_true_raw = y_scaler.inverse_transform(y_true_all).flatten()
    y_pred_raw = y_scaler.inverse_transform(y_pred_all).flatten()
    
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    mape = mean_absolute_percentage_error(y_true_raw, y_pred_raw)
    return mae, mape

def evaluate_pruned_branch(model, branch_name, keep_ratio, loader, y_scaler, device):
    orig_state = copy.deepcopy(model.state_dict())
    branch_module = getattr(model, branch_name)
    
    for module in branch_module.modules():
        if isinstance(module, EQLLayer):
            abs_w = module.weight.data.abs().cpu().numpy()
            percentile_val = max(0.0, min(100.0, (1.0 - keep_ratio) * 100.0))
            thresh = np.percentile(abs_w, percentile_val)
            thresh = max(thresh, 1e-5)
            
            with torch.no_grad():
                module.weight.data[module.weight.data.abs() <= thresh] = 0.0
                module.bias.data[module.bias.data.abs() <= thresh] = 0.0
                
    mae, mape = evaluate_model(model, loader, y_scaler, device)
    model.load_state_dict(orig_state)
    return mae, mape


def get_dynamic_threshold(weights, keep_ratio):
    abs_weights = np.abs(weights)
    percentile_val = max(0.0, min(100.0, (1.0 - keep_ratio) * 100.0))
    dynamic_threshold = np.percentile(abs_weights, percentile_val)
    return max(dynamic_threshold, 1e-5)


def count_layer_operator_usage(layer, keep_ratio):
    weights = layer.weight.detach().cpu().numpy()
    threshold = get_dynamic_threshold(weights, keep_ratio)

    n_in = layer.in_features
    n_funcs = layer.n_funcs
    op_labels = OPERATOR_LABELS[:n_funcs]

    counts = {op_name: 0 for op_name in op_labels}
    for f_idx, op_name in enumerate(op_labels):
        op_block = weights[:, f_idx * n_in:(f_idx + 1) * n_in]
        counts[op_name] = int(np.sum(np.abs(op_block) > threshold))

    return counts, float(threshold)


def count_branch_operator_usage(branch_module, keep_ratio):
    branch_counts = {op_name: 0 for op_name in OPERATOR_LABELS}
    layer_records = []

    for layer_name, module in branch_module.named_modules():
        if not isinstance(module, EQLLayer):
            continue

        layer_counts, threshold = count_layer_operator_usage(module, keep_ratio)
        layer_label = layer_name if layer_name else 'root'

        for op_name, count in layer_counts.items():
            branch_counts[op_name] += count
            layer_records.append({
                'Layer': layer_label,
                'Operator': op_name,
                'Usage_Count': int(count),
                'Dynamic_Threshold': threshold,
            })

    return branch_counts, layer_records

def compute_raw_complexity(formula_str, early_complexities=None):
    if not isinstance(formula_str, str) or formula_str.strip() in ["0", "(0)", "0.0"]:
        return 0
    c = 0
    c += formula_str.count('+')
    c += formula_str.count('-')
    star_count = formula_str.count('*')
    power_count = formula_str.count('**')
    c += (star_count - 2 * power_count)
    c += power_count
    c += formula_str.count('/')
    c += formula_str.count('exp')
    c += formula_str.count('sqrt')
    c += formula_str.count('abs')
    c += formula_str.count('sign')
    c += formula_str.count('ln')
    c += formula_str.count('log10')
    c += formula_str.count('cbrt')
    
    if early_complexities:
        matches = re.findall(r'EarlyZInorg_Dim\d+|EarlyZOrg_Dim\d+', formula_str)
        for match in matches:
            if match in early_complexities:
                c += early_complexities[match]
                
    return c

def extract_layer_formula(layer, in_names, keep_ratio=0.1):
    """
    Extract mathematical expressions from a single EQLLayer module.
    """
    weights = layer.weight.detach().cpu().numpy() 
    bias = layer.bias.detach().cpu().numpy()      
    
    n_in = layer.in_features
    n_out = layer.out_features
    n_funcs = layer.n_funcs
    
    dynamic_threshold = get_dynamic_threshold(weights, keep_ratio)
    
    if hasattr(layer, 'func_names'):
        func_names = layer.func_names
    else:
        func_names = [
            "{}",              
            "({})**2",         
            "sign({})*sqrt(abs({}))", 
            "cbrt({})",        
            "1/({})",          
            "exp({})",         
            "sign({})*ln(abs({}) + 1.0)",   
            "sign({})*log10(abs({}) + 1.0)" 
        ]
    
    out_formulas = []
    for i in range(n_out):
        terms = []
        for f_idx in range(n_funcs):
            for in_idx in range(n_in):
                w_idx = f_idx * n_in + in_idx
                w = weights[i, w_idx]
                
                if abs(w) > dynamic_threshold:
                    feat_name = in_names[in_idx]
                    term_str = func_names[f_idx].format(feat_name)
                    if w > 0:
                        terms.append(f"{w:.4f} * {term_str}")
                    else:
                        terms.append(f"- {abs(w):.4f} * {term_str}")
        
        b = bias[i]
        if abs(b) > dynamic_threshold:
            if b > 0:
                terms.append(f"{b:.4f}")
            else:
                terms.append(f"- {abs(b):.4f}")
                
        if not terms:
            out_formulas.append("0")
        else:
            out_formulas.append(" + ".join(terms).replace("+ -", "-"))
            
    return out_formulas

def expand_formula(formula_str, intermediate_dict, feature_scaler_dict=None):
    if not intermediate_dict:
        return formula_str
        
    expanded_str = formula_str
    pattern = re.compile(r'\bL\d+_N\d+\b')
    
    max_depth = 20
    for _ in range(max_depth):
        matches = pattern.findall(expanded_str)
        if not matches:
            break
            
        matches = list(set(matches))
        matches.sort(key=len, reverse=True)
        
        for match in matches:
            if match in intermediate_dict:
                sub_formula = intermediate_dict[match]
                expanded_str = expanded_str.replace(match, f"({sub_formula})")
    
    sympy_str = expanded_str
    
    if feature_scaler_dict:
        sorted_feats = sorted(feature_scaler_dict.keys(), key=lambda x: len(x), reverse=True)
        for feat_name in sorted_feats:
            center, scale = feature_scaler_dict[feat_name]
            inv_expr = f"(({feat_name} - {center:.4f}) / {scale:.4f})"
            
            escaped_feat = re.escape(feat_name)
            pattern = re.compile(r'(?<![a-zA-Z0-9_])' + escaped_feat + r'(?![a-zA-Z0-9_])')
            
            sympy_str = pattern.sub(inv_expr, sympy_str)
                
    return sympy_str

def extract_sequential_formula(seq_module, in_names, keep_ratio=0.1, log_lines=None, feature_scaler_dict=None, verbose=True):
    """
    Extract the complete mathematical formula for an nn.Sequential network containing multiple EQLLayer modules.
    """
    current_names = in_names
    intermediate_dict = {}
    
    for i, layer in enumerate(seq_module):
        if verbose:
            print(f"  -> Processing EQLLayer {i+1}/{len(seq_module)}...")
        out_names = []
        formulas = extract_layer_formula(layer, current_names, keep_ratio)
        for j, form in enumerate(formulas):
            if i < len(seq_module) - 1:
                name = f"L{i}_N{j}"
                log_lines.append(f"{name} = {form}")
                out_names.append(name)
                intermediate_dict[name] = form
            else:
                out_names.append(f"({form})")
        current_names = out_names
        
    final_form_list = current_names
    expanded_form_list = [expand_formula(f, intermediate_dict, feature_scaler_dict) for f in final_form_list]
    
    return final_form_list, expanded_form_list

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # ================= Configuration Area =================
    MODEL_FILENAME = 'eql_model.pth'
    METRICS_FILENAME = 'stage1_metrics.txt'
    # Branch Retention Ratio Configuration (Retain the weights of the top X% with the highest absolute values at each level; for example, 0.1 represents 10%)
    KEEP_RATIOS = {
        'h_prod': 0.003,
        'early_g_inorg':0.007,
        'early_g_org': 0.003,
        'g_joint': 0.003,
        'g_ops': 0.008,
        'g_inorg': 0.01,  # legacy
        'g_org': 0.01,    # legacy
        'default': 0.01
    }
    PREDICTOR_TYPE = 'glm'
    
    # ------------------ Module Output Switches ------------------
    GENERATE_EQL_WEIGHTS_DISTRIBUTION = False      # [2.8/5] eql_weights_distribution.csv
    GENERATE_FORMULAS_JSON = False                 # [3.5/5] formulas_by_keep_ratio.json
    GENERATE_PRUNING_METRICS_CSV = False             # [3.6/5] pruning_metrics_0.1_to_100.csv
    GENERATE_OPERATOR_USAGE_COUNTS_CSV = False     # [3.7/5] operator_usage_counts_0.1_to_100.csv
    GENERATE_STAGE1_FORMULAS_TXT = True           # [4/5] stage1_formulas.txt
    GENERATE_INTERMEDIATE_WEIGHTS_CSV = False      # [5/5] intermediate_weights.csv
    # ================================================
    
    print("\n[1/5] Loading dataset features...")
    inorg_df = pd.read_csv(os.path.join(BASE_DIR, 'inorganic_descriptors_vif_filtered.csv'))
    org_df = pd.read_csv(os.path.join(BASE_DIR, 'organic_descriptors_vif_filtered.csv'))
    
    feature_recon_path = os.path.join(BASE_DIR, 'Feature_Reconstruction.csv')
    
    model_path = os.path.join(BASE_DIR, MODEL_FILENAME)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print("[2/5] Loading model weights and inferring architecture...")
    state_dict = torch.load(model_path)
    
    n_funcs = 8 
    
    full_inorg_names = list(inorg_df.drop('filename', axis=1).columns)
    full_org_names = list(org_df.drop('filename', axis=1).columns)
    
    if 'early_g_inorg.0.weight' in state_dict:
        actual_dim_inorg = state_dict['early_g_inorg.0.weight'].shape[1] // n_funcs
        actual_dim_org = state_dict['early_g_org.0.weight'].shape[1] // n_funcs
    elif 'g_inorg.0.weight' in state_dict:
        pna_multiplier = 8 
        actual_dim_inorg = state_dict['g_inorg.0.weight'].shape[1] // (n_funcs * pna_multiplier)
        actual_dim_org = state_dict['g_org.0.weight'].shape[1] // (n_funcs * pna_multiplier)
    else:
        actual_dim_inorg = len(full_inorg_names)
        actual_dim_org = len(full_org_names)
    
    if actual_dim_inorg < len(full_inorg_names) and os.path.exists(feature_recon_path):
        print("  Detected model trained with TOP 10 features. Applying same filtering to feature names...")
        feature_recon_df = pd.read_csv(feature_recon_path)
        
        prod_inorg_features = feature_recon_df[feature_recon_df['Is a product feature?'] == 'yes']['Raw feature name'].tolist()
        react_inorg_features = feature_recon_df[feature_recon_df['Is a inorganic reactant feature?'] == 'yes']['Raw feature name'].tolist()
        react_org_features = feature_recon_df[feature_recon_df['Is a organic reactant feature?'] == 'yes']['Raw feature name'].tolist()
        
        top_inorg_features = list(set(prod_inorg_features + react_inorg_features))
        top_org_features = list(set(react_org_features))
        
        inorg_names = [col for col in top_inorg_features if col in inorg_df.columns]
        org_names = [col for col in top_org_features if col in org_df.columns]
    else:
        print("  Detected model trained with ALL features.")
        inorg_names = full_inorg_names
        org_names = full_org_names
    
    if len(inorg_names) != actual_dim_inorg:
        print(f"  Warning: Extracted inorg names ({len(inorg_names)}) does not match model weights ({actual_dim_inorg}). Using placeholders.")
        inorg_names = [f"InorgFeature_{i}" for i in range(actual_dim_inorg)]
        
    if len(org_names) != actual_dim_org:
        print(f"  Warning: Extracted org names ({len(org_names)}) does not match model weights ({actual_dim_org}). Using placeholders.")
        org_names = [f"OrgFeature_{i}" for i in range(actual_dim_org)]
    
    mapping_path = os.path.join(BASE_DIR, 'morphology_mapping.csv')
    try:
        shape_df = pd.read_csv(mapping_path, sep='\t')
        if 'Circularity' not in shape_df.columns:
            shape_df = pd.read_csv(mapping_path, sep=',')
        defined_shapes = ['Circularity', 'Aspect_Ratio', 'Vertices']
    except Exception:
        defined_shapes = ['Circularity', 'Aspect_Ratio', 'Vertices']
                
    prod_names = inorg_names + defined_shapes
    
    import ast
    metrics_path = os.path.join(BASE_DIR, METRICS_FILENAME)
    best_params = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            content = f.read()
            if 'Best Parameters:\n' in content:
                dict_str = content.split('Best Parameters:\n')[1].strip()
                dict_str = dict_str.split('\n')[0].strip() # Fix parsing by taking only the first line
                try:
                    best_params = ast.literal_eval(dict_str)
                except Exception as e:
                    print(f"Error parsing best params: {e}")
                    
    actual_early_dim = best_params.get('early_dim', 16)
    actual_eql_depth = best_params.get('eql_depth', 3)
    actual_mlp_depth = best_params.get('mlp_depth', 2)
    actual_hidden_dim = best_params.get('hidden_dim', 16)
    actual_mlp_dropout = best_params.get('mlp_dropout', 0.0)
    actual_activation = best_params.get('activation', 'relu')
    actual_predictor_type = best_params.get('predictor_type', PREDICTOR_TYPE)
    
    model = NanoEQLModel(
        dim_inorg=len(inorg_names), 
        dim_org=len(org_names), 
        dim_prod=len(prod_names),
        dim_ops=3, 
        hidden_dim=actual_hidden_dim,
        early_dim=actual_early_dim,
        latent_dim=1,
        eql_depth=actual_eql_depth,
        mlp_depth=actual_mlp_depth,
        mlp_dropout=actual_mlp_dropout,
        activation=actual_activation,
        predictor_type=actual_predictor_type
    )
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: Failed to load state_dict strictly. Details: {e}")
        return
    model.eval()
    
    if GENERATE_EQL_WEIGHTS_DISTRIBUTION:
        print("\n[2.8/5] Exporting all EQL network weights to CSV for distribution analysis...")
        from model import EQLLayer
        weight_records = []
        for name, module in model.named_modules():
            if isinstance(module, EQLLayer):
                branch = name.split('.')[0] if '.' in name else name
                w_vals = module.weight.detach().cpu().numpy().flatten()
                b_vals = module.bias.detach().cpu().numpy().flatten()
                
                for val in w_vals:
                    weight_records.append({'Branch': branch, 'Layer': name, 'Type': 'weight', 'Value': val})
                for val in b_vals:
                    weight_records.append({'Branch': branch, 'Layer': name, 'Type': 'bias', 'Value': val})
                    
        if weight_records:
            weights_df = pd.DataFrame(weight_records)
            weights_csv_path = os.path.join(BASE_DIR, 'eql_weights_distribution.csv')
            weights_df.to_csv(weights_csv_path, index=False)
            print(f"  Saved {len(weight_records)} weight values to {weights_csv_path}")
    else:
        print("\n[2.8/5] Skipped exporting EQL weights distribution (Disabled in config).")

    log_lines = []
    
    def expand_pna_names(base_names):
        expanded = []
        for name in base_names:
            expanded.append(name) 
            expanded.append(f"{name}_Sum")
            expanded.append(f"{name}_Mean")
            expanded.append(f"{name}_Range")
            expanded.append(f"{name}_Std")
            expanded.append(f"{name}_Rss")
            expanded.append(f"{name}_Max")
            expanded.append(f"{name}_Min")
        return expanded
        
    pna_inorg_names = expand_pna_names(inorg_names)
    pna_org_names = expand_pna_names(org_names)
    
    print("\n[2.5/5] Loading Dataset to get Scalers for Inverse Transformation...")
    from dataset import NanoDataset
    train_path = os.path.join(BASE_DIR, 'train_data.xlsx')
    test_path = os.path.join(BASE_DIR, 'test_data.xlsx')
    train_ds = NanoDataset(
        train_path,
        os.path.join(BASE_DIR, 'inorganic_descriptors_vif_filtered.csv'),
        os.path.join(BASE_DIR, 'organic_descriptors_vif_filtered.csv'),
        fit_scaler=True,
        target_transform='robust',
    )
    test_ds = NanoDataset(
        test_path,
        os.path.join(BASE_DIR, 'inorganic_descriptors_vif_filtered.csv'),
        os.path.join(BASE_DIR, 'organic_descriptors_vif_filtered.csv'),
        fit_scaler=False,
        scalers=(
            train_ds.inorg_scaler,
            train_ds.org_scaler,
            train_ds.y_scaler,
            train_ds.ops_imputer,
            train_ds.ops_scaler,
            train_ds.shape_encoder,
        ),
        target_transform='robust',
    )
    
    feature_scaler_dict = {}
    for i, name in enumerate(inorg_names):
        feature_scaler_dict[name] = (train_ds.inorg_scaler.center_[i], train_ds.inorg_scaler.scale_[i])
    for i, name in enumerate(org_names):
        feature_scaler_dict[name] = (train_ds.org_scaler.center_[i], train_ds.org_scaler.scale_[i])
    ops_names = ['T_inj', 'T_rea', 't']
    for i, name in enumerate(ops_names):
        feature_scaler_dict[name] = (train_ds.ops_scaler.center_[i], train_ds.ops_scaler.scale_[i])
    for name in defined_shapes:
        feature_scaler_dict[name] = (0.0, 1.0)
        
    print("\n[3/5] Extracting sequential formulas from branches...")
    log_lines.append("="*50)
    log_lines.append("Extracting Z_product Formula (h_prod network)")
    log_lines.append("="*50)
    
    if hasattr(model, 'h_prod'):
        print("-> Extracting h_prod...")
        z_prod_forms, z_prod_simps = extract_sequential_formula(model.h_prod, prod_names, KEEP_RATIOS.get('h_prod', KEEP_RATIOS['default']), log_lines, feature_scaler_dict)
        z_prod_form = z_prod_forms[0]
        z_prod_simp = z_prod_simps[0]
        log_lines.append(f"\nFinal Z_product_base = {z_prod_form}\n")
        log_lines.append(f"-> Expanded Z_product_base = {z_prod_simp}\n")
    
    if hasattr(model, 'early_g_inorg'):
        print("-> Extracting early_g_inorg...")
        log_lines.append("="*50)
        log_lines.append("Extracting Early_Z_inorg Formula (early_g_inorg network)")
        log_lines.append("="*50)
        early_z_inorg_forms, early_z_inorg_simps = extract_sequential_formula(model.early_g_inorg, inorg_names, KEEP_RATIOS.get('early_g_inorg', KEEP_RATIOS['default']), log_lines, feature_scaler_dict)
        for i, (form, simp) in enumerate(zip(early_z_inorg_forms, early_z_inorg_simps)):
            log_lines.append(f"\nFinal Early_Z_inorg_Dim{i} = {form}\n")
            log_lines.append(f"-> Expanded Early_Z_inorg_Dim{i} = {simp}\n")
        
        print("-> Extracting early_g_org...")
        log_lines.append("="*50)
        log_lines.append("Extracting Early_Z_org Formula (early_g_org network)")
        log_lines.append("="*50)
        early_z_org_forms, early_z_org_simps = extract_sequential_formula(model.early_g_org, org_names, KEEP_RATIOS.get('early_g_org', KEEP_RATIOS['default']), log_lines, feature_scaler_dict)
        for i, (form, simp) in enumerate(zip(early_z_org_forms, early_z_org_simps)):
            log_lines.append(f"\nFinal Early_Z_org_Dim{i} = {form}\n")
            log_lines.append(f"-> Expanded Early_Z_org_Dim{i} = {simp}\n")
        
        pna_early_inorg_names = []
        pna_early_org_names = []
        for i in range(actual_early_dim):
            base_inorg = f"EarlyZInorg_Dim{i}"
            base_org = f"EarlyZOrg_Dim{i}"
            # 只保留 7 个池化统计量的名字
            for stat in ["Sum", "Mean", "Range", "Std", "Rss", "Max", "Min"]:
                pna_early_inorg_names.append(f"{base_inorg}_{stat}")
                pna_early_org_names.append(f"{base_org}_{stat}")
                
        print("-> Extracting g_joint...")
        log_lines.append("="*50)
        log_lines.append("Extracting Z_joint Formula (g_joint network from PNA features)")
        log_lines.append("="*50)
        joint_input_names = pna_early_inorg_names + pna_early_org_names
        z_joint_forms, z_joint_simps = extract_sequential_formula(model.g_joint, joint_input_names, KEEP_RATIOS.get('g_joint', KEEP_RATIOS['default']), log_lines, None)
        z_joint_form = z_joint_forms[0]
        z_joint_simp = z_joint_simps[0]
        log_lines.append(f"\nFinal Z_joint = {z_joint_form}\n")
        log_lines.append(f"-> Expanded Z_joint = {z_joint_simp}\n")
    elif hasattr(model, 'g_inorg'):
        print("-> Extracting g_inorg (legacy)...")
        log_lines.append("="*50)
        log_lines.append("Extracting Z_inorg Formula (g_inorg network with PNA features)")
        log_lines.append("="*50)
        z_inorg_forms, z_inorg_simps = extract_sequential_formula(model.g_inorg, pna_inorg_names, KEEP_RATIOS.get('g_inorg', KEEP_RATIOS['default']), log_lines, None)
        z_inorg_form = z_inorg_forms[0]
        z_inorg_simp = z_inorg_simps[0]
        log_lines.append(f"\nFinal Z_inorg = {z_inorg_form}\n")
        log_lines.append(f"-> Expanded Z_inorg = {z_inorg_simp}\n")
        
        print("-> Extracting g_org (legacy)...")
        log_lines.append("="*50)
        log_lines.append("Extracting Z_org Formula (g_org network with PNA features)")
        log_lines.append("="*50)
        z_org_forms, z_org_simps = extract_sequential_formula(model.g_org, pna_org_names, KEEP_RATIOS.get('g_org', KEEP_RATIOS['default']), log_lines, None)
        z_org_form = z_org_forms[0]
        z_org_simp = z_org_simps[0]
        log_lines.append(f"\nFinal Z_org = {z_org_form}\n")
        log_lines.append(f"-> Expanded Z_org = {z_org_simp}\n")
        
    if hasattr(model, 'g_ops'):
        print("-> Extracting g_ops...")
        log_lines.append("="*50)
        log_lines.append("Extracting Z_ops Formula (g_ops network)")
        log_lines.append("="*50)
        ops_names = ['T_inj', 'T_rea', 't']
        z_ops_forms, z_ops_simps = extract_sequential_formula(model.g_ops, ops_names, KEEP_RATIOS.get('g_ops', KEEP_RATIOS['default']), log_lines, feature_scaler_dict)
        z_ops_form = z_ops_forms[0]
        z_ops_simp = z_ops_simps[0]
        log_lines.append(f"\nFinal Z_ops = {z_ops_form}\n")
        log_lines.append(f"-> Expanded Z_ops = {z_ops_simp}\n")
        
    print("-> Extracting Final Top Predictor Formula...")
    log_lines.append("="*50)
    log_lines.append("Extracting Final Size Formula (top_predictor)")
    log_lines.append("="*50)
    
    if actual_predictor_type == 'glm':
        weights = model.top_predictor.weight.detach().cpu().numpy()[0]
        bias = model.top_predictor.bias.detach().cpu().numpy()[0]
        
        branch_names = []
        if model.use_h_prod:
            branch_names.append('Z_product')
        if model.use_g_joint:
            branch_names.append('Z_joint')
        if model.use_g_ops:
            branch_names.append('Z_ops')
            
        terms = []
        for w, name in zip(weights, branch_names):
            terms.append(f"{w:.4f} * {name}")
            
        if bias > 0:
            terms.append(f"+ {bias:.4f}")
        elif bias < 0:
            terms.append(f"- {abs(bias):.4f}")
            
        final_formula = " ".join(terms).replace("+ -", "-")
        log_lines.append(f"\nFinal Size (Scaled) = {final_formula}\n")
        print(f"  Extracted Scaled Size Formula: {final_formula}")
        
        # --- Only transform the Size (Y) ---
        try:
            import sympy as sp
            from sympy import Float
            sym_Z_prod = sp.Symbol('Z_product')
            sym_Z_joint = sp.Symbol('Z_joint')
            sym_Z_ops = sp.Symbol('Z_ops')
            
            # 原始 scaled 表达式
            expr_scaled = weights[branch_names.index('Z_product')] * sym_Z_prod if 'Z_product' in branch_names else 0
            expr_scaled += weights[branch_names.index('Z_joint')] * sym_Z_joint if 'Z_joint' in branch_names else 0
            expr_scaled += weights[branch_names.index('Z_ops')] * sym_Z_ops if 'Z_ops' in branch_names else 0
            expr_scaled += bias
            
            if getattr(train_ds.y_scaler, 'method', None) == 'box-cox':
                y_mean = float(train_ds.y_scaler.standard_mean_[0])
                y_scale = float(train_ds.y_scaler.standard_scale_[0])
                y_lambda = float(train_ds.y_scaler.lambdas_[0])

                expr_boxcox = y_scale * expr_scaled + y_mean
                if abs(y_lambda) < 1e-8:
                    expr_unscaled_y = sp.exp(expr_boxcox)
                else:
                    expr_unscaled_y = (y_lambda * expr_boxcox + 1.0) ** (1.0 / y_lambda)
            else:
                y_center = float(train_ds.y_scaler.center_[0])
                y_scale = float(train_ds.y_scaler.scale_[0])

                # Y inverse transform: Y_raw = Y_scale * Y_scaled + Y_center
                expr_unscaled_y = y_scale * expr_scaled + y_center
            
            # Round to 4 decimal places
            expr_unscaled_y = expr_unscaled_y.xreplace({n: Float(n, 4) for n in expr_unscaled_y.atoms(Float)})
            
            final_raw_formula = str(expr_unscaled_y).replace('**', '^')
            log_lines.append(f"Final Size (Raw Physical Scale) = {final_raw_formula}\n")
            print(f"  Extracted Raw Physical Size Formula: {final_raw_formula}")
            
            eval_path = os.path.join(BASE_DIR, 'eval_comp6.py')
            if os.path.exists(eval_path):
                with open(eval_path, 'w') as f:
                    f.write(f"""import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

test_df = pd.read_csv('/home/ubuntu/project/nano-sr/test_features.csv')

Z_joint = test_df['Z_joint'].values
Z_ops = test_df['Z_ops'].values
Z_product = test_df['Z_product'].values
y_true = test_df['Size'].values

y_pred = {str(expr_unscaled_y)}

print("R2: ", r2_score(y_true, y_pred))
""")
                print("  [Auto-Fix] Updated eval_comp6.py with the Raw Physical Scale formula.")
                
        except Exception as e:
            print(f"  Warning: Failed to inverse transform the formula. {e}")
            
    else:
        msg = f"[Note] top_predictor is of type '{actual_predictor_type}', which is not a simple linear GLM. Cannot extract a simple weighted formula."
        log_lines.append(f"\n{msg}\n")
        print(f"  {msg}")
        
    # --- [3.5/5] or [3.6] ---
    if GENERATE_FORMULAS_JSON or GENERATE_PRUNING_METRICS_CSV:
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

    if GENERATE_FORMULAS_JSON:
        print("\n[3.5/5] Extracting formulas and evaluating pruning errors across KEEP_RATIOS (0.1% to 2%)...")
        
        base_mae, base_mape = evaluate_model(model, train_loader, train_ds.y_scaler, device)
        print(f"  -> Base Model Performance (Unpruned): MAE={base_mae:.4f}, MAPE={base_mape:.4f}")
        
        keep_ratios_test = np.arange(0.001, 0.021, 0.001)
        ratio_records = []
        for kr in keep_ratios_test:
            kr_str = f"{kr*100:.1f}%"
            record = {'Keep_Ratio': kr_str}
            dummy_logs = []
            early_complexities = {}
            
            if hasattr(model, 'h_prod'):
                _, simps = extract_sequential_formula(model.h_prod, prod_names, kr, dummy_logs, feature_scaler_dict, verbose=False)
                record['h_prod'] = " | ".join(simps) if len(simps) > 1 else simps[0]
                record['h_prod_Complexity'] = compute_raw_complexity(record['h_prod'])
                mae, mape = evaluate_pruned_branch(model, 'h_prod', kr, train_loader, train_ds.y_scaler, device)
                record['h_prod_MAE'] = float(mae)
                record['h_prod_MAPE'] = float(mape)
                
            if hasattr(model, 'early_g_inorg'):
                _, simps = extract_sequential_formula(model.early_g_inorg, inorg_names, kr, dummy_logs, feature_scaler_dict, verbose=False)
                early_inorg_dict = {}
                early_inorg_comp = {}
                for i, simp in enumerate(simps):
                    k = f"EarlyZInorg_Dim{i}"
                    early_inorg_dict[k] = simp
                    early_inorg_comp[k] = compute_raw_complexity(simp)
                    early_complexities[k] = early_inorg_comp[k]
                record['early_g_inorg'] = early_inorg_dict
                record['early_g_inorg_Complexity'] = early_inorg_comp
                mae, mape = evaluate_pruned_branch(model, 'early_g_inorg', kr, train_loader, train_ds.y_scaler, device)
                record['early_g_inorg_MAE'] = float(mae)
                record['early_g_inorg_MAPE'] = float(mape)
                
                _, simps = extract_sequential_formula(model.early_g_org, org_names, kr, dummy_logs, feature_scaler_dict, verbose=False)
                early_org_dict = {}
                early_org_comp = {}
                for i, simp in enumerate(simps):
                    k = f"EarlyZOrg_Dim{i}"
                    early_org_dict[k] = simp
                    early_org_comp[k] = compute_raw_complexity(simp)
                    early_complexities[k] = early_org_comp[k]
                record['early_g_org'] = early_org_dict
                record['early_g_org_Complexity'] = early_org_comp
                mae, mape = evaluate_pruned_branch(model, 'early_g_org', kr, train_loader, train_ds.y_scaler, device)
                record['early_g_org_MAE'] = float(mae)
                record['early_g_org_MAPE'] = float(mape)
                
                joint_input_names = pna_early_inorg_names + pna_early_org_names
                _, simps = extract_sequential_formula(model.g_joint, joint_input_names, kr, dummy_logs, None, verbose=False)
                record['g_joint'] = " | ".join(simps) if len(simps) > 1 else simps[0]
                record['g_joint_Complexity'] = compute_raw_complexity(record['g_joint'], early_complexities)
                mae, mape = evaluate_pruned_branch(model, 'g_joint', kr, train_loader, train_ds.y_scaler, device)
                record['g_joint_MAE'] = float(mae)
                record['g_joint_MAPE'] = float(mape)
                
            elif hasattr(model, 'g_inorg'):
                _, simps = extract_sequential_formula(model.g_inorg, pna_inorg_names, kr, dummy_logs, None, verbose=False)
                record['g_inorg'] = " | ".join(simps) if len(simps) > 1 else simps[0]
                record['g_inorg_Complexity'] = compute_raw_complexity(record['g_inorg'])
                mae, mape = evaluate_pruned_branch(model, 'g_inorg', kr, train_loader, train_ds.y_scaler, device)
                record['g_inorg_MAE'] = float(mae)
                record['g_inorg_MAPE'] = float(mape)
                
                _, simps = extract_sequential_formula(model.g_org, pna_org_names, kr, dummy_logs, None, verbose=False)
                record['g_org'] = " | ".join(simps) if len(simps) > 1 else simps[0]
                record['g_org_Complexity'] = compute_raw_complexity(record['g_org'])
                mae, mape = evaluate_pruned_branch(model, 'g_org', kr, train_loader, train_ds.y_scaler, device)
                record['g_org_MAE'] = float(mae)
                record['g_org_MAPE'] = float(mape)
                
            if hasattr(model, 'g_ops'):
                _, simps = extract_sequential_formula(model.g_ops, ops_names, kr, dummy_logs, feature_scaler_dict, verbose=False)
                record['g_ops'] = " | ".join(simps) if len(simps) > 1 else simps[0]
                record['g_ops_Complexity'] = compute_raw_complexity(record['g_ops'])
                mae, mape = evaluate_pruned_branch(model, 'g_ops', kr, train_loader, train_ds.y_scaler, device)
                record['g_ops_MAE'] = float(mae)
                record['g_ops_MAPE'] = float(mape)
                
            ratio_records.append(record)
            
        import json
        ratios_json_path = os.path.join(BASE_DIR, 'formulas_by_keep_ratio.json')
        with open(ratios_json_path, 'w', encoding='utf-8') as f:
            json.dump(ratio_records, f, ensure_ascii=False, indent=4)
        print(f"  Saved formulas across different KEEP_RATIOS to {ratios_json_path}")
    else:
        print("\n[3.5/5] Skipped extracting formulas and evaluating pruning errors across KEEP_RATIOS (0.1% to 2%) (Disabled in config).")
    
    if GENERATE_PRUNING_METRICS_CSV:
        print("\n[3.6/5] Evaluating pruning performance across KEEP_RATIOS (0.1% to 100%)...")
        metrics_records = []
        # 0.1% to 100% with step 0.1%
        keep_ratios_full = np.arange(0.001, 1.001, 0.001)
        
        from tqdm import tqdm
        for kr in tqdm(keep_ratios_full, desc="Evaluating pruning MAE/MAPE"):
            kr_str = f"{kr*100:.1f}%"
            record = {'Keep_Ratio': kr_str}
            
            if hasattr(model, 'h_prod'):
                train_mae, train_mape = evaluate_pruned_branch(model, 'h_prod', kr, train_loader, train_ds.y_scaler, device)
                test_mae, test_mape = evaluate_pruned_branch(model, 'h_prod', kr, test_loader, train_ds.y_scaler, device)
                record['h_prod_Train_MAE'] = float(train_mae)
                record['h_prod_Train_MAPE'] = float(train_mape)
                record['h_prod_Test_MAE'] = float(test_mae)
                record['h_prod_Test_MAPE'] = float(test_mape)
                
            if hasattr(model, 'early_g_inorg'):
                train_mae, train_mape = evaluate_pruned_branch(model, 'early_g_inorg', kr, train_loader, train_ds.y_scaler, device)
                test_mae, test_mape = evaluate_pruned_branch(model, 'early_g_inorg', kr, test_loader, train_ds.y_scaler, device)
                record['early_g_inorg_Train_MAE'] = float(train_mae)
                record['early_g_inorg_Train_MAPE'] = float(train_mape)
                record['early_g_inorg_Test_MAE'] = float(test_mae)
                record['early_g_inorg_Test_MAPE'] = float(test_mape)
                
                train_mae, train_mape = evaluate_pruned_branch(model, 'early_g_org', kr, train_loader, train_ds.y_scaler, device)
                test_mae, test_mape = evaluate_pruned_branch(model, 'early_g_org', kr, test_loader, train_ds.y_scaler, device)
                record['early_g_org_Train_MAE'] = float(train_mae)
                record['early_g_org_Train_MAPE'] = float(train_mape)
                record['early_g_org_Test_MAE'] = float(test_mae)
                record['early_g_org_Test_MAPE'] = float(test_mape)
                
                train_mae, train_mape = evaluate_pruned_branch(model, 'g_joint', kr, train_loader, train_ds.y_scaler, device)
                test_mae, test_mape = evaluate_pruned_branch(model, 'g_joint', kr, test_loader, train_ds.y_scaler, device)
                record['g_joint_Train_MAE'] = float(train_mae)
                record['g_joint_Train_MAPE'] = float(train_mape)
                record['g_joint_Test_MAE'] = float(test_mae)
                record['g_joint_Test_MAPE'] = float(test_mape)
                
            elif hasattr(model, 'g_inorg'):
                train_mae, train_mape = evaluate_pruned_branch(model, 'g_inorg', kr, train_loader, train_ds.y_scaler, device)
                test_mae, test_mape = evaluate_pruned_branch(model, 'g_inorg', kr, test_loader, train_ds.y_scaler, device)
                record['g_inorg_Train_MAE'] = float(train_mae)
                record['g_inorg_Train_MAPE'] = float(train_mape)
                record['g_inorg_Test_MAE'] = float(test_mae)
                record['g_inorg_Test_MAPE'] = float(test_mape)
                
                train_mae, train_mape = evaluate_pruned_branch(model, 'g_org', kr, train_loader, train_ds.y_scaler, device)
                test_mae, test_mape = evaluate_pruned_branch(model, 'g_org', kr, test_loader, train_ds.y_scaler, device)
                record['g_org_Train_MAE'] = float(train_mae)
                record['g_org_Train_MAPE'] = float(train_mape)
                record['g_org_Test_MAE'] = float(test_mae)
                record['g_org_Test_MAPE'] = float(test_mape)
                
            if hasattr(model, 'g_ops'):
                train_mae, train_mape = evaluate_pruned_branch(model, 'g_ops', kr, train_loader, train_ds.y_scaler, device)
                test_mae, test_mape = evaluate_pruned_branch(model, 'g_ops', kr, test_loader, train_ds.y_scaler, device)
                record['g_ops_Train_MAE'] = float(train_mae)
                record['g_ops_Train_MAPE'] = float(train_mape)
                record['g_ops_Test_MAE'] = float(test_mae)
                record['g_ops_Test_MAPE'] = float(test_mape)
                
            metrics_records.append(record)
            
        df_metrics = pd.DataFrame(metrics_records)
        metrics_csv_path = os.path.join(BASE_DIR, 'pruning_metrics_0.1_to_100.csv')
        df_metrics.to_csv(metrics_csv_path, index=False)
        print(f"  Saved full range pruning metrics to {metrics_csv_path}")
    else:
        print("\n[3.6/5] Skipped evaluating pruning performance across KEEP_RATIOS (0.1% to 100%) (Disabled in config).")

    if GENERATE_OPERATOR_USAGE_COUNTS_CSV:
        print("\n[3.7/5] Counting operator usage across KEEP_RATIOS (0.1% to 100%) directly from model weights...")
        usage_records = []
        branch_names_to_check = []

        if hasattr(model, 'h_prod'):
            branch_names_to_check.append('h_prod')
        if hasattr(model, 'early_g_inorg'):
            branch_names_to_check.extend(['early_g_inorg', 'early_g_org', 'g_joint'])
        elif hasattr(model, 'g_inorg'):
            branch_names_to_check.extend(['g_inorg', 'g_org'])
        if hasattr(model, 'g_ops'):
            branch_names_to_check.append('g_ops')

        keep_ratios_full = np.arange(0.001, 1.001, 0.001)

        from tqdm import tqdm
        for kr in tqdm(keep_ratios_full, desc="Counting operator usage"):
            for branch_name in branch_names_to_check:
                branch_module = getattr(model, branch_name)
                branch_counts, layer_records = count_branch_operator_usage(branch_module, kr)

                for op_name in OPERATOR_LABELS:
                    usage_records.append({
                        'Keep_Ratio': f"{kr*100:.1f}%",
                        'Keep_Ratio_Value': float(kr),
                        'Branch': branch_name,
                        'Layer': 'ALL',
                        'Operator': op_name,
                        'Usage_Count': int(branch_counts.get(op_name, 0)),
                    })

                for layer_record in layer_records:
                    usage_records.append({
                        'Keep_Ratio': f"{kr*100:.1f}%",
                        'Keep_Ratio_Value': float(kr),
                        'Branch': branch_name,
                        'Layer': layer_record['Layer'],
                        'Operator': layer_record['Operator'],
                        'Usage_Count': int(layer_record['Usage_Count']),
                        'Dynamic_Threshold': float(layer_record['Dynamic_Threshold']),
                    })

        usage_df = pd.DataFrame(usage_records)
        usage_csv_path = os.path.join(BASE_DIR, 'operator_usage_counts_0.1_to_100.csv')
        usage_df.to_csv(usage_csv_path, index=False)
        print(f"  Saved operator usage counts to {usage_csv_path}")
    else:
        print("\n[3.7/5] Skipped counting operator usage across KEEP_RATIOS (0.1% to 100%) (Disabled in config).")
    
    if GENERATE_STAGE1_FORMULAS_TXT:
        print("\n[4/5] Saving formulas to txt...")
        out_file = os.path.join(BASE_DIR, 'stage1_formulas.txt')
        with open(out_file, 'w') as f:
            f.write("\n".join(log_lines))
            f.write("\n\nNote: Adjust the KEEP_RATIOS dictionary in extract_formulas.py to control sparsity per branch.\n")
            
        print(f"  All formulas successfully extracted and saved to {out_file}")
    else:
        print("\n[4/5] Skipped saving formulas to txt (Disabled in config).")
    
    if not GENERATE_INTERMEDIATE_WEIGHTS_CSV:
        print("\n[5/5] Skipped running inference to extract intermediate variables to CSV (Disabled in config).")
        print("\n🎉 DONE! All processes finished successfully.")
        return

    print("\n[5/5] Running inference to extract intermediate variables to CSV...")
    try:
        model.eval()
        
        records = []
        
        with torch.no_grad():
            total_batches = len(train_loader)
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                    print(f"  -> Processing batch {batch_idx + 1}/{total_batches}...")
                    
                x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled, ops_raw, y = [b.to(device) for b in batch]
                pred, z_prod, z_inorg_out, z_org_out, z_joint, z_ops, attn_params = model(x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled)
                
                y_np = y.cpu().numpy()
                size_raw = train_ds.y_scaler.inverse_transform(y_np).flatten()
                ops_raw_np = ops_raw.cpu().numpy()
                
                B = x_prod.size(0)
                for i in range(B):
                    global_idx = batch_idx * 128 + i
                    T_inj = ops_raw_np[i, 0]
                    T_rea = ops_raw_np[i, 1]
                    t_time = ops_raw_np[i, 2]
                    size = size_raw[i]
                    
                    product_name = train_ds.product_names[global_idx]
                    inorg_names_list = train_ds.inorg_reactant_names[global_idx]
                    org_names_list = train_ds.org_reactant_names[global_idx]
                    
                    if attn_params.get('inorg_interm') is not None:
                        alpha_inorg = attn_params['alpha_inorg'][i].item()
                        beta_inorg = attn_params['beta_inorg'][i].item()
                        inorg_interm = attn_params['inorg_interm']
                        early_z_inorg = attn_params['early_z_inorg']
                        
                        N = x_inorg.size(1)
                        for j in range(N):
                            m_val = m_inorg[i, j, 0].item()
                            if m_val > 1e-6:
                                reactant_name = inorg_names_list[j] if j < len(inorg_names_list) else 'Unknown'
                                record = {
                                    'Sample_Idx': global_idx,
                                    'Product_Name': product_name,
                                    'Type': 'Inorganic',
                                    'Reactant_Idx': j,
                                    'Reactant_Name': reactant_name,
                                    'T_inj': T_inj,
                                    'T_rea': T_rea,
                                    'Time': t_time,
                                    'Size': size,
                                    'alpha': alpha_inorg,
                                    'beta': beta_inorg,
                                    'm_tilde': inorg_interm['m_tilde'][i, j, 0].item(),
                                    'a_tilde': inorg_interm['a_tilde'][i, j, 0].item(),
                                    'a': inorg_interm['a'][i, j, 0].item(),
                                    'w': inorg_interm['w'][i, j, 0].item(),
                                }
                                for d in range(actual_early_dim):
                                    record[f'z_early_{d}'] = early_z_inorg[i, j, d].item()
                                records.append(record)
                                
                    if attn_params.get('org_interm') is not None:
                        alpha_org = attn_params['alpha_org'][i].item()
                        beta_org = attn_params['beta_org'][i].item()
                        org_interm = attn_params['org_interm']
                        early_z_org = attn_params['early_z_org']
                        
                        M = x_org.size(1)
                        for j in range(M):
                            m_val = m_org[i, j, 0].item()
                            if m_val > 1e-6:
                                reactant_name = org_names_list[j] if j < len(org_names_list) else 'Unknown'
                                record = {
                                    'Sample_Idx': global_idx,
                                    'Product_Name': product_name,
                                    'Type': 'Organic',
                                    'Reactant_Idx': j,
                                    'Reactant_Name': reactant_name,
                                    'T_inj': T_inj,
                                    'T_rea': T_rea,
                                    'Time': t_time,
                                    'Size': size,
                                    'alpha': alpha_org,
                                    'beta': beta_org,
                                    'm_tilde': org_interm['m_tilde'][i, j, 0].item(),
                                    'a_tilde': org_interm['a_tilde'][i, j, 0].item(),
                                    'a': org_interm['a'][i, j, 0].item(),
                                    'w': org_interm['w'][i, j, 0].item(),
                                }
                                for d in range(actual_early_dim):
                                    record[f'z_early_{d}'] = early_z_org[i, j, d].item()
                                records.append(record)
                                
        df_interm = pd.DataFrame(records)
        out_csv = os.path.join(BASE_DIR, 'intermediate_weights.csv')
        df_interm.to_csv(out_csv, index=False)
        print(f"  Intermediate variables successfully extracted and saved to {out_csv}")
        print("\n🎉 DONE! All processes finished successfully.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to extract intermediate variables to CSV: {e}")

if __name__ == '__main__':
    main()
