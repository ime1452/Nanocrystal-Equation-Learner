import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
import random
import optuna
import glob
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from dataset import NanoDataset
from model import NanoEQLModel


def get_target_transform_name(y_scaler):
    if hasattr(y_scaler, 'method'):
        return str(y_scaler.method)
    if y_scaler.__class__.__name__ == 'RobustScaler':
        return 'robust'
    return y_scaler.__class__.__name__


def resolve_descriptor_paths(base_dir, descriptor_set):
    descriptor_key = descriptor_set.lower()
    if descriptor_key == 'old':
        return (
            os.path.join(base_dir, 'inorganic_descriptors_processed_filled.csv'),
            os.path.join(base_dir, 'organic_descriptors_processed_filled.csv'),
        )
    if descriptor_key == 'new':
        return (
            os.path.join(base_dir, 'inorganic_descriptors_vif_filtered.csv'),
            os.path.join(base_dir, 'organic_descriptors_vif_filtered.csv'),
        )
    raise ValueError("descriptor_set must be 'old' or 'new'.")


def read_table_file(file_path):
    _, ext = os.path.splitext(str(file_path).lower())
    if ext in {'.xlsx', '.xls'}:
        return pd.read_excel(file_path)
    if ext == '.csv':
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported data file format: {file_path}. Please use .xlsx, .xls, or .csv.")


def write_table_file(df, file_path):
    _, ext = os.path.splitext(str(file_path).lower())
    if ext in {'.xlsx', '.xls'}:
        df.to_excel(file_path, index=False)
        return
    if ext == '.csv':
        df.to_csv(file_path, index=False)
        return
    raise ValueError(f"Unsupported data file format: {file_path}. Please use .xlsx, .xls, or .csv.")


def resolve_data_paths(base_dir, data_split, train_data_file=None, test_data_file=None):
    presets = {
        'default': ('train_data.xlsx', 'test_data.xlsx'),
        'upto20': ('train_data_upto20_split.xlsx', 'test_data_upto20_split.xlsx'),
    }

    if data_split not in presets:
        raise ValueError("data_split must be 'default' or 'upto20'.")

    default_train, default_test = presets[data_split]
    train_name = train_data_file or default_train
    test_name = test_data_file or default_test

    train_path = train_name if os.path.isabs(train_name) else os.path.join(base_dir, train_name)
    test_path = test_name if os.path.isabs(test_name) else os.path.join(base_dir, test_name)
    return train_path, test_path


def create_model(dim_inorg, dim_org, dim_prod, dim_ops, params, predictor_type, use_h_prod, use_g_joint, use_g_ops, device):
    return NanoEQLModel(
        dim_inorg=dim_inorg,
        dim_org=dim_org,
        dim_prod=dim_prod,
        dim_ops=dim_ops,
        hidden_dim=params['hidden_dim'],
        early_dim=params.get('early_dim', 16),
        latent_dim=1,
        eql_depth=params.get('eql_depth', 3),
        mlp_depth=params.get('mlp_depth', 2),
        mlp_dropout=params.get('mlp_dropout', 0.0),
        activation=params.get('activation', 'relu'),
        predictor_type=predictor_type,
        use_h_prod=use_h_prod,
        use_g_joint=use_g_joint,
        use_g_ops=use_g_ops,
    ).to(device)


def create_optimizer_and_scheduler(model, params):
    gate_lr_mult = params.get('gate_lr_mult', 25)
    gate_params = []
    base_params = []
    for name, param in model.named_parameters():
        if 'gate' in name:
            gate_params.append(param)
        else:
            base_params.append(param)

    param_groups = [
        {'params': base_params, 'lr': params['lr']},
        {'params': gate_params, 'lr': params['lr'] * gate_lr_mult},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=params['t_0'],
        T_mult=params['t_mult'],
    )
    return optimizer, scheduler


def compute_raw_space_mse(model, dataloader, y_scaler, device):
    model.eval()
    total_sq_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled, _ops_raw, y = [b.to(device) for b in batch]
            pred, *_ = model(x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled)

            pred_raw = y_scaler.inverse_transform(pred.detach().cpu().numpy()).reshape(-1)
            y_raw = y_scaler.inverse_transform(y.detach().cpu().numpy()).reshape(-1)
            total_sq_error += np.sum((pred_raw - y_raw) ** 2)
            total_samples += y.size(0)

    if total_samples == 0:
        return float('inf')
    return total_sq_error / total_samples


def collect_oof_epoch_curve(
    dataset,
    dim_inorg,
    dim_org,
    dim_prod,
    dim_ops,
    params,
    predictor_type,
    use_h_prod,
    use_g_joint,
    use_g_ops,
    y_scaler,
    device,
    seed,
    epoch_upper,
    k_folds=5,
):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    num_workers = 0
    pin_memory = torch.cuda.is_available()

    models = []
    optimizers = []
    schedulers = []
    train_loaders = []
    val_loaders = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_sub = Subset(dataset, train_idx.tolist())
        val_sub = Subset(dataset, val_idx.tolist())

        fold_generator = torch.Generator()
        fold_generator.manual_seed(seed + fold_idx)

        train_loaders.append(
            DataLoader(
                train_sub,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                generator=fold_generator,
            )
        )
        val_loaders.append(
            DataLoader(
                val_sub,
                batch_size=params['batch_size'],
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        )

        model = create_model(
            dim_inorg=dim_inorg,
            dim_org=dim_org,
            dim_prod=dim_prod,
            dim_ops=dim_ops,
            params=params,
            predictor_type=predictor_type,
            use_h_prod=use_h_prod,
            use_g_joint=use_g_joint,
            use_g_ops=use_g_ops,
            device=device,
        )
        optimizer, scheduler = create_optimizer_and_scheduler(model, params)

        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    criterion = nn.SmoothL1Loss(beta=params['beta'])
    per_fold_curve = np.zeros((k_folds, epoch_upper), dtype=np.float64)

    for epoch in range(epoch_upper):
        for fold_idx in range(k_folds):
            models[fold_idx].train()
            for batch in train_loaders[fold_idx]:
                x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled, _ops_raw, y = [b.to(device) for b in batch]
                optimizers[fold_idx].zero_grad()
                pred, *_ = models[fold_idx](x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled)

                main_loss = criterion(pred, y)
                l1_base = models[fold_idx].get_l1_loss()
                loss = main_loss + params['l1_lambda'] * l1_base

                loss.backward()
                torch.nn.utils.clip_grad_norm_(models[fold_idx].parameters(), max_norm=params['max_norm'])
                optimizers[fold_idx].step()

            per_fold_curve[fold_idx, epoch] = compute_raw_space_mse(
                models[fold_idx],
                val_loaders[fold_idx],
                y_scaler,
                device,
            )
            schedulers[fold_idx].step()

    mean_curve = per_fold_curve.mean(axis=0)
    best_epoch_idx = int(np.argmin(mean_curve))
    best_epoch = best_epoch_idx + 1
    best_mse = float(mean_curve[best_epoch_idx])

    return {
        'per_fold_curve': per_fold_curve,
        'mean_curve': mean_curve,
        'best_epoch': best_epoch,
        'best_mse': best_mse,
    }


def save_oof_curve(save_path, oof_result):
    data = {'epoch': np.arange(1, len(oof_result['mean_curve']) + 1)}
    for fold_idx in range(oof_result['per_fold_curve'].shape[0]):
        data[f'fold_{fold_idx + 1}_val_mse'] = oof_result['per_fold_curve'][fold_idx]
    data['mean_val_mse'] = oof_result['mean_curve']
    pd.DataFrame(data).to_csv(save_path, index=False)


def evaluate_metrics(model, dataloader, y_scaler, device):
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch in dataloader:
            x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled, _ops_raw, y = [b.to(device) for b in batch]
            pred, *_ = model(x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled)
            
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y.cpu().numpy())
            
    preds_scaled = np.concatenate(all_preds, axis=0)
    trues_scaled = np.concatenate(all_trues, axis=0)
    
    preds_raw = y_scaler.inverse_transform(preds_scaled).flatten()
    trues_raw = y_scaler.inverse_transform(trues_scaled).flatten()
    
    if np.isnan(preds_raw).any() or np.isinf(preds_raw).any():
        return float('inf'), float('inf'), -float('inf')
    
    mse = np.mean((preds_raw - trues_raw) ** 2)
    mae = np.mean(np.abs(preds_raw - trues_raw))
    r2 = r2_score(trues_raw, preds_raw)
    
    return mse, mae, r2

def export_features(model, dataloader, scaler, output_path, device):
    z_prods, z_joints, z_ops_list, ops_list, ys = [], [], [], [], []
    alphas_inorg, betas_inorg = [], []
    alphas_org, betas_org = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled, ops_raw, y = [b.to(device) for b in batch]
            _, z_prod, _, _, z_joint, z_ops, attn_params = model(x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled)
            z_prods.append(z_prod.cpu().numpy())
            z_joints.append(z_joint.cpu().numpy())
            z_ops_list.append(z_ops.cpu().numpy())
            ops_list.append(ops_raw.cpu().numpy())
            ys.append(y.cpu().numpy())
            
            alphas_inorg.append(attn_params['alpha_inorg'].cpu().numpy())
            betas_inorg.append(attn_params['beta_inorg'].cpu().numpy())
            alphas_org.append(attn_params['alpha_org'].cpu().numpy())
            betas_org.append(attn_params['beta_org'].cpu().numpy())
            
    z_prods = np.concatenate(z_prods, axis=0)
    z_joints = np.concatenate(z_joints, axis=0)
    z_ops_all = np.concatenate(z_ops_list, axis=0)
    ops_all = np.concatenate(ops_list, axis=0)
    ys_scaled = np.concatenate(ys, axis=0)
    
    alphas_inorg = np.concatenate(alphas_inorg, axis=0)
    betas_inorg = np.concatenate(betas_inorg, axis=0)
    alphas_org = np.concatenate(alphas_org, axis=0)
    betas_org = np.concatenate(betas_org, axis=0)
    
    ys_raw = scaler.inverse_transform(ys_scaled)
    
    export_dict = {}
    latent_dim = z_prods.shape[1]
    
    if latent_dim == 1:
        export_dict['Z_product'] = z_prods.flatten()
        export_dict['Z_joint'] = z_joints.flatten()
        export_dict['Z_ops'] = z_ops_all.flatten()
    else:
        for i in range(latent_dim):
            export_dict[f'Z_product_{i+1}'] = z_prods[:, i]
            export_dict[f'Z_joint_{i+1}'] = z_joints[:, i]
            export_dict[f'Z_ops_{i+1}'] = z_ops_all[:, i]
            
    export_dict['T_inj'] = ops_all[:, 0]
    export_dict['T_rea'] = ops_all[:, 1]
    export_dict['t'] = ops_all[:, 2]
    
    export_dict['alpha_inorg'] = alphas_inorg.flatten()
    export_dict['beta_inorg'] = betas_inorg.flatten()
    export_dict['alpha_org'] = alphas_org.flatten()
    export_dict['beta_org'] = betas_org.flatten()
    
    export_dict['Size'] = ys_raw.flatten()
    
    export_df = pd.DataFrame(export_dict)
    export_df.to_csv(output_path, index=False)
    print(f"Features exported to {output_path}")

def train_stage1(
    use_augmentation=True,
    use_top10_features=True,
    predictor_type='glm',
    use_h_prod=True,
    use_g_joint=True,
    use_g_ops=True,
    run_name='default_run',
    skip_optuna=False,
    descriptor_set='new',
    target_transform='box-cox',
    data_split='default',
    train_data_file=None,
    test_data_file=None,
    epoch_selection_mode='oof',
    fixed_train_epochs=None,
    oof_curve_epochs=120,
    final_max_epochs=None,
    final_patience=None,
):
    """
    First Stage Training Script with Industrial Grade Reproducibility Lock and Optuna Skipping.
    """
    # ================= 100% Industrial Grade Reproducibility Lock =================
    seed = 43
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # ===================================================

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(BASE_DIR, 'results', run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, Saving to: {save_dir}")
    
    print("Loading prepared train and test datasets...")
    inorg_descriptor_path, org_descriptor_path = resolve_descriptor_paths(BASE_DIR, descriptor_set)
    train_path, test_path = resolve_data_paths(
        BASE_DIR,
        data_split=data_split,
        train_data_file=train_data_file,
        test_data_file=test_data_file,
    )
    print(f"Descriptor Set: {descriptor_set}")
    print(f" - Inorganic descriptors: {inorg_descriptor_path}")
    print(f" - Organic descriptors: {org_descriptor_path}")
    print(f"Target Transform: {target_transform}")
    print(f"Data Split: {data_split}")
    print(f" - Train data: {train_path}")
    print(f" - Test data: {test_path}")

    aug_path = os.path.join(BASE_DIR, 'rawag_2x.xlsx')
    train_ext = os.path.splitext(train_path)[1] or '.xlsx'
    combined_path = os.path.join(save_dir, f'combined_train_data{train_ext}')
    
    if use_augmentation and os.path.exists(aug_path):
        print(f"Data augmentation ENABLED. Merging rawag_2x.xlsx into {os.path.basename(train_path)}...")
        train_df = read_table_file(train_path)
        aug_df = pd.read_excel(aug_path)
        write_table_file(pd.concat([train_df, aug_df], ignore_index=True), combined_path)
    else:
        if not use_augmentation:
            print(f"Data augmentation DISABLED. Using original {os.path.basename(train_path)} only.")
        combined_path = train_path

    train_val_ds = NanoDataset(
        combined_path, 
        inorg_descriptor_path,
        org_descriptor_path,
        fit_scaler=True, 
        use_top10_features=use_top10_features,
        target_transform=target_transform,
    )
    
    test_ds = NanoDataset(
        test_path,
        inorg_descriptor_path,
        org_descriptor_path,
        fit_scaler=False, 
        scalers=(train_val_ds.inorg_scaler, train_val_ds.org_scaler, train_val_ds.y_scaler, train_val_ds.ops_imputer, train_val_ds.ops_scaler, train_val_ds.shape_encoder),
        use_top10_features=use_top10_features,
        target_transform=target_transform,
    )
    
    dim_inorg = train_val_ds.dim_inorg
    dim_org = train_val_ds.dim_org
    dim_prod = dim_inorg + train_val_ds.dim_shape
    dim_ops = 3 
    
    if final_max_epochs is not None:
        oof_curve_epochs = final_max_epochs
    if final_patience is not None:
        print("Warning: final_patience is deprecated and ignored under OOF epoch selection.")
    if epoch_selection_mode not in {'oof', 'fixed'}:
        raise ValueError("epoch_selection_mode must be 'oof' or 'fixed'.")

    def objective(trial):
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32])
        early_dim = trial.suggest_categorical("early_dim", [4, 8])
        lr = trial.suggest_float("lr", 5e-6, 3e-4, log=True)
        l1_lambda = trial.suggest_float("l1_lambda", 5e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        max_norm = trial.suggest_float("max_norm", 0.5, 5.0)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        t_0 = trial.suggest_categorical("t_0", [2, 5, 10, 20])
        t_mult = trial.suggest_int("t_mult", 1, 6) 
        beta = trial.suggest_float("beta", 0.1, 2.0) 
        gate_lr_mult = trial.suggest_categorical("gate_lr_mult", [10, 25, 50])
        
        eql_depth = trial.suggest_int("eql_depth", 2, 3)
        
        if predictor_type.lower() == 'mlp':
            mlp_depth = trial.suggest_int("mlp_depth", 1, 5)
            mlp_dropout = trial.suggest_float("mlp_dropout", 0.0, 0.5)
            activation = trial.suggest_categorical("activation", ["relu", "gelu", "silu"])
        else:
            mlp_depth = 1
            mlp_dropout = 0.0
            activation = 'relu'
        
        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        
        models = []
        optimizers = []
        schedulers = []
        train_loaders = []
        val_loaders = []
        
        num_workers = 0
        pin_memory = torch.cuda.is_available()
        
        g_kfold = torch.Generator()
        g_kfold.manual_seed(seed)
        
        for train_idx, val_idx in kf.split(train_val_ds):
            train_sub = Subset(train_val_ds, train_idx)
            val_sub = Subset(train_val_ds, val_idx)
            train_loaders.append(DataLoader(train_sub, batch_size=batch_size, shuffle=True, 
                                            num_workers=num_workers, pin_memory=pin_memory, generator=g_kfold))
            val_loaders.append(DataLoader(val_sub, batch_size=batch_size, shuffle=False, 
                                          num_workers=num_workers, pin_memory=pin_memory))
            
            trial_params = {
                'hidden_dim': hidden_dim,
                'early_dim': early_dim,
                'lr': lr,
                'l1_lambda': l1_lambda,
                'batch_size': batch_size,
                'max_norm': max_norm,
                'weight_decay': weight_decay,
                't_0': t_0,
                't_mult': t_mult,
                'beta': beta,
                'gate_lr_mult': gate_lr_mult,
                'eql_depth': eql_depth,
                'mlp_depth': mlp_depth,
                'mlp_dropout': mlp_dropout,
                'activation': activation,
            }

            m = create_model(
                dim_inorg=dim_inorg,
                dim_org=dim_org,
                dim_prod=dim_prod,
                dim_ops=dim_ops,
                params=trial_params,
                predictor_type=predictor_type,
                use_h_prod=use_h_prod,
                use_g_joint=use_g_joint,
                use_g_ops=use_g_ops,
                device=device,
            )
            opt, sch = create_optimizer_and_scheduler(m, trial_params)
            
            models.append(m)
            optimizers.append(opt)
            schedulers.append(sch)
            
        criterion = nn.SmoothL1Loss(beta=beta)
        epochs = 200 
        patience = 50
        best_fold_mses = [float('inf')] * k_folds
        epochs_no_improve = [0] * k_folds
        for epoch in range(epochs):
            epoch_val_mses = []
            for i in range(k_folds):
                if epochs_no_improve[i] >= patience:
                    epoch_val_mses.append(best_fold_mses[i])
                    continue
                    
                models[i].train()
                for batch in train_loaders[i]:
                    x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled, _ops_raw, y = [b.to(device) for b in batch]
                    optimizers[i].zero_grad()
                    pred, *_ = models[i](x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled)
                    
                    main_loss = criterion(pred, y)
                    l1_base = models[i].get_l1_loss()
                    loss = main_loss + l1_lambda * l1_base
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(models[i].parameters(), max_norm=max_norm)
                    optimizers[i].step()
                    
                models[i].eval()
                fold_val_mse = compute_raw_space_mse(models[i], val_loaders[i], train_val_ds.y_scaler, device)
                epoch_val_mses.append(fold_val_mse)
                
                if fold_val_mse < best_fold_mses[i]:
                    best_fold_mses[i] = fold_val_mse
                    epochs_no_improve[i] = 0
                else:
                    epochs_no_improve[i] += 1
                    
                schedulers[i].step()
                
            avg_val_mse = np.mean(epoch_val_mses)
            trial.report(avg_val_mse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
            if all(e >= patience for e in epochs_no_improve):
                break
        return np.mean(best_fold_mses)

    # ================= OPTUNA Hyperparameter Optimization or Skip Logic =================
    if not skip_optuna:
        print("Starting Optuna Hyperparameter Optimization...")
        sampler = optuna.samplers.TPESampler(seed=seed)
        
        old_weights = glob.glob(os.path.join(save_dir, "temp_trial_*_weights.pth"))
        for f in old_weights:
            try:
                os.remove(f)
            except:
                pass
        
        db_path = os.path.join(save_dir, "optuna_study.db")
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                print(f"Removed previous Optuna database at {db_path}.")
            except:
                pass
                
        storage_name = f"sqlite:///{db_path}"
        study = optuna.create_study(
            study_name="nano_eql_study", 
            direction="minimize", 
            sampler=sampler, 
            pruner=optuna.pruners.HyperbandPruner(),
            storage=storage_name
        )
        
        n_jobs = 4
        print(f"Running Optuna with {n_jobs} parallel job to ensure reproducibility...")
        
        study.optimize(objective, n_trials=800, n_jobs=n_jobs)
        
        optuna_best_mse = study.best_value
        global_best_params = study.best_params
        
        print("\n" + "="*50)
        print("Optimization finished!")
        print(f"Global Best Validation MSE: {optuna_best_mse:.4f}")
        print(f"Best Parameters: {global_best_params}")
        print("="*50 + "\n")
        
    else:
        print("\n" + "="*50)
        print("Skipping Optuna. Using Hardcoded Golden Parameters...")
        print("="*50 + "\n")
        
        # Replace the values here to test the results for different architectures.
        global_best_params = {'hidden_dim': 32, 'early_dim': 4, 'lr': 0.0001964515812554643, 'l1_lambda': 0.0005654325985956404, 'batch_size': 32, 'max_norm': 4.3886594848266425, 'weight_decay': 0.0002524525741413435, 't_0': 2, 't_mult': 2, 'beta': 0.8435849279110961, 'gate_lr_mult': 10, 'eql_depth': 3}
        optuna_best_mse = float('nan')

    if epoch_selection_mode == 'oof':
        # ================= Use OOF to determine the epoch =================
        print(f"Running pure 5-Fold OOF training for {oof_curve_epochs} epochs to select E_best...")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        oof_result = collect_oof_epoch_curve(
            dataset=train_val_ds,
            dim_inorg=dim_inorg,
            dim_org=dim_org,
            dim_prod=dim_prod,
            dim_ops=dim_ops,
            params=global_best_params,
            predictor_type=predictor_type,
            use_h_prod=use_h_prod,
            use_g_joint=use_g_joint,
            use_g_ops=use_g_ops,
            y_scaler=train_val_ds.y_scaler,
            device=device,
            seed=seed,
            epoch_upper=oof_curve_epochs,
        )
        oof_curve_path = os.path.join(save_dir, 'oof_val_curve.csv')
        save_oof_curve(oof_curve_path, oof_result)
        best_epoch = oof_result['best_epoch']
        oof_best_mse = oof_result['best_mse']
        print(f"OOF curve saved to {oof_curve_path}")
        print(f"Selected E_best from OOF curve: epoch={best_epoch}, mean_val_mse={oof_best_mse:.4f}")
        final_training_label = f"E_best={best_epoch}"
    else:
        if fixed_train_epochs is None or fixed_train_epochs <= 0:
            raise ValueError("fixed_train_epochs must be a positive integer when epoch_selection_mode='fixed'.")
        best_epoch = int(fixed_train_epochs)
        oof_best_mse = float('nan')
        oof_curve_path = None
        print(f"Skipping OOF. Using fixed training epochs: {best_epoch}")
        final_training_label = f"fixed_epoch={best_epoch}"

    # ================= Retraining on the Full Dataset with a Fixed Number of Epochs =================
    print(f"Retraining final model on FULL dataset for {final_training_label}...")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    g_full = torch.Generator()
    g_full.manual_seed(seed)

    full_train_loader_shuffle = DataLoader(
        train_val_ds,
        batch_size=global_best_params['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=g_full,
    )

    best_model = create_model(
        dim_inorg=dim_inorg,
        dim_org=dim_org,
        dim_prod=dim_prod,
        dim_ops=dim_ops,
        params=global_best_params,
        predictor_type=predictor_type,
        use_h_prod=use_h_prod,
        use_g_joint=use_g_joint,
        use_g_ops=use_g_ops,
        device=device,
    )
    opt, sch = create_optimizer_and_scheduler(best_model, global_best_params)
    criterion = nn.SmoothL1Loss(beta=global_best_params['beta'])

    for _epoch in range(best_epoch):
        best_model.train()
        for batch in full_train_loader_shuffle:
            x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled, _ops_raw, y = [b.to(device) for b in batch]
            opt.zero_grad()
            pred, *_ = best_model(x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled)

            main_loss = criterion(pred, y)
            l1_base = best_model.get_l1_loss()
            loss = main_loss + global_best_params['l1_lambda'] * l1_base

            loss.backward()
            torch.nn.utils.clip_grad_norm_(best_model.parameters(), max_norm=global_best_params['max_norm'])
            opt.step()
        sch.step()

    best_model.eval()
    
    num_workers = 0 
    pin_memory = torch.cuda.is_available()
    full_train_loader = DataLoader(train_val_ds, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    train_mse, train_mae, train_r2 = evaluate_metrics(best_model, full_train_loader, train_val_ds.y_scaler, device)
    test_mse, test_mae, test_r2 = evaluate_metrics(best_model, test_loader, train_val_ds.y_scaler, device)
    
    best_model.eval()
    dummy_batch = next(iter(test_loader))
    x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled, _ops_raw, _y = [b.to(device) for b in dummy_batch]
    with torch.no_grad():
        _, _, _, _, _, _, attn_params = best_model(x_prod, x_inorg, m_inorg, x_org, m_org, ops_scaled)
        
    attn_str = (
        "============= Hybrid Attention Weights (Mean over batch) =============\n"
        f"Inorg: Mass (alpha) = {attn_params['alpha_inorg'].mean().item():.4f}, Activity (beta) = {attn_params['beta_inorg'].mean().item():.4f}\n"
        f"Org:   Mass (alpha) = {attn_params['alpha_org'].mean().item():.4f}, Activity (beta) = {attn_params['beta_org'].mean().item():.4f}\n"
        "======================================================================\n"
    )
    
    metrics_str = (
        "================ Stage 1 Metrics ================\n"
        f"Descriptor Set    - {descriptor_set}\n"
        f"Data Split        - {data_split}\n"
        f"Train Data File   - {os.path.basename(train_path)}\n"
        f"Test Data File    - {os.path.basename(test_path)}\n"
        f"Target Transform  - {get_target_transform_name(train_val_ds.y_scaler)}\n"
        f"Epoch Mode       - {epoch_selection_mode}\n"
        f"Train Set (Full) - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f}\n"
        f"Test Set         - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}\n"
        f"Optuna Best MSE  - {optuna_best_mse:.4f}\n"
        f"OOF Curve Min MSE- {oof_best_mse:.4f}\n"
        f"Final Train Epoch - {best_epoch}\n"
        "=================================================\n"
    )
    
    print("\n" + metrics_str)
    print(attn_str)
    with open(os.path.join(save_dir, 'stage1_metrics.txt'), 'w') as f:
        f.write(metrics_str)
        f.write(attn_str)
        f.write(f"\nBest Parameters:\n{global_best_params}\n")
    
    export_features(best_model, full_train_loader, train_val_ds.y_scaler, os.path.join(save_dir, 'train_features.csv'), device)
    export_features(best_model, test_loader, train_val_ds.y_scaler, os.path.join(save_dir, 'test_features.csv'), device)
    
    torch.save(best_model.state_dict(), os.path.join(save_dir, 'eql_model.pth'))
    print(f"Best Model saved to {os.path.join(save_dir, 'eql_model.pth')}")
    return {
        'descriptor_set': descriptor_set,
        'data_split': data_split,
        'train_data_file': train_path,
        'test_data_file': test_path,
        'target_transform': get_target_transform_name(train_val_ds.y_scaler),
        'epoch_selection_mode': epoch_selection_mode,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'optuna_best_mse': optuna_best_mse,
        'oof_best_mse': oof_best_mse,
        'best_final_epoch': best_epoch,
        'oof_curve_path': oof_curve_path,
        'save_dir': save_dir,
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train Stage 1 with dynamic plugin control")
    parser.add_argument('--run_name', type=str, default='default_run', help='Name of the run for output isolation')
    parser.add_argument('--use_augmentation', action='store_true', help='Use augmented dataset')
    parser.add_argument('--use_top10_features', action='store_true', help='Use only top 10 features')
    parser.add_argument('--predictor_type', type=str, default='glm', choices=['mlp', 'glm', 'quad', 'eql'])
    
    parser.add_argument('--skip_optuna', action='store_true', help='Skip Optuna and use hardcoded best parameters')
    parser.add_argument('--descriptor_set', type=str, default='new', choices=['old', 'new'], help='Choose descriptor family')
    parser.add_argument('--target_transform', type=str, default='robust', choices=['robust', 'box-cox'], help='Choose target transform')
    parser.add_argument('--data_split', type=str, default='default', choices=['default', 'upto20'], help='Choose prepared train/test split preset')
    parser.add_argument('--train_data_file', type=str, default=None, help='Optional custom train data file (.xlsx/.xls/.csv). Overrides --data_split train preset when provided.')
    parser.add_argument('--test_data_file', type=str, default=None, help='Optional custom test data file (.xlsx/.xls/.csv). Overrides --data_split test preset when provided.')
    parser.add_argument('--epoch_selection_mode', type=str, default='fixed', choices=['oof', 'fixed'], help='Choose whether to use OOF to select epoch or directly use a fixed training epoch.')
    parser.add_argument('--fixed_train_epochs', type=int, default=29, help='Required when --epoch_selection_mode fixed. Final model trains on full dataset for this many epochs.')
    parser.add_argument('--oof_curve_epochs', type=int, default=200, help='Upper epoch bound for pure 5-Fold OOF epoch selection')
    parser.add_argument('--final_max_epochs', type=int, default=None, help='Deprecated alias for --oof_curve_epochs')
    parser.add_argument('--final_patience', type=int, default=None, help='Deprecated and ignored under OOF epoch selection')
    
    parser.add_argument('--disable_h_prod', action='store_true', help='Disable product feature network')
    parser.add_argument('--disable_g_joint', action='store_true', help='Disable joint macro feature network')
    parser.add_argument('--disable_g_ops', action='store_true', help='Disable operational descriptors network')
    
    args = parser.parse_args()
    
    train_stage1(
        use_augmentation=args.use_augmentation, 
        use_top10_features=args.use_top10_features, 
        predictor_type=args.predictor_type,
        use_h_prod=not args.disable_h_prod, 
        use_g_joint=not args.disable_g_joint, 
        use_g_ops=not args.disable_g_ops,
        run_name=args.run_name,
        skip_optuna=args.skip_optuna,
        descriptor_set=args.descriptor_set,
        target_transform=args.target_transform,
        data_split=args.data_split,
        train_data_file=args.train_data_file,
        test_data_file=args.test_data_file,
        epoch_selection_mode=args.epoch_selection_mode,
        fixed_train_epochs=args.fixed_train_epochs,
        oof_curve_epochs=args.oof_curve_epochs,
        final_max_epochs=args.final_max_epochs,
        final_patience=args.final_patience,
    )
