import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EQLLayer(nn.Module):
    """
    Equation Learner (EQL) layer.
    This is the fundamental building block of the symbolic regression neural network surrogate.
    It processes inputs through multiple elementary mathematical functions, then combines
    them linearly, enabling the subsequent extraction of mathematical analytical expressions.
    """
    def __init__(self, in_features, out_features, op_set='all'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Unified damping constants (control maximum gradient near 0)
        c_inv = 0.1    # Maximum gradient 10
        c_sqrt = 0.01  # Maximum gradient 10
        c_cbrt = 0.001 # Maximum gradient 10
        
        # =====================================================================
        # Zero-Centered, Gradient-Bounded operator library
        # Completely abandoned autograd.Function prone to bugs; using native 
        # operators to ensure stability
        # =====================================================================
        all_funcs = [
            lambda x: x,                                                    # 0. Identity
            lambda x: torch.clamp(x, min=-3.0, max=3.0)**2,                 # 1. Square (y=0 at x=0)
            lambda x: x / torch.sqrt(torch.abs(x) + c_sqrt),                # 2. Damped square root
            lambda x: x / torch.pow(x**2 + c_cbrt, 1.0/3.0),                # 3. Damped cube root
            lambda x: x / (x**2 + c_inv),                                   # 4. Damped inverse
            lambda x: torch.exp(torch.clamp(x, min=-3.0, max=3.0)) - 1.0,   # 5. Exponential (shifted to pass origin)
            lambda x: torch.sign(x) * torch.log(torch.abs(x) + 1.0),        # 6. Natural log (origin-passing by construction)
            lambda x: torch.sign(x) * torch.log10(torch.abs(x) + 1.0)       # 7. Common log (origin-passing by construction)
        ]
        
        # Pure algebraic expression mapping for extract_formulas.py to use during extraction
        # (Ensure constants in strings match the c defined above)
        all_names = [
            "{0}",                                      # 0. Identity
            "({0})**2",                                 # 1. Square
            "({0}) / sqrt(abs({0}) + 0.01)",            # 2. Damped square root
            "({0}) / (({0})**2 + 0.001)**(1/3)",        # 3. Damped cube root
            "({0}) / (({0})**2 + 0.1)",                 # 4. Damped inverse
            "(exp({0}) - 1.0)",                         # 5. Exponential
            "sign({0})*ln(abs({0}) + 1.0)",             # 6. Natural log
            "sign({0})*log10(abs({0}) + 1.0)"           # 7. Common log
        ]
        
        # Removed physical rule routing; apply all defined operators to the EQL network
        indices = list(range(8))
            
        self.funcs = [all_funcs[i] for i in indices]
        self.func_names = [all_names[i] for i in indices]
        self.n_funcs = len(self.funcs)
        
        # Weight definition with Kaiming uniform initialization
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features * self.n_funcs))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        fan_in = in_features * self.n_funcs
        bound = min(0.1, 1.0 / (fan_in ** 0.5)) if fan_in > 0 else 0.1
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.zeros_(self.bias)
    def forward(self, x):
        # Record original dimensions and flatten to (Batch, in_features)
        x_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        
        # Pass input through every elementary mathematical operator
        outs = [f(x_flat) for f in self.funcs]
        # Concatenate all operator outputs along the feature dimension, shape becomes (Batch, in_features * n_funcs)
        x_funcs = torch.cat(outs, dim=1) 
        
        # Feature combination via linear layer
        out = F.linear(x_funcs, self.weight, self.bias)
        
        # Restore original tensor dimensions (preserving batch/sequence dimensions)
        out = out.view(*x_shape[:-1], self.out_features)
        return out

class AttentivePNAPooling(nn.Module):
    """
    Early attention, late fusion multi-channel statistical pooling (Attentive PNA Pooling).
    Concatenates operation descriptors with physical features for activity scoring,
    then computes mixed weighting and performs weighted statistical pooling.
    """
    def __init__(self, feature_dim, ops_dim=3):
        super().__init__()
        # Simple linear layer for computing feature activity (scorer)
        self.attn_proj = nn.Sequential(
            nn.Linear(feature_dim + ops_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Temperature gating network (input: T_inj, T_rea, t; output: alpha_logit, beta_logit)
        self.gate = nn.Linear(3, 2)
                        
        # Make alpha (molar mass) initially very large, beta (network activity) initially very small
        nn.init.uniform_(self.gate.weight, -0.05, 0.05)
        nn.init.constant_(self.gate.bias[0], 1.0)
        nn.init.constant_(self.gate.bias[1], 0.0)

    def forward(self, X, m, valid_mask, ops):
        """
        X: (Batch, Seq_len, Dim) raw features
        m: (Batch, Seq_len, 1) corresponding mole amounts
        valid_mask: (Batch, Seq_len, 1) mask for non-padding elements, 1 for valid, 0 for padding
        ops: (Batch, ops_dim) containing T_inj, T_rea, t
        """
        B, Seq, Dim = X.shape
        
        # 1. Operation descriptor concatenation and activity scoring
        ops_expanded = ops.unsqueeze(1).expand(B, Seq, -1) # (B, Seq, ops_dim)
        c = torch.cat([X, ops_expanded], dim=-1) # (B, Seq, Dim + ops_dim)
        
        a_tilde = self.attn_proj(c) # (B, Seq, 1)
        
        # Softmax normalization (only for valid positions; set invalid positions to extremely small values)
        a_tilde_masked = a_tilde.masked_fill(valid_mask < 0.5, -1e9)
        a = F.softmax(a_tilde_masked, dim=1) # (B, Seq, 1)
        a = torch.where(torch.isnan(a), torch.zeros_like(a), a) # Prevent NaN when all positions are padding
        
        # Force scores at invalid positions to zero to prevent all-padding samples from receiving uniform fake weights
        a = a * valid_mask
        
        # 2. Molar distribution
        m_sum = torch.sum(m * valid_mask, dim=1, keepdim=True).clamp(min=1e-6)
        m_tilde = (m * valid_mask) / m_sum # (B, Seq, 1)
        
        # 3. Temperature gating
        ops_gate = ops[:, :3] # T_inj, T_rea, t
        gate_logits = self.gate(ops_gate) # (B, 2)
        alpha_beta = F.softmax(gate_logits, dim=-1) # (B, 2)
        alpha = alpha_beta[:, 0].unsqueeze(1).unsqueeze(2) # (B, 1, 1)
        beta = alpha_beta[:, 1].unsqueeze(1).unsqueeze(2) # (B, 1, 1)
        
        # 4. Mixed weights
        w = alpha * m_tilde + beta * a # (B, Seq, 1)
        
        # 5. Weighted aggregation
        count = torch.sum(valid_mask, dim=1).clamp(min=1e-6) # (B, 1)
        
        # (1) Mean (Weighted Mean)
        X_mean = torch.sum(w * X, dim=1) # (B, Dim)
        
        # (2) Sum (Sum = Mean * count)
        X_sum = X_mean * count
        
        # (3) Max and Min
        mask_bool = valid_mask.squeeze(-1) > 0.5
        X_masked_max = torch.where(mask_bool.unsqueeze(-1), X, torch.tensor(float('-inf'), device=X.device))
        X_max, _ = torch.max(X_masked_max, dim=1)
        X_max = torch.where(torch.isinf(X_max), torch.zeros_like(X_max), X_max)
        
        X_masked_min = torch.where(mask_bool.unsqueeze(-1), X, torch.tensor(float('inf'), device=X.device))
        X_min, _ = torch.min(X_masked_min, dim=1)
        X_min = torch.where(torch.isinf(X_min), torch.zeros_like(X_min), X_min)
        
        # (4) Range
        X_range = X_max - X_min
        
        # (5) Standard Deviation (Weighted Standard Deviation)
        X_var = torch.sum(w.double() * ((X.double() - X_mean.unsqueeze(1).double()) ** 2), dim=1)
        X_std = torch.sqrt(X_var + 1e-8).float()
        
        # (6) Root Sum Square (Weighted Root Sum Square)
        X_rss = torch.sqrt(torch.sum(w.double() * (X.double() ** 2), dim=1) + 1e-8).float()
        
        # Concatenate macroscopic features (7 statistical measures)
        X_fused = torch.cat([X_sum, X_mean, X_range, X_std, X_rss, X_max, X_min], dim=-1) # (B, Dim * 7)
        
        intermediates = {'a_tilde': a_tilde, 'a': a, 'm_tilde': m_tilde, 'w': w}
        return X_fused, alpha.squeeze(-1).squeeze(-1), beta.squeeze(-1).squeeze(-1), intermediates

class QuadraticLayer(nn.Module):
    """
    Quadratic polynomial layer.
    Introduces squared terms in top-level fusion without the black-box nonlinearity
    brought by deep MLPs.
    y = w_1 * x + w_2 * x^2 + b
    """
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.quad = nn.Linear(in_features, 1, bias=False)
        
    def forward(self, x):
        return self.linear(x) + self.quad(x ** 2)

class NanoEQLModel(nn.Module):
    """
    Nanocrystal multi-branch representation learning model (Phase I surrogate network).
    Implements a heterogeneous graph architecture with categorical representation,
    intra-group aggregation, and top-level fusion.
    Introduces a mixed attention mechanism before feature fusion via AttentivePNAPooling.
    """
    def __init__(self, dim_inorg, dim_org, dim_prod, dim_ops=3, hidden_dim=16, early_dim=16, latent_dim=1, eql_depth=3, mlp_depth=2, mlp_dropout=0.0, activation='relu', predictor_type='mlp',
                 use_h_prod=True, use_g_joint=True, use_g_ops=True):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.use_h_prod = use_h_prod
        self.use_g_joint = use_g_joint
        self.use_g_ops = use_g_ops
        
        def build_eql_branch(in_d, out_dim=latent_dim, op_set='all'):
            layers = []
            if eql_depth == 1:
                layers.append(EQLLayer(in_d, out_dim, op_set=op_set))
            else:
                layers.append(EQLLayer(in_d, hidden_dim, op_set=op_set))
                for _ in range(eql_depth - 2):
                    layers.append(EQLLayer(hidden_dim, hidden_dim, op_set=op_set))
                layers.append(EQLLayer(hidden_dim, out_dim, op_set=op_set))
            return nn.Sequential(*layers)
            
        # Product inorganic feature extractor h_prod (input dimension includes shape)
        if self.use_h_prod:
            self.h_prod = build_eql_branch(dim_prod, out_dim=latent_dim, op_set='props')
        
        # Joint macroscopic feature extractor g_joint
        if self.use_g_joint:
            # 1. Early lightweight EQL (Early-EQL): Independently compresses high-dimensional features into early_dim dimensions
            self.early_g_inorg = build_eql_branch(dim_inorg, out_dim=early_dim, op_set='props')
            self.early_g_org = build_eql_branch(dim_org, out_dim=early_dim, op_set='props')
            
            # 2. Mixed attention pooling receives dimensionality-reduced features
            self.attn_pool_inorg = AttentivePNAPooling(early_dim, dim_ops)
            self.attn_pool_org = AttentivePNAPooling(early_dim, dim_ops)
            
            # 3. Late-Fusion EQL (Late-Fusion): PNA_pooling generates early_dim * 7 statistical measures
            self.g_joint = build_eql_branch(7 * early_dim + 7 * early_dim, out_dim=latent_dim, op_set='all')
        
        # Operation descriptor feature extractor g_ops
        if self.use_g_ops:
            self.g_ops = build_eql_branch(dim_ops, out_dim=latent_dim)
        # if self.use_g_ops:
        #     # Strategy: strictly lock width to 16, never use global 64/128 to prevent Taylor expansion explosion!
        #     ops_hidden_dim = 16
        #     ops_layers = []
        #     if eql_depth == 1:
        #         ops_layers.append(EQLLayer(dim_ops, latent_dim, op_set='thermo'))
        #     else:
        #         ops_layers.append(EQLLayer(dim_ops, ops_hidden_dim, op_set='thermo'))
        #         for _ in range(eql_depth - 2):
        #             ops_layers.append(EQLLayer(ops_hidden_dim, ops_hidden_dim, op_set='thermo'))
        #         ops_layers.append(EQLLayer(ops_hidden_dim, latent_dim, op_set='thermo'))
        #     self.g_ops = nn.Sequential(*ops_layers) 


        # Top-level predictor
        enabled_branches = sum([self.use_h_prod, self.use_g_joint, self.use_g_ops])
        in_dim = latent_dim * enabled_branches
        
        if predictor_type.lower() == 'mlp':
            act_layer = nn.ReLU() if activation.lower() == 'relu' else (nn.GELU() if activation.lower() == 'gelu' else nn.SiLU())
            mlp_layers = []
            for _ in range(mlp_depth - 1):
                mlp_layers.append(nn.Linear(in_dim, hidden_dim))
                mlp_layers.append(act_layer)
                if mlp_dropout > 0:
                    mlp_layers.append(nn.Dropout(mlp_dropout))
                in_dim = hidden_dim
            mlp_layers.append(nn.Linear(in_dim, 1))
            self.top_predictor = nn.Sequential(*mlp_layers)
        elif predictor_type.lower() == 'glm':
            self.top_predictor = nn.Linear(in_dim, 1)
        elif predictor_type.lower() == 'quad':
            self.top_predictor = QuadraticLayer(in_dim)
        elif predictor_type.lower() == 'eql':
            self.top_predictor = EQLLayer(in_dim, 1, op_set='all')
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}. Please choose from 'mlp', 'glm', 'quad', 'eql'.")
        
    def forward(self, x_prod, x_inorg, m_inorg, x_org, m_org, ops):
        """
        Forward propagation
        """
        B = x_prod.size(0)
        z_list = []
        
        # 1. Product representation Z_product
        if self.use_h_prod:
            z_prod = self.h_prod(x_prod)
            z_list.append(z_prod)
        else:
            z_prod = torch.zeros(B, self.latent_dim, device=x_prod.device)
            
        # 2. Joint macroscopic representation Z_joint
        if self.use_g_joint:
            valid_mask_inorg = (m_inorg > 1e-6).float() # (Batch, max_inorg, 1)
            valid_mask_org = (m_org > 1e-6).float() # (Batch, max_org, 1)
            
            # (1) Early nonlinear extraction and dimensionality reduction (Early-EQL)
            # To handle sequences, flatten features for computation
            N = x_inorg.size(1)
            M = x_org.size(1)
            
            # Early-EQL extraction for inorganic features
            x_inorg_flat = x_inorg.view(B * N, -1)
            early_z_inorg_flat = self.early_g_inorg(x_inorg_flat)
            early_z_inorg = early_z_inorg_flat.view(B, N, -1) # (B, N, hidden_dim)
            
            # Early-EQL extraction for organic features
            x_org_flat = x_org.view(B * M, -1)
            early_z_org_flat = self.early_g_org(x_org_flat)
            early_z_org = early_z_org_flat.view(B, M, -1) # (B, M, hidden_dim)
            
            # (2) Mixed attention pooling (at this point, early_z imprinted with nonlinear patterns is passed in)
            x_inorg_fused, alpha_inorg, beta_inorg, inorg_interm = self.attn_pool_inorg(early_z_inorg, m_inorg, valid_mask_inorg, ops)
            x_org_fused, alpha_org, beta_org, org_interm = self.attn_pool_org(early_z_org, m_org, valid_mask_org, ops)
            
            # (3) Late-Fusion EQL
            x_joint = torch.cat([x_inorg_fused, x_org_fused], dim=-1)
            z_joint = self.g_joint(x_joint)
            z_list.append(z_joint)
        else:
            z_joint = torch.zeros(B, self.latent_dim, device=x_prod.device)
            alpha_inorg = beta_inorg = alpha_org = beta_org = torch.zeros(B, device=x_prod.device)
            inorg_interm = org_interm = None
            early_z_inorg = early_z_org = None
            
        # 3. Operation descriptor representation Z_ops
        if self.use_g_ops:
            z_ops = self.g_ops(ops)
            z_list.append(z_ops)
        else:
            z_ops = torch.zeros(B, self.latent_dim, device=x_prod.device)
            
        # 4. Top-level fusion prediction
        if len(z_list) > 0:
            z_all = torch.cat(z_list, dim=1)
            pred = self.top_predictor(z_all)
        else:
            pred = torch.zeros(B, 1, device=x_prod.device)
            
        attn_params = {
            'alpha_inorg': alpha_inorg, 'beta_inorg': beta_inorg,
            'alpha_org': alpha_org, 'beta_org': beta_org,
            'inorg_interm': inorg_interm,
            'org_interm': org_interm,
            'early_z_inorg': early_z_inorg,
            'early_z_org': early_z_org
        }
        
        # Return zero tensors to maintain compatibility with external feature extraction interfaces
        z_inorg = torch.zeros(B, self.latent_dim, device=x_prod.device)
        z_org = torch.zeros(B, self.latent_dim, device=x_prod.device)
        
        return pred, z_prod, z_inorg, z_org, z_joint, z_ops, attn_params
        
    def get_l1_loss(self):
        """
        Get L1 regularization penalty term.
        Only applies L1 penalty to base EQLLayer weights to extract sparse formulas.
        (Note: AttentivePNAPooling attention scorer is not included in the penalty to prevent attention collapse)
        """
        l1_base = 0.0
        for name, m in self.named_modules():
            if isinstance(m, EQLLayer):
                l1_base += torch.sum(torch.abs(m.weight)) + torch.sum(torch.abs(m.bias))
        return l1_base        
    def get_ortho_loss(self, z_inorg, z_org, z_joint):
        """
        As g_inorg and g_org have been removed, the orthogonality penalty mechanism 
        is no longer applicable; directly returns 0.
        """
        return torch.tensor(0.0, device=z_joint.device)