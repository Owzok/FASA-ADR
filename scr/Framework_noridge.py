import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score, accuracy_score, auc, precision_score, recall_score, matthews_corrcoef
from model import build_ae
import numpy as np
import gc
from metrics import compute_all_metrics

def set_seeds(seed=42):
    """Control de aleatoriedad para reproducibilidad"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma)
        
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        
        focal = alpha_t * focal_weight * bce
        
        return tf.reduce_mean(focal)
    return loss

bce_loss = tf.keras.losses.BinaryCrossentropy(
    from_logits=False,
    reduction='sum_over_batch_size'
)

def mora_loss(lambda_=0.1):
    bce = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        reduction='sum_over_batch_size'
    )

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)    # windows error
        bce_loss = tf.reduce_mean(bce(y_true, y_pred))

        true_sum = tf.reduce_sum(y_true, axis=1)
        pred_sum = tf.reduce_sum(y_pred, axis=1)

        sum_mse = tf.reduce_mean(tf.square(true_sum - pred_sum))

        total_loss = bce_loss + lambda_ * sum_mse
        return total_loss

    return loss


def hybrid_stacking(X, n_folds=5, n_runs=5, random_state=42, bias_factor=-3.5):
    """
    1. Múltiples runs con diferentes seeds
    2. Entrena múltiples base models con pseudo-labeling
    3. Stack con meta-learner
    """

    base_configs = [
        {'latent_dim': 2048, 'alpha': 0.25, 'gamma': 2.0},
        {'latent_dim': 1024, 'alpha': 0.25, 'gamma': 2.0},
        {'latent_dim': 2048, 'alpha': 0.25, 'gamma': 2.0},
        {'latent_dim': 1024, 'alpha': 0.25, 'gamma': 2.0}
    ]

    print(f"\n{'='*70}")
    print("STABLE MULTI-RUN HYBRID STACKING")
    print(f"{'='*70}")
    print(f"Configurations: {len(base_configs)}")
    print(f"Folds per run: {n_folds}")
    print(f"Total runs: {n_runs}")
    print(f"Total models trained: {len(base_configs) * n_folds * n_runs}")
    print(f"{'='*70}")

    all_runs_results = []

    for run_idx in range(n_runs):
        run_seed = 42 + run_idx * 100
        print(f"\n{'#'*70}")
        print(f"# RUN {run_idx + 1}/{n_runs} (seed={run_seed})")
        print(f"{'#'*70}")
        
        set_seeds(run_seed)
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=run_seed)
        
        all_fold_predictions = []
        
        # =============================================
        # STAGE 1: Entrenar base models con PL
        # =============================================
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            print(f"\n{'='*70}")
            print(f"FOLD {fold_idx}/{n_folds}")
            print(f"{'='*70}")
            
            X_train_clean = X[train_idx]
            X_val_clean = X[val_idx]
            
            fold_predictions = []
            
            for config_idx, config in enumerate(base_configs, 1):
                print(f"\n--- Model {config_idx}/{len(base_configs)}: L={config['latent_dim']}, α={config['alpha']} ---")
                
                # Initialize tracking variables BEFORE the loop
                best_val_aupr = 0.0
                best_model_weights = None  # Changed from best_model
                best_iteration = 0
                best_predictions = None  # This must be None initially

                # Pseudo-labeling iterations
                X_train_current = X_train_clean.copy()
                
                pl_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

                for pl_iter in range(len(pl_thresholds) + 1):  # +1 for final training
                    # Build model
                    model = build_ae(
                        n_adrs=X_train_current.shape[1], 
                        latent_dim=config['latent_dim'],
                        bias_factor=bias_factor
                    )
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss=mora_loss(lambda_=1e-6),
                        metrics=[tf.keras.metrics.AUC(name='auc', curve='PR')]
                    )
                    
                    early_stop = callbacks.EarlyStopping(
                        monitor='val_auc', 
                        patience=10, 
                        mode='max',
                        restore_best_weights=True, 
                        verbose=0
                    )
                    
                    # Train
                    history = model.fit(
                        X_train_current, X_train_clean,
                        validation_data=(X_val_clean, X_val_clean),
                        epochs=50, 
                        batch_size=512,
                        callbacks=[early_stop], 
                        verbose=0
                    )

                    # Predict on validation
                    y_pred_val = model.predict(X_val_clean, batch_size=512, verbose=0)
                    val_aupr = average_precision_score(X_val_clean.flatten(), y_pred_val.flatten())
                    
                    print(f"    PL iter {pl_iter}: Val AUPR = {val_aupr:.4f}", end="")

                    # Track best iteration
                    if val_aupr > best_val_aupr:
                        best_val_aupr = val_aupr
                        best_iteration = pl_iter
                        # Save predictions and weights
                        best_predictions = y_pred_val.copy()
                        best_model_weights = [w.copy() for w in model.get_weights()]
                        print(" ✓ NEW BEST")
                    else:
                        print()

                    # Early stopping for PL
                    if pl_iter > 0 and (pl_iter - best_iteration) >= 2:
                        print(f"    Early Stopping PL: No improvement for 2 iterations")
                        tf.keras.backend.clear_session()
                        del model, y_pred_val
                        gc.collect()
                        break
                    
                    # Pseudo-labeling (if not last iteration)
                    if pl_iter < len(pl_thresholds):
                        threshold = pl_thresholds[pl_iter]
                        y_pred_train = model.predict(X_train_current, batch_size=512, verbose=0)
                        mask_unknown = (X_train_current == 0) & (X_train_clean == 0)
                        
                        unknown_scores = y_pred_train[mask_unknown]
                        if len(unknown_scores) > 0:
                            adaptive_threshold = max(np.percentile(unknown_scores, 99), threshold)
                            pseudo_mask = (y_pred_train > adaptive_threshold) & mask_unknown
                            n_pseudo = pseudo_mask.sum()
                            
                            if n_pseudo > 0:
                                X_train_current[pseudo_mask] = 1
                                print(f"    → Added {n_pseudo} pseudo-labels (threshold={adaptive_threshold:.3f})")
                            else:
                                print(f"    → No pseudo-labels (max_score={unknown_scores.max():.3f})")
                        
                        del y_pred_train, mask_unknown, unknown_scores, pseudo_mask
                        gc.collect()
                    
                    # Clean up current iteration
                    tf.keras.backend.clear_session()
                    del model, y_pred_val
                    gc.collect()
    
                # Use best model's predictions
                print(f"  ✅ Selected iteration {best_iteration} with AUPR: {best_val_aupr:.4f}")
                
                # Verify we have predictions before appending
                if best_predictions is None:
                    raise RuntimeError(f"No valid predictions found for model {config_idx}")
                
                fold_predictions.append(best_predictions)
                
                # Cleanup
                del best_model_weights, best_predictions, X_train_current
                gc.collect()

        
            all_fold_predictions.append(fold_predictions)
            del X_train_clean, X_val_clean, fold_predictions
            gc.collect()
        
        # =============================================
        # STAGE 2: Simple Averaging (No Meta-Learner)
        # =============================================
        print(f"\n{'='*70}")
        print("STAGE 2: AVERAGING BASE MODEL PREDICTIONS")
        print(f"{'='*70}")
        
        # Compute average AUPR across folds
        avg_auprs_per_fold = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            fold_preds = all_fold_predictions[fold_idx]  # list of arrays (n_val, n_adrs)
            fold_avg = np.mean(fold_preds, axis=0)  # average across models
            y_true_fold = X[val_idx].flatten()
            y_pred_fold = fold_avg.flatten()
            aupr_fold = average_precision_score(y_true_fold, y_pred_fold)
            avg_auprs_per_fold.append(aupr_fold)
            print(f"  Fold {fold_idx+1}/{n_folds} Avg AUPR: {aupr_fold:.4f}")
        
        # Aggregate CV score
        aupr_meta = float(np.mean(avg_auprs_per_fold))
        print(f"\n{'='*70}")
        print(f"RUN {run_idx + 1} AVERAGED AUPR (mean over folds): {aupr_meta:.4f}")
        print(f"{'='*70}")
        
        print("\nUsing simple averaging (equal weights) for all base models")

        final_predictions = np.zeros_like(X)

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            fold_preds = all_fold_predictions[fold_idx]  # lista de modelos base
            fold_avg = np.mean(fold_preds, axis=0)  # average across 4 models -> (n_val, n_adrs)
            final_predictions[val_idx] = fold_avg

        
        # Save run results (use averaged AUPR)
        all_runs_results.append({
            'run_idx': run_idx + 1,
            'seed': run_seed,
            'aupr': aupr_meta,
            'predictions': all_fold_predictions,
            'final_matrix': final_predictions
        })
        tf.keras.backend.clear_session()
        del X_meta_folds, y_meta_folds, meta_coefs_cv, meta_cv_auprs, all_fold_predictions, final_predictions
        gc.collect()
    


    # =============================================
    # ANÁLISIS FINAL DE ESTABILIDAD
    # =============================================
    print(f"\n{'#'*70}")
    print("# FINAL STABILITY ANALYSIS")
    print(f"{'#'*70}")
    
    auprs = [r['aupr'] for r in all_runs_results]
    
    print(f"\nAUPR across {n_runs} independent runs:")
    for i, result in enumerate(all_runs_results, 1):
        print(f"  Run {i} (seed={result['seed']}): {result['aupr']:.4f}")
    
    print(f"\n{'='*70}")
    print("STATISTICS")
    print(f"{'='*70}")
    print(f"Mean AUPR:   {np.mean(auprs):.4f}")
    print(f"Std AUPR:    {np.std(auprs):.4f}")
    print(f"Min AUPR:    {np.min(auprs):.4f}")
    print(f"Max AUPR:    {np.max(auprs):.4f}")
    print(f"Range:       {np.max(auprs) - np.min(auprs):.4f}")
    print(f"CV (coef. variation): {(np.std(auprs)/np.mean(auprs))*100:.2f}%")
    
    # Encontrar mejor run
    best_run = max(all_runs_results, key=lambda x: x['aupr'])
    
    print(f"\n{'='*70}")
    print("BEST RUN")
    print(f"{'='*70}")
    print(f"Run #{best_run['run_idx']} (seed={best_run['seed']})")
    print(f"AUPR: {best_run['aupr']:.4f}")
    
    print("\nEnsemble: Simple average of 4 base autoencoders")
    
    # Recomendación para reportar
    print(f"\n{'='*70}")
    print("RECOMMENDATION FOR REPORTING")
    print(f"{'='*70}")
    print(f"Report as: {np.mean(auprs):.4f} Â± {np.std(auprs):.4f}")
    print(f"Best result: {np.max(auprs):.4f}")
    print(f"Methodology: 5-fold CV averaged over {n_runs} runs with different seeds")
    

    print(f"\n{'='*70}")
    print("AGGREGATED METRICS ACROSS RUNS")
    print(f"{'='*70}")

    per_run_metrics = []
    for r in all_runs_results:
        metrics_dict, _ = compute_all_metrics(X, r['final_matrix'], k=15, beta=1.0)
        per_run_metrics.append(metrics_dict)

    keys = sorted({k for d in per_run_metrics for k in d.keys()})
    summary = {}
    for k in keys:
        vals = np.array([d.get(k, np.nan) for d in per_run_metrics], dtype=float)
        mean_v = float(np.nanmean(vals)) if vals.size > 0 else float('nan')
        std_v = float(np.nanstd(vals)) if vals.size > 0 else float('nan')
        summary[k] = {'mean': mean_v, 'std': std_v}
        print(f"  {k}: mean={mean_v:.4f}, std={std_v:.4f}")

    best_run['metrics_summary'] = summary

    return best_run, all_runs_results