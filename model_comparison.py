import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scikit_posthocs as sp
import scipy.stats as stats
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, confusion_matrix
from artifact_utils import load_artifact_bundle

def main():
    """
    Main function to run the model comparison analysis.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette('husl')

    MODEL_LABELS = {
        'saint': 'SAINT',
        'lightgbm': 'LightGBM',
        'xgboost': 'XGBoost',
        'catboost': 'CatBoost',
        'autogluon': 'AutoGluon',
        'askl2': 'ASKL 2.0'
    }

    def g_mean_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return the geometric mean of per-class recalls, robust to empty classes."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            row_sums = np.sum(cm, axis=1)
            mask = row_sums > 0
            recalls = np.diag(cm)[mask] / row_sums[mask]
            if len(recalls) == 0:
                return 0.0
            return np.prod(recalls) ** (1.0 / len(recalls))
        except Exception:
            return 0.0

    def get_auc_ovo(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute ROC AUC handling binary and multiclass (one-vs-one) probability arrays."""
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    return roc_auc_score(y_true, y_prob[:, 1])
                return roc_auc_score(y_true, y_prob)
            return roc_auc_score(y_true, y_prob, multi_class='ovo')
        except Exception:
            return np.nan

    bundles_by_model = {
        key: load_artifact_bundle(key)
        for key in MODEL_LABELS
    }

    evaluation_rows = []
    missing_models = []

    for model_key, bundles in bundles_by_model.items():
        model_label = MODEL_LABELS.get(model_key, model_key.upper())
        if not bundles:
            missing_models.append(model_label)
            continue

        for bundle in bundles:
            arrays = bundle.arrays
            y_true = arrays['y_true']
            y_pred = arrays.get('y_pred')
            y_prob = arrays.get('y_prob')

            acc = accuracy_score(y_true, y_pred) if y_pred is not None else np.nan
            gmean = g_mean_multiclass(y_true, y_pred) if y_pred is not None else np.nan
            auc = get_auc_ovo(y_true, y_prob) if y_prob is not None else np.nan
            try:
                ce = log_loss(y_true, y_prob) if y_prob is not None else np.nan
            except Exception:
                ce = np.nan

            runtime = bundle.metadata.get('runtime_seconds', np.nan)
            metrics_meta = bundle.metadata.get('metrics', {})

            evaluation_rows.append({
                'Dataset': bundle.dataset_name,
                'Dataset Slug': bundle.dataset_slug,
                'Model Key': model_key,
                'Model': model_label,
                'AUC_OVO': auc,
                'Accuracy': acc,
                'G-Mean': gmean,
                'Cross_Entropy': ce,
                'Time (s)': runtime,
                'CV Score': metrics_meta.get('cv_accuracy'),
                'Test Score': metrics_meta.get('test_accuracy')
            })

    if missing_models:
        print('⚠ Nenhum artefato encontrado para:')
        for model_name in missing_models:
            print(f"  - {model_name}")
    else:
        print('✓ Artefatos encontrados para todos os modelos configurados.')

    if evaluation_rows:
        results_df = pd.DataFrame(evaluation_rows)
        results_df.to_csv('evaluation_results.csv', index=False)
        print("Resultados salvos em 'evaluation_results.csv'.")
    else:
        results_df = pd.DataFrame()
        print('Nenhum resultado disponível. Verifique os diretórios de artefatos.')

    if results_df.empty:
        print('Sem resultados para resumir.')
    else:
        cols = ['AUC_OVO', 'Accuracy', 'G-Mean', 'Cross_Entropy', 'Time (s)']
        print("\n## Médias por modelo")
        print(results_df.groupby('Model')[cols].mean())

    ranking_metrics = [
        ('Accuracy', True),
        ('AUC_OVO', True),
        ('G-Mean', True),
        ('Cross_Entropy', False),
        ('Time (s)', False)
    ]

    if results_df.empty:
        print('⚠ Resultados indisponíveis. Execute a célula de carregamento primeiro.')
    else:
        valid_datasets = results_df['Dataset'].nunique()
        if valid_datasets == 0:
            print('Nenhum dataset disponível para ranqueamento.')
        else:
            fig, axes = plt.subplots(1, len(ranking_metrics), figsize=(5.2 * len(ranking_metrics), 4.5))
            fig.suptitle('Rankings dos Modelos por Dataset', fontsize=16, fontweight='bold')

            for ax, (metric, maximize) in zip(np.atleast_1d(axes), ranking_metrics):
                pivot = results_df.pivot(index='Dataset', columns='Model', values=metric).dropna()
                if pivot.empty:
                    ax.set_axis_off()
                    ax.set_title(f'Sem dados: {metric}')
                    continue

                ranks = pivot.apply(lambda row: (-row).rank() if maximize else row.rank(), axis=1)
                sns.heatmap(
                    ranks.T,
                    annot=True,
                    fmt='.1f',
                    cmap='RdYlGn_r',
                    ax=ax,
                    cbar_kws={'label': 'Ranking'}
                )
                ax.set_title(metric)
                ax.set_xlabel('Dataset')
                ax.set_ylabel('Modelo')

            plt.tight_layout()
            plt.savefig('model_rankings_heatmap.png')
            plt.close()
            print("\nHeatmaps de rankings salvos em 'model_rankings_heatmap.png'.")


    metric_panels = [
        ('AUC_OVO', 'AUC OVO'),
        ('Accuracy', 'Accuracy'),
        ('G-Mean', 'G-Mean'),
        ('Cross_Entropy', 'Cross-Entropy'),
        ('Time (s)', 'Tempo Total (s)')
    ]

    if results_df.empty:
        print('⚠ Resultados indisponíveis. Execute a célula de carregamento primeiro.')
    else:
        rows, cols = 2, 3
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.5))
        fig.suptitle('Comparação de Métricas por Modelo', fontsize=16, fontweight='bold')

        for idx, (metric, title) in enumerate(metric_panels):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            data = results_df.groupby('Model')[metric].agg(['mean', 'std']).reset_index()
            ax.bar(data['Model'], data['mean'], yerr=data['std'], capsize=5, alpha=0.8)
            ax.set_title(title)
            ax.set_ylabel(title)
            ax.set_xlabel('Modelo')
            ax.tick_params(axis='x', rotation=35)
            ax.grid(True, alpha=0.3)

        if len(metric_panels) < rows * cols:
            for k in range(len(metric_panels), rows * cols):
                fig.delaxes(axes.flatten()[k])

        plt.tight_layout()
        plt.savefig('model_metrics_panel.png')
        plt.close()
        print("\nPainel de métricas salvo em 'model_metrics_panel.png'.")

    def analyze_results(results_df, metric='Accuracy', maximize=True):
        """Run Friedman + Nemenyi analysis for a metric and print ranks, p-values, and CD diagram."""
        print(f"\n{'#'*60}")
        print(f"ANÁLISE ESTATÍSTICA: {metric}")
        print(f"{ '#'*60}")

        df_metric = results_df.copy()
        df_metric[metric] = pd.to_numeric(df_metric[metric], errors='coerce')

        pivot_df = df_metric.pivot(index='Dataset', columns='Model', values=metric)
        original_count = len(pivot_df)
        pivot_df = pivot_df.dropna()
        final_count = len(pivot_df)

        if final_count < original_count:
            print(f"Aviso: {original_count - final_count} datasets removidos por dados incompletos.")

        if final_count < 5:
            print('Aviso: Número de datasets muito baixo para testes estatísticos confiáveis (>=5 recomendado).')

        if final_count == 0:
            print('Erro: Nenhum dataset válido para análise.')
            return

        data_for_test = pivot_df.values
        statistic, p_value = stats.friedmanchisquare(*[data_for_test[:, i] for i in range(data_for_test.shape[1])])

        print(f"Estatística χ²: {statistic:.4f}")
        print(f"p-value: {p_value:.4e}")
        alpha = 0.05
        reject_h0 = p_value < alpha
        if reject_h0:
            print(f"✓ REJEITA H0 (p < {alpha}): Existem diferenças significativas entre os modelos.")
        else:
            print(f"✗ FALHA AO REJEITAR H0 (p >= {alpha}): Não há evidência suficiente de diferença.")

        ranks = pivot_df.rank(axis=1, ascending=not maximize)
        avg_ranks = ranks.mean().sort_values()

        print('\nRanks Médios (1 = melhor):')
        for i, (model, rank) in enumerate(avg_ranks.items(), 1):
            print(f"  {i}. {model:15s} → Rank Médio: {rank:.2f}")

        if reject_h0:
            print("\n--- Pivot Table for Nemenyi Test ---")
            print(pivot_df)
            nemenyi_matrix = sp.posthoc_nemenyi_friedman(pivot_df)
            nemenyi_result = pd.DataFrame(
                nemenyi_matrix,
                index=pivot_df.columns,
                columns=pivot_df.columns
            )
            print('\nMatriz de p-values (Teste de Nemenyi):')
            print(nemenyi_result.round(4))

            print('\nPARES SIGNIFICATIVAMENTE DIFERENTES (p < 0.05):')
            found_diff = False
            models = nemenyi_result.columns.tolist()
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    p_val = nemenyi_result.iloc[i, j]
                    if p_val < 0.05:
                        rank_diff = abs(avg_ranks[models[i]] - avg_ranks[models[j]])
                        print(f"  • {models[i]} vs {models[j]}: p={p_val:.4f} (Δrank={rank_diff:.2f})")
                        found_diff = True
            if not found_diff:
                print('  Nenhum par apresentou diferença estatisticamente significativa.')

            # Substituído pelo plot da biblioteca
            fig = sp.critical_difference_diagram(avg_ranks, nemenyi_result)
            plt.title(f'Diagrama de Diferença Crítica para {metric} (α={alpha})', fontsize=14, fontweight='bold')
            filename = f'cd_diagram_{metric.lower().replace(" ", "_")}.png'
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            print(f"\nDiagrama de Diferença Crítica salvo como '{filename}'.")
        else:
            print('\nTeste post-hoc não executado (Friedman não rejeitou H0).')


    print("\n" + "="*80)
    print("ANÁLISE ESTATÍSTICA BASEADA NO PROTOCOLO DE DEMŠAR (2006)")
    print("="*80)

    if 'results_df' in locals() and not results_df.empty:
        analyze_results(results_df, metric='Accuracy', maximize=True)
        analyze_results(results_df, metric='AUC_OVO', maximize=True)
        analyze_results(results_df, metric='Cross_Entropy', maximize=False)
    else:
        print('⚠ Nenhum resultado disponível para análise. Carregue os artefatos primeiro.')


if __name__ == '__main__':
    main()