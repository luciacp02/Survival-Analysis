from lifelines.statistics import logrank_test
import pandas as pd
import numpy as np

def perform_log_rank_test(df, group_by, alpha: float = 0.05):
    # verificar que haya exactamente 2 grupos
    vals = sorted(pd.unique(df[group_by].dropna()))
    if len(vals) != 2:
        return pd.DataFrame({
            'Covariable':[group_by],
            'Grupo A':[vals[0] if len(vals)>0 else None],
            'Grupo B':[vals[1] if len(vals)>1 else None],
            'n A':[np.nan],'n B':[np.nan],
            'test_statistic':[np.nan],'p_value':[np.nan],'-log2(p)':[np.nan],
            'Decisión':['No evaluable'],
            'Conclusión':['La covariable no tiene exactamente dos grupos.']
        })

    a, b = vals[0], vals[1]
    gA = df[df[group_by] == a]
    gB = df[df[group_by] == b]

    res = logrank_test(
        gA['date'], gB['date'],
        event_observed_A=gA['final_result'],
        event_observed_B=gB['final_result']
    )

    p = float(res.p_value)
    stat = float(res.test_statistic)
    decision = 'Rechazar H0' if p < alpha else 'No rechazar H0'
    interpretacion = ('Hay diferencias significativas entre las curvas'
                      if p < alpha else
                      'No se observan diferencias significativas entre las curvas')

    return pd.DataFrame({
        'Covariable':[group_by],
        'Grupo A':[a],
        'Grupo B':[b],
        'n A':[int(len(gA))],'n B':[int(len(gB))],
        'test_statistic':[stat],'p_value':[p],
        '-log2(p)':[(-np.log2(p) if p>0 else np.inf)],
        'Decisión':[decision],
        'Conclusión':[interpretacion],
    })
