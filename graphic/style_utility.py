import itertools
from copy import deepcopy

import pandas as pd
from matplotlib.markers import MarkerStyle


def generate_map_df():
    values_dict = {}
    model_names = ['hybrid_1', 'hybrid_2', 'hybrid_3', 'hybrid_4', 'hybrid_5', 'hybrid_6']
    unconstrained = [True, False]
    active_sampling = [True, False]
    run_linprog = [True, False]
    grid_mode = ['sqrt', 'gf_1']
    to_iter = itertools.product(model_names, unconstrained, active_sampling, grid_mode)

    for t_varing in ['exp', 'gri', 'eps']:
        for t_run_lp in run_linprog:
            for t_model_name, t_unconstrained, t_active_sampling, t_grid_mode in deepcopy(to_iter):
                name = 'sub_' if t_active_sampling else ''
                name += t_model_name + ('_U' if t_unconstrained else '')
                name += '_LP_off' if t_run_lp else ''
                name += f'_{t_varing}'
                if t_grid_mode == 'gf_1':
                    name += f'_gf_1'
                    t_grid_mode = '1'

                label = f'EXPGRAD=' + ('adaptive' if t_active_sampling else 'static')
                label += ' GS=' + (
                    t_grid_mode if any('_' + x in t_model_name for x in ['1', '2', '3', '4', '6']) else 'No')
                label += ' LP=Yes'
                label += ' +U' if t_unconstrained else ''
                if 'hybrid_6' in t_model_name:
                    label += ' *e&g'
                label += ' run_linprog=F' if t_run_lp else ''
                values_dict[name] = label.strip()

            rlp_name = "_LP_off" if t_run_lp else ""
            rlp_label = ' run_linprog=F' if t_run_lp else ''
            values_dict[f'hybrid_7{rlp_name}_{t_varing}'] = 'EXPGRAD=adaptive GS=No LP=Yes' + rlp_label
            values_dict[f'expgrad_fracs{rlp_name}_{t_varing}'] = 'EXPGRAD=static GS=No LP=No' + rlp_label
        values_dict[f'unconstrained_{t_varing}'] = 'UNMITIGATED full'
        values_dict[f'unconstrained_frac_{t_varing}'] = 'UNMITIGATED=static'

    return pd.DataFrame.from_dict(values_dict, orient='index', columns=['label'])


linewidth = 0.5
markersize = 8
base_config = dict(linewidth=linewidth, elinewidth=linewidth / 2,
                   marker=MarkerStyle('1', 'left', 0, ), s=markersize, markevery=1,  # mew=linewidth / 2,
                   )


class StyleUtility:
    linewidth = linewidth
    markersize = markersize

    common_keys = ['color']
    line_keys = common_keys + ['linestyle', 'linewidth', 'elinewidth']
    marker_keys = common_keys + ['marker', 's']
    label_keys = common_keys + ['fmt', 'linewidth', 'elinewidth', 'marker', 'label']

    map_df = generate_map_df()
    other_models = ['ThresholdOptimizer', 'Calmon', 'ZafarDI']
    map_df = pd.concat([map_df,
                        pd.DataFrame.from_dict({x: x for x in other_models},
                                               orient='index', columns=['label'])
                        ])

    graphic_style_map = {
        'EXPGRAD': {'color': 'tab:blue', 'marker': 'o', 'linestyle': '-.'},
        'EXPGRAD=adaptive GS=No LP=Yes': {'color': 'tab:blue', 'marker': 'o', 'linestyle': '-.'},
        'EXPGRAD=static GS=No LP=Yes': {'color': 'tab:orange', 'marker': 'o', 'linestyle': '-.'},
        'EXPGRAD=adaptive GS=sqrt LP=Yes': {'color': 'tab:brown', 'marker': 'o', 'linestyle': '-.'},
        'EXPGRAD=adaptive GS=1 LP=Yes': {'color': 'tab:red', 'marker': 'o', 'linestyle': '-.'},
        'EXPGRAD=static GS=1 LP=Yes': {'color': 'tab:purple', 'marker': 'o', 'linestyle': '-.'},
        'EXPGRAD=static GS=sqrt LP=Yes': {'color': 'tab:green', 'marker': 'o', 'linestyle': '-.'},

        'UNMITIGATED full': {'color': 'tab:brown', 'marker': 'o', 'linestyle': '-'},
        'UNMITIGATED=static': {'color': 'tab:pink', 'marker': 'o', 'linestyle': '-.'},

        'Threshold': {'color': 'black', 'linestyle': 'solid', 'linewidth': linewidth * .5},

        'RLP=F': {'color': 'tab:red', 'marker': 's', 'linestyle': '-.', },
        'RLP=F eta0==1.0': {'color': 'tab:brown', 'marker': 'o', 'linestyle': '-.'},
        'RLP=F eta0==2.0': {'color': 'tab:orange', 'marker': 'o', 'linestyle': '-.'},
        'RLP=T eta0==1.0': {'color': 'tab:green', 'marker': 'o', 'linestyle': '-.'},
        'RLP=T eta0==2.0': {'color': 'tab:red', 'marker': 'o', 'linestyle': '-.'},
        'RLP=T eta0==3.0': {'color': 'tab:purple', 'marker': 'o', 'linestyle': '-.'},



        'ThresholdOptimizer': {'color': 'tab:green', 'marker': 'o', 'linestyle': '-.'},
        'Calmon': {'color': 'tab:red', 'marker': 'o', 'linestyle': '--'},
        'ZafarDI': {'color': 'tab:purple', 'marker': 'o', 'linestyle': 'dotted'},


    }

    graphic_style_map = {key: dict(base_config, label=key, **value) for key, value in graphic_style_map.items()}

    @staticmethod
    def get_label(model_code):
        if model_code in StyleUtility.map_df.index:
            return StyleUtility.map_df.loc[model_code, 'label']
        else:
            return model_code

    @staticmethod
    def get_style(key, index=None):
        label = StyleUtility.get_label(key)
        if label not in StyleUtility.graphic_style_map.keys():
            raise KeyError(f'key {label} not in graphic_style_map')
        return StyleUtility.graphic_style_map[label]
