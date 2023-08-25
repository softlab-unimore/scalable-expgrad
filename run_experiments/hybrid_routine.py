import run

if __name__ == "__main__":
    # run.launch_experiment_by_id('s_h_1.0.TEST')
    # run.launch_experiment_by_id('acs_h_gs1_EO_1.0')

    # run.launch_experiment_by_id('acs_h_eps_1.0')

    # run.launch_experiment_by_id('acs_h_eps_1.E0')
    # run.launch_experiment_by_id('acs_h_eps_1.LGBM0')
    # run.launch_experiment_by_id('sigmod_h_exp_1.0')
    # run.launch_experiment_by_id('acs_h_gs1_EO_1.0')
    # run.launch_experiment_by_id('s_h_EO_1.0')
    # run.launch_experiment_by_id('acs_eps_EO_1.0')

    # doing
    # run.launch_experiment_by_id('acsE_h_gs1_1.0')

    done_conf = [
        'sigmod_h_exp_1.0',
        's_h_exp_EO_1.0',
        's_h_EO_1.0',
        'acs_eps_EO_1.0',
        'acs_eps_EO_1.1',
        'acs_eps_EO_2.0',
        'acs_eps_EO_2.1',
        'acs_h_gs1_EO_2.0',
        'acs_h_gsSR_1.0',
        'acs_h_gsSR_2.0',
        'acs_h_gsSR_1.1',
        'acsE_h_gsSR_1.1',
        'acsE_h_gs1_1.0',

        's_h_EO_1.0',
        'acs_h_gsSR_2.1',
        'acs_h_gs1_1.1',
        'sigmod_h_exp_2.0',
        'sigmod_h_exp_3.0',
        'acs_eps_EO_2.0',
        'acs_eps_EO_2.1',
        'acs_h_gs1_EO_1.0',
        'acs_h_gs1_EO_2.0',
        'acs_h_gsSR_2.0',
        'acs_h_gsSR_2.1',

        's_tr_1.1',
        's_tr_2.0',
        's_tr_2.1',
        'f_eta0_1.0',
        'f_eta0_2.0',


        's_h_1.0r',
        's_h_EO_1.0r',

        'acs_eps_EO_1.0r',
        'acs_h_eps_1.0r',

        's_h_exp_1.0r',
        's_h_exp_EO_1.0r',
        's_h_exp_2.0r',
        's_h_exp_EO_2.0r',
        'acs_h_gs1_1.0r',
        'acs_h_gs1_EO_1.0r',
        'acs_h_gsSR_1.0r',
        'acs_h_gsSR_EO_1.0r',
        # to update
        's_c_1.0r',
        's_tr_1.0r',
        's_tr_1.1r',
        's_tr_2.0r',
        's_tr_2.1r',
        'f_eta0_eps.1',
        'f_eta0_eps.2',
    ]

    conf_todo = [
        # 'acs_h_gs1_1.test',
        #'f_eta0_1.0.test',



        #'s_zDI_1.1',



    ]

    for x in conf_todo:
        run.launch_experiment_by_id(x)
