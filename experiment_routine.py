import run

if __name__ == "__main__":
    done_conf = [
        # done on fairlearn-2
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
        's_c_1.0r',
        's_tr_1.0r',
        's_tr_1.1r',
        's_tr_2.0r',
        's_tr_2.1r',
        'f_eta0_eps.1',
        'f_eta0_eps.2',
        'most_frequent_sig.0r',
        'most_frequent_ACS.0r',
        'acs_to_binary1.0r',
        'acs_to_binaryEO1.0r',
        'acsE_eps_EO_1.0r',
        'acsE_h_eps_1.0r',
        'f_eta0_1.1',
        'f_eta0_2.1',
        'acsE_h_gs1_1.0r',
        'acsE_h_gs1_EO_1.0r',
        'acsE_h_gsSR_1.0r',
        'acsE_h_gsSR_1.1r',


        # doing in fairlearn-2
        'acsE_h_gs1_EO_1.1r',


        # done on fairlearn-3
        's_zDI_1.2',
        's_zDI_1.22',
        's_zEO_1.1',
        's_f_1.0r',
        's_f_1.1r',
        'acsER_bin2.0r',
        'acsER_bin2.1r',
        'acsER_bin2.2r',
        'acs_to_binary1.1r',
        'acsER_bin3.0r',
        'acsER_bin3.1r',
        'acsE_h_gs1_1.1r',
        'acsE_h_gs1_1.2r',

        # to update on fairlearn-3


        # doing on fairlearn-3

    ]



    conf_todo = [
        # next on fairlearn-2
        # 'acsER_bin3.0rtest',
        # next on fairlearn-3



        # testing
        # 'acs_h_gs1_1.test',
        # 'f_eta0_1.0.test',


    ]

    for x in conf_todo:
        run.launch_experiment_by_id(x)
