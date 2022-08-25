This dataset contains information about defendants arrested in 2013-2014 in Florida, 
and their subsequent assessment by the COMPAS risk assessment tool. More details: https://github.com/propublica/compas-analysis

Numrical attributes:
age
juv_fel_count: number of juvenile felonies
juv_misd_count: number of juvenile misdemeanors
juv_other_count: number of other juvenile charges
priors_count: total number of prior charges
c_days_from_compas: charged how many days before COMPAS assessment (I am not 100% sure about this)

Categorical attributes:
sex
age_cat
race
priors_count_cat
c_charge_degree: charge at the time of COMPAS assessment, M = misdemeanor and F = felony.


Class label: two_year_recid
1 = Did not re-offend/recidivate within 2 years
0 = Recidivated within 2 years