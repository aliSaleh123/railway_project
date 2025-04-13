import pickle
from pathlib import Path
import os
import crack_initiation as cr_init
import crack_propagation as cr_prop
from config import BASE_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR, RESULTS_DIR

# -------------------------------------------------------------------------------------------------------------------- #
# Crack Initiation Steps


# # step1: combine raw data into one data frame and save it
# cr_init.combine_raw_data(
#     raw_data_dir=RAW_DATA_DIR / "usage",
#     combined_raw_data_path=PROCESSED_DATA_DIR / "usage" / "combined_raw_data.pkl"
#     )
#
#
# # step2: create MGT data frame and save it
# cr_init.create_mgt_df(
#     combined_raw_data_path=PROCESSED_DATA_DIR / "usage" / "combined_raw_data.pkl",
#     mgt_data_path=RESULTS_DIR / "mgt_df.pkl"
# )
#
#
# # step3: create Damage Index (DI) data frames for different cases and save it
# cr_init.create_di_dfs(
#     combined_raw_data_path=PROCESSED_DATA_DIR / "usage" / "combined_raw_data.pkl",
#     materials_properties_path=RAW_DATA_DIR / "materials_properties.xlsx",
#     crack_init_coefficients_path=RAW_DATA_DIR / "crack_init_coefficients.xlsx",
#     results_dir=PROCESSED_DATA_DIR / "usage" / "di_dfs1"
#     )
#
# # step4: create crack initiation data
# cr_init.create_crack_init_times(
#     di_dfs_dir=PROCESSED_DATA_DIR / "usage" / "di_dfs",
#     crack_init_times_path=RESULTS_DIR / "crack_init_times.pkl",
# )
#
# # step5: create crack initiation dists
# cr_init.create_crack_init_dists(
#     crack_init_times_path=RESULTS_DIR / "crack_init_times.pkl",
#     crack_init_dists_path=PROCESSED_DATA_DIR / "usage" / "crack_init_dists.pkl"
# )

#
# # step5: creating some plots to visualize the results
# cr_init.plot_DI_vs_days()
# cr_init.plot_MGT_vs_days()
# cr_init.plot_accumulated_DI_vs_days()
# cr_init.plot_accumulated_MGT_vs_days()
# cr_init.plot_crack_init_dists(figures_path=LOGS_DIR / "crack_init_dists")

# -------------------------------------------------------------------------------------------------------------------- #
# Crack Propagation Steps


# # step1: combine tonnage raw data
# cr_prop.combine_tonnage_data(
#     RAW_DATA_DIR / 'tonnage',
#     PROCESSED_DATA_DIR / 'cracks' / 'tonnage_dfs.pkl'
#     )
#
#
#
#
# # step2: process eddy current data
# cr_prop.process_ec_data(
#     raw_eddy_current_path = RAW_DATA_DIR / "cracks" / "eddy_current_measurements.xls",
#     tonnage_dfs_path = PROCESSED_DATA_DIR / 'cracks' / 'tonnage_dfs.pkl',
#     processed_eddy_current_path = PROCESSED_DATA_DIR / "cracks" / "eddy_current_measurements.pkl"
#     )
#
# # step3: save all the names of spoortaks in a file in order to use them to relate US to tonnage data
# cr_prop.get_spoortak_names(
#     raw_eddy_current_path = RAW_DATA_DIR / "cracks" / "eddy_current_measurements.xls",
#     spoortak_dict_path = PROCESSED_DATA_DIR / "cracks" / "spoortak_dict.pkl"
# )
#
# # step4: process ultrasonic measurements data
# cr_prop.process_us_data(
#     raw_ultrasonic_path=RAW_DATA_DIR / "cracks" / "ultrasonic_measurements.xlsx",
#     processed_ultrsonic_path=PROCESSED_DATA_DIR / "cracks" / "ultrasonic_measurements.pkl",
#     spoortak_dict_path=PROCESSED_DATA_DIR / "cracks" / "spoortak_dict.pkl",
#     tonnage_dfs_path=PROCESSED_DATA_DIR / 'cracks' / 'tonnage_dfs.pkl',
# )
#



# ---------------------------------------------------------------------------------------------------------------------#
# Create transitions

# # Create transitions based on different maximum backward depth limits
# # the maximum backward depth filters the backward transitions that exceeds its value
# for max_back_depth in [0, 1, 2, 3, 4, 5]:
#     cr_prop.create_transitions(
#         processed_ec_sections_path=PROCESSED_DATA_DIR / "cracks" / "eddy_current_measurements.pkl",
#         processed_us_sections_path=PROCESSED_DATA_DIR / "cracks" / "ultrasonic_measurements.pkl",
#         results_path=RESULTS_DIR / "transitions" / f"at_depth_limit_{max_back_depth}.pkl",
#         max_back_depth=max_back_depth
#     )
#
# # plot the distributions of transitions of max_back_depth = 1
# cr_prop.plot_transitions(
#     transitions_path=RESULTS_DIR / "transitions" / "at_depth_limit_1.pkl",
#     figs_dir=LOGS_DIR / 'distributions'
# )
#
#
# # test the propagation model using the different transitions results
# for max_back_depth in [0, 1, 2, 3, 4, 5]:
#     cr_prop.test_propagation_model(
#         transitions_path=RESULTS_DIR / "transitions" / f"at_depth_limit_{max_back_depth}.pkl",
#         results_dir=LOGS_DIR / 'transitions',
#         name=max_back_depth,
#         num_simulations=2000,
#         mgt_end=200,
#     )
#
# # plot the propagation of cracks of the propagation tests
# cr_prop.plot_test_simulations(results_dir=LOGS_DIR / 'transitions')


""" 
max_back_depth = 1 is chosen, this results in excluding all backward transitions that are greater than 2 mm when 
creating the transitions and the propagation model.

backward transitions that result in less than max_back_depth are kept to include the effect of rail wear on reducing 
cracks and due to the possibility of having measurements errors.
"""

# # ---------------------------------------------------------------------------------------------------------------------#
# # get the affected length portion
#
# cr_prop.calculate_affected_length_portion(
#     processed_eddy_current_path=PROCESSED_DATA_DIR / "cracks" / "eddy_current_measurements.pkl",
#     results_file_path=RESULTS_DIR / "affected_length_portion.pkl",
# )



# ---------------------------------------------------------------------------------------------------------------------#
# create the PN model


# import maintenance_model as model
# model.main_procedure(
#     mgt_data_path=RESULTS_DIR / "mgt_df.pkl",
#     crack_init_times_path=RESULTS_DIR / "crack_init_times.pkl",
#     transitions_path=RESULTS_DIR / "transitions" / "at_depth_limit_1.pkl",
#     affected_length_portion_path=RESULTS_DIR / "affected_length_portion.pkl"
# )









