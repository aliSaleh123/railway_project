import dill
import matplotlib.pyplot as plt


def calculate_affected_length_portion(processed_eddy_current_path, results_file_path):
    # read the processed sections that contain eddy current measurements
    with (open(processed_eddy_current_path, "rb")) as f:
        processed_sections_ec = dill.load(f)

    # get the lengths and the affected lengths of different sections
    lengths, affected_lengths = get_affected_lengths(processed_sections_ec)

    # # get the percentages of different sections
    # percentages = [c / l for l, c in zip(lengths, affected_lengths)]

    # # plot percentages of affected lengths as a function of section length
    # plt.scatter(lengths, percentages)
    # plt.xlabel("Length of section (m)")
    # plt.ylabel("Affected length percentage")
    # plt.show()

    # get the portion of affected length to the total length of sections
    affected_length_portion = sum(affected_lengths) / sum(lengths)

    with open(results_file_path, 'wb') as f:
        dill.dump(affected_length_portion, f)

    return affected_length_portion


def get_affected_lengths(processed_sections_ec):
    affected_lengths = []
    lengths = []
    for geo_code in processed_sections_ec:
        for sec in processed_sections_ec[geo_code]:

            # define a dictionary to collect cracks at each date
            dates_dict = {}
            length_section = 0
            for crack in sec.cracks:

                length = max(abs(crack.end - crack.start), 1)
                length_section += length
                for date in crack.conditions:
                    if date in dates_dict:
                        dates_dict[date].append([crack.conditions[date], length])
                    else:
                        dates_dict[date] = [[crack.conditions[date], length]]

            # get the percentage of cracks at each date
            for date in dates_dict:

                # check if there is any crack of length greater than 0 and
                # get the length of the cracks within the section
                length_cracks = 0
                contain_cracks = False
                for crck in dates_dict[date]:
                    if crck[0] != 0 and crck[0] != 'unknown':
                        length_cracks += crck[1]
                        contain_cracks = True

                # if the section contains cracks at that date, get the percentage
                if contain_cracks:
                    affected_lengths.append(length_cracks)
                    lengths.append(length_section)

    return lengths, affected_lengths
