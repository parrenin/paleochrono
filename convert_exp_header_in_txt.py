import os
import sys
import csv

dir = sys.argv[1]
inclusive = True

def convert_file(pattern, labels):
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in files:
            if name == pattern:
                path = os.path.join(root, name)
                print(path)
                f = open(path)
                lines_comments = []
                lines_data = []
                for line in f:
                    line1 = line.strip(" \t")            
                    line = line.strip(" \t\n")            
                    if len(line1) == 0:
                        pass
                    elif line1[0] == "#":
                        lines_comments.append(line)
                    elif line1[0:5] =="depth" or line1[0:9] == "air_depth" or line1 == '\n':
                        pass
                    else:                    
                        lines_data.append(line)
                f.close()
                if len(lines_data) > 0:
                    # print(repr(lines_data[0]))
                    dialect = csv.Sniffer().sniff(lines_data[0])
                    sep = dialect.delimiter
                else:
                    sep = "\t"
                if name == 'isotopes.txt':
                    nb_cols = len(lines_data[0].split(sep))
                    if nb_cols == 4:
                        header = "depth"+sep+"d18O"+sep+"deut"+sep+"d18Osw\n"
                        print('4 columns file.')
                    elif nb_cols == 2:
                        header = "depth"+sep+"deut\n"
                        print("2 columns file.")
                    else:
                        print('incorrect number of columns in isotopes.txt file')
                else:
                    header = ''                
                    for label in labels:
                        header += label+sep
                    header += 'comment\n'
                f = open(path, 'w')
                for line in lines_comments:
                    f.write(line+'\n')
                f.write(header)
                for line in lines_data:
                    line1 = sep
                    line = line1.join(line.split())
                    f.write(line+'\n')
                f.close()

                
labels = ["depth","age","age_unc"]
for name in ['age_horizons.txt', 'ice_age_horizons.txt', 'air_age_horizons.txt']:
    convert_file(name, labels)
convert_file('density.txt', ["depth", "rel_dens"])
convert_file('solid_fraction.txt', ["depth", "rel_dens"])
convert_file('deposition.txt', ["depth", "deporate", "rel_unc"])
convert_file('thinning.txt', ["depth", "thinning", "rel_unc"])
convert_file('lock_in_depth.txt', ["depth", "LID", "rel_unc"])
labels = ["depth_top", "depth_bot", "duration", "dur_unc"]
for name in ['ice_age_intervals.txt', 'air_age_intervals.txt', 'age_intervals.txt']:
    convert_file(name, labels)
labels = ["depth1", "depth2", "age_unc"]
for name in ['synchro_horizons.txt', 'iceice_synchro_horizons.txt', 'airair_synchro_horizons.txt',
             "iceair_synchro_horizons.txt", "airice_synchro_horizons.txt",
             "ice_synchro_horizons.txt", "air_synchro_horizons.txt"]:
    convert_file(name, labels)
convert_file('delta_depths.txt', ["air_depth", "Ddepth", "Ddepth_unc"])
convert_file('isotopes.txt', ['depth', 'd18O', 'deut', 'd18Osw'])

