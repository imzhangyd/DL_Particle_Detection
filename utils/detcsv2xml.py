import glob
import pandas as pd
import os
import argparse


__author__ = "Yudong Zhang"


def parse_args_():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--det_folder', type=str, default='./Log/20240301_11_12_50_MICROTUBULE_SNR7_deepBlink_eval/prediction_0.5')
    parser.add_argument('--detfor_track', type=str, default='./detfor_track')   

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    opt = parse_args_()
    os.makedirs(opt.detfor_track, exist_ok=True)
    allcsvpaths = glob.glob(os.path.join(opt.det_folder,'**.csv'))
    allmvnames = set([os.path.split(pa)[-1].split(' t')[0] for pa in allcsvpaths])

    for mvna in allmvnames:

        sc = mvna.split(' ')[0]
        sn = mvna.split(' ')[2]
        de = mvna.split(' ')[4]

        thispalist = glob.glob(os.path.join(opt.det_folder,'{} snr {} density {}**.csv'.format(sc,sn,de)))
        all_df = pd.DataFrame(columns=["pos_x","pos_y","frame"])

        # merge csv
        for thispa in thispalist:
            temp_df = pd.read_csv(thispa)
            temp_df['frame'] = int(thispa.split(' ')[-2].replace('t',''))
            temp_df = temp_df[["pos_x",'pos_y','frame']]
            all_df = all_df.append(temp_df)

        all_df.reset_index(drop=True, inplace=True)
        csvsavepa = os.path.join(opt.detfor_track,'{} snr {} density {}.csv'.format(sc,sn,de))
        all_df.to_csv(csvsavepa)

        # csv to xml
        pos_np_total = all_df.values.tolist()
        xmlsavepa = csvsavepa.replace('csv','xml')
        with open(xmlsavepa, "w+") as output:
            output.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
            output.write('<root>\n')
            output.write('<TrackContestISBI2012 SNR="' + sn + '" density="' + de + '" scenario="' + sc + \
                            '" ' + 'Deepblinkdet' + '="' + '0000000' + '">\n')


            output.write('<particle>\n')
            for pos in pos_np_total:
                output.write('<detection t="' + str(int(pos[-1])) +
                                '" x="' + str(pos[0]) +
                                '" y="' + str(pos[1]) + '" z="0"/>\n')
            output.write('</particle>\n')
            output.write('</TrackContestISBI2012>\n')
            output.write('</root>\n')
            output.close()
