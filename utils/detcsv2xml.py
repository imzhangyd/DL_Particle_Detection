import glob
import pandas as pd
import os
import argparse


__author__ = "Yudong Zhang"


def parse_args_():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--det_folder', type=str, default='./Log/20240301_11_12_50_MICROTUBULE_SNR7_deepBlink_eval/prediction_0.5')
    parser.add_argument('--detfor_track', type=str, default='./detfor_track')
    parser.add_argument('--mvnames',type=str,nargs='+',default=[])
    

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    opt = parse_args_()
    os.makedirs(opt.detfor_track, exist_ok=True)
    
    if opt.mvnames == []:
        allcsvpaths = glob.glob(os.path.join(opt.det_folder,'**.csv'))
        allmvnames = set([os.path.split(pa)[-1].split(' t')[0] for pa in allcsvpaths])
        assert len(allmvnames) > 0, print('Error: No movies will be processed!')
    else:
        allmvnames = opt.mvnames

    for mvna in allmvnames:
        print('[Info] Collate {}'.format(mvna))
        if 'snr' in mvna and 'density' in mvna:
            sc = mvna.split(' ')[0]
            sn = mvna.split(' ')[2]
            de = mvna.split(' ')[4]
            mvna = '{} snr {} density {}'.format(sc,sn,de)
            # thispalist = glob.glob(os.path.join(opt.det_folder,'{} snr {} density {}**.csv'.format(sc,sn,de)))
        # else:
        thispalist = glob.glob(os.path.join(opt.det_folder, f'{mvna}**.csv'))
        all_df = pd.DataFrame(columns=["pos_x","pos_y","frame"])
        thispalist.sort()
        # merge csv
        for thispa in thispalist:
            temp_df = pd.read_csv(thispa)
            if 'snr' in mvna and 'density' in mvna:
                temp_df['frame'] = int(thispa.split(' ')[-2].replace('t',''))
            else:
                temp_df['frame'] = int(thispa.split(mvna)[-1].replace('.csv',''))
            temp_df = temp_df[["pos_x",'pos_y','frame']]
            all_df = all_df.append(temp_df)

        all_df.reset_index(drop=True, inplace=True)
        if 'snr' in mvna and 'density' in mvna:
            csvsavepa = os.path.join(opt.detfor_track,'{} snr {} density {}.csv'.format(sc,sn,de))
        else:
            csvsavepa = os.path.join(opt.detfor_track,'{}.csv'.format(mvna))
        all_df.to_csv(csvsavepa)
        print(f'merge csv is saved to {csvsavepa}')
        # csv to xml
        pos_np_total = all_df.values.tolist()
        xmlsavepa = csvsavepa.replace('csv','xml')
        with open(xmlsavepa, "w+") as output:
            output.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
            output.write('<root>\n')
            if 'snr' in mvna and 'density' in mvna:
                output.write('<TrackContestISBI2012 SNR="' + sn + '" density="' + de + '" scenario="' + sc + \
                                '" ' + 'Deepblinkdet' + '="' + '0000000' + '">\n')
            else:
                output.write('<TrackContestISBI2012 SNR="' + '0' + '" density="' + 'unknown' + '" scenario="' + mvna + \
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
