import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os, sys

sys.path.append('./src')
import sr_functions as srf
import PySimpleGUI as sg

print = sg.Print # Set print to go to a window rather than the terminal

## Set theme colors
sg.theme('Light Green 3')

## default ScaredyRat settings
inpath = 'D:/Work/ShanskyLab/TestData/ECB_SPR2020/GUI_RAW' #'./'
outpath = 'D:/Work/ShanskyLab/TestData/ECB_SPR2020/SR_GUI_OUT_TEST' #'./'
outpath2 = 'D:/Work/ShanskyLab/TestData/ECB_SPR2020/SR_GUI_CompOut' #'./'
raw_sheetSettings = {0: 'Track-Arena 1-Subject 1, Track-Arena 2-Subject 1, Track-Arena 3-Subject 1, Track-Arena 4-Subject 1',
                    1: 'Context A, Context B, FC, Extinction, EXTINCTION RETRIEVAL',
                    2: 'Context A, Context B, FC, Extinction, EXTINCTION RETRIEVAL',
                    3: 'CTXA, CTXB, FC, EXT, EXTRET',
                    4: '0, 0, 7, 20, 5'}
raw_epochSettings = {0: 'Tone',
                    1: True,
                    2: 'PreTone, Shock, PostShock',
                    3: '0,-30,0',
                    4:'-1,0,5',
                    5:'-1,5,35'}
raw_trialSettings = {0: '1',
                    1: '120',
                    2: '0.1',
                    3: '20'}


## Setup menu layouts
layout_MainWindow = [ [sg.Text('Main Window.\nUse buttons to navigate ScaredyRat settings')],
                [sg.Button('Select Input Folder')],
                [sg.Button('Select Per-Animal Output Folder')],
                [sg.Button('Select Compiled Output Folder')],
                [sg.Button('Sheet Settings')],
                [sg.Button('Trial Settings')],
                [sg.Button('Epoch and Derived Epoch Settings')],
                [sg.Button('View Settings')],
                [sg.Text('')],
                [sg.Button('Run')],
                # [sg.Button('Compile Files')],
                [sg.Text('')],
                [sg.Button('Exit')]]
sheetCol =  [
            [sg.Text('List of sheet names', size=(60,1)), sg.Input('Track-Arena 1-Subject 1, Track-Arena 2-Subject 1, Track-Arena 3-Subject 1, Track-Arena 4-Subject 1', size=(120,1))],
            [sg.Text('Detection Settings Labels', size=(60,1)), sg.Input('Context A, Context B, FC, Extinction, 2 Tone Test, Ext. Retrieval', size=(120,1))],
            [sg.Text('Trial Type List', size=(60,1)), sg.Input('Context A Habituation, Context B, FC, Extinction, Extinction Retrieval', size=(120,1))],
            [sg.Text('Trial Type Abbreviation List', size=(60,1)), sg.Input('CTXA, CTXB, FC, EXT, EXTRET', size=(120,1))],
            [sg.Text('Number of epochs per trial', size=(60,1)), sg.Input('0, 0, 7, 20, 5', size=(120,1))]
            ]
layout_SheetWin = [  [sg.Text('Sheet Information')],
            [sg.Column(sheetCol)],
            [sg.Button('Apply'), sg.Button('Ok'), sg.Button('Cancel')] ]

trialCol = [
            [sg.Text('Time Bin Duration', size=(15,1)), sg.Input('1', size=(20,1)), sg.Button('Help', key='TimeHelp')],
            [sg.Text('Baseline Duration', size=(15,1)), sg.Input('120', size=(20,1)), sg.Button('Help', key='baselineHelp')],
            [sg.Text('Freezing Threshold', size=(15,1)), sg.Input('0.1', size=(20,1)), sg.Button('Help', key='FreezeTHelp')],
            [sg.Text('Darting Threshold', size=(15,1)), sg.Input('20', size=(20,1)), sg.Button('Help', key='dartTHelp')],
            ]
layout_TrialWin = [ [sg.Text('Trial Information')],
                    [sg.Column(trialCol)],
                    [sg.Button('Apply'), sg.Button('Ok'), sg.Button('Cancel')] ]
################################################################################################
## Functions
################################################################################################
def dEpochTimeEntry(dEpochItem, dEpochTimes):
    return[sg.Text(dEpochItem, size=(20,1)), sg.Input(dEpochTimes, size=(40,1)), sg.Button('Help', key='dEpochTimeHelp')]



def parseSheetSettings(raw_sheet_setting):
    tmp1 = raw_sheet_setting[0].split(',')
    for i in range(0,len(tmp1)):
        tmp1[i] = tmp1[i].lstrip()
    sheetlist = tmp1

    tmp2 = raw_sheet_setting[1].split(',')
    for i in range(0,len(tmp2)):
        tmp2[i] = tmp2[i].lstrip()
    detectionSettingsLabel = tmp2

    tmp3 = raw_sheet_setting[2].split(',')
    for i in range(0,len(tmp3)):
        tmp3[i] = tmp3[i].lstrip()
    trialTypeFull_list = tmp3

    tmp4 = raw_sheet_setting[3].split(',')
    for i in range(0,len(tmp4)):
        tmp4[i] = tmp4[i].lstrip()
    trialType_list = tmp4

    tmp5 = raw_sheet_setting[4].split(',')
    for i in range(0,len(tmp5)):
        tmp5[i] = int(tmp5[i].lstrip())
    epochNum_list = tmp5
    return(sheetlist, detectionSettingsLabel, trialTypeFull_list, trialType_list, epochNum_list)

def parseTrialSettings(raw_trial_settings):
    binSecs = int(raw_trial_settings[0])
    baselineDuration = float(raw_trial_settings[1])
    freezeThresh = float(raw_trial_settings[2])
    dartThresh = float(raw_trial_settings[3])
    return(binSecs, baselineDuration, freezeThresh, dartThresh)

def parseEpochSettings(raw_epoch_settings):
    tmp1 = raw_epoch_settings[0].split(',')
    for i in range(0,len(tmp1)):
        if raw_epoch_settings[1]:
            tmp1[i] = tmp1[i].lstrip() + ' '
        else:
            tmp1[i] = tmp1[i].lstrip()
    epochLabel = tmp1

    tmp2 = raw_epoch_settings[2].split(',')
    for i in range(0,len(tmp2)):
        tmp2[i] = tmp2[i].lstrip()
    derivedEpoch_list = tmp2

    dEpochTime = []
    for iter01 in range(3,3+len(derivedEpoch_list)):
        dEpochTime.append((raw_epochSettings[iter01].split(',')))
    return(epochLabel, derivedEpoch_list, dEpochTime)

def printSettings(inpath, outpath, raw_sheet_settings, raw_trial_settings, raw_epoch_settings):

    print(parseSheetSettings(raw_sheet_settings))

    print(parseTrialSettings(raw_trial_settings))

    print(parseEpochSettings(raw_epoch_settings))

def run_SR(inpath, outpath, raw_sheet_settings, raw_trial_settings, raw_epoch_settings):
    sheetlist, detectionSettingsLabel, trialTypeFull_list, trialType_list, epochNum_list = parseSheetSettings(raw_sheet_settings)
    # print(detectionSettingsLabel)
    binSize, baselineDuration, freezeThresh, dartThresh = parseTrialSettings(raw_trial_settings)

    epochLabel, derivedEpoch_list, derivedEpochTiming_list = parseEpochSettings(raw_epoch_settings)

    ## Search the inpath for files
    filelist = []
    for entry in os.scandir(inpath):
        if entry.is_file():
            filelist.append(entry.path)

    for file in filelist:
        for sheet in sheetlist:
            ## set input/output info
            ID,ctx,anim = srf.animal_read(inpath,file,sheet)
            # print(ctx)
            # print(ID)

            # Check the context, skipping if the file isn't an excel sheet or properly labeled
            if(ID == "-1" or ID=="nan" or (isinstance(ID,float) and math.isnan(float(ID)))or (isinstance(ID,int) and ID==-1)):
                print('Animal Detection Failure')
                continue
            else:
                for i in range(0,len(detectionSettingsLabel)):
                    # print(detectionSettingsLabel[i])
                    if(ctx==detectionSettingsLabel[i]):
                        # print(ctx)
                        # print(detectionSettingsLabel[i])
                        trialTypeFull = trialTypeFull_list[i]
                        trialType = trialType_list[i]
                        nEpochEvents = epochNum_list[i]
                        # print(nEpochEvents)

            #Get baseline data
            baseline = {}
            label = 'Recording time'
            baseline = anim[anim[label]<=baselineDuration]
            baselineFreezing, bFTs = srf.get_baseline_freezing(baseline, freezingThreshold=freezeThresh, binSecs=binSize)
            BaselineOutfile = outpath + '/' + trialType + '-baseline-freezing-{}.csv'
            BaselineOutfile = BaselineOutfile.format(ID)
            bFreezing = pd.concat([baselineFreezing],axis=1)
            bFreezing.to_csv(BaselineOutfile)

            # print(nEpochEvents)
            if(nEpochEvents==0):
                continue
            epoch_maxVel = [0]*len(epochLabel)
            epoch_meanVel = [0]*len(epochLabel)
            epoch_medVel = [0]*len(epochLabel)
            epoch_semVel = [0]*len(epochLabel)
            epochFreezing = [0]*len(epochLabel)
            epochFTs = [0]*len(epochLabel)
            epochDarting = [0]*len(epochLabel)
            epochDTs = [0]*len(epochLabel)

            dEpoch_maxVel = [[0]*len(derivedEpoch_list)]*len(epochLabel)
            dEpoch_meanVel = [[0]*len(derivedEpoch_list)]*len(epochLabel)
            dEpoch_medVel = [[0]*len(derivedEpoch_list)]*len(epochLabel)
            dEpoch_semVel = [[0]*len(derivedEpoch_list)]*len(epochLabel)
            dEpochFreezing = [[0]*len(derivedEpoch_list)]*len(epochLabel)
            dEpochFTs = [[0]*len(derivedEpoch_list)]*len(epochLabel)
            dEpochDarting = [[0]*len(derivedEpoch_list)]*len(epochLabel)
            dEpochDTs = [[0]*len(derivedEpoch_list)]*len(epochLabel)

            #find each epoch and derived epoch
            for i in range(0,len(epochLabel)):
                epc = srf.find_delim_segment(anim,nEpochEvents,epochLabel[i])
                # print('EPC')
                # print(epc)
                #Epoch max velocity
                epoch_maxVel[i] = srf.get_top_vels(epc,1,nEpochEvents)#
                # print('EPC')
                # print(epc)
                # print(nEpochEvents)
                epoch_maxVel_file = outpath + '/' + trialType + '-' + epochLabel[i] + '-max-vels-{}.csv'
                epoch_maxVel_file = epoch_maxVel_file.format(ID)
                epoch_maxVel[i].to_csv(epoch_maxVel_file)

                #Epoch Mean velocity
                epoch_meanVel[i] = srf.get_means(epc,epochLabel[i],nEpochEvents)#
                epoch_meanVel_file = outpath + '/' + trialType + '-' + epochLabel[i] + '-mean-vels-{}.csv'
                epoch_meanVel_file = epoch_meanVel_file.format(ID)
                epoch_meanVel[i].to_csv(epoch_meanVel_file)

                #Epoch Median velocity
                epoch_medVel[i] = srf.get_meds(epc,epochLabel[i],nEpochEvents)#
                epoch_medVel_file = outpath + '/' + trialType + '-' + epochLabel[i] + '-med-vels-{}.csv'
                epoch_medVel_file = epoch_medVel_file.format(ID)
                epoch_medVel[i].to_csv(epoch_medVel_file)

                #Epoch Velocity SEM
                epoch_semVel[i] = srf.get_SEMs(epc,epochLabel[i],nEpochEvents)#
                epoch_semVel_file = outpath + '/' + trialType + '-' + epochLabel[i] + '-SEM-vels-{}.csv'
                epoch_semVel_file = epoch_semVel_file.format(ID)
                epoch_semVel[i].to_csv(epoch_semVel_file)
                # print(type(epoch_semVel[i]))

                #Epoch freezing
                epochFreezing[i], epochFTs[i] = srf.get_freezing(epc,nEpochEvents,freezeThresh, binSize)

                #Epoch Darting
                epochDarting[i], epochDTs[i] = srf.get_darting(epc,nEpochEvents,dartThresh, binSize)

                #Epoch Plots
                srf.plot_outputs(anim, ID, trialTypeFull, outpath, trialType, nEpochEvents, epochFTs[i], epochDTs[i])  #Needs to be updated to insert

                # print(epochDTs[i])
                for m in range(0,len(derivedEpoch_list)):
                    if(derivedEpoch_list[m])=='':
                        continue;

                    #Get derived-epoch data frame
                    dEpoch_df = srf.find_delim_based_time(anim, nEpochEvents, epochLabel[i], int(derivedEpochTiming_list[m][0]), int(derivedEpochTiming_list[m][1]), int(derivedEpochTiming_list[m][2]))
                    #Derived-epoch max velocity
                    dEpoch_maxVel[i][m] = srf.get_top_vels(dEpoch_df,1,nEpochEvents)#
                    dEpoch_maxVel_file = outpath + '/' + trialType + '-' + epochLabel[i] + '_' + derivedEpoch_list[m] + '-max-vels-{}.csv'
                    # dEpoch_maxVel_file = dEpoch_maxVel_file.format(ID)

                    # print('derived epoch')

                    #Derived-epoch Mean velocity
                    dEpoch_meanVel[i][m] = srf.get_means(dEpoch_df,derivedEpoch_list[m],nEpochEvents)
                    dEpoch_meanVel_file = outpath + '/' + trialType + '-' + epochLabel[i] + '_' + derivedEpoch_list[m] + '-mean-vels-{}.csv'
                    dEpoch_meanVel_file = dEpoch_meanVel_file.format(ID)
                    # dEpoch_meanVel[i][m].to_csv(dEpoch_meanVel_file)

                    #Derived-epoch Median velocity
                    dEpoch_medVel[i][m] = srf.get_meds(dEpoch_df,derivedEpoch_list[m],nEpochEvents)#
                    dEpoch_medVel_file = outpath + '/' + trialType + '-' + epochLabel[i]+ '_' + derivedEpoch_list[m] + '-med-vels-{}.csv'
                    dEpoch_medVel_file = dEpoch_medVel_file.format(ID)
                    # dEpoch_medVel[i][m].to_csv(dEpoch_medVel_file)

                    #Derived-epoch Velocity SEM
                    dEpoch_semVel[i][m] = srf.get_SEMs(dEpoch_df,derivedEpoch_list[m],nEpochEvents)#
                    dEpoch_semVel_file = outpath + '/' + trialType + '-' + epochLabel[i]+ '_' + derivedEpoch_list[m] + '-SEM-vels-{}.csv'
                    dEpoch_semVel_file = dEpoch_semVel_file.format(ID)
                    # dEpoch_semVel[i][m].to_csv(dEpoch_semVel_file)

                    #Derived-epoch freezing

                    dEpochFreezing[i][m], dEpochFTs[i][m] = srf.get_freezing(dEpoch_df,nEpochEvents,freezeThresh, binSize)

                    #Derived-epoch Darting

                    dEpochDarting[i][m], dEpochDTs[i][m] = srf.get_darting(dEpoch_df,nEpochEvents,dartThresh, binSize)

                    #Derived-epoch Plots
                    # srf.plot_outputs(anim, ID, trialTypeFull, outpath, trialType, nEpochEvents, dEpochFTs[i][m], dEpochDTs[i][m])  #Needs to be updated to insert

            #Concatinate the full mean file per-animal
            # print(dEpoch_meanVel[0][0])
            # print('m=1')
            # print(dEpoch_meanVel[0][1])
            # print('m=2')
            # print(dEpoch_meanVel[0][2])
            allMeans = pd.concat([epoch_meanVel[0]], axis=1)
            for i in range(0,len(dEpoch_meanVel)):
                allMeans = pd.concat([allMeans,epoch_meanVel[i]], axis=1)
                for j in range(0,len(dEpoch_meanVel[i])):
                    allMeans = pd.concat([allMeans, dEpoch_meanVel[i][j]],axis=1)

            meanOutFile = outpath + '/' + trialType + '-mean-vels-{}.csv'
            meanOutFile=meanOutFile.format(ID)
            allMeans.to_csv(meanOutFile)

            #Concatinate the full median file per-animal
            # allMedians = pd.concat([epoch_medVel, dEpoch_medVel],axis=1)
            allMedians = pd.concat([epoch_medVel[0]], axis=1)
            for i in range(0,len(epoch_medVel)):
                allMedians = pd.concat([allMedians,epoch_medVel[i]], axis=1)
                for j in range(0,len(dEpoch_medVel[i])):
                    allMedians = pd.concat([allMedians, dEpoch_medVel[i][j]],axis=1)
            medOutFile = outpath + '/' + trialType + '-median-vels-{}.csv'
            medOutFile=medOutFile.format(ID)
            allMedians.to_csv(medOutFile)

            #Concatinate the full SEM file per-animal
            # allsem = pd.concat([epoch_semVel, dEpoch_semVel],axis=1)
            allsem = pd.concat([epoch_semVel[0]], axis=1)
            for i in range(0,len(epoch_semVel)):
                allsem = pd.concat([allsem,epoch_semVel[i]], axis=1)
                for j in range(0,len(dEpoch_semVel[i])):
                    allsem = pd.concat([allsem, dEpoch_semVel[i][j]],axis=1)
            semOutFile = outpath + '/' + trialType + '-SEM-vels-{}.csv'
            semOutFile=semOutFile.format(ID)
            allsem.to_csv(semOutFile)

            #Concatinate the full freezing file per-animal
            # allFreeze = pd.concat([epochFreezing, dEpochFreezing],axis=1)
            allFreeze = pd.concat([epochFreezing[0]], axis=1)
            for i in range(0,len(epochFreezing)):
                allFreeze = pd.concat([allFreeze,epochFreezing[i]], axis=1)
                for j in range(0,len(dEpochFreezing[i])):
                    allFreeze = pd.concat([allFreeze, dEpochFreezing[i][j]],axis=1)
            freezeOutFile = outpath + '/' + trialType + '-Freezing-{}.csv'
            freezeOutFile=freezeOutFile.format(ID)
            allFreeze.to_csv(freezeOutFile)

            #Concatinate the full darting file per-animal
            # allDart = pd.concat([epochDarting, dEpochDarting],axis=1)
            allDart = pd.concat([epochDarting[0]], axis=1)
            for i in range(0,len(epochDarting)):
                allDart = pd.concat([allDart,epochDarting[i]], axis=1)
                for j in range(0,len(dEpochDarting[i])):
                    allDart = pd.concat([allDart, dEpochDarting[i][j]],axis=1)
            dartOutFile = outpath + '/' + trialType + '-Darting-{}.csv'
            dartOutFile=dartOutFile.format(ID)
            allDart.to_csv(dartOutFile)

    for k in range(0,len(trialType_list)):
        srf.compile_SR(trialType_list[k],epochNum_list[k], len(derivedEpoch_list), derivedEpoch_list, 'Darting', outpath, outpath2)
        srf.compile_SR(trialType_list[k],epochNum_list[k], len(derivedEpoch_list), derivedEpoch_list, 'Freezing', outpath, outpath2)


################################################################################################
## Execution
################################################################################################
if __name__ == '__main__':
    sg.Print(size=(180,25))
    sg.Print()
    mainWindow = sg.Window('ScaredyRat', layout_MainWindow)

    while True:
        event, values = mainWindow.read()
        if event == sg.WIN_CLOSED or event == 'Exit':	# if user closes window or clicks cancel
            break
        elif event == 'Select Input Folder':
            inpath = sg.popup_get_folder('Please select the directory from which to load the Ethovision-style data', initial_folder=inpath)
        elif event == 'Select Per-Animal Output Folder':
            outpath = sg.popup_get_folder('Please select the directory in which to output the files', initial_folder=outpath)
        elif event == 'Select Compiled Output Folder':
            outpath2 = sg.popup_get_folder('Please select the directory in which to output the Compiled files', initial_folder=outpath2)
        elif event == 'Sheet Settings':
            sheetWindow = sg.Window('Sheet Settings', layout_SheetWin)
            # Event Loop to process "events" and get the "values" of the inputs
            while True:
                sheet_event, sheet_values = sheetWindow.read()
                if sheet_event == sg.WIN_CLOSED or sheet_event == 'Cancel':	# if user closes window or clicks cancel
                    break

                if sheet_event == 'Ok':	# if user closes window or clicks cancel
                    raw_sheetSettings = sheet_values
                    break
            sheetWindow.close()
        elif event == 'Trial Settings':
            trialWindow = sg.Window('Trial Settings', layout_TrialWin)
            # Event Loop to process "events" and get the "values" of the inputs
            while True:
                trial_event, trial_values = trialWindow.read()
                if trial_event == sg.WIN_CLOSED or trial_event == 'Cancel':	# if user closes window or clicks cancel
                    break
                if trial_event == 'Ok':	# if user closes window or clicks cancel
                    raw_trialSettings = trial_values
                    break
            trialWindow.close()
        elif event == 'Epoch and Derived Epoch Settings':
            tmp1 = raw_epochSettings[2].split(',')
            for i in range(0,len(tmp1)):
                tmp1[i] = tmp1[i].lstrip()
            dEpochName = tmp1
            dEpochTime = []
            for iter01 in range(3,len(raw_epochSettings)):
                dEpochTime.append(raw_epochSettings[iter01])

            epochCol = [
                        [sg.Text('Epoch Base Label', size=(20,1)), sg.Input('Tone ', size=(40,1)), sg.Button('Help', key='EpochHelp')],
                        [sg.CBox('Space Present',pad=(180,0), default=True) ],
                        [sg.Text('')],
                        [sg.Text('Derived Epoch List', size=(20,1)), sg.Input('PreTone, Shock, PostShock', size=(40,1)), sg.Button('Help', key='dEpochHelp')],
                        ] + [dEpochTimeEntry(dEpochName[iter],dEpochTime[iter]) for iter in range(0,len(dEpochName))]
            layout_EpochWin = [ [sg.Text('Epoch Information')],
                                [sg.Column(epochCol)],
                                [sg.Button('Apply'), sg.Button('Ok'), sg.Button('Cancel')] ]

            epochWindow = sg.Window('Epoch Settings (Click Apply to update the derived epoch timing list)', layout_EpochWin)
            # Event Loop to process "events" and get the "values" of the inputs
            while True:
                epoch_event, epoch_values = epochWindow.read()
                if epoch_event == sg.WIN_CLOSED or epoch_event == 'Cancel':	# if user closes window or clicks cancel
                    break
                if epoch_event == 'Ok':	# if user clicks Ok
                    raw_epochSettings = epoch_values
                    break
                if epoch_event == 'Apply':	# if user clicks Apply to update the window
                    raw_epochSettings = epoch_values
                    # Rebuild
                    tmp1 = raw_epochSettings[2].split(',')
                    for i in range(0,len(tmp1)):
                        tmp1[i] = tmp1[i].lstrip()
                    dEpochName = tmp1

                    dEpochTime = []
                    for iter01 in range(3,len(raw_epochSettings)):
                        dEpochTime.append(raw_epochSettings[iter01])
                    if len(dEpochTime) < len(dEpochName):
                        for iter02 in range(len(dEpochTime), len(dEpochName)):
                            dEpochTime += ['0,-30,0']
                    if len(dEpochTime) > len(dEpochName):
                        for iter02 in range(len(dEpochName), len(dEpochTime)):
                            print(len(raw_epochSettings))
                            print(3+iter02)
                            print(raw_epochSettings[3+iter02])
                    epochCol = [
                                [sg.Text('Epoch Base Label', size=(20,1)), sg.Input('Tone ', size=(40,1)), sg.Button('Help', key='EpochHelp')],
                                [sg.CBox('Space Present',pad=(180,0), default=raw_epochSettings[1]) ],
                                [sg.Text('')],
                                [sg.Text('Derived Epoch List', size=(20,1)), sg.Input(raw_epochSettings[2], size=(40,1)), sg.Button('Help', key='dEpochHelp')],
                                ] + [dEpochTimeEntry(dEpochName[iter],dEpochTime[iter]) for iter in range(0,len(dEpochName))]
                    layout_EpochWin = [ [sg.Text('Epoch Information')],
                                        [sg.Column(epochCol)],
                                        [sg.Button('Apply'), sg.Button('Ok'), sg.Button('Cancel')] ]


                    epochWindow.close()
                    epochWindow = sg.Window('Epoch Settings', layout_EpochWin)
            epochWindow.close()
        elif event == 'View Settings':
            printSettings(inpath, outpath, raw_sheetSettings, raw_trialSettings, raw_epochSettings)

        elif event == 'Run':
            print(raw_sheetSettings)
            print(raw_trialSettings)
            print(raw_epochSettings)
            run_SR(inpath, outpath, raw_sheetSettings, raw_trialSettings, raw_epochSettings)
            sg.popup('ScaredyRat Complete')
        # elif event == 'Compile Files':
        #     compile_SR(trialType, num_dEpoch,dEpoch_list, behavior, inpath, outpath)
    mainWindow.close()