import os
import PyInstaller.__main__
import sys
sys.setrecursionlimit(50000)

PyInstaller.__main__.run([
    '--name=%s' %'ScaredyRat',
    '--onefile',
    '--hidden-import=%s' %'tkinter,scipy,numpy,pandas,matplotlib,seaborn,PySimpleGUI',
    '--clean',
 #   '--exclude-module=torch',
    '--upx-dir=%s\,%''./UPX/upx-3.96-win64',
    # '--windowed',
    '--paths=%s' %'./src',
    './SR_GUI.py',
    './src/sr_functions.py'
])