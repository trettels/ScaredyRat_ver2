import os
import PyInstaller.__main__
import sys
sys.setrecursionlimit(50000)

PyInstaller.__main__.run([
    '--name=%s' %'ScaredyRat',
    '--onefile',
    '--hidden-import=%s' %'tkinter',
    '--clean',
    '--windowed',
    '--paths=%s' %'./src',
    './SR_GUI.py',
    './src/sr_functions.py'
])