# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['SR_GUI.py', 'src\\sr_functions.py'],
             pathex=['./src', 'D:\\GITHUB\\ScaredyRat_ver2'],
             binaries=[],
             datas=[],
             hiddenimports=['tkinter,scipy,numpy,pandas,matplotlib,seaborn,PySimpleGUI'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='ScaredyRat',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
