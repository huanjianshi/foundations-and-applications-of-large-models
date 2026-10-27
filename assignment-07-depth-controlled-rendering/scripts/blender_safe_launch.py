"""Run existing Blender with the Windows CPU identifier restored in its child environment.

No installation, registry, preferences, or global environment variables are changed.
Example: python blender_safe_launch.py -- --background --factory-startup --python probe.py
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import winreg
import shutil

def blender_environment():
    env=os.environ.copy()
    if not env.get('PROCESSOR_IDENTIFIER'):
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            identifier=winreg.QueryValueEx(key,'Identifier')[0]
            vendor=winreg.QueryValueEx(key,'VendorIdentifier')[0]
        env['PROCESSOR_IDENTIFIER']=f'{identifier}, {vendor}'
    return env

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender',type=Path,default=Path(os.environ.get('BLENDER_EXE','blender')))
    args,remaining=parser.parse_known_args()
    if remaining and remaining[0]=='--':remaining=remaining[1:]
    if not args.blender.is_file() and not shutil.which(str(args.blender)):parser.error(f'Blender not found: {args.blender}')
    result=subprocess.run([str(args.blender),*remaining],env=blender_environment())
    return result.returncode

if __name__=='__main__':sys.exit(main())
