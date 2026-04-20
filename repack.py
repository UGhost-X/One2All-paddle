#!/usr/bin/env python3
"""重新打包（用于增量更新后）"""
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
package_dir = PROJECT_ROOT / 'build' / 'One2All-Paddle'
zip_path = PROJECT_ROOT / 'dist' / 'One2All-Paddle-Deploy.zip'

# 确保 dist 目录存在
zip_path.parent.mkdir(exist_ok=True)

if zip_path.exists():
    zip_path.unlink()

print('Creating ZIP package...')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for file_path in package_dir.rglob('*'):
        if file_path.is_file():
            arcname = file_path.relative_to(package_dir)
            zf.write(file_path, arcname)

print(f'Package created: {zip_path}')
print(f'Package size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB')
