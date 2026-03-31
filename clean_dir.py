"""
===============================================================================
BISTRO Archive Standardization Pipeline
===============================================================================

Description:
    This script automates the migration and standardization of raw, unstructured 
    polarimetric data from the BISTRO (JCMT) survey into an IVOA-compliant 
    hierarchical archive. It mitigates directory entropy by separating pixel maps, 
    vector catalogs, error maps, and ancillary data into designated subdirectories.
    
Algorithm:
    1. Recursive Traversal: Scans the source directory tree for all files.
    2. Filtration: Excludes proprietary formats (.sdf) and system artifacts (.DS_Store).
    3. Categorization: Inspects filenames using predefined substring markers to 
       classify files into 'maps', 'catalogs', 'errors', or 'ancillary'.
    4. Nomenclature Standardization: Parses original filenames of FITS maps to 
       extract the Stokes parameter (I, Q, U) and observing band (450um / 850um). 
       Renames files to a unified format: <Target>_<Band>_<Stokes>_<Type>.fits.
    5. Safe Migration: Copies processed files to the target directory while 
       preserving the target object's root folder name. Ensures non-destructive 
       copying (avoids overwriting existing files).

Expected Input Structure:
    /Archive/<Object_Name>/[mixed unclassified files]
    
Resulting Output Structure:
    /BISTRO_Clean/<Object_Name>/
        ├── maps/       (Standardized FITS maps: I, Q, U)
        ├── catalogs/   (Vector polarization catalogs: .FIT)
        ├── errors/     (Variance/error maps: DI, DQ, DU)
        └── ancillary/  (Masks, weights, exposure times, multi-wavelength data)
        
===============================================================================
"""

import os
import shutil
from pathlib import Path

# Конфигурация путей
SRC_DIR = Path("/Users/ildana/Downloads/BISTRO")
DST_DIR = Path("/Users/ildana/Downloads/BISTRO")

# Категоризация на основе строковых маркеров в названиях файлов
CATEGORIES = {
    'maps': ['iext', 'qext', 'uext', '_i.fits', '_q.fits', '_u.fits', 'imap', 'qmap', 'umap'],
    'catalogs': ['cat', 'catalogue'],
    'errors': ['_di.', '_dq.', '_du.'],
    'ancillary': ['mask', 'exp_time', 'weights', 'herschel']
}

def determine_stokes_and_band(filename: str) -> tuple:
    """Извлекает параметр Стокса и предполагаемую длину волны из старого имени."""
    fn_lower = filename.lower()
    
    # Стоксов параметр
    stokes = "Unknown"
    if any(m in fn_lower for m in ['iext', '_i.', 'imap']): stokes = "I"
    elif any(m in fn_lower for m in ['qext', '_q.', 'qmap']): stokes = "Q"
    elif any(m in fn_lower for m in ['uext', '_u.', 'umap']): stokes = "U"
    
    # Диапазон (BISTRO по умолчанию использует 850 микрон, если не указано 450)
    band = "450um" if "450" in fn_lower else "850um"
    
    return stokes, band

def assemble_archive():
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Исходный архив не найден: {SRC_DIR}")
        
    for item in SRC_DIR.rglob('*'):
        if item.is_dir() or item.name == '.DS_Store' or item.suffix.lower() == '.sdf':
            continue # Игнорируем мусор и проприетарные форматы Starlink
            
        obj_name = item.relative_to(SRC_DIR).parts[0]
        if obj_name == 'Filtered_Herschel_Maps': # Обработка вложенной папки L43_Karoly_2023
            obj_name = item.relative_to(SRC_DIR).parts[1]
            
        old_name = item.name
        
        # Определение категории
        category = 'ancillary' # по умолчанию
        for cat, markers in CATEGORIES.items():
            if any(m in old_name.lower() for m in markers):
                category = cat
                break
                
        # Формирование нового имени для карт
        if category == 'maps' and item.suffix.lower() == '.fits':
            stokes, band = determine_stokes_and_band(old_name)
            new_name = f"{obj_name}_{band}_{stokes}_map.fits"
        elif category == 'catalogs':
            new_name = f"{obj_name}_vector_cat{item.suffix}"
        else:
            new_name = f"{obj_name}_{old_name}" # Сохраняем исходное имя для нетипичных файлов
            
        # Создание структуры и копирование
        target_dir = DST_DIR / obj_name / category
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = target_dir / new_name
        
        # Защита от перезаписи
        if not target_path.exists():
            shutil.copy2(item, target_path)
            print(f"[{obj_name}] {old_name} -> {category}/{new_name}")

if __name__ == "__main__":
    print(f"Запуск миграции из {SRC_DIR} в {DST_DIR}...")
    assemble_archive()
    print("Миграция завершена. Энтропия снижена.")

