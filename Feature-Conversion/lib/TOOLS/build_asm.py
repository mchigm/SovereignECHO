"""
Assembly Build Script
Automates the compilation of x86_64 assembly files
"""

import os
import subprocess
import platform
import sys


def find_nasm():
    """Find NASM assembler in PATH"""
    try:
        result = subprocess.run(['nasm', '-v'], capture_output=True, text=True)
        if result.returncode == 0:
            return 'nasm'
    except FileNotFoundError:
        pass
    
    # Common installation paths
    if platform.system() == "Windows":
        common_paths = [
            r"C:\Program Files\NASM\nasm.exe",
            r"C:\Program Files (x86)\NASM\nasm.exe",
            r"C:\nasm\nasm.exe"
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
    
    return None


def find_linker():
    """Find Microsoft linker (link.exe)"""
    try:
        result = subprocess.run(['link', '/?'], capture_output=True, text=True)
        if result.returncode == 0:
            return 'link'
    except FileNotFoundError:
        pass
    
    return None


def build_windows():
    """Build assembly for Windows x64"""
    print("Building for Windows x64...")
    
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    asm_file = os.path.join(src_dir, "bandpass_filter.asm")
    obj_file = os.path.join(src_dir, "bandpass_filter.obj")
    dll_file = os.path.join(src_dir, "bandpass_filter.dll")
    
    # Check if source file exists
    if not os.path.exists(asm_file):
        print(f"Error: Source file not found: {asm_file}")
        return False
    
    # Find NASM
    nasm = find_nasm()
    if not nasm:
        print("Error: NASM assembler not found!")
        print("Please install NASM from https://www.nasm.us/")
        return False
    
    print(f"Using NASM: {nasm}")
    
    # Assemble
    print(f"Assembling {asm_file}...")
    cmd = [nasm, '-f', 'win64', asm_file, '-o', obj_file]
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Assembly failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    print(f"✓ Created object file: {obj_file}")
    
    # Find linker
    linker = find_linker()
    if not linker:
        print("Warning: Microsoft linker not found!")
        print("Please run from Visual Studio Developer Command Prompt")
        print("Or manually link with:")
        print(f"  link /DLL /OUT:{dll_file} {obj_file} msvcrt.lib")
        return False
    
    print(f"Using linker: {linker}")
    
    # Link
    print(f"Linking {dll_file}...")
    cmd = [
        linker,
        '/DLL',
        f'/OUT:{dll_file}',
        obj_file,
        'msvcrt.lib',      # C runtime
        'legacy_stdio_definitions.lib'
    ]
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Linking failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    print(f"✓ Created DLL: {dll_file}")
    return True


def build_linux():
    """Build assembly for Linux x64"""
    print("Building for Linux x64...")
    
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    asm_file = os.path.join(src_dir, "bandpass_filter.asm")
    obj_file = os.path.join(src_dir, "bandpass_filter.o")
    so_file = os.path.join(src_dir, "bandpass_filter.so")
    
    # Check if source file exists
    if not os.path.exists(asm_file):
        print(f"Error: Source file not found: {asm_file}")
        return False
    
    # Find NASM
    nasm = find_nasm()
    if not nasm:
        print("Error: NASM assembler not found!")
        print("Install with: sudo apt-get install nasm")
        return False
    
    # Assemble
    print(f"Assembling {asm_file}...")
    cmd = [nasm, '-f', 'elf64', asm_file, '-o', obj_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Assembly failed!")
        print(result.stderr)
        return False
    
    print(f"✓ Created object file: {obj_file}")
    
    # Link
    print(f"Linking {so_file}...")
    cmd = ['gcc', '-shared', '-o', so_file, obj_file, '-lm']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Linking failed!")
        print(result.stderr)
        return False
    
    print(f"✓ Created shared library: {so_file}")
    return True


def main():
    """Main build function"""
    print("=" * 60)
    print("Assembly Build Script")
    print("=" * 60)
    print()
    
    system = platform.system()
    
    if system == "Windows":
        success = build_windows()
    elif system == "Linux":
        success = build_linux()
    else:
        print(f"Unsupported platform: {system}")
        success = False
    
    print()
    if success:
        print("=" * 60)
        print("BUILD SUCCESSFUL!")
        print("=" * 60)
        print()
        print("You can now use the assembly functions from Python:")
        print("  from asm_interface import create_processor")
        print("  processor = create_processor()")
        print("  filtered = processor.apply_bandpass_filter(data, 300, 3000, 44100)")
    else:
        print("=" * 60)
        print("BUILD FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
