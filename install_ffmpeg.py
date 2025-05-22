#!/usr/bin/env python3
"""
Script to check if FFmpeg is installed and provide installation instructions.
"""

import os
import subprocess
import sys
import platform
import webbrowser

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible in PATH."""
    try:
        # Try to run ffmpeg -version
        process = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # If the command was successful, FFmpeg is installed
        if process.returncode == 0:
            version_info = process.stdout.split('\n')[0]
            print(f"✅ FFmpeg is installed: {version_info}")
            return True
        else:
            print("❌ FFmpeg is installed but returned an error.")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg is not installed or not in PATH.")
        return False

def provide_installation_instructions():
    """Provide platform-specific installation instructions for FFmpeg."""
    system = platform.system()
    
    print("\n=== FFmpeg Installation Instructions ===\n")
    
    if system == "Windows":
        print("For Windows:")
        print("1. Download FFmpeg from https://ffmpeg.org/download.html")
        print("   - Click on 'Windows' under 'Get packages & executable files'")
        print("   - Download the latest build (e.g., 'ffmpeg-release-essentials.zip')")
        print("2. Extract the ZIP file to a location like C:\\ffmpeg")
        print("3. Add the bin folder to your PATH environment variable:")
        print("   - Right-click on 'This PC' or 'My Computer' and select 'Properties'")
        print("   - Click on 'Advanced system settings'")
        print("   - Click on 'Environment Variables'")
        print("   - Under 'System variables', find and select 'Path', then click 'Edit'")
        print("   - Click 'New' and add the path to the bin folder (e.g., C:\\ffmpeg\\bin)")
        print("   - Click 'OK' on all dialogs")
        print("4. Restart your command prompt or terminal")
        print("5. Run 'ffmpeg -version' to verify the installation")
        
        # Ask if user wants to open the download page
        open_browser = input("\nWould you like to open the FFmpeg download page? (y/n): ")
        if open_browser.lower() == 'y':
            webbrowser.open("https://ffmpeg.org/download.html")
            
    elif system == "Darwin":  # macOS
        print("For macOS:")
        print("1. Install Homebrew if you don't have it: https://brew.sh/")
        print("2. Run: brew install ffmpeg")
        print("3. Run 'ffmpeg -version' to verify the installation")
        
    elif system == "Linux":
        print("For Ubuntu/Debian:")
        print("1. Run: sudo apt update")
        print("2. Run: sudo apt install ffmpeg")
        print("3. Run 'ffmpeg -version' to verify the installation")
        
        print("\nFor CentOS/RHEL:")
        print("1. Run: sudo yum install epel-release")
        print("2. Run: sudo yum install ffmpeg ffmpeg-devel")
        print("3. Run 'ffmpeg -version' to verify the installation")
    
    else:
        print(f"Unsupported platform: {system}")
        print("Please visit https://ffmpeg.org/download.html for installation instructions.")

def main():
    """Main function."""
    print("Checking if FFmpeg is installed...\n")
    
    if not check_ffmpeg():
        provide_installation_instructions()
        
        print("\n=== Why FFmpeg is Important ===")
        print("FFmpeg is essential for the accent detection application because it:")
        print("1. Processes and converts video files to extract audio")
        print("2. Normalizes audio for better transcription and accent detection")
        print("3. Handles various audio and video formats")
        print("4. Segments audio for analysis")
        print("\nWithout FFmpeg, the application falls back to using sample audio,")
        print("which significantly reduces the accuracy of accent detection.")
    else:
        print("\n✅ Your system is ready for accent detection!")
        print("The application will now be able to properly process audio from videos.")

if __name__ == "__main__":
    main()
