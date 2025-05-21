from setuptools import setup, find_packages

setup(
    name="accent-detector",
    version="0.1.0",
    description="A tool that analyzes a speaker's accent from a video",
    author="Sookchand Harripersad",
    author_email="sookchand38@gmail.com",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.31.0",
        "yt-dlp>=2023.11.16",
        "ffmpeg-python>=0.2.0",
        "openai>=1.6.1",
        "scikit-learn>=1.3.2",
        "transformers>=4.36.2",
        "torch>=2.2.0",
        "pydub>=0.25.1",
        "python-dotenv>=1.0.0",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
