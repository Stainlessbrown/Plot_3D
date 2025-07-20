# Plot 3D - macOS Builds

This repository contains the Plot 3D application configured for building on both macOS architectures: Apple Silicon (ARM) and Intel.

## About Plot 3D

Plot 3D is a Python application for 3D data visualization and analysis, featuring:
- 3D plotting capabilities
- K-means clustering
- Delta E color calculations
- Interactive data manipulation

## Automated Builds

This repository uses GitHub Actions to automatically build the application for both macOS architectures. The workflow:

1. Sets up Python 3.11 on both macOS ARM and Intel runners
2. Installs all dependencies including `ezodf`, `lxml`, and `odfpy`
3. Packages the application using PyInstaller with comprehensive hidden imports
4. Creates `.app` bundles with proper permissions
5. Uploads builds as separate artifacts for each architecture

## Download

To get the latest build:

1. Go to the [Actions](https://github.com/Stainlessbrown/Plot_3D/actions) tab
2. Click on the latest successful workflow run
3. Download the appropriate artifact for your Mac:
   - **Apple Silicon (M1/M2/M3)**: Download "Plot3D-macOS-ARM" artifact
   - **Intel Mac**: Download "Plot3D-macOS-Intel" artifact
4. Extract the zip file to get the `Plot3D.app` bundle

## Running the Application

### macOS Security Note

When running the app for the first time, macOS may block it with a security warning. To allow it:

1. **Right-click** on `Plot3D.app` and select "Open"
2. Click "Open" in the security dialog that appears
3. Or go to System Preferences → Security & Privacy → General and click "Open Anyway"

### Running from Terminal (for debugging)

If you encounter issues, you can run the app from Terminal to see error messages:

```bash
cd /path/to/Plot3D.app/Contents/MacOS
./Plot3D
```

## Dependencies

The application requires these main dependencies (automatically included in builds):
- Python 3.11
- NumPy
- Matplotlib
- Pandas
- scikit-learn
- ezodf (for Excel file handling)
- lxml (for XML processing)

## Building Locally

If you want to build locally instead of using GitHub Actions:

```bash
# Install dependencies
pip install -r requirements.txt
pip install ezodf lxml pyinstaller

# Build with PyInstaller
pyinstaller --onedir --windowed --name "Plot3D" Plot_3D.py
```

## Issues

If you encounter problems with the built application:

1. Check that you're downloading the latest artifact from a successful build
2. Try running from Terminal to see error messages
3. Ensure you've allowed the app in macOS Security settings
4. File permissions should be automatically fixed, but you can manually run:
   ```bash
   chmod +x Plot3D.app/Contents/MacOS/Plot3D
   ```
