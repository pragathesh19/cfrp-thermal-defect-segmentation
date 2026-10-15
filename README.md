# CFRP Thermal Defect Segmentation

Streamlit application for CFRP thermal defect segmentation using a trained U-Net model.

## Run locally

1. Install Python.
2. Open the project folder in a terminal.
3. Create a virtual environment.
4. Install requirements:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   python -m streamlit run app.py
   ```

## Model

The trained model file is `unet_best.keras`.  
Input shape: 256 x 256 x 1.
