# Waste Detection and Classification System

This project implements an AI-powered waste detection and classification system using deep learning. The system can detect waste in images and classify them into different categories, making it useful for automated waste sorting and recycling applications.

## Features

- Waste Detection: Identifies waste objects in images
- Waste Classification: Categorizes waste into multiple categories including:
  - Plastic
  - Glass
  - Wood
  - Metal
  - Paper
  - Other

## Project Structure

```
├── app.py                         # Streamlit web application
├── waste_detection_model.py       # Waste detection model implementation
├── waste_type_model.py           # Waste classification model implementation
├── train_waste_classifier.py      # Training script for classifier
├── train_waste_classifier_v2.py   # Updated training script
├── prepare_dataset.py            # Dataset preparation script
├── prepare_dataset_v2.py         # Enhanced dataset preparation
├── reduce_dataset.py             # Dataset optimization script
├── test_app_integration.py       # Integration tests
├── test_model_prediction.py      # Model prediction tests
└── datasets/
    ├── waste_categories_data/    # Classification dataset
    ├── waste_data/              # Detection dataset
    └── sample_data/             # Sample test data
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Information

The project uses three optimized datasets:
- waste_categories_data (105MB): Contains categorized waste images for classification
- waste_data (235MB): Contains images for waste detection training
- sample_data (1.1MB): Contains sample images for testing

Note: All datasets have been optimized to stay within GitHub's storage limits while maintaining functionality.

## Usage

### Running the Web Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload an image through the web interface to:
   - Detect waste objects in the image
   - Classify the type of waste detected

### Training Models

To train the waste classification model:
```bash
python train_waste_classifier.py
# or use the enhanced version
python train_waste_classifier_v2.py
```

### Preparing Datasets

To prepare or optimize datasets:
```bash
# For initial dataset preparation
python prepare_dataset.py

# For enhanced dataset preparation
python prepare_dataset_v2.py

# To reduce dataset size while maintaining quality
python reduce_dataset.py
```

## Model Architecture

- Waste Detection: Uses a deep learning model to identify waste objects in images
- Waste Classification: Implements a CNN-based classifier to categorize waste into different categories
- Both models are optimized for efficient inference while maintaining high accuracy

## Testing

The project includes comprehensive tests:
```bash
# Run integration tests
python test_app_integration.py

# Run model prediction tests
python test_model_prediction.py
```

## Requirements

Core dependencies include:
- Python 3.x
- TensorFlow
- OpenCV
- Streamlit
- NumPy
- scikit-learn
- PIL

See `requirements.txt` for a complete list of dependencies.

## Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or suggestions
- Submit pull requests for improvements
- Help with documentation or testing

## License

[Add your chosen license here]