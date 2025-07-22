# Tower Jump Analyzer

## Overview
The Tower Jump Analyzer is a Python application designed to analyze location data and detect anomalies known as "tower jumps." It processes input data, calculates distances and speeds, and generates reports on the findings. This tool is useful for understanding movement patterns and ensuring data quality in location tracking systems.

## Installation
To set up the Tower Jump Analyzer, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/tower-jump-analyzer.git
   cd tower-jump-analyzer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the Tower Jump Analyzer, execute the following command in your terminal:

```
python -m src.main
```

Make sure that the `CarrierData.csv` file is located in the `localization` directory. The application will load the data, analyze it for tower jumps, and generate a report.

## Project Structure
```
tower-jump-analyzer/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── analyzer.py
│   └── utils.py
├── localization/
│   └── CarrierData.csv
├── requirements.txt
├── README.md
└── setup.py
```

## Contributing
Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
