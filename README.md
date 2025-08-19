# TDS Data Analyst Agent

A fully compliant Python API for automated data analysis and visualization, designed for evaluation systems.

## Features

- **Multi-format Data Processing**: Handles CSV, Excel, JSON, Parquet, ZIP, TAR.GZ files
- **Web Scraping**: Automated data collection from web sources
- **AI-Powered Analysis**: Uses AI Pipe for intelligent reasoning and code generation
- **Visualization**: Generates compliant plots with regression analysis
- **Secure File Handling**: Safe extraction and processing of compressed archives
- **Fast Response**: Completes analysis within 180 seconds

## API Endpoint

### POST /api/

Accepts multipart form data with:
- `questions.txt` (required): Problem statement and analysis requirements
- Additional files: Data files in various formats (.csv, .xlsx, .json, .parquet, .zip, .tar.gz, etc.)

### Response Formats

**Case A - 4-element JSON Array:**
```json
[1, "Titanic", 0.485782, "data:image/png;base64,..."]
```

**Case B - JSON Object:**
```json
{
  "Which high court disposed the most cases from 2019 - 2022?": "Delhi High Court",
  "What's the regression slope ...": "0.234",
  "Plot the year ...": "data:image/webp;base64,..."
}
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export OPENAI_API_KEY="your_ai_pipe_token"
export OPENAI_BASE_URL="https://aipipe.org/openai/v1"
```

## Usage

### Local Development
```bash
python run.py
```

The API will be available at `http://localhost:8000`

### Testing
```bash
curl -X POST "http://localhost:8000/api/" \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv"
```

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set the following:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run.py`
   - **Environment Variables**:
     - `OPENAI_API_KEY`: Your AI Pipe token
     - `OPENAI_BASE_URL`: `https://aipipe.org/openai/v1`

## Supported File Types

- **Text**: .txt
- **Spreadsheets**: .csv, .xlsx, .xls
- **Data**: .json, .parquet
- **Images**: .png, .jpg, .jpeg
- **Archives**: .zip, .tar.gz, .tgz
- **Documents**: .pdf

## Technical Stack

- **Framework**: FastAPI + Uvicorn
- **Data Processing**: Pandas, NumPy, DuckDB
- **Visualization**: Matplotlib, Scikit-learn
- **Web Scraping**: Requests, BeautifulSoup4
- **AI Integration**: OpenAI (via AI Pipe)
- **File Handling**: Pillow, aiofiles

## Performance Features

- Asynchronous file processing
- Efficient memory usage for large datasets
- Automatic image compression (<100KB)
- Graceful error handling
- Deterministic output generation

## Security

- Secure file extraction with path validation
- Temporary file cleanup
- Input sanitization
- Safe default responses for edge cases

## License

MIT License - see LICENSE file for details


## Author : Sana Bint Salim
