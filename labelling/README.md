
## Project Structure

- `labelling.py`: Main script for classifying HTTP requests.
- `requirements.txt`: List of dependencies.
- `.env`: Environment variables for API keys.
- `data/evernote_test.csv`: Sample input data for testing.

## Setup

1. Clone the repository.
2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
3. Create a `.env` file in the root directory with your API keys:
    ```dotenv
    OPENAI_API_KEY=your_openai_api_key
    GOOGLE_API_KEY=your_google_api_key
    ```

## Usage

1. Prepare your input CSV file with HTTP request data.
2. Run the `labelling.py` script:
    ```sh
    python labelling.py
    ```
3. The script will classify the requests and save the results back to the CSV file.

## Configuration

- `openai_model`: The model name to use for OpenAI API (e.g., `gpt-4o-mini`).
- `gemini_model`: The model name to use for Google Gemini API (e.g., `gemini-2.0-flash-thinking-exp`).
- `batch_size`: Number of rows to process before saving progress (e.g., `10`).
- `test_rows`: Number of rows to read for testing (e.g., `20`).
- `use_openai`: Set to `True` to use OpenAI API, `False` to use Google Gemini API.


## Example

To classify the first 20 rows of `data/evernote_test.csv` using Google Gemini API:
```sh
python labelling.py
```

## Results

The script will output the classified services and activities, and save the results in the input CSV file.

## Dependencies

- pandas
- google-generativeai
- openai
- dotenv
