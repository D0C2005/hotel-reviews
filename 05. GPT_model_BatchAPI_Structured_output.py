import os
import pandas as pd
import json
from dotenv import load_dotenv
from openai import OpenAI
import time

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Load the dataset
# The dataset should have columns: "ID" and "raw_review"
df = pd.read_csv("GPT_SA_dataset.csv")


# We will process 1000 reviews per request
batch_size = 1000
num_reviews = len(df)
num_batches = (num_reviews // batch_size) + (1 if num_reviews % batch_size != 0 else 0)

# Define the structured output schema for a batch of reviews
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_analysis_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reviews": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "integer",
                                "description": "The ID of the review."
                            },
                            "sentiment_code": {
                                "type": "integer",
                                "enum": [0, 1],
                                "description": "Sentiment of the review: 0 for Negative, 1 for Positive."
                            }
                        },
                        "required": ["id", "sentiment_code"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["reviews"],
            "additionalProperties": False
        }
    }
}

# Create a .jsonl file with one request per batch of 1000 reviews
jsonl_file_path = "GPT_all_reviews_batch.jsonl"
with open(jsonl_file_path, 'w') as f:
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_reviews)
        batch_df = df.iloc[start_idx:end_idx]

        # Construct the prompt for this batch
        # We'll include instructions and then list all reviews with their IDs.
        prompt_lines = [
            "You are a marketing expert specialized in tourism.",
            "Analyze the following hotel reviews and determine their sentiment.",
            "Sentiment codes: 0 for Negative, 1 for Positive.",
            "Return the results as a JSON object with a 'reviews' key containing an array of objects.",
            "Each object in the 'reviews' array should have keys 'id' and 'sentiment_code'.",
            "The 'id' must exactly match the provided ID of the review."
            "",
            "Reviews:"
        ]

        for _, row in batch_df.iterrows():
            review_id = row['ID']
            review_text = row['review_text']
            prompt_lines.append(f"ID: {review_id}\nReview: {review_text}")

        prompt = "\n".join(prompt_lines)

        batch_request = {
            "custom_id": f"batch-{batch_idx+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "response_format": response_format,
            }
        }
        f.write(json.dumps(batch_request) + "\n")

print("JSONL file created with all batch requests.")



#%%

# Upload the JSONL file
with open(jsonl_file_path, 'rb') as file_obj:
    batch_input_file = client.files.create(file=file_obj, purpose="batch")

print(f"Uploaded file ID: {batch_input_file.id}")

# Create a batch job
batch_job = client.batches.create(
    input_file_id=batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "Sentiment analysis for hotel reviews with structured output"}
)

print(f"Batch job ID: {batch_job.id}")

# Function to check job status periodically
def check_job_status(batch_job_id, interval=60, timeout=86400):
    """Check the status of a batch job until it completes or times out."""
    start_time = time.time()
    while True:
        batch_status = client.batches.retrieve(batch_job_id)
        print(f"Job Status: {batch_status.status}")
        if batch_status.status == "completed":
            output_file_id = batch_status.output_file_id
            print(f"Output File ID: {output_file_id}")
            # Download results
            result = client.files.content(output_file_id)
            output_file_path = "batch_output_results.jsonl"
            with open(output_file_path, 'w') as f:
                f.write(result.text)
            print(f"Results saved to {output_file_path}")
            break
        elif batch_status.status in ["failed", "cancelled"]:
            print(f"Batch job {batch_job_id} failed or was cancelled.")
            break
        elif time.time() - start_time > timeout:
            print(f"Batch job {batch_job_id} timed out after {timeout} seconds.")
            break
        else:
            print(f"Waiting for {interval} seconds before checking again...")
            time.sleep(interval)

# Start checking the job status
check_job_status(batch_job.id)


#%%

# Process the output file into a CSV

import json
import pandas as pd
import re

# Define the JSONL file and output CSV file paths
jsonl_file_path = "batch_675d503fcc60819088559b27bd489774_output.jsonl"
csv_output_path = "predicted_sentiments.csv"


log_file_path = "regex_extraction_errors.log"

# Initialize lists to store results and errors
results = []
error_log = []

# Regular expression pattern to extract id and sentiment_code
pattern = r'\{"id":\s*(\d+),\s*"sentiment_code":\s*(\d+)\}'

# Read the JSONL file line by line
with open(jsonl_file_path, 'r') as f:
    for line in f:
        try:
            # Extract the "content" field using regex without parsing the full JSON
            content_match = re.search(r'"content":\s*"(.*?)"}', line)
            if content_match:
                content = content_match.group(1)

                # Replace escaped quotes for proper regex parsing
                content = content.replace('\\"', '"')

                # Use regex to extract all id and sentiment_code pairs from the "content" field
                matches = re.findall(pattern, content)
                for match in matches:
                    review_id, sentiment_code = match
                    results.append({
                        "ID": int(review_id),
                        "predicted_sentiment": int(sentiment_code)
                    })
            else:
                error_log.append(f"No content found in line: {line[:200]}\n")
        except Exception as e:
            error_log.append(f"Failed to process line: {line[:200]}\nError: {e}\n")

# Convert results to a DataFrame
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv(csv_output_path, index=False)

# Save errors to a log file
with open(log_file_path, 'w') as log_file:
    log_file.writelines(error_log)


print(f"Extracted results have been saved to {csv_output_path}")
print(f"Errors have been logged to {log_file_path}")


#%%
# Determine quality indicators for each category

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the CSV file path
csv_file_path = "GPT_SA_results.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Ensure the columns are correctly named and available
if not all(col in df.columns for col in ["sentiment", "predicted_sentiment"]):
    raise ValueError("CSV file must contain 'sentiment' and 'predicted_sentiment' columns.")

# Define category ranges
categories = {
    "Positive Raw": slice(0, 6000),
    "Negative Raw": slice(6000, 12000),
    "Positive Clean": slice(12000, 18000),
    "Negative Clean": slice(18000, 24000),
}

# Calculate and print metrics for each category
for category, indices in categories.items():
    predicted = df.loc[indices, "predicted_sentiment"]
    actual = df.loc[indices, "sentiment"]

    # Accuracy is unaffected by the class labels
    accuracy = accuracy_score(actual, predicted)
    
    # Calculate metrics for positive sentiment (1) and negative sentiment (0)
    if "Positive" in category:
        pos_label = 1
    elif "Negative" in category:
        pos_label = 0
    else:
        raise ValueError("Category name must include 'Positive' or 'Negative'.")

    precision = precision_score(actual, predicted, pos_label=pos_label, zero_division=0)
    recall = recall_score(actual, predicted, pos_label=pos_label, zero_division=0)
    f1 = f1_score(actual, predicted, pos_label=pos_label, zero_division=0)

    print(f"\nMetrics for {category}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
