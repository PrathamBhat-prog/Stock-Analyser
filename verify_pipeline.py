import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.pipelines.inference_pipeline import StockAnalysisPipeline
import traceback

def verify_pipeline():
    print("Initializing pipeline...")
    try:
        pipeline = StockAnalysisPipeline()
        print("Pipeline initialized.")
        
        print("Running pipeline for MSFT...")
        result = pipeline.run("MSFT", "1mo")
        print("Pipeline run successful!")
        print("Decision:", result.get("final_decision"))
        
    except Exception:
        print("Pipeline failed.")
        traceback.print_exc()

if __name__ == "__main__":
    verify_pipeline()
