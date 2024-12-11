import requests
import time
import statistics
from typing import List, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import argparse
from math import ceil

@dataclass
class BenchmarkConfig:
    total_requests: int = 50    
    max_batch_size: int = 64  
    concurrent_requests: int = 5 
    warm_up_iterations: int = 2 

TEST_MESSAGES = [
    "Open source for the win! 🤗 Contributing to community projects is the way forward",
    "Neural networks are awesome and revolutionizing every industry sector imaginable",
    "FastAPI makes servers easy and scalable with great documentation",
    "Programming is fun especially when solving complex problems",
    "AI is the future and will transform how we live and work",
    "Validation complete with all test cases passing successfully",
    "Vectors are neat and essential for modern machine learning",
    "Transformer models have completely changed the NLP landscape forever",
    "Computer vision is enabling unprecedented applications",
    "Foundation models are reshaping AI development worldwide"
]

class EmbeddingBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.url = "http://localhost:11434/api/embed"
        self.model = "intfloat/multilingual-e5-small"
        
    def generate_batch_messages(self) -> List[str]:
        """Generate a batch of messages, repeating TEST_MESSAGES as needed."""
        repeats = ceil(self.config.max_batch_size / len(TEST_MESSAGES))
        messages = TEST_MESSAGES * repeats
        return messages[:self.config.max_batch_size]

    def send_request(self) -> Dict:
        """Send a single request with batch_size messages."""
        messages = self.generate_batch_messages()
        payload = {
            "model": self.model,
            "documents": [{"content": message} for message in messages]
        }
        
        start_time = time.time()
        try:
            response = requests.post(self.url, json=payload, timeout=30)
            response.raise_for_status()
            duration = time.time() - start_time
            return {
                "success": True,
                "duration": duration,
                "status_code": response.status_code,
                "num_messages": len(messages)
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e),
                "num_messages": len(messages)
            }

    def run_batch_benchmark(self) -> List[Dict]:
        """Run benchmark with multiple batched requests concurrently."""
        results = []
        total_batches = self.config.total_requests
        
        print(f"\nRunning {total_batches} requests with {self.config.max_batch_size} documents each...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.config.concurrent_requests) as executor:
            futures = [executor.submit(self.send_request) for _ in range(total_batches)]
            
            # Process results as they complete
            for i, future in enumerate(futures, 1):
                result = future.result()
                results.append(result)
                
                # Print progress every 10%
                if i % (total_batches // 10) == 0 or i == total_batches:
                    progress = (i / total_batches) * 100
                    elapsed = time.time() - start_time
                    rate = i / elapsed
                    print(f"Progress: {progress:.1f}% ({i}/{total_batches}) - Rate: {rate:.2f} requests/sec")
        
        return results

    def warm_up(self):
        """Perform warm-up requests."""
        print("\nPerforming warm-up requests...")
        for i in range(self.config.warm_up_iterations):
            result = self.send_request()
            print(f"Warm-up {i+1}/{self.config.warm_up_iterations}: "
                  f"{'Success' if result['success'] else 'Failed'} "
                  f"({result['duration']:.3f}s)")
        print("Warm-up complete.")

    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze benchmark results."""
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        if not successful_requests:
            return {"error": "No successful requests to analyze"}
            
        durations = [r["duration"] for r in successful_requests]
        total_duration = max(r["duration"] for r in results)  # Time from start to last completion
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "documents_per_request": self.config.max_batch_size,
            "total_documents_processed": len(successful_requests) * self.config.max_batch_size,
            "average_duration": statistics.mean(durations),
            "median_duration": statistics.median(durations),
            "p95_duration": sorted(durations)[int(len(durations) * 0.95)],
            "min_duration": min(durations),
            "max_duration": max(durations),
            "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0,
            "total_duration": total_duration,
            "requests_per_second": len(successful_requests) / total_duration,
            "documents_per_second": (len(successful_requests) * self.config.max_batch_size) / total_duration
        }

    def print_results(self, results: Dict):
        """Print formatted benchmark results."""
        print("\nBenchmark Results:")
        print("-----------------")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")

def main():
    parser = argparse.ArgumentParser(description='Embedding Service Benchmark')
    parser.add_argument('--requests', type=int, default=50,
                       help='Total number of requests to make')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Maximum number of documents per request')
    parser.add_argument('--concurrent', type=int, default=5,
                       help='Number of concurrent requests')
    parser.add_argument('--warm-up', type=int, default=2,
                       help='Number of warm-up iterations')
    args = parser.parse_args()

    config = BenchmarkConfig(
        total_requests=args.requests,
        max_batch_size=args.batch_size,
        concurrent_requests=args.concurrent,
        warm_up_iterations=args.warm_up
    )

    benchmark = EmbeddingBenchmark(config)
    
    # Perform warm-up
    benchmark.warm_up()
    
    # Run batch benchmark
    results = benchmark.run_batch_benchmark()
    analysis = benchmark.analyze_results(results)
    
    # Print results
    benchmark.print_results(analysis)

if __name__ == "__main__":
    main()
