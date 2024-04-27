import time
import multiprocessing
import platform
import psutil
import numpy as np
import os
import subprocess
import speedtest
from tqdm import tqdm
import json
import sys


def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

def prime_numbers(numbers):
    return [n for n in numbers if is_prime(n)]


def test_memory_copy(n_elements: int):
    arr = np.arange(n_elements)
    copy_arr = arr.copy()
    return True

def memory_copy_multiple(n_elements, n_threads, n_runs):
    results = []
    for _ in range(n_runs):
        with multiprocessing.Pool(n_threads) as pool:
            tmp_results = pool.map(test_memory_copy, [n_elements for _ in range(n_threads)])
            results.extend(tmp_results)
    return results


class Vps:
    # General info
    provider: str
    location: str
    tier: str
    usd_per_month: float
    __info_gathered: bool = False
    __fully_benchmarked: bool = False

    # Cpu
    n_cores: int
    clock_speed_mhz: int
    cpu_name: str

    cpu_benchmark_score: int

    # Memory
    memory_mb: int
    memory_speed_mhz: int

    memory_benchmark_score: int

    # Disk
    disk_size_gb: int

    disk_benchmark_score: tuple[int, int]

    # Network
    inet_benchmark_score: tuple[int, int, float]

    def __init__(self, provider, tier, location, usd_per_month):
        self.provider = provider
        self.tier = tier
        self.location = location
        self.usd_per_month = usd_per_month
    
    def __str__(self) -> str:
        if not self.__info_gathered:
            return f"VPS: {self.provider} ({self.location}) - ${self.usd_per_month}/month\n"
        
        if not self.__fully_benchmarked:
            # only show machine info, no benchmark scores
            return f"VPS: {self.provider} ({self.location}) - ${self.usd_per_month}/month\n" + \
                f"  CPU: {self.n_cores} cores @ {self.clock_speed_mhz} MHz ({self.cpu_name})\n" + \
                f"  Memory: {self.memory_mb} MB @ {self.memory_speed_mhz} MHz\n" + \
                f"  Disk: {self.disk_size_gb} GB\n"
        
        else:
            return f"VPS: {self.provider} ({self.location}) - ${self.usd_per_month}/month\n" + \
                f"  CPU: {self.n_cores} cores @ {self.clock_speed_mhz} MHz ({self.cpu_name}) - {self.cpu_benchmark_score} primes/s\n" + \
                f"  Memory: {self.memory_mb} MB @ {self.memory_speed_mhz} MHz - {self.memory_benchmark_score} MB/s\n" + \
                f"  Disk: {self.disk_size_gb} GB - {self.disk_benchmark_score} MB/s\n" + \
                f"  Network: {self.inet_benchmark_score} MB/s\n"
        

    def get_info(self):
        self.n_cores = multiprocessing.cpu_count()
        try:
            self.clock_speed_mhz = int(psutil.cpu_freq().current)
        except:
            self.clock_speed_mhz = 3200
        self.cpu_name = platform.processor()

        self.memory_mb = psutil.virtual_memory().total // 1024 // 1024
        self.memory_speed_mhz = 3200


        self.disk_size_gb = psutil.disk_usage('/').total // 1024 // 1024 // 1024

        self.__info_gathered = True
    

    def benchmark(self):
        for i in tqdm(range(4), desc="Benchmarking"):
            match i:
                case 0:
                    self.benchmark_cpu()
                case 1:
                    self.benchmark_memory()
                case 2:
                    self.benchmark_disk()
                case 3:
                    self.benchmark_network()
        
        self.__fully_benchmarked = True

    def write_to_json(self, fp):
        if not self.__fully_benchmarked or not self.__info_gathered:
            print("Error: cannot write to json file, benchmark not completed")
            return
        
        if not fp.endswith(".json"):
            fp += ".json"
        
        data = {
            "provider": self.provider,
            "location": self.location,
            "usd_per_month": self.usd_per_month,
            "n_cores": self.n_cores,
            "clock_speed_mhz": self.clock_speed_mhz,
            "cpu_name": self.cpu_name,
            "cpu_benchmark_score": self.cpu_benchmark_score,
            "memory_mb": self.memory_mb,
            "memory_speed_mhz": self.memory_speed_mhz,
            "memory_benchmark_score": self.memory_benchmark_score,
            "disk_size_gb": self.disk_size_gb,
            "disk_benchmark_score": self.disk_benchmark_score,
            "inet_benchmark_score": self.inet_benchmark_score
        }

        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


    def benchmark_cpu(self):
        start = time.perf_counter()

        # multithreaded prime numbers
        number_range = range(0, 250_000)
        thread_data = []
        for i in range(self.n_cores):
            thread_data.append(number_range[i::self.n_cores])
        
        
        
        with multiprocessing.Pool(self.n_cores) as pool:
            results = pool.map(prime_numbers, thread_data)
        
        n_primes = sum(len(r) for r in results)

        end = time.perf_counter()
        elapsed = end - start

        primes_per_second = n_primes / elapsed

        self.cpu_benchmark_score = int(primes_per_second)



    def benchmark_memory(self):
        n_runs = 10
        n_elements_per_mb = 50_000
        n_elements = n_elements_per_mb * self.memory_mb
        n_elements_per_thread = n_elements // self.n_cores
        n_threads = self.n_cores

        start = time.perf_counter()
        results = memory_copy_multiple(n_elements_per_thread, n_threads, n_runs)
        end = time.perf_counter()
        elapsed = end - start

        time_per_run = elapsed / n_runs
        
        score = n_elements / time_per_run / n_elements_per_mb
        self.memory_benchmark_score = int(score)



    def benchmark_disk(self):
        MB = 1024 * 1024
        GB = 1024 * MB

        file_size = 1 * GB

        file_bin = b"hello!"
        file_bin += b"\x00" * (file_size - len(file_bin))
        file_bin += b"world!"

        file_path = "testfile.bin"
        n_runs = 10

        t_write = 0
        t_read = 0

        for _ in range(n_runs):
            t_start_write = time.perf_counter()

            with open(file_path, "wb") as f:
                f.write(file_bin)
            
            t_end_write = time.perf_counter()

            t_write += t_end_write - t_start_write

            t_start_read = time.perf_counter()

            with open(file_path, "rb") as f:
                file_bin_out = f.read()
            
            t_end_read = time.perf_counter()

            t_read += t_end_read - t_start_read

            if file_bin != file_bin_out:
                print("Error: file read/write mismatch")
                return
            
            os.remove(file_path)

        

        
        
        write_speed = int(file_size / (t_write / n_runs) / MB)
        read_speed = int(file_size / (t_read / n_runs) / MB)

        self.disk_benchmark_score = (read_speed, write_speed)


    def benchmark_network(self):
        s = speedtest.Speedtest()
        s.get_servers([])
        s.get_best_server()
        s.download(threads=self.n_cores)
        s.upload(threads=self.n_cores)
        s.results.share()

        results_dict = s.results.dict()
        downloads_bit_p_s = results_dict["download"]
        uploads_bit_p_s = results_dict["upload"]
        latency = results_dict["server"]["latency"]

        download_speed = int(downloads_bit_p_s / 1024 / 1024)
        upload_speed = int(uploads_bit_p_s / 1024 / 1024)

        self.inet_benchmark_score = (download_speed, upload_speed, latency)


if __name__ == "__main__":
    argc = len(sys.argv)
    # args:
    # python benchmark_core.py [provider] [tier] [location] [usd_per_month]
    if argc != 5:
        print("Usage: python3 benchmark_core.py [provider] [tier] [location] [usd_per_month]")
        print("Example: python3 benchmark_core.py AWS t2.micro us-east-1 10.0")
        exit(1)
    
    provider = sys.argv[1]
    tier = sys.argv[2]
    location = sys.argv[3]
    usd_per_month = float(sys.argv[4])

    vps = Vps(provider, tier, location, usd_per_month)
    vps.get_info()
    vps.benchmark()
    fp = f"{provider}_{tier}_{location}.json"
    vps.write_to_json(fp)