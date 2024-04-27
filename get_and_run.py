import os


ext_packages = [
    "psutil",
    "numpy",
    "speedtest-cli",
    "tqdm"
]

for package in ext_packages:
    os.system(f"pip install {package}")

provider = input("Provider: ")
tier = input("Tier: ")
location = input("Location: ")
usd_per_month = float(input("USD per month: "))

cmd = f"git clone https://github.com/MixoMax/vps-benchmark.git && cd vps-benchmark && python3 benchmark_core.py {provider} {tier} {location} {usd_per_month}"
os.system(cmd)

fp = f"./vps-benchmark/{provider}_{tier}_{location}.json"
os.system(f"cat {fp}")
os.system(f"cat {fp} | pbcopy")
